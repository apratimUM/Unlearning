"""Greedy extraction of frozen probes or an arbitrary text prefix.

For fixed probes, score exact 50-token continuation against the frozen corpus.
For --text, generate a continuation without ground truth (no match score).
"""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from data import PackedSampler, file_sha, verify_data
from model import GPT, GPTConfig


def score(generated, target):
    if not target or len(generated) != len(target):
        raise ValueError('Invalid continuation lengths')
    lcp = 0
    for a, b in zip(generated, target):
        if a != b:
            break
        lcp += 1
    return dict(exact_match=(generated == target), longest_matching_prefix=lcp,
                aligned_token_accuracy=sum(a == b for a, b in zip(generated, target))/len(target))


def validate_probe(probe, sampler, documents, probe_length):
    k, within = probe['block_id'], probe['training_position']
    i = probe['document_index']
    if not 0 <= k < sampler.n_blocks or not 0 <= within < sampler.block:
        raise ValueError('Bad probe block')
    if not 0 <= i < len(documents):
        raise ValueError('Bad probe document')
    rec = documents[i]
    start = k*sampler.block + within
    ids = probe['tokens']
    if (len(ids) != probe_length or within+len(ids) > sampler.block or
        start != rec['offset']+probe['document_token_offset'] or
        probe['document_token_offset']+len(ids) > rec['text_tokens'] or
        rec['sha256'] != probe['document_sha256']):
        raise ValueError('Probe boundary or identity mismatch')
    actual = sampler.tokens[start:start+len(ids)].astype('int64').tolist()
    preceding = sampler.tokens[k*sampler.block:start].astype('int64').tolist()
    if ids != actual or ('preceding_context_ids' in probe and
                          probe['preceding_context_ids'] != preceding):
        raise ValueError('Probe does not match frozen corpus')
    return preceding


@torch.inference_mode()
def evaluate(model, probe, preceding, prefix_length, continuation_length, mode, device):
    prefix = probe['tokens'][:prefix_length]
    target = probe['tokens'][prefix_length:prefix_length+continuation_length]
    prompt = (preceding if mode == 'matched' else []) + prefix
    if len(prompt)+continuation_length > model.config.block_size:
        raise ValueError('Prompt + continuation longer than model context')
    prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)
    generated = model.generate_greedy(prompt_tensor, continuation_length)[0, len(prompt):].tolist()
    complete = torch.tensor([prompt+target], dtype=torch.long, device=device)
    logits, _ = model(complete[:, :-1])
    suffix = logits[:, len(prompt)-1:, :].float()
    y = complete[:, len(prompt):]
    nll = F.cross_entropy(suffix.reshape(-1, suffix.size(-1)), y.reshape(-1)).item()
    accuracy = (suffix.argmax(-1) == y).float().mean().item()
    return dict(**score(generated, target), mode=mode, prefix_length=prefix_length,
                continuation_length=continuation_length,
                evaluation_position=len(preceding) if mode == 'matched' else 0,
                teacher_forced_nll=nll, teacher_forced_accuracy=accuracy,
                prefix_ids=prefix, expected_ids=target, generated_ids=generated,
                prompt_ids=prompt)


def summarize(rows):
    result = {}
    for mode in sorted(set(row['mode'] for row in rows)):
        result[mode] = {}
        for group in ('train', 'val', 'train_exposed'):
            chosen = [r for r in rows if r['mode'] == mode and
                      (r['split'] == group if group != 'train_exposed' else
                       r['split'] == 'train' and r['probe_block_exposures'] > 0)]
            n = len(chosen)
            result[mode][group] = dict(n=n, exact_match_count=sum(r['exact_match'] for r in chosen),
                exact_match_rate=sum(r['exact_match'] for r in chosen)/n if n else None,
                mean_lcp=sum(r['longest_matching_prefix'] for r in chosen)/n if n else None)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='data/pile')
    p.add_argument('--prefix', type=int, default=50)
    p.add_argument('--continuation', type=int, default=50)
    p.add_argument('--mode', choices=['standalone', 'matched', 'both'], default='both')
    p.add_argument('--text', help='Optional arbitrary text prompt; no reference continuation')
    p.add_argument('--out')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--precision', choices=['float32', 'auto'], default='float32')
    a = p.parse_args()
    if min(a.prefix, a.continuation) < 1 or a.gpu < 0:
        p.error('Prefix/continuation must be positive and GPU index nonnegative')
    folder = Path(a.data)
    m = verify_data(folder)
    digest = file_sha(folder / 'manifest.json')
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ckpt['manifest_sha256'] != digest:
        raise ValueError('Checkpoint belongs to a different dataset')
    cfg = GPTConfig(**ckpt['config'])
    if cfg.block_size != m['block_size']:
        raise ValueError('Checkpoint block size mismatch')
    if torch.cuda.is_available():
        if a.gpu >= torch.cuda.device_count():
            raise ValueError('GPU index out of range')
        device = torch.device(f'cuda:{a.gpu}')
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(ckpt['model'])
    if a.precision == 'auto' and device.type == 'cuda' and torch.cuda.is_bf16_supported():
        model = model.to(dtype=torch.bfloat16)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(folder / 'tokenizer', local_files_only=True)
    if a.text is not None:
        ids = tok.encode(a.text, add_special_tokens=False)
        if not ids:
            p.error('Provided text prompt tokenizes to nothing')
        if len(ids)+a.continuation > cfg.block_size:
            p.error('Text prompt + requested continuation exceed model context length')
        prompt = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model.generate_greedy(prompt, a.continuation)[0, len(ids):].tolist()
        print('PROMPT:\n'+a.text+'\n\nGREEDY CONTINUATION:\n'+tok.decode(output))
        return
    if a.prefix+a.continuation > min(cfg.block_size, m['probe_length']):
        p.error('Prefix and continuation exceed prepared probe length or context')
    if not 0 <= ckpt['step'] or min(ckpt['signature']['batch'], ckpt['signature']['accum']) < 1:
        raise ValueError('Bad checkpoint exposure metadata')
    probes = [json.loads(line) for line in (folder/'probes.jsonl').read_text().splitlines()]
    if not probes:
        raise ValueError('No saved evaluation probes')
    samplers = {s: PackedSampler(folder, s, cfg.block_size, ckpt['signature']['seed'])
                for s in ('train', 'val')}
    try:
        consumed = ckpt['step'] * ckpt['signature']['batch'] * ckpt['signature']['accum']
        counts = samplers['train'].exposure_counts([r['block_id'] for r in probes if r['split']=='train'], consumed)
        records = {s: json.loads((folder/f'{s}_index.json').read_text()) for s in samplers}
        modes = ('standalone', 'matched') if a.mode == 'both' else (a.mode,)
        results = []
        for probe in probes:
            split = probe['split']
            if split not in samplers:
                raise ValueError('Invalid probe split')
            preceding = validate_probe(probe, samplers[split], records[split], m['probe_length'])
            for mode in modes:
                row = evaluate(model, probe, preceding, a.prefix, a.continuation, mode, device)
                row.update(split=split, document_index=probe['document_index'],
                           document_sha256=probe['document_sha256'], block_id=probe['block_id'],
                           checkpoint_step=ckpt['step'],
                           probe_block_exposures=counts[probe['block_id']] if split == 'train' else 0,
                           prefix=tok.decode(row['prefix_ids']), expected=tok.decode(row['expected_ids']),
                           generated=tok.decode(row['generated_ids']))
                results.append(row)
    finally:
        for sampler in samplers.values():
            sampler.close()
    checkpoint = Path(a.checkpoint)
    out = Path(a.out) if a.out else checkpoint.parent / f'{checkpoint.stem}_extraction_p{a.prefix}_c{a.continuation}_{a.mode}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    summary = dict(metrics=summarize(results), checkpoint_step=ckpt['step'],
                   checkpoint_sha256=file_sha(checkpoint), manifest_sha256=digest,
                   primary_metric='standalone greedy 50-token exact-match rate')
    out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('Saved:', out)


if __name__ == '__main__':
    main()
