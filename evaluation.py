#!/usr/bin/env python3
"""Evaluate scratch GPT-2 Small checkpoints on WikiText-103, LAMBADA, BLiMP, MMLU.

Compatible with train.py's full-state step_*.pt/latest.pt, model.py, runtime.py,
and checkpoints.py's evaluate_one(argparse.Namespace(...)) interface.
Checks schema-v3 checkpoint provenance, saved/legacy sampler identity and block exposure.

Model inference requires CUDA. Does not train or change weights.
Results are protocol-specific; Pile/benchmark contamination has not been ruled out.

Smoke test:
  python evaluate.py --checkpoint runs/gpt2_small_pile_full_1024/latest.pt \
      --tasks all --limit 10 --wikitext-max-tokens 2000 \
      --blimp-configs adjunct_island --mmlu-subjects abstract_algebra
Full evaluation:
  python evaluate.py --checkpoint runs/gpt2_small_pile_full_1024/latest.pt --tasks all
Across saved steps:
  python evaluate.py --checkpoint runs/gpt2_small_pile_full_1024 \
      --all-checkpoints --tasks wikitext --wikitext-max-tokens 2000
"""

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from data import SAMPLER_ORDER, verify_data

# Historical checkpoint identities, kept independent of current data.py.
LEGACY_FEISTEL_ORDER = 'feistel_cyclewalk_v1'
LIVE_SAMPLER_ORDER = 'sequential_committed_v1'
from runtime import require_cuda, choose_precision, autocast_context

WIKITEXT_REPO = 'Salesforce/wikitext'
LAMBADA_REPO = 'EleutherAI/lambada_openai'
BLIMP_REPO = 'nyu-mll/blimp'
MMLU_REPO = 'cais/mmlu'
TASKS = ('wikitext', 'lambada', 'blimp', 'mmlu')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--all-checkpoints', action='store_true')
    p.add_argument('--tasks', nargs='+', choices=(*TASKS, 'all'), default=list(TASKS))
    p.add_argument('--tokenizer', type=Path, default=None)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--device', default='auto', help='auto, cuda, or cuda:<selected --gpu>; no CPU/MPS')
    p.add_argument('--precision', choices=['auto', 'float32', 'float16', 'bfloat16'], default='auto')
    p.add_argument('--limit', type=int, default=0, help='Scored examples per BLiMP phenomenon/MMLU subject or LAMBADA; 0=all')
    p.add_argument('--wikitext-max-tokens', type=int, default=0, help='0=full')
    p.add_argument('--stride', type=int, default=0, help='WikiText stride; 0=half context')
    p.add_argument('--blimp-configs', nargs='*', default=None)
    p.add_argument('--mmlu-subjects', nargs='*', default=None)
    p.add_argument('--mmlu-shots', type=int, choices=range(6), default=0)
    p.add_argument('--lambada-max-new-tokens', type=int, default=16)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--trusted-checkpoint', action='store_true', help='Unsafe pickle fallback for YOUR trusted checkpoint only')
    a = p.parse_args()
    a.tasks = list(TASKS) if 'all' in a.tasks else list(dict.fromkeys(a.tasks))
    if a.all_checkpoints:
        if not a.checkpoint.is_dir():
            p.error('--all-checkpoints requires --checkpoint RUN_DIRECTORY')
        if a.output is not None:
            p.error('--output applies only to a single checkpoint')
    elif not a.checkpoint.is_file():
        p.error('Checkpoint file not found (use --all-checkpoints for a directory)')
    if min(a.gpu, a.limit, a.wikitext_max_tokens, a.stride) < 0 or a.lambada_max_new_tokens < 1:
        p.error('Invalid GPU index, limits, stride or LAMBADA generation cap')
    return a


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _read_checkpoint(path, trusted=False):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        if not trusted:
            raise RuntimeError(
                'Safe checkpoint loading failed. For YOUR OWN trusted checkpoint only, '
                'pass --trusted-checkpoint.') from exc
        return torch.load(path, map_location='cpu', weights_only=False)


def inspect_training_checkpoint(ckpt, *, tokenizer_override=None):
    """Read data provenance without needing a GPU or finalized manifest."""
    if not isinstance(ckpt, dict) or not {'model', 'config', 'signature', 'step', 'blocks_consumed'} <= ckpt.keys():
        raise ValueError('Checkpoint missing model/config/signature/step/blocks_consumed')
    cfg = GPTConfig(**ckpt['config'])
    if (cfg.block_size, cfg.vocab_size, cfg.n_layer, cfg.n_head, cfg.n_embd) != (1024, 50257, 12, 12, 768):
        raise ValueError('Expected GPT-2 Small from scratch, 1024 context and GPT-2 vocabulary')
    schema = int(ckpt.get('schema_version', -1))
    sig = ckpt['signature']
    if not isinstance(sig, dict):
        raise ValueError('Missing checkpoint signature dictionary')
    consumed = int(ckpt['blocks_consumed'])
    step = int(ckpt['step'])
    if consumed < 0 or step < 0:
        raise ValueError('Negative checkpoint progress')
    if schema == 3:
        sampler_order = sig.get('sampler_order')
        if sampler_order not in (SAMPLER_ORDER, LEGACY_FEISTEL_ORDER):
            raise ValueError(f'Unsupported prepared sampler: {sampler_order!r}')
        folder = Path(sig['data'])
        shuffled = bool(sig.get('shuffled', False))
        manifest_file = folder / ('manifest_shuffled.json' if shuffled else 'manifest.json')
        if not manifest_file.is_file():
            raise FileNotFoundError(manifest_file)
        digest = _sha256(manifest_file)
        if ckpt.get('manifest_sha256') != digest or sig.get('manifest_sha256') != digest:
            raise ValueError('Prepared checkpoint manifest checksum mismatch')
        manifest = verify_data(folder, shuffled=shuffled)
        n_blocks = int(manifest['train_blocks'])
        batch = int(sig['batch'])
        accum = int(sig['accum'])
        world = int(sig.get('world_size', 4))
        effective = batch * accum
        if effective <= 0 or world <= 0:
            raise ValueError('Invalid effective batch or world size')
        capacity = math.ceil(n_blocks / effective)
        if sampler_order == SAMPLER_ORDER:
            # Validate exactly the saved permutation used by the new training run.
            for filename, key in (
                (('train_chunk_order_shuffled.u32' if shuffled else 'train_chunk_order.u32'), 'chunk_order_sha256'),
                (('train_chunk_inverse_shuffled.u32' if shuffled else 'train_chunk_inverse.u32'), 'chunk_inverse_sha256'),
            ):
                path = folder / filename
                if not path.is_file() or _sha256(path) != sig.get(key):
                    raise ValueError(f'Chunk permutation missing or hash mismatch: {filename}')
            order_info = json.loads((folder / ('train_chunk_order_shuffled.json' if shuffled else 'train_chunk_order.json')).read_text())
            if (order_info.get('algorithm') != sampler_order or
                int(order_info['n_blocks']) != n_blocks or
                int(order_info['seed']) != int(sig['seed'])):
                raise ValueError('Saved chunk permutation metadata mismatch')
            remainder = n_blocks % effective
            if 0 < (remainder % batch) < world:
                capacity -= 1
        consumed = int(ckpt['blocks_consumed'])
        expected = min(step * effective, n_blocks)
        if consumed != expected or not 0 <= step <= int(sig['steps']) <= capacity:
            raise ValueError('Prepared checkpoint consumption/step mismatch')
        # The new safe final-update guard may deliberately leave a tiny tail.
        complete = step == capacity and int(sig['steps']) == capacity
        return dict(data_mode='prepared', schema_version=schema, data=folder,
                    manifest_sha256=digest, sampler_order=sampler_order,
                    sampler_seed=int(sig['seed']), blocks_consumed=consumed,
                    dataset_blocks=n_blocks, block_size=1024,
                    completed_full_pass=complete,
                    all_dataset_blocks_visited=consumed == n_blocks,
                    unvisited_blocks=n_blocks - consumed)
    if schema == 4 and ckpt.get('data_mode') == 'streaming':
        if sig.get('data_mode') != 'streaming' or sig.get('sampler_order') != LIVE_SAMPLER_ORDER or ckpt.get('sampler_order') != LIVE_SAMPLER_ORDER:
            raise ValueError('Streaming checkpoint sampler identity mismatch')
        folder = Path(sig['data'])
        args = sig.get('args', {})
        effective = int(args.get('batch', 0)) * int(args.get('accum', 0))
        if effective <= 0 or consumed > step * effective or (step > 0 and consumed <= (step - 1) * effective):
            raise ValueError('Streaming checkpoint has inconsistent block cursor')
        if step == 0 and consumed != 0:
            raise ValueError('Initial checkpoint cannot have consumed blocks')
        live_file = folder / 'live_state.json'
        if not live_file.is_file():
            raise FileNotFoundError(live_file)
        live = json.loads(live_file.read_text(encoding='utf-8'))
        if (live.get('source_revision') != sig.get('source_revision') or
            live.get('block_size') != 1024 or
            int(live.get('train_ready_blocks', -1)) < consumed or
            live.get('tokenizer') != 'gpt2'):
            raise ValueError('Streaming checkpoint does not match live source/watermark')
        digest = None
        n_blocks = None
        if (folder / 'manifest.json').is_file():
            manifest = verify_data(folder)
            if manifest.get('source_revision') != sig['source_revision']:
                raise ValueError('Final source revision mismatch')
            n_blocks = int(manifest['train_blocks'])
            digest = _sha256(folder / 'manifest.json')
            if consumed > n_blocks:
                raise ValueError('Consumed more blocks than dataset contains')
        if ckpt.get('completed_full_pass'):
            if (not live.get('finished') or n_blocks is None or consumed != n_blocks or
                not manifest.get('source_scan_complete')):
                raise ValueError('Completed full-pass claim is not verified')
        return dict(data_mode='streaming', schema_version=schema, data=folder,
                    manifest_sha256=digest, sampler_order=LIVE_SAMPLER_ORDER,
                    sampler_seed=None, blocks_consumed=consumed,
                    dataset_blocks=n_blocks, block_size=1024,
                    completed_full_pass=bool(ckpt.get('completed_full_pass')))
    raise ValueError(f'Unsupported checkpoint schema={schema}, mode={ckpt.get("data_mode")}')


def load_model_and_tokenizer(args):
    from transformers import AutoTokenizer
    gpu = getattr(args, 'gpu', 0)
    device_requested = getattr(args, 'device', 'auto')
    if device_requested not in ('auto', 'cuda', f'cuda:{gpu}'):
        raise ValueError('Evaluation requires --device auto, cuda, or matching cuda:N')
    ckpt = _read_checkpoint(args.checkpoint, getattr(args, 'trusted_checkpoint', False))
    metadata = inspect_training_checkpoint(ckpt)
    folder = metadata['data']
    tok_path = getattr(args, 'tokenizer', None) or (folder / 'tokenizer')
    tok = AutoTokenizer.from_pretrained(str(tok_path), use_fast=True,
        local_files_only=getattr(args, 'local_files_only', False))
    if not tok.is_fast or tok.vocab_size != 50257 or tok.eos_token_id != 50256:
        raise ValueError('Expected official fast GPT-2 BPE tokenizer with EOS=50256')
    device = require_cuda(gpu)
    precision = choose_precision(args.precision, device)
    model = GPT(GPTConfig(**ckpt['config']))
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(device).eval()
    return model, tok, device, precision, int(ckpt['step']), str(tok_path), metadata


@torch.inference_mode()
def conditional_loglikelihood(model, prefix, continuation, device, precision):
    """Sum log P(continuation | prefix), with a shared prompt and no reset."""
    if not prefix or not continuation or len(prefix) + len(continuation) > model.config.block_size:
        raise ValueError('Empty prefix/continuation or context overflow')
    ids = prefix + continuation
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    with autocast_context(precision):
        logits, _ = model(x)
    wanted = logits[0, len(prefix) - 1:].float()
    y = torch.tensor(continuation, dtype=torch.long, device=device)
    assert wanted.shape[0] == y.numel()
    return F.log_softmax(wanted, dim=-1).gather(-1, y[:, None]).sum().item()


@torch.inference_mode()
def loglikelihood_ids(model, ids, device, precision, bos_id=None):
    """Sum per-token likelihood; reset context after each nonoverlapping window."""
    ids = ([bos_id] if bos_id is not None else []) + list(ids)
    if len(ids) < 2:
        return 0.0, 0
    total = 0.0
    count = 0
    block = model.config.block_size
    for start in range(0, len(ids) - 1, block):
        seq = ids[start:start + block + 1]
        x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([seq[1:]], dtype=torch.long, device=device)
        with autocast_context(precision):
            logits, _ = model(x)
        lp = F.log_softmax(logits.float(), dim=-1)
        total += lp.gather(-1, y.unsqueeze(-1)).sum().item()
        count += y.numel()
    return total, count


def run_wikitext(model, tokenizer, device, precision, args):
    from datasets import load_dataset
    ds = load_dataset(WIKITEXT_REPO, 'wikitext-103-raw-v1', split='test')
    text = ''.join(s if s.endswith('\n') else s + '\n' for s in (r['text'] for r in ds))
    ids = tokenizer.encode(text, add_special_tokens=False)
    if args.wikitext_max_tokens:
        ids = ids[:args.wikitext_max_tokens]
    if len(ids) < 2:
        raise ValueError('Too few WikiText test tokens')

    block = model.config.block_size
    stride = args.stride or block // 2
    if not 1 <= stride < block:
        raise ValueError('WikiText stride must satisfy 1 <= stride < block_size')
    nll_sum = 0.0
    count = 0
    last_target = 0
    end = min(block + 1, len(ids))
    while True:
        begin = max(0, end - (block + 1))
        chunk = ids[begin:end]
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        with torch.inference_mode(), autocast_context(precision):
            logits, _ = model(x)
        first_target = max(last_target + 1, begin + 1)
        if first_target < end:
            chosen = logits[0, first_target - begin - 1:end - begin - 1].float()
            targets = torch.tensor(ids[first_target:end], dtype=torch.long, device=device)
            nll_sum += F.cross_entropy(chosen, targets, reduction='sum').item()
            count += targets.numel()
            last_target = end - 1
        if end == len(ids):
            break
        end = min(end + stride, len(ids))
    if count != len(ids) - 1:
        raise RuntimeError(f'WikiText coverage mismatch: {count} != {len(ids)-1}')
    mean_nll = nll_sum / count
    return dict(dataset=WIKITEXT_REPO, config='wikitext-103-raw-v1', split='test',
                total_tokens=len(ids), scored_tokens=count, stride=stride,
                mean_nll=mean_nll, perplexity=math.exp(mean_nll),
                method='sliding window; each next-token target scored once; window uses <=1024 input tokens')


def lambada_split(text, tokenizer):
    """Tokenize whole passage and locate final word using offsets."""
    text = text.rstrip()
    match = re.search(r'\S+$', text)
    if not match:
        raise ValueError('No final word')
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc['input_ids']
    first = next((i for i, (_, end) in enumerate(enc['offset_mapping']) if end > match.start()), None)
    if first is None or first == 0 or first >= len(ids):
        raise ValueError('Cannot isolate LAMBADA last word')
    return ids[:first], ids[first:], match.group(0)


@torch.inference_mode()
def generate_first_word(model, tokenizer, prefix, device, precision, max_new_tokens):
    """Greedy decoding until complete first whitespace-delimited word or EOS."""
    tokens = torch.tensor([prefix], dtype=torch.long, device=device)
    output = []
    ended = False
    ending = 'cap'
    budget = min(max_new_tokens, model.config.block_size - len(prefix))
    for _ in range(budget):
        with autocast_context(precision):
            logits, _ = model(tokens)
        next_id = int(logits[0, -1].argmax().item())
        output.append(next_id)
        if next_id == tokenizer.eos_token_id:
            ended = True
            ending = 'eos'
            break
        decoded = tokenizer.decode(output, skip_special_tokens=False)
        if re.match(r'^\s*\S+\s', decoded):
            ended = True
            ending = 'whitespace'
            break
        # Never run a forward pass beyond the trained context.
        if tokens.size(1) + 1 <= model.config.block_size:
            tokens = torch.cat((tokens, torch.tensor([[next_id]], device=device)), dim=1)
    decoded = tokenizer.decode(output, skip_special_tokens=True)
    match = re.match(r'^\s*(\S+)', decoded)
    return (match.group(1) if match else ''), output, ended, ending


def run_lambada(model, tokenizer, device, precision, args):
    from datasets import load_dataset
    ds = load_dataset(LAMBADA_REPO, 'en', split='test')
    correct_word = correct_tokens = total = skipped = boundary_count = 0
    nll_total = 0.0
    for row in ds:
        if args.limit and total >= args.limit:
            break
        try:
            prefix, target, expected = lambada_split(row['text'], tokenizer)
            reserve = max(len(target), args.lambada_max_new_tokens)
            if reserve >= model.config.block_size:
                raise ValueError('LAMBADA target/generation cap too long')
            prefix = prefix[-(model.config.block_size - reserve):]
            if not prefix:
                raise ValueError('No prefix')
        except ValueError:
            skipped += 1
            continue
        word, generated, boundary, _ = generate_first_word(
            model, tokenizer, prefix, device, precision, args.lambada_max_new_tokens)
        # No claimed correct word if generation stopped without a confirmed boundary.
        correct_word += int(boundary and word == expected)
        correct_tokens += int(len(generated) >= len(target) and generated[:len(target)] == target)
        boundary_count += int(boundary)
        nll_total -= conditional_loglikelihood(model, prefix, target, device, precision)
        total += 1
    if total == 0:
        raise RuntimeError('No LAMBADA examples scored')
    return dict(dataset=LAMBADA_REPO, split='test', examples=total, skipped=skipped,
                correct=correct_word, accuracy=correct_word / total,
                word_accuracy=correct_word / total,
                token_sequence_accuracy=correct_tokens / total,
                word_boundaries_observed=boundary_count,
                word_boundary_not_seen=total - boundary_count,
                mean_target_word_nll=nll_total / total,
                method='capped greedy first word; count word correct only on whitespace/EOS boundary; case-sensitive',
                caveat='Not guaranteed identical to official LAMBADA protocol; token metric is an auxiliary measure')


def run_blimp(model, tokenizer, device, precision, args):
    from datasets import load_dataset, get_dataset_config_names
    configs = args.blimp_configs or get_dataset_config_names(BLIMP_REPO)
    per = {}
    correct_total = count_total = 0
    for config in configs:
        ds = load_dataset(BLIMP_REPO, config, split='train')
        correct = count = 0
        for row in ds:
            if args.limit and count >= args.limit:
                break
            good = tokenizer.encode(row['sentence_good'], add_special_tokens=False)
            bad = tokenizer.encode(row['sentence_bad'], add_special_tokens=False)
            lg, ng = loglikelihood_ids(model, good, device, precision, bos_id=tokenizer.eos_token_id)
            lb, nb = loglikelihood_ids(model, bad, device, precision, bos_id=tokenizer.eos_token_id)
            if ng < 1 or nb < 1:
                raise ValueError(f'BLiMP {config}: empty sentence')
            correct += int(lg > lb)
            count += 1
        if count == 0:
            raise RuntimeError(f'BLiMP {config}: no scored pairs')
        per[config] = dict(correct=correct, pairs=count, accuracy=correct / count)
        correct_total += correct
        count_total += count
        print(f'[BLiMP] {config}: {correct}/{count}', flush=True)
    return dict(dataset=BLIMP_REPO, split='train (challenge dataset)', pairs=count_total,
                correct=correct_total, accuracy=correct_total / count_total,
                configurations=len(per), per_phenomenon=per,
                method='raw summed sentence log-likelihood given GPT-2 EOS as BOS; length sensitive')


def mmlu_prompt(row, examples):
    def fmt(r, answered):
        lines = [f"Question: {r['question']}"]
        lines.extend(f'{label}. {choice}' for label, choice in zip('ABCD', r['choices']))
        lines.append(f"Answer: {'ABCD'[int(r['answer'])]}" if answered else 'Answer:')
        return '\n'.join(lines)
    demos = '\n\n'.join(fmt(r, True) for r in examples)
    return (demos + '\n\n' if demos else '') + fmt(row, False)


def mmlu_candidates(prompt, tokenizer):
    """Score exact, fully tokenized prompt+answer strings after their common token prefix."""
    encoded = [tokenizer.encode(prompt + ' ' + letter, add_special_tokens=False) for letter in 'ABCD']
    common = 0
    while all(len(x) > common for x in encoded) and len({x[common] for x in encoded}) == 1:
        common += 1
    if common == 0 or any(len(x) == common for x in encoded):
        raise ValueError('No common token prefix or candidate-specific continuation')
    return encoded[0][:common], [x[common:] for x in encoded]


def run_mmlu(model, tokenizer, device, precision, args):
    from datasets import load_dataset, get_dataset_config_names
    subjects = args.mmlu_subjects or [s for s in get_dataset_config_names(MMLU_REPO) if s not in ('all', 'auxiliary_train')]
    per = {}
    correct_all = total_all = skipped_all = 0
    for subject in subjects:
        test = load_dataset(MMLU_REPO, subject, split='test')
        dev = list(load_dataset(MMLU_REPO, subject, split='dev'))[:args.mmlu_shots] if args.mmlu_shots else []
        correct = scored = skipped = used_shots = 0
        for row in test:
            if args.limit and scored >= args.limit:
                break
            examples = list(dev)
            while True:
                try:
                    prefix, suffixes = mmlu_candidates(mmlu_prompt(row, examples), tokenizer)
                    fits = all(len(prefix) + len(s) <= model.config.block_size for s in suffixes)
                except ValueError:
                    fits = False
                if fits:
                    break
                if not examples:
                    skipped += 1
                    break
                examples.pop(0)
            if not fits:
                continue
            scores = [conditional_loglikelihood(model, prefix, suffix, device, precision) for suffix in suffixes]
            prediction = max(range(4), key=scores.__getitem__)
            correct += int(prediction == int(row['answer']))
            scored += 1
            used_shots += len(examples)
        if not scored:
            raise RuntimeError(f'MMLU {subject}: all questions skipped')
        per[subject] = dict(correct=correct, questions=scored, accuracy=correct / scored,
                            skipped=skipped, mean_used_shots=used_shots / scored)
        correct_all += correct
        total_all += scored
        skipped_all += skipped
        print(f'[MMLU] {subject}: {correct}/{scored} (skipped={skipped})', flush=True)
    return dict(dataset=MMLU_REPO, split='test', questions=total_all, correct=correct_all,
                accuracy=correct_all / total_all, skipped=skipped_all, shots_requested=args.mmlu_shots,
                subjects=per, method='conditional summed log-likelihood of four fully tokenized candidates after shared prefix',
                caveat='Custom prompt and variable feasible few-shot demonstrations; not necessarily harness-comparable')


def evaluate_one(args):
    """Stable public integration API used by checkpoints.py."""
    model, tokenizer, device, precision, step, tok_path, training = load_model_and_tokenizer(args)
    print(f'[CHECKPOINT] {args.checkpoint} step={step} device={device} context={model.config.block_size}', flush=True)
    runners = {'wikitext': run_wikitext, 'lambada': run_lambada, 'blimp': run_blimp, 'mmlu': run_mmlu}
    results = {}
    for task in args.tasks:
        print(f'[RUN] {task}', flush=True)
        t0 = time.monotonic()
        results[task] = runners[task](model, tokenizer, device, precision, args)
        results[task]['elapsed_seconds'] = round(time.monotonic() - t0, 2)
        print(f"[DONE] {task}: {results[task].get('perplexity', results[task].get('accuracy'))}", flush=True)
    out = (args.output or args.checkpoint.parent / 'evaluations' /
           (args.checkpoint.stem + '_' + '_'.join(args.tasks) + '.json'))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(checkpoint=str(args.checkpoint), step=step, tokenizer=tok_path,
                   checkpoint_schema_version=training['schema_version'],
                   dataset_manifest_sha256=training['manifest_sha256'],
                   training_exposure=dict(data_mode=training['data_mode'],
                       sampler_order=training['sampler_order'],
                       sampler_seed=training['sampler_seed'],
                       blocks_consumed=training['blocks_consumed'],
                       dataset_blocks=training['dataset_blocks'],
                       predicted_training_tokens=training['blocks_consumed'] * training['block_size'],
                       epoch_fraction=(training['blocks_consumed'] / training['dataset_blocks']
                           if training['dataset_blocks'] else None),
                       completed_full_pass=training['completed_full_pass']),
                   device=str(device), precision=str(precision or torch.float32),
                   evaluation_settings=dict(tasks=args.tasks, limit=args.limit,
                       wikitext_max_tokens=args.wikitext_max_tokens, stride=args.stride,
                       blimp_configs=args.blimp_configs, mmlu_subjects=args.mmlu_subjects,
                       mmlu_shots=args.mmlu_shots,
                       lambada_max_new_tokens=getattr(args, 'lambada_max_new_tokens', 16)),
                   caveat='Benchmark leakage into Pile not audited; custom benchmark protocols may differ from published evaluations.',
                   results=results)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'[SAVED] {out}', flush=True)
    return out


def main():
    args = parse_args()
    if args.all_checkpoints:
        import re as _re
        paths = sorted(args.checkpoint.glob('step_*.pt'),
                       key=lambda path: (int(_re.search(r'^step_(\d+)', path.stem).group(1)), path.name))
        if not paths:
            raise FileNotFoundError('No historical step_*.pt files')
        for path in paths:
            args.checkpoint = path
            evaluate_one(args)
    else:
        evaluate_one(args)


if __name__ == '__main__':
    main()
