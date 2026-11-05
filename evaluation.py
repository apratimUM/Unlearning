#!/usr/bin/env python3
"""Evaluate this repository's GPT checkpoints on WikiText-103, LAMBADA, BLiMP, MMLU.

Examples:
  python evaluate.py --checkpoint runs/myrun/step_0001000.pt --tasks wikitext lambada blimp --limit 20
  python evaluate.py --checkpoint runs/myrun/latest.pt --tasks all
  python evaluate.py --checkpoint runs/myrun/step_0001000.pt --tasks mmlu --mmlu-shots 5

Outputs JSON to runs/<run>/evaluations/<checkpoint_stem>_<tasks>.json by default.
All benchmarks use held-out splits and never update model parameters.
"""
import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

WIKITEXT_REPO = 'Salesforce/wikitext'
LAMBADA_REPO = 'EleutherAI/lambada_openai'
BLIMP_REPO = 'nyu-mll/blimp'
MMLU_REPO = 'cais/mmlu'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='A step_*.pt/latest.pt, or a run directory with --all-checkpoints')
    p.add_argument('--all-checkpoints', action='store_true',
                   help='Evaluate each distinct historical step_*.pt in the selected run directory')
    p.add_argument('--tasks', nargs='+', choices=['wikitext', 'lambada', 'blimp', 'mmlu', 'all'], default=['wikitext', 'lambada', 'blimp'])
    p.add_argument('--tokenizer', type=str, default=None, help='Tokenizer directory; default: <data>/tokenizer from checkpoint signature')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--precision', choices=['auto', 'float32', 'bfloat16', 'float16'], default='auto')
    p.add_argument('--limit', type=int, default=0, help='Maximum examples PER task (and per BLiMP/MMLU category); 0=full')
    p.add_argument('--wikitext-max-tokens', type=int, default=0, help='Maximum test tokens; 0=full')
    p.add_argument('--stride', type=int, default=0, help='WikiText sliding window stride (0=block_size//2)')
    p.add_argument('--blimp-configs', nargs='*', default=None, help='Optional list of BLiMP phenomena')
    p.add_argument('--mmlu-subjects', nargs='*', default=None, help='Optional list of MMLU subjects')
    p.add_argument('--mmlu-shots', type=int, default=0, choices=range(0, 6), metavar='0..5', help='Number of per-subject dev examples in prompt')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--local-files-only', action='store_true', help='Use cached tokenizer; datasets are loaded through datasets cache')
    p.add_argument('--trusted-checkpoint', action='store_true', help='Allow full pickle loading of your own latest.pt if safe weights-only loading fails')
    a = p.parse_args()
    if 'all' in a.tasks: a.tasks = ['wikitext', 'lambada', 'blimp', 'mmlu']
    else: a.tasks = list(dict.fromkeys(a.tasks))
    if a.all_checkpoints and not a.checkpoint.is_dir():
        p.error('--all-checkpoints requires --checkpoint RUN_DIRECTORY')
    if not a.all_checkpoints and not a.checkpoint.is_file():
        p.error('--checkpoint must be an existing file')
    if a.all_checkpoints and a.output is not None:
        p.error('--output is only supported for a single checkpoint')
    if a.limit < 0 or a.wikitext_max_tokens < 0 or a.stride < 0:
        p.error('Limits and stride must be nonnegative')
    return a


def load_model_and_tokenizer(args):
    from transformers import AutoTokenizer
    # Load only checkpoints you trust. The project stores Python objects such as RNG
    # states in latest.pt; PyTorch pickle loading is unsafe for untrusted files.
    try:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    except Exception as exc:
        if not args.trusted_checkpoint:
            raise RuntimeError('Safe checkpoint loading failed. For your own trusted latest.pt, rerun with --trusted-checkpoint (permits pickle loading).') from exc
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or 'model' not in ckpt or 'config' not in ckpt:
        raise ValueError('Not a project checkpoint: expected model and config')
    signature = ckpt.get('signature') or {}
    tokenizer_path = args.tokenizer or str(Path(signature.get('data', 'data/pile_large')) / 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=args.local_files_only)
    if not tokenizer.is_fast:
        raise ValueError('Need a FAST tokenizer for LAMBADA word-to-token alignment')
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if args.precision == 'auto':
        precision = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    else:
        precision = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.precision]
    if device.type != 'cuda' and precision != torch.float32:
        raise ValueError('Use --precision float32 for CPU')
    model = GPT(GPTConfig(**ckpt['config']))
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(device)
    model.eval()
    if tokenizer.vocab_size > model.config.vocab_size:
        raise ValueError('Tokenizer vocabulary exceeds model vocabulary')
    return model, tokenizer, device, precision, int(ckpt.get('step', -1)), tokenizer_path


def cast_context(device, precision):
    if device.type == 'cuda' and precision != torch.float32:
        return torch.autocast(device_type='cuda', dtype=precision)
    from contextlib import nullcontext
    return nullcontext()


@torch.inference_mode()
def loglikelihood_ids(model, ids, device, precision, bos_id=None):
    """Score all tokens (except first without BOS) using non-overlapping windows.

    For lengths > model block size, context resets each block. Returns total
    log-likelihood, evaluated token count. Ideal for short BLiMP sentences.
    """
    if not ids: raise ValueError('Empty sequence')
    # Include sentence-initial token consistently: GPT-2 uses EOS as a
    # beginning-of-text marker for scoring isolated sentences.
    if bos_id is not None:
        ids = [bos_id] + ids
    elif len(ids) == 1:
        return 0.0, 0
    block = model.config.block_size
    total = 0.0
    count = 0
    for start in range(0, len(ids)-1, block):
        chunk = ids[start:start+block+1]
        if len(chunk) < 2: break
        x = torch.tensor(chunk[:-1], dtype=torch.long, device=device)[None]
        y = torch.tensor(chunk[1:], dtype=torch.long, device=device)[None]
        with cast_context(device, precision):
            logits, _ = model(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        total += float(log_probs.gather(-1, y.unsqueeze(-1)).sum().item())
        count += y.numel()
    return total, count


@torch.inference_mode()
def conditional_loglikelihood(model, prefix_ids, continuation_ids, device, precision):
    """Teacher-force continuation and score exactly its tokens, rejecting overflow.

    Handles continuation shorter than block_size; returns sum log probability.
    """
    if not prefix_ids or not continuation_ids:
        raise ValueError('Both prompt and continuation must contain tokens')
    block = model.config.block_size
    if len(continuation_ids) >= block:
        raise ValueError('Candidate continuation too long for model context')
    if len(prefix_ids) + len(continuation_ids) > block:
        raise ValueError('Prompt and continuation exceed model context; trim prompt explicitly')
    prompt = prefix_ids
    ids = prompt + continuation_ids
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device)[None]
    with cast_context(device, precision):
        logits, _ = model(x)
    start = len(prompt)-1
    chosen = logits[0, start:start+len(continuation_ids)].float()
    targets = torch.tensor(continuation_ids, dtype=torch.long, device=device)
    return float(F.log_softmax(chosen, -1).gather(-1, targets[:, None]).sum().item())


def run_wikitext(model, tokenizer, device, precision, args):
    from datasets import load_dataset
    ds = load_dataset(WIKITEXT_REPO, 'wikitext-103-raw-v1', split='test')
    # WikiText source rows already preserve their own newlines. Add a single
    # newline only when the line is not already newline-terminated.
    text = ''.join(row['text'] if row['text'].endswith('\n') else row['text'] + '\n' for row in ds)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if args.wikitext_max_tokens: ids = ids[:args.wikitext_max_tokens]
    if len(ids) < 2: raise ValueError('WikiText test text contains fewer than 2 tokens')
    block = model.config.block_size
    stride = args.stride or max(1, block // 2)
    if not 1 <= stride < block:
        raise ValueError('WikiText stride must be in [1, block_size-1]')
    total_nll = 0.0
    total_count = 0
    # Each window scores only unseen (not previously scored) targets, with left context.
    # First token cannot be predicted without a preceding BOS token.
    last_target = 0
    for end in range(min(block, len(ids)), len(ids)+block, stride):
        end = min(end, len(ids))
        begin = max(0, end-block)
        chunk = ids[begin:end]
        x = torch.tensor(chunk[:-1], dtype=torch.long, device=device)[None]
        with torch.inference_mode(), cast_context(device, precision):
            logits, _ = model(x)
        # target absolute positions p in (last_target, end-1]
        target_begin = max(last_target+1, begin+1)
        if target_begin < end:
            token_logits = logits[0, target_begin-begin-1:end-begin-1].float()
            targets = torch.tensor(ids[target_begin:end], dtype=torch.long, device=device)
            nll = F.cross_entropy(token_logits, targets, reduction='sum')
            total_nll += float(nll.item())
            total_count += len(targets)
            last_target = end-1
        if end == len(ids): break
    if total_count != len(ids)-1:
        raise RuntimeError(f'WikiText target coverage error: {total_count} vs {len(ids)-1}')
    avg_nll = total_nll / total_count
    return {'dataset': WIKITEXT_REPO, 'config': 'wikitext-103-raw-v1', 'split': 'test',
            'total_tokens': len(ids), 'scored_tokens': total_count, 'stride': stride,
            'mean_nll': avg_nll, 'perplexity': math.exp(avg_nll),
            'method': 'sliding-window causal cross-entropy, weighted by scored tokens'}


def lambada_split(text, tokenizer):
    """Return tokenized context / final-word target using offset mappings.

    The first target token may include a preceding space due to GPT-2 byte BPE.
    This uses token boundaries of the WHOLE passage, avoiding a boundary mismatch.
    """
    match = re.search(r'\S+\s*$', text)
    if not match: raise ValueError('No final word')
    # Use end of final non-whitespace word for full-reference comparison.
    final_word = match.group(0).strip()
    word_start = match.start()
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc['input_ids']
    offsets = enc['offset_mapping']
    first = next((i for i, (a,b) in enumerate(offsets) if b > word_start), None)
    if first is None or first == 0 or first >= len(ids):
        raise ValueError('Could not isolate LAMBADA last word')
    return ids[:first], ids[first:], final_word


def run_lambada(model, tokenizer, device, precision, args):
    from datasets import load_dataset
    ds = load_dataset(LAMBADA_REPO, 'en', split='test')
    correct = 0
    total = 0
    failures = 0
    total_word_nll = 0.0
    for row in ds:
        if args.limit and total >= args.limit: break
        try:
            prefix, target, expected = lambada_split(row['text'], tokenizer)
        except ValueError:
            failures += 1
            continue
        # Model's sliding context: keep as much left context as is permitted.
        if len(target) >= model.config.block_size:
            failures += 1
            continue
        prefix = prefix[-(model.config.block_size-len(target)):]
        with torch.inference_mode():
            generated = model.generate_greedy(torch.tensor(prefix, dtype=torch.long, device=device)[None], len(target))
        generated_ids = generated[0, len(prefix):].tolist()
        # The benchmark metric is complete final-word exact match. Word
        # tokenization is fixed from the original passage, including spaces.
        correct += int(generated_ids == target)
        total_word_nll += -conditional_loglikelihood(model, prefix, target, device, precision)
        total += 1
    if not total: raise RuntimeError('No LAMBADA examples evaluated')
    return {'dataset': LAMBADA_REPO, 'split': 'test', 'examples': total,
            'accuracy': correct/total, 'correct': correct,
            'mean_target_word_nll': total_word_nll/total, 'skipped': failures,
            'method': 'greedy exact last-word-token sequence match; full text token offsets'}


def run_blimp(model, tokenizer, device, precision, args):
    from datasets import load_dataset, get_dataset_config_names
    configs = args.blimp_configs or get_dataset_config_names(BLIMP_REPO)
    per = {}
    all_correct = 0
    all_pairs = 0
    for config in configs:
        ds = load_dataset(BLIMP_REPO, config, split='train')
        good = 0
        count = 0
        for row in ds:
            if args.limit and count >= args.limit: break
            a = tokenizer.encode(row['sentence_good'], add_special_tokens=False)
            b = tokenizer.encode(row['sentence_bad'], add_special_tokens=False)
            ll_good, n_good = loglikelihood_ids(model, a, device, precision,
                                                bos_id=tokenizer.eos_token_id)
            ll_bad, n_bad = loglikelihood_ids(model, b, device, precision,
                                              bos_id=tokenizer.eos_token_id)
            if n_good == 0 or n_bad == 0: continue
            good += int(ll_good > ll_bad)
            count += 1
        if not count: raise RuntimeError(f'No BLiMP pairs for {config}')
        per[config] = {'accuracy': good/count, 'correct': good, 'pairs': count}
        all_correct += good
        all_pairs += count
        print(f'  BLiMP {config}: {good}/{count} = {good/count:.3%}', flush=True)
    return {'dataset': BLIMP_REPO, 'pairs': all_pairs, 'correct': all_correct,
            'accuracy': all_correct/all_pairs, 'configurations': len(per),
            'per_phenomenon': per, 'method': 'raw summed log-likelihood including sentence-initial token conditioned on GPT-2 EOS/BOS'}


def mmlu_prompt(row, examples):
    def format_one(r, include_answer):
        lines = [f"Question: {r['question']}"]
        lines += [f'{letter}. {answer}' for letter, answer in zip('ABCD', r['choices'])]
        lines += [f"Answer: {'ABCD'[int(r['answer'])]}" if include_answer else 'Answer:']
        return '\n'.join(lines)
    return ('\n\n'.join(format_one(ex, True) for ex in examples) + '\n\n' if examples else '') + format_one(row, False)


def run_mmlu(model, tokenizer, device, precision, args):
    from datasets import load_dataset, get_dataset_config_names
    subjects = args.mmlu_subjects or [c for c in get_dataset_config_names(MMLU_REPO) if c not in ('all', 'auxiliary_train')]
    per = {}
    correct = 0
    total = 0
    for subject in subjects:
        ds = load_dataset(MMLU_REPO, subject, split='test')
        few = []
        if args.mmlu_shots:
            dev = load_dataset(MMLU_REPO, subject, split='dev')
            few = list(dev)[:args.mmlu_shots]
        sub_correct = 0
        sub_total = 0
        used_shots = 0
        for row in ds:
            if args.limit and sub_total >= args.limit: break
            # Preserve complete question and options. Drop oldest few-shot
            # demonstrations if the small GPT-2 context cannot fit them.
            examples = list(few)
            while True:
                prompt_text = mmlu_prompt(row, examples)
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                max_answer = max(len(tokenizer.encode(' '+letter,
                                   add_special_tokens=False)) for letter in 'ABCD')
                if len(prompt_ids) + max_answer <= model.config.block_size:
                    break
                if not examples:
                    raise ValueError(f'MMLU question exceeds context size for {subject}')
                examples.pop(0)
            if not prompt_ids: raise ValueError('Empty MMLU prompt')
            used_shots += len(examples)
            scores = []
            for letter in 'ABCD':
                answer_ids = tokenizer.encode(' '+letter, add_special_tokens=False)
                # Check tokenization at the BPE boundary.
                combined = tokenizer.encode(prompt_text+' '+letter,
                                            add_special_tokens=False)
                if combined != prompt_ids + answer_ids:
                    raise ValueError('MMLU prompt/answer token boundary mismatch')
                scores.append(conditional_loglikelihood(model, prompt_ids, answer_ids, device, precision))
            pred = max(range(4), key=lambda idx: scores[idx])
            sub_correct += int(pred == int(row['answer']))
            sub_total += 1
        if not sub_total: raise RuntimeError(f'No MMLU examples for {subject}')
        per[subject] = {'correct': sub_correct, 'questions': sub_total,
                        'accuracy': sub_correct/sub_total,
                        'mean_used_shots': used_shots/sub_total}
        correct += sub_correct
        total += sub_total
        print(f'  MMLU {subject}: {sub_correct}/{sub_total} = {sub_correct/sub_total:.3%}', flush=True)
    return {'dataset': MMLU_REPO, 'questions': total, 'correct': correct,
            'accuracy': correct/total, 'shots': args.mmlu_shots,
            'subjects': per,
            'method': 'conditional summed log likelihood of answer labels A/B/C/D; common question+options prompt'}


def evaluate_one(args):
    model, tokenizer, device, precision, step, tokenizer_path = load_model_and_tokenizer(args)
    print(f'Loaded {args.checkpoint}: step={step}, device={device}, context={model.config.block_size}', flush=True)
    results = {}
    for task in args.tasks:
        print(f'\n[RUN] {task}', flush=True)
        begin = time.time()
        fn = {'wikitext': run_wikitext, 'lambada': run_lambada,
              'blimp': run_blimp, 'mmlu': run_mmlu}[task]
        results[task] = fn(model, tokenizer, device, precision, args)
        results[task]['elapsed_seconds'] = round(time.time()-begin, 2)
        headline = results[task].get('perplexity', results[task].get('accuracy'))
        print(f'[DONE] {task}: {headline}', flush=True)
    output = args.output or args.checkpoint.parent / 'evaluations' / (args.checkpoint.stem + '_' + '_'.join(args.tasks) + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {'checkpoint': str(args.checkpoint), 'step': step, 'tokenizer': tokenizer_path,
              'device': str(device), 'precision': str(precision),
              'evaluation_settings': {'tasks': args.tasks, 'limit': args.limit,
                  'wikitext_max_tokens': args.wikitext_max_tokens, 'stride': args.stride,
                  'blimp_configs': args.blimp_configs, 'mmlu_subjects': args.mmlu_subjects,
                  'mmlu_shots': args.mmlu_shots}, 'results': results}
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(f'\nSaved results: {output}', flush=True)
    return output


def main():
    args = parse_args()
    if args.all_checkpoints:
        steps = sorted(args.checkpoint.glob('step_*.pt'))
        if not steps:
            raise FileNotFoundError('No step_*.pt snapshots in run directory')
        for checkpoint in steps:
            args.checkpoint = checkpoint
            evaluate_one(args)
    else:
        evaluate_one(args)


if __name__ == '__main__':
    main()
