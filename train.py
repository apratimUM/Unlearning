#!/usr/bin/env python3

# IMPORTANT: Before starting a NEW training run, prepare the document-shuffled
# dataset and fixed chunk order ONCE using a CPU compute node:
#
# python train.py --prepare-order \
#   --data /coe-aroyc/apratim/pile_full_data \
#   --out /coe-aroyc/apratim/pile_full_docshuf42 \
#   --seed 42
#
# This reuses existing token IDs (no retokenization), shuffles documents,
# forms 1024-token chunks, and saves a fixed shuffled chunk order.
# Original document IDs and chunk IDs remain traceable through saved metadata.
#
# Run this ONLY ONCE for each new dataset/seed. Use --resume if interrupted.
# After preparation completes, start GPU training using the NEW dataset path.
# Do not rerun preparation when resuming training from a checkpoint.
# No shuffling occurs inside the training batch loop.

import argparse
import contextlib
import hashlib
import sqlite3
import copy
import fcntl
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Existing [TRAIN] logs still report ETA without tqdm.

from data import PackedSampler, SAMPLER_ORDER, file_sha, verify_data
from model import GPT, GPTConfig, make_model
from runtime import require_cuda, choose_precision, autocast_context, make_scaler, provenance, checked_optimizer_step



def get_rng():
    name, keys, pos, has_gauss, cached = np.random.get_state()

    return {
        "python": random.getstate(),

        "numpy": (
            name,
            torch.tensor(keys.astype(np.int64)),
            pos,
            has_gauss,
            cached,
        ),

        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }

def set_rng(state):
    random.setstate(state["python"])

    name, keys, pos, has_gauss, cached = state["numpy"]

    np.random.set_state((
        name,
        keys.cpu().numpy().astype(np.uint32),
        pos,
        has_gauss,
        cached,
    ))

    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])

def lr_schedule(step, args):
    """Prepared-mode learning rate; independent run configs define decay."""
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    if args.lr_schedule == 'constant':
        return args.lr
    fraction = (step - args.warmup) / max(1, args.steps - args.warmup - 1)
    fraction = max(0.0, min(1.0, fraction))
    return args.min_lr + (args.lr - args.min_lr) * (1 + math.cos(math.pi * fraction)) / 2

def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")

    tmp.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )

    os.replace(tmp, path)

def trim_log(path, step):
    """Remove logs beyond the restored checkpoint."""

    if not path.exists():
        return

    kept = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        try:
            if int(json.loads(line)["step"]) <= step:
                kept.append(line)

        except (ValueError, KeyError, TypeError):
            pass

    tmp = path.with_name(path.name + ".tmp")

    tmp.write_text(
        "".join(s + "\n" for s in kept),
        encoding="utf-8",
    )

    os.replace(tmp, path)

def update_training_plots(run):
    """Save PNG figures to the run directory without opening a GUI window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load_rows(filename):
        path = run / filename
        if not path.is_file():
            return []
        rows = []
        with path.open("r", encoding="utf-8") as inp:
            for line in inp:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except (ValueError, TypeError):
                        continue
        return rows

    training = load_rows("train_loss_history.jsonl")
    validation = load_rows("metrics.jsonl")

    def save_plot(filename, title, ylabel, series):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        has_data = False
        for label, records, key in series:
            points = [(int(r["step"]), float(r[key])) for r in records
                      if r.get("step") is not None and r.get(key) is not None
                      and math.isfinite(float(r[key]))]
            if points:
                x, y = zip(*points)
                ax.plot(x, y, label=label)
                has_data = True
        if has_data:
            ax.set_title(title)
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            target = run / filename
            temp = run / (filename + ".tmp.png")
            fig.savefig(temp, dpi=150)
            os.replace(temp, target)
            print(f"[PLOT] {target}", flush=True)
        plt.close(fig)

    save_plot("loss_curve.png", "Training and validation loss", "Cross-entropy loss", [
        ("Training", training, "train_loss"),
        ("Validation", validation, "val_loss"),
    ])
    save_plot("perplexity_curve.png", "Validation perplexity", "Perplexity", [
        ("Validation", validation, "val_perplexity"),
    ])

def block_count_after(step, args, n_blocks):
    """Number of unique blocks consumed after step updates."""

    return min(
        step * args.batch * args.accum,
        n_blocks,
    )

@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompts,
    device,
    cast,
    *,
    temperature=0.8,
    max_new_tokens=50,
    top_k=50,
    do_sample=True,
    seed=42,
):
    """Generate from one string or a list of strings; preserve input/output shape.

    Equal-token-length prompts run together in actual GPU batches, without
    padding or changed absolute-position indices. Different prompt lengths are
    grouped separately. Each prompt has its own seeded RNG, so sampling does
    not depend on how other prompts are grouped.

    Returns one decoded continuation string, or a list in the original order.
    This is a qualitative training check, NOT verbatim-extraction scoring.
    """
    single = isinstance(prompts, str)
    if single:
        prompt_list = [prompts]
    elif isinstance(prompts, (list, tuple)) and all(isinstance(p, str) for p in prompts):
        prompt_list = list(prompts)
    else:
        raise TypeError("prompts must be a string or a list/tuple of strings")
    if not prompt_list or any(not p.strip() for p in prompt_list):
        raise ValueError("Generation prompts cannot be empty")
    if (not math.isfinite(float(temperature)) or temperature <= 0 or
            max_new_tokens < 1 or top_k is not None and (top_k < 0 or top_k > model.config.vocab_size)):
        raise ValueError("Invalid generation settings")

    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_list]
    for i, ids in enumerate(encoded):
        if not ids or len(ids) + max_new_tokens > model.config.block_size:
            raise ValueError(f"Prompt {i} is empty or exceeds model context with continuation")

    previous_mode = model.training
    model.eval()
    answers = [None] * len(prompt_list)
    groups = {}
    for index, ids in enumerate(encoded):
        groups.setdefault(len(ids), []).append(index)

    try:
        for indices in groups.values():
            tokens = torch.tensor([encoded[i] for i in indices], dtype=torch.long, device=device)
            generated = [[] for _ in indices]
            generators = []
            for i in indices:
                generator = torch.Generator(device=device)
                generator.manual_seed(int(seed) + i)
                generators.append(generator)

            for _ in range(max_new_tokens):
                with cast():
                    logits, _ = model(tokens)
                scores = logits[:, -1, :].float()

                if do_sample:
                    scores = scores / float(temperature)
                    if top_k is not None and top_k > 0:
                        threshold = torch.topk(scores, min(top_k, scores.size(-1)), dim=-1).values[:, -1:]
                        scores = scores.masked_fill(scores < threshold, float('-inf'))
                    probs = torch.softmax(scores, dim=-1)
                    # Distinct RNG stream for each prompt; batch forward stays vectorized.
                    next_ids = torch.cat([
                        torch.multinomial(probs[j:j+1], 1, generator=generators[j])
                        for j in range(len(indices))
                    ], dim=0)
                else:
                    next_ids = scores.argmax(dim=-1, keepdim=True)

                tokens = torch.cat((tokens, next_ids), dim=1)
                for j in range(len(indices)):
                    generated[j].append(int(next_ids[j, 0]))

            for local, index in enumerate(indices):
                answers[index] = tokenizer.decode(generated[local], skip_special_tokens=False)
    finally:
        model.train(previous_mode)

    return answers[0] if single else answers

def generation_check(model, tokenizer, prompts, args, device, cast, step, run):
    """Run configured generation check and append one record per input prompt."""
    outputs = generate_text(
        model, tokenizer, prompts, device, cast,
        temperature=args.check_temperature,
        max_new_tokens=args.check_max_new_tokens,
        top_k=args.check_top_k,
        do_sample=args.check_do_sample,
        seed=args.seed + step,
    )
    if isinstance(outputs, str):
        outputs = [outputs]
        prompts = [prompts]
    with (run / "generation_checks.jsonl").open("a", encoding="utf-8") as f:
        for index, (prompt, output) in enumerate(zip(prompts, outputs)):
            print(f"[GENERATE step={step}] {prompt!r} -> {output!r}", flush=True)
            f.write(json.dumps({
                "step": step,
                "prompt_index": index,
                "prompt": prompt,
                "generated": output,
                "do_sample": args.check_do_sample,
                "temperature": args.check_temperature,
                "max_new_tokens": args.check_max_new_tokens,
                "top_k": args.check_top_k,
                "seed": args.seed + step + index,
            }, ensure_ascii=False) + "\n")


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--model', default='small', choices=['small'])
    p.add_argument('--block-size', type=int, default=1024)
    p.add_argument('--batch', type=int, default=128, help='GLOBAL microbatch across all DDP ranks (32 per GPU on 4 GPUs)')
    p.add_argument('--accum', type=int, default=1)
    p.add_argument('--steps', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--precision', default='auto', choices=['auto', 'bfloat16', 'float16', 'float32'])
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--min-lr', type=float, default=1e-5)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--lr-schedule', choices=['cosine', 'constant'], default='cosine')
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--eval-every', type=int, default=10000)
    p.add_argument('--eval-batches', type=int, default=50)
    p.add_argument('--save-every', type=int, default=0, help='optional step-based checkpoint interval; 0 disables it')
    p.add_argument('--save-hours', type=float, default=6.0, help='wall-clock hours between checkpoint snapshots')
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--eta-window', type=int, default=200, help='number of recent optimizer steps for rolling ETA')
    p.add_argument('--max-loss', type=float, default=15.0, help='abort when loss exceeds this after 300 steps; 0 disables guard')
    p.add_argument('--preflight-only', action='store_true', help='verify prepared data and sampler without training')
    p.add_argument('--check-begin', type=int, default=10000)
    p.add_argument('--check-every', type=int, default=10000)
    p.add_argument('--check-temperature', type=float, default=0.8)
    p.add_argument('--check-max-new-tokens', type=int, default=50)
    p.add_argument('--check-top-k', type=int, default=50)
    p.add_argument('--check-do-sample', action='store_true', default=True)
    p.add_argument('--check-prompts', nargs='+', default=[
        'Once upon a time', 'The capital of France is', 'Artificial intelligence is'])
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.block_size != 1024 or a.batch < 2 or a.accum < 1 or a.steps < 0:
        p.error('Expected block size 1024, batch >= 2, accum >= 1, steps >= 0')
    if a.warmup < 0 or a.lr <= 0 or not 0 <= a.min_lr <= a.lr:
        p.error('Invalid learning rate/warmup')
    if a.save_every < 0 or not math.isfinite(a.save_hours) or a.save_hours <= 0:
        p.error('Expected nonnegative --save-every and positive finite --save-hours')
    if min(a.eval_every, a.eval_batches, a.log_every, a.eta_window) < 1 or a.max_loss < 0:
        p.error('Intervals must be positive')
    return a


def snapshot_to_cpu(value):
    """Detach tensors from the live training state before the writer can see them."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        snapshot = copy.copy(value)
        for key, child in value.items():
            snapshot[key] = snapshot_to_cpu(child)
        if hasattr(value, '_metadata'):
            snapshot._metadata = copy.deepcopy(value._metadata)
        return snapshot
    if isinstance(value, list):
        return [snapshot_to_cpu(child) for child in value]
    if isinstance(value, tuple):
        return tuple(snapshot_to_cpu(child) for child in value)
    return copy.deepcopy(value)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_checkpoint(run, state, n_blocks):
    step = state['step']
    consumed = state['blocks_consumed']
    path = run / f'step_{step:07d}.pt'
    tmp = run / '.checkpoint.pt.tmp'
    link = run / '.latest.pt.tmp'
    try:
        # A step file left ahead of latest.pt by an interrupted write can be
        # replaced when the resumed run reaches that step again.
        with tmp.open('wb') as file:
            torch.save(state, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
        write_json(path.with_suffix('.json'), {
            'step': step, 'checkpoint_file': path.name,
            'model_config': state['config'],
            'dataset_manifest_sha256': state['manifest_sha256'],
            'token_exposure': {
                'blocks_consumed': consumed,
                'unique_blocks_visited': consumed,
                'dataset_blocks': n_blocks,
                'sampler_order': SAMPLER_ORDER,
                'prediction_tokens': consumed * 1024,
                'epochs': consumed / n_blocks,
            },
            'blocks_consumed': consumed, 'unique_blocks_visited': consumed,
            'train_loss': state['last_train_loss'], 'optimizer_state_saved': True,
            'rng_state_saved': True, 'parallelism': 'ddp',
        })
        link.unlink(missing_ok=True)
        try:
            os.link(path, link)
        except OSError:
            with path.open('rb') as source, link.open('wb') as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
        sync_directory(run)
        os.replace(link, run / 'latest.pt')
        sync_directory(run)
        print(f'[SAVE] step={step:,} blocks={consumed:,}/{n_blocks:,}', flush=True)
    finally:
        tmp.unlink(missing_ok=True)
        link.unlink(missing_ok=True)


class CheckpointWriter:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint')
        self.pending = None

    def flush(self):
        if self.pending is not None:
            self.pending.result()
            self.pending = None

    def failed(self):
        return (
            self.pending is not None
            and self.pending.done()
            and self.pending.exception() is not None
        )

    def close(self):
        self.executor.shutdown(wait=True)


def ensure_chunk_order(folder, manifest, seed, *, allow_create=False):
    """Verify a saved chunk order; create only in explicit CPU preparation mode."""
    folder = Path(folder)
    if manifest.get('document_shuffled') is not True or manifest.get('document_shuffle_index') != 'document_order.u32':
        raise RuntimeError('Document shuffle missing: run train.py --prepare-order first')
    doc_order = folder / 'document_order.u32'
    if not doc_order.is_file() or doc_order.stat().st_size != 4 * int(manifest['train_docs']):
        raise RuntimeError('Document order file missing or wrong size')
    n = int(manifest['train_blocks'])
    if not 1 <= n < 2**32:
        raise ValueError('Fixed uint32 order requires 1..2**32-1 chunks')
    paths = [folder / 'train_chunk_order.u32', folder / 'train_chunk_inverse.u32',
             folder / 'train_chunk_order.json']
    present = [path.exists() for path in paths]
    if all(present):
        info = json.loads(paths[2].read_text(encoding='utf-8'))
        if (info.get('algorithm') != SAMPLER_ORDER or info.get('seed') != seed or
                info.get('n_blocks') != n or
                info.get('source_manifest_sha256') != file_sha(folder / 'manifest.json')):
            raise ValueError('Existing fixed chunk order does not match manifest/seed')
        if any(path.stat().st_size != 4*n for path in paths[:2]):
            raise ValueError('Fixed chunk order files have incorrect sizes')
        print(f'[ORDER] reusing fixed shuffled order: {n:,} chunks', flush=True)
        return
    if any(present):
        raise RuntimeError('Partial fixed chunk order found; inspect files before retrying')
    if not allow_create:
        raise RuntimeError('Chunk order missing: run train.py --prepare-order before GPU training')

    print(f'[ORDER] generating {n:,} fixed chunk IDs before training', flush=True)
    rng = np.random.default_rng(seed)
    order64 = rng.permutation(n)
    order = order64.astype('<u4')
    del order64
    inverse = np.empty(n, dtype='<u4')
    inverse[order] = np.arange(n, dtype='<u4')
    for i in np.linspace(0, n-1, min(n, 5000), dtype=np.int64):
        if int(inverse[order[i]]) != int(i):
            raise RuntimeError('Chunk-order inverse verification failed')
    temps = [path.with_name(path.name + '.tmp') for path in paths]
    try:
        for arr, tmp in ((order, temps[0]), (inverse, temps[1])):
            with tmp.open('wb') as out:
                arr.tofile(out)
                out.flush()
                os.fsync(out.fileno())
        info = dict(algorithm=SAMPLER_ORDER, seed=seed, n_blocks=n,
                    source_manifest='manifest.json',
                    source_manifest_sha256=file_sha(folder / 'manifest.json'),
                    source_manifest_train_blocks=n)
        with temps[2].open('w', encoding='utf-8') as out:
            json.dump(info, out, indent=2)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(temps[0], paths[0])
        os.replace(temps[1], paths[1])
        os.replace(temps[2], paths[2])  # completion marker published last
    finally:
        for tmp in temps:
            tmp.unlink(missing_ok=True)
    print(f'[ORDER] fixed order ready ({n:,} chunks, seed={seed})', flush=True)


def rank_zero_io(rank, action):
    """Propagate rank-zero I/O errors before other ranks start another collective."""
    error = [None]
    cause = None
    if rank == 0:
        try:
            action()
        except Exception as exc:
            cause = exc
            error[0] = f'{type(exc).__name__}: {exc}'
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0]) from cause


def checkpoint_ddp(model, optimizer, scaler, run, step, signature, args, n_blocks, loss, rank, world, writer):
    rng_states = [None] * world
    dist.all_gather_object(rng_states, get_rng())

    def stage():
        # One in-flight snapshot bounds host RAM. This only waits for disk
        # when the previous write is still running at the next checkpoint.
        writer.flush()
        consumed = block_count_after(step, args, n_blocks)
        state = {
            'schema_version': 3,
            'parallelism': 'ddp',
            'model': snapshot_to_cpu(model.state_dict()),
            'config': copy.deepcopy(vars(model.config)),
            'optimizer': snapshot_to_cpu(optimizer.state_dict()),
            'scaler': snapshot_to_cpu(scaler.state_dict()),
            'rng': rng_states[0],
            'rng_per_rank': rng_states,
            'step': step,
            'blocks_consumed': consumed,
            'epoch_fraction': consumed / n_blocks,
            'last_train_loss': loss,
            'scheduler': {
                'type': 'manual_linear_warmup_cosine_decay' if args.lr_schedule == 'cosine' else 'constant',
                'completed_optimizer_steps': step,
                'next_step_index': step,
                'total_steps': args.steps,
                'warmup': args.warmup,
                'base_lr': args.lr,
                'min_lr': args.min_lr,
                'last_applied_lr': optimizer.param_groups[0]['lr'],
                'scheduler_state_dict': None,
            },
            'signature': copy.deepcopy(signature),
            'manifest_sha256': signature['manifest_sha256'],
        }
        writer.pending = writer.executor.submit(write_checkpoint, run, state, n_blocks)
        print(f'[SAVE_QUEUED] step={step:,}', flush=True)

    rank_zero_io(rank, stage)


def validate_ddp(model, sampler, args, device, cast, rank, world):
    """Evaluate identical sequential prefix as old validate(), partitioned by rank."""
    was_training = model.training
    model.eval()
    local_nll = torch.zeros(1, device=device, dtype=torch.float64)
    local_tokens = torch.zeros(1, device=device, dtype=torch.float64)
    try:
        limit = min(sampler.n_blocks, args.eval_batches * args.batch)
        for first in range(0, limit, args.batch):
            n = min(args.batch, limit - first)
            a = n * rank // world
            b = n * (rank + 1) // world
            if a == b:
                continue
            # The existing PackedSampler.batch() is the authoritative data format.
            # Both ranks read the global microbatch, then process disjoint slices.
            # This preserves the tested shuffled Feistel visit order and shifted targets.
            gx, gy = sampler.batch_range(first + a, b - a)
            x, y = gx.to(device), gy.to(device)
            with torch.inference_mode(), cast():
                logits, _ = model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       y.reshape(-1), reduction='sum')
            local_nll += loss.detach().to(torch.float64)
            local_tokens += y.numel()
            del logits, loss
    finally:
        model.train(was_training)
    dist.all_reduce(local_nll, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_tokens, op=dist.ReduceOp.SUM)
    return (local_nll / local_tokens).item(), int(local_tokens.item())


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def create_db(db):
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('''CREATE TABLE IF NOT EXISTS documents (
        source_row INTEGER PRIMARY KEY, split TEXT NOT NULL,
        token_offset INTEGER NOT NULL, text_tokens INTEGER NOT NULL,
        stored_tokens INTEGER NOT NULL, text_sha256 TEXT NOT NULL)''')
    db.execute('''CREATE TABLE IF NOT EXISTS progress (
        key TEXT PRIMARY KEY, value TEXT NOT NULL)''')
    db.commit()


def prepare_document_shuffle(a):
    """One-time CPU-only document reorder from existing token IDs (no retokenization)."""
    class _Parser:
        @staticmethod
        def error(message):
            raise ValueError(message)
    p = _Parser()
    src = a.data.expanduser().resolve()
    dst = a.out.expanduser().resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        p.error('Source and destination must be separate non-nested directories')
    if a.seed < 0 or a.checkpoint_every <= 0:
        p.error('Invalid seed or checkpoint interval')
    if not (src / 'manifest.json').is_file():
        p.error('Source manifest.json not found')
    m = json.loads((src / 'manifest.json').read_text())
    if m.get('schema_version') != 4 or m.get('shuffling') is not False:
        p.error('Expected original, unshuffled schema-v4 source')
    if m.get('document_shuffled') or m.get('document_shuffle_seed') is not None:
        p.error('Source already has document shuffling; use original source')
    n = int(m['train_docs'])
    if n < 1 or n >= 2**32:
        p.error('Document order uses uint32 and requires 1..2^32-1 training docs')
    if not (src / 'train.bin').is_file() or not (src / 'documents.sqlite').is_file():
        p.error('Source train.bin or documents.sqlite missing')
    if (src / 'train.bin').stat().st_size != 2 * int(m['train_tokens']):
        p.error('Source train.bin size disagrees with manifest')
    if (dst / 'manifest.json').is_file():
        completed = json.loads((dst / 'manifest.json').read_text())
        if (completed.get('document_shuffle_seed') != a.seed or
                completed.get('document_shuffle_source_manifest_sha256') != sha256_file(src / 'manifest.json')):
            p.error('Completed dataset does not match source/seed')
        verify_data(dst)
        ensure_chunk_order(dst, completed, a.seed, allow_create=True)
        print('[ORDER] Both saved shuffles already exist; no work needed', flush=True)
        return
    if dst.exists() and not a.resume:
        p.error('Output exists: use --resume or a NEW output directory')
    if not dst.exists() and a.resume:
        p.error('Cannot resume a missing destination directory')
    source_hash = sha256_file(src / 'manifest.json')
    print(f'[SOURCE] {n:,} documents, {int(m["train_tokens"]):,} tokens; loading document metadata', flush=True)
    # Fast indexed lookup by original source_row during randomized repacking.
    offsets = np.empty(n, dtype=np.int64)
    stored = np.empty(n, dtype=np.int64)
    textlens = np.empty(n, dtype=np.int64)
    hashes = np.empty(n, dtype='S64')
    original_db = sqlite3.connect(f'file:{src / "documents.sqlite"}?mode=ro', uri=True)
    try:
        last_end = 0
        for i, (sid, off, text_tokens, count, digest) in enumerate(original_db.execute('''
            SELECT source_row, token_offset, text_tokens, stored_tokens, text_sha256
            FROM documents WHERE split='train' ORDER BY source_row''')):
            if i >= n or int(sid) != i or int(off) != last_end or int(count) != int(text_tokens) + 1:
                raise RuntimeError(f'Non-contiguous source document metadata at index {i}')
            if len(digest) != 64:
                raise RuntimeError(f'Invalid hash at document {i}')
            offsets[i] = off
            stored[i] = count
            textlens[i] = text_tokens
            hashes[i] = digest.encode('ascii')
            last_end += count
        if i + 1 != n or last_end != int(m['train_tokens']):
            raise RuntimeError('Source document count/token totals disagree with manifest')
        print('[INDEX] Source documents are contiguous and complete', flush=True)
        order = np.random.default_rng(a.seed).permutation(n).astype('<u4')
        # Check all expected IDs via the bijection guaranteed by numpy permutation.
        if not a.resume:
            dst.mkdir(parents=True, exist_ok=False)
        db = sqlite3.connect(dst / 'documents.sqlite', timeout=120)
        try:
            create_db(db)
            existing = db.execute("SELECT value FROM progress WHERE key='shuffle_state'").fetchone()
            if existing:
                if not a.resume:
                    raise RuntimeError('Output already initialized')
                state = json.loads(existing[0])
                if state['seed'] != a.seed or state['source_manifest_sha256'] != source_hash:
                    raise RuntimeError('Resume source/seed mismatch')
            else:
                if a.resume:
                    raise RuntimeError('No durable progress in output')
                state = dict(next_doc=0, tokens=0, seed=a.seed, source_manifest_sha256=source_hash)
                db.execute('INSERT INTO progress(key,value) VALUES (?,?)',
                           ('shuffle_state', json.dumps(state)))
                db.commit()
            # Output is only finalized when manifest is published.
            if (dst / 'manifest.json').exists():
                raise RuntimeError('Destination already finalized; do not shuffle twice')
            next_doc = int(state['next_doc'])
            written = int(state['tokens'])
            count, high = db.execute("SELECT COUNT(*), COALESCE(MAX(token_offset+stored_tokens),0) FROM documents WHERE split='train'").fetchone()
            if count != next_doc or high != written:
                raise RuntimeError('Output SQLite metadata and progress disagree')
            output_file = dst / 'train.bin'
            if next_doc and not output_file.exists():
                raise RuntimeError('Committed output binary missing')
            if output_file.exists() and output_file.stat().st_size < 2 * written:
                raise RuntimeError('Output binary shorter than durable checkpoint')
            source_tokens = np.memmap(src / 'train.bin', dtype='<u2', mode='r')
            if next_doc == 0 and not output_file.exists():
                output_file.touch()
            with output_file.open('r+b', buffering=4 << 20) as output:
                output.truncate(2 * written)  # discard interrupted, uncommitted writes
                output.seek(2 * written)
                start_time = time.monotonic()
                for batch_start in range(next_doc, n, a.checkpoint_every):
                    batch_end = min(n, batch_start + a.checkpoint_every)
                    entries = []
                    for j in range(batch_start, batch_end):
                        old_id = int(order[j])
                        old_offset = int(offsets[old_id])
                        length = int(stored[old_id])
                        # Byte-for-byte copy: no tokenizer invocation or EOS insertion.
                        output.write(source_tokens[old_offset:old_offset + length].tobytes())
                        entries.append((j, 'train', written, int(textlens[old_id]),
                                        length, hashes[old_id].decode('ascii')))
                        written += length
                    output.flush()
                    os.fsync(output.fileno())
                    db.executemany('''INSERT INTO documents
                        (source_row,split,token_offset,text_tokens,stored_tokens,text_sha256)
                        VALUES (?,?,?,?,?,?)''', entries)
                    state['next_doc'], state['tokens'] = batch_end, written
                    db.execute('UPDATE progress SET value=? WHERE key=?',
                               (json.dumps(state), 'shuffle_state'))
                    db.commit()
                    elapsed = time.monotonic() - start_time
                    print(f'[SHUFFLE] {batch_end:,}/{n:,} docs; {written:,} tokens; '
                          f'{(batch_end-next_doc)/max(1,elapsed):.1f} docs/s', flush=True)
            del source_tokens
            if written != int(m['train_tokens']):
                raise RuntimeError('Shuffled training token count changed')
            # Persist random visit order of original training-document source IDs.
            with (dst / 'document_order.u32.tmp').open('wb') as f:
                order.tofile(f)
                f.flush(); os.fsync(f.fileno())
            os.replace(dst / 'document_order.u32.tmp', dst / 'document_order.u32')
            # Validation is NEVER shuffled. Preserve exact validation bytes and metadata.
            src_val_hash = sha256_file(src / 'val.bin')
            dst_val = dst / 'val.bin'
            if not dst_val.exists() or sha256_file(dst_val) != src_val_hash:
                tmp_val = dst / 'val.bin.tmp'
                shutil.copyfile(src / 'val.bin', tmp_val)
                if sha256_file(tmp_val) != src_val_hash:
                    raise RuntimeError('Validation copy checksum failed')
                os.replace(tmp_val, dst_val)
            already_val = db.execute("SELECT COUNT(*) FROM documents WHERE split='val'").fetchone()[0]
            if already_val == 0:
                batch = []
                for j, (_, off, text_tokens, count, digest) in enumerate(original_db.execute('''
                    SELECT source_row, token_offset, text_tokens, stored_tokens, text_sha256
                    FROM documents WHERE split='val' ORDER BY token_offset''')):
                    batch.append((n + j, 'val', off, text_tokens, count, digest))
                if len(batch) != int(m['val_docs']):
                    raise RuntimeError('Source validation document count mismatch')
                db.executemany('''INSERT INTO documents
                    (source_row,split,token_offset,text_tokens,stored_tokens,text_sha256)
                    VALUES (?,?,?,?,?,?)''', batch)
                db.commit()
            elif already_val != int(m['val_docs']):
                raise RuntimeError('Partially inserted validation metadata')
            if not (dst / 'tokenizer').exists():
                tmp_tok = dst / 'tokenizer.tmp'
                if tmp_tok.exists():
                    shutil.rmtree(tmp_tok)
                shutil.copytree(src / 'tokenizer', tmp_tok)
                os.replace(tmp_tok, dst / 'tokenizer')
            db.execute('CREATE INDEX IF NOT EXISTS documents_split_offset ON documents(split,token_offset)')
            db.commit()
            manifest = dict(m)
            # Existing data.py expects `shuffling=False` to mean prepared *blocks*
            # are not pre-shuffled; document order is recorded separately here.
            manifest.update(document_shuffled=True,
                            document_shuffle_seed=a.seed,
                            document_shuffle_source_manifest_sha256=source_hash,
                            document_shuffle_index='document_order.u32',
                            document_shuffle_index_semantics='new_document_position_to_original_source_row',
                            document_shuffle_stage='after_tokenization_before_1024_token_chunking')
            # Existing checkpoint verification works with the new contiguous row IDs.
            atomic_json(dst / 'manifest.json', manifest)
            verify_data(dst)
            ensure_chunk_order(dst, manifest, a.seed, allow_create=True)
            print(f'[DONE] Both document and fixed chunk ordering ready: {dst}', flush=True)
        finally:
            db.close()
    finally:
        original_db.close()



def preparation_cli():
    p = argparse.ArgumentParser(description='One-time CPU preparation of document + chunk shuffles')
    p.add_argument('--prepare-order', action='store_true', required=True)
    p.add_argument('--data', type=Path, required=True, help='Existing ordered, tokenized source dataset')
    p.add_argument('--out', type=Path, required=True, help='New shuffled dataset directory')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint-every', type=int, default=20_000)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    prepare_document_shuffle(a)


def main():
    args = get_args()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 4 or torch.cuda.device_count() != 4:
        raise RuntimeError('Launch with torchrun --standalone --nproc_per_node=4 on four GPUs')
    if args.batch % world != 0:
        raise ValueError('--batch must be divisible by 4')
    torch.cuda.set_device(local_rank)
    device = require_cuda(local_rank)
    dist.init_process_group(backend='nccl')
    stop = {'requested': False}
    stop_file = os.environ.get('TRAIN_STOP_FILE', '')
    result_file = os.environ.get('TRAIN_SEGMENT_RESULT', '')
    test_stop_step = int(os.environ.get('TRAIN_TEST_STOP_STEP', '0'))
    expected_resume = os.environ.get('TRAIN_EXPECTED_RESUME_STEP', '')
    writer = CheckpointWriter() if rank == 0 else None
    run_lock = None
    def request_stop(sig, frame):
        stop['requested'] = True
        if rank == 0:
            print(f'[SIGNAL] {sig}: save after next optimizer update', flush=True)
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        folder = Path(args.data)
        manifest = verify_data(folder)
        # Rank zero generates the one-time chunk permutation BEFORE training.
        # A broadcast makes other ranks wait until all order files are complete.
        rank_zero_io(rank, lambda: ensure_chunk_order(folder, manifest, args.seed))
        train = PackedSampler(folder, 'train', args.block_size, args.seed)
        val = PackedSampler(folder, 'val', args.block_size, args.seed + 100)
        try:
            capacity = math.ceil(train.n_blocks / (args.batch * args.accum))
            # Each *microbatch* must contain >=world chunks so all ranks
            # participate in DDP backward. With accumulation, the last update
            # can be large yet its final microbatch may contain only 1..3 chunks.
            # Drop that entire final update rather than silently skip its last
            # microbatch and misreport token exposure / loss normalization.
            remainder = train.n_blocks % (args.batch * args.accum)
            final_micro_remainder = remainder % args.batch
            if 0 < final_micro_remainder < world:
                capacity -= 1
                if rank == 0:
                    print(f'[DATA] skipping final update with a microbatch of '
                          f'{final_micro_remainder} (<{world}) chunks; '
                          f'{remainder} final chunks will remain unvisited', flush=True)
            if args.steps == 0:
                args.steps = capacity
            if args.steps < 1 or args.steps > capacity or args.warmup >= args.steps:
                raise ValueError(f'Invalid step count or warmup: steps={args.steps} full_pass={capacity}')
            # Dataset preflight via public sampler interface (no private mmap assumptions).
            gx, gy = train.batch(0, 0, args.batch, args.accum)
            if gx.shape != (min(args.batch, train.n_blocks), 1024) or gy.shape != gx.shape:
                raise RuntimeError('Preflight: sampler returned incorrect batch shape')
            if not torch.equal(gx[:, 1:], gy[:, :-1]):
                raise RuntimeError('Preflight: GPT-2 next-token targets are misaligned')
            if int(gx.min()) < 0 or int(gy.min()) < 0 or int(gx.max()) >= 50257 or int(gy.max()) >= 50257:
                raise RuntimeError('Preflight: token ID outside GPT-2 vocabulary')
            logical = train.block_ids(0, args.batch)
            if len(logical) != gx.shape[0] or len(set(logical)) != len(logical):
                raise RuntimeError('Preflight: initial shuffled block IDs repeat or mismatch')
            if rank == 0:
                print('[PREFLIGHT] sample shapes, target shift, vocabulary, block IDs PASS', flush=True)
            if rank == 0:
                print(f'[DATA] train_blocks={train.n_blocks:,} val_blocks={val.n_blocks:,} '
                      f'optimizer_steps={args.steps:,} effective_batch={args.batch * args.accum:,}', flush=True)
            if args.preflight_only:
                return
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            amp_dtype = choose_precision(args.precision, device)
            cast = lambda: autocast_context(amp_dtype)
            run = args.run_root / args.run
            signature = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                         if k not in ('resume', 'eta_window', 'preflight_only')}
            root = Path(__file__).resolve().parent
            signature['sampler_order'] = SAMPLER_ORDER
            signature['sampler_seed'] = args.seed
            signature['world_size'] = world
            signature['parallelism'] = 'ddp'
            signature['manifest_sha256'] = file_sha(folder / 'manifest.json')
            # Pin the actual 1-time shuffled chunk order in the checkpoint signature.
            signature['chunk_order_sha256'] = file_sha(folder / 'train_chunk_order.u32') if rank == 0 else None
            signature['chunk_inverse_sha256'] = file_sha(folder / 'train_chunk_inverse.u32') if rank == 0 else None
            signature['implementation_sha256'] = {name: file_sha(root / name)
                                                   for name in (Path(__file__).name, 'model.py', 'data.py', 'runtime.py')}
            signature['precision_dtype'] = str(amp_dtype)  # Stable across nodes; runtime details saved separately
            # Device-specific metadata may differ by rank; rank zero provides the canonical signature.
            items = [signature if rank == 0 else None]
            dist.broadcast_object_list(items, src=0)
            signature = items[0]
            def initialize_run():
                nonlocal run_lock
                if not args.resume:
                    run.mkdir(parents=True, exist_ok=False)
                run_lock = (run / '.trainer.lock').open('a')
                fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

            rank_zero_io(rank, initialize_run)
            ckpt = None
            if args.resume:
                ckpt = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
                if ckpt['signature'] != signature:
                    raise ValueError('Resume settings, source, environment, or implementation changed')
                start = int(ckpt['step'])
                if not 0 <= start <= args.steps:
                    raise ValueError(f'Checkpoint step outside training budget: {start}')
                if expected_resume and start != int(expected_resume):
                    raise ValueError(f'Expected checkpoint step {expected_resume}, loaded {start}')
                if ckpt['blocks_consumed'] != block_count_after(start, args, train.n_blocks):
                    raise ValueError('Checkpoint sample cursor mismatch')
                if len(ckpt.get('rng_per_rank', [])) != world:
                    raise ValueError('Checkpoint missing rank-specific RNG states')
                model = GPT(GPTConfig(**ckpt['config'])).to(device)
                model.load_state_dict(ckpt['model'], strict=True)
            else:
                model = make_model('small', 1024).to(device)
                start = 0
            if ((model.config.n_layer, model.config.n_head, model.config.n_embd, model.config.block_size) != (12, 12, 768, 1024) or model.config.dropout != 0.0 or model.config.vocab_size != manifest['vocab_size']):
                raise RuntimeError('Unexpected GPT-2 Small configuration')
            if model.wte.weight is not model.head.weight:
                raise RuntimeError('GPT-2 embedding/head weights must be tied')
            optimizer = torch.optim.AdamW([
                {'params': [p for p in model.parameters() if p.ndim >= 2], 'weight_decay': args.weight_decay},
                {'params': [p for p in model.parameters() if p.ndim < 2], 'weight_decay': 0.0}],
                lr=args.lr, betas=(args.beta1, args.beta2), fused=True)
            scaler = make_scaler(amp_dtype)
            if args.resume:
                optimizer.load_state_dict(ckpt['optimizer'])
                scaler.load_state_dict(ckpt['scaler'])
                set_rng(ckpt['rng_per_rank'][rank])
                scheduler = ckpt['scheduler']
                if (scheduler['next_step_index'] != start
                        or scheduler['total_steps'] != args.steps
                        or scheduler['last_applied_lr'] != optimizer.param_groups[0]['lr']):
                    raise ValueError('Checkpoint LR progress disagrees with optimizer or step')
                if start and not optimizer.state:
                    raise ValueError('Checkpoint is missing AdamW state')
                del ckpt

                def restore_logs():
                    for name in ('train_loss_history.jsonl', 'metrics.jsonl', 'generation_checks.jsonl'):
                        trim_log(run / name, start)
                    print(f'[RESUME] step={start} next_step={start + 1} '
                          f'optimizer_states={len(optimizer.state)} rng_ranks={world}', flush=True)

                rank_zero_io(rank, restore_logs)
            else:
                # Identical initial weights, independent dropout RNG streams.
                torch.manual_seed(args.seed + rank * 100003)
                torch.cuda.manual_seed_all(args.seed + rank * 100003)
                if rank == 0:
                    write_json(run / 'config.json', signature)
                    write_json(run / 'environment.json', provenance(device, amp_dtype))
            # Compile the model used for training; keep the raw model for state_dict,
            # gradient clipping, evaluation and text generation.
            compiled_model = torch.compile(model)
            ddp = DDP(compiled_model, device_ids=[local_rank], output_device=local_rank,
                      broadcast_buffers=False, gradient_as_bucket_view=True)
            if not args.resume:
                checkpoint_ddp(model, optimizer, scaler, run, 0, signature, args,
                               train.n_blocks, None, rank, world, writer)
            checkpoint_interval = args.save_hours * 3600
            next_checkpoint = time.monotonic() + checkpoint_interval

            def finish_segment(status, done):
                def publish_result():
                    writer.flush()
                    if result_file:
                        write_json(Path(result_file), {
                            'status': status, 'start_step': start, 'step': done,
                            'target_steps': args.steps, 'optimizer_states': len(optimizer.state),
                        })
                    if status == 'stopped':
                        print(f'[WALLTIME_STOP] checkpointed step={done:,}; resume with --resume', flush=True)
                    else:
                        print(f'[DONE] step={done:,} latest={run / "latest.pt"}', flush=True)

                rank_zero_io(rank, publish_result)
            if rank == 0:
                print(f'[GPU] DDP world_size={world} total_microbatch={args.batch} '
                      f'per_gpu={args.batch//world} accum={args.accum} '
                      f'effective_batch={args.batch*args.accum} local_loss=True', flush=True)
                print(f'[MODEL] parameters={sum(p.numel() for p in model.parameters()):,} '
                      f'start_step={start:,} precision={amp_dtype}', flush=True)
            tokenizer = None
            if args.check_every and rank == 0:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(folder / 'tokenizer', local_files_only=True)
            # Wall-clock ETA from a rolling window, with a whole-segment fallback.
            dist.barrier(device_ids=[local_rank])
            torch.cuda.synchronize(device)
            t0 = time.monotonic()
            samples = deque([(start, t0)])
            # Rank zero owns the progress bar; other ranks never write progress UI.
            # tqdm uses elapsed wall time to estimate the remainder of this one-epoch run.
            # A restarted Slurm segment begins with initial=start (not zero).
            progress = (tqdm(total=args.steps, initial=start, desc='Pile epoch 1/1',
                             unit='step', mininterval=30, miniters=20,
                             dynamic_ncols=False, ascii=True, file=sys.stderr,
                             disable=False)
                        if rank == 0 and tqdm is not None else None)
            if rank == 0 and tqdm is None:
                print('[ETA] tqdm unavailable; using periodic [TRAIN] ETA logs instead', flush=True)
            for step in range(start, args.steps):
                data_ms = 0.0
                forward_ms = 0.0
                backward_ms = 0.0
                ddp.train()
                lr = lr_schedule(step, args)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                global_start = step * args.batch * args.accum
                expected_this_step = min(train.n_blocks - global_start, args.batch * args.accum)
                if expected_this_step <= 0:
                    raise RuntimeError('No blocks remain for training')
                total_targets = expected_this_step * args.block_size
                total_loss = torch.zeros((), device=device, dtype=torch.float64)
                for micro in range(args.accum):
                    first = global_start + micro * args.batch
                    n = max(0, min(args.batch, train.n_blocks - first))
                    if not n:
                        continue
                    a = n * rank // world
                    b = n * (rank + 1) // world
                    # The final partial batch is split evenly; this avoids DDP rank mismatch.
                    if a == b:
                        raise RuntimeError('Unsupported final partial update: a GPU has no samples')
                    data_start = time.perf_counter()
                    gx, gy = train.batch_range(first + a, b - a)
                    if gx.shape != (b-a, args.block_size) or gy.shape != gx.shape:
                        raise RuntimeError('PackedSampler.batch_range returned inconsistent shape')
                    x, y = gx.to(device, non_blocking=True), gy.to(device, non_blocking=True)
                    data_ms += (time.perf_counter() - data_start) * 1000
                    forward_start = time.perf_counter()
                    sync = (micro == args.accum - 1 or first + args.batch >= train.n_blocks)
                    ctx = contextlib.nullcontext() if sync else ddp.no_sync()
                    with ctx:
                        with cast():
                            logits, _ = ddp(x)
                            loss_sum = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                                       y.reshape(-1), reduction='sum')
                        if not torch.isfinite(loss_sum):
                            raise FloatingPointError(f'Nonfinite loss at step {step}, rank {rank}')
                        forward_ms += (time.perf_counter() - forward_start) * 1000
                        backward_start = time.perf_counter()
                        # DDP averages gradients across ranks. Compensate with world_size,
                        # normalize by actual GLOBAL tokens (including a final partial batch).
                        scaled_loss = loss_sum * (world / total_targets)
                        scaler.scale(scaled_loss).backward()
                        total_loss += loss_sum.detach().to(torch.float64)
                    backward_ms += (time.perf_counter() - backward_start) * 1000
                    del logits, loss_sum, scaled_loss
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f'Nonfinite global gradient at step {step}')
                # All ranks decide whether to abort BEFORE applying this update.
                # This is a stability check, not a guarantee of convergence.
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                current_loss = float((total_loss / total_targets).item())
                done = step + 1
                if not math.isfinite(current_loss):
                    raise RuntimeError(f'Nonfinite global training loss at step {done}')
                if args.max_loss and done >= 300 and current_loss > args.max_loss:
                    raise RuntimeError(f'[DIVERGENCE] step={done} train_loss={current_loss:.4f} exceeds max_loss={args.max_loss:.4f}; optimizer update NOT applied')
                optimizer_start = time.perf_counter()
                checked_optimizer_step(scaler, optimizer)
                # Synchronize only on reporting steps for accurate optimizer latency.
                # Do not add a synchronization to every training step.
                if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
                    torch.cuda.synchronize(device)
                optimizer_ms = (time.perf_counter() - optimizer_start) * 1000
                consumed = block_count_after(done, args, train.n_blocks)
                if done % args.log_every == 0 or done == args.steps:
                    torch.cuda.synchronize(device)
                    stamp = time.monotonic()
                    samples.append((done, stamp))
                    while len(samples) > 2 and done - samples[1][0] >= args.eta_window:
                        samples.popleft()
                    prior_step, prior_time = samples[0]
                    recent_sps = (done - prior_step) / max(0.001, stamp - prior_time)
                    overall_sps = (done - start) / max(0.001, stamp - t0)
                    # Report both estimates; actual elapsed includes training and any validation/checkpoint
                    # that occurred during the measured interval.
                    eta_rolling = (args.steps - done) / recent_sps / 3600
                    eta_segment = (args.steps - done) / overall_sps / 3600
                    if rank == 0:
                        # UTC completion timestamp estimated from recent measured throughput.
                        estimated_finish = datetime.fromtimestamp(
                            time.time() + eta_rolling * 3600, tz=timezone.utc
                        ).isoformat()
                        print(f'[ETA] completed={done:,}/{args.steps:,} '
                              f'epoch={consumed/train.n_blocks:.6f}/1 '
                              f'rolling_hours={eta_rolling:.2f} '
                              f'finish_utc={estimated_finish}', flush=True)
                        with (run / 'train_loss_history.jsonl').open('a') as f:
                            f.write(json.dumps({'step': done, 'train_loss': current_loss}) + '\n')
                        print(f'[TRAIN] {datetime.now(timezone.utc).isoformat()} '
                              f'{done:,}/{args.steps:,} loss={current_loss:.5f} lr={lr:.3e} '
                              f'grad_norm={grad_norm.item():.3f} blocks={consumed:,}/{train.n_blocks:,} '
                              f'data_ms={data_ms:.2f} forward_ms={forward_ms:.2f} '
                               f'backward_ms={backward_ms:.2f} optimizer_ms={optimizer_ms:.2f} '
                               f'steps_per_second={recent_sps:.4f} tokens_per_second={recent_sps*args.batch*args.accum*1024:,.0f} '
                              f'eta_hours={eta_rolling:.2f} eta_segment_hours={eta_segment:.2f}', flush=True)
                if args.check_every and done >= args.check_begin and (done - args.check_begin) % args.check_every == 0:
                    if rank == 0:
                        generation_check(model, tokenizer, args.check_prompts, args, device, cast, done, run)
                    dist.barrier(device_ids=[local_rank])
                if done % args.eval_every == 0 or done == args.steps:
                    vl, vt = validate_ddp(model, val, args, device, cast, rank, world)
                    if rank == 0:
                        record = {'step': done, 'train_loss': current_loss, 'val_loss': vl,
                                  'val_perplexity': math.exp(min(vl, 50)), 'val_scored_tokens': vt,
                                  'lr': lr, 'prediction_tokens': consumed * 1024,
                                  'unique_training_blocks_visited': consumed,
                                  'training_block_count': train.n_blocks, 'sampler_order': SAMPLER_ORDER}
                        with (run / 'metrics.jsonl').open('a') as f:
                            f.write(json.dumps(record) + '\n')
                        print(f'[VAL] step={done:,} loss={vl:.5f} ppl={record["val_perplexity"]:.3f}', flush=True)
                # A single collective coordinates write errors, stop requests,
                # and the rank-zero wall-clock checkpoint deadline.
                control = 0
                if rank == 0 and time.monotonic() >= next_checkpoint:
                    control = 1
                if stop['requested'] or (rank == 0 and (
                    (stop_file and Path(stop_file).is_file())
                    or (test_stop_step and done == test_stop_step)
                )):
                    control = 2
                if rank == 0 and writer.failed():
                    control = 3
                decision = torch.tensor(control, device=device)
                dist.all_reduce(decision, op=dist.ReduceOp.MAX)
                control = decision.item()
                if control == 3:
                    rank_zero_io(rank, lambda: writer.flush())
                stopping = control == 2 and done < args.steps
                step_due = args.save_every and done % args.save_every == 0
                if control or step_due or done == args.steps:
                    checkpoint_ddp(model, optimizer, scaler, run, done, signature, args,
                                   train.n_blocks, current_loss, rank, world, writer)
                    next_checkpoint = time.monotonic() + checkpoint_interval
                    if done == args.steps or (step_due and done % 50000 == 0):
                        rank_zero_io(rank, lambda: update_training_plots(run))
                if progress is not None:
                    progress.update(1)
                    if done % args.log_every == 0 or done == args.steps:
                        progress.set_postfix_str(
                            f'loss={current_loss:.4f} lr={lr:.2e}', refresh=False
                        )
                if stopping:
                    if progress is not None:
                        progress.close()
                    finish_segment('stopped', done)
                    return
            if progress is not None:
                progress.close()
            finish_segment('complete', args.steps)
        finally:
            train.close()
            val.close()
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            if run_lock is not None:
                run_lock.close()
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == '__main__':
    if '--prepare-order' in sys.argv:
        preparation_cli()
    else:
        main()
