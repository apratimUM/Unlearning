#!/usr/bin/env python3

from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import nullcontext
import inspect
import math

EXPECTED_SOURCE_DOCS = 134_318_121  # From Unlearning/prepare.py, deduplicated Pile rows

# Frances: each independent process receives exactly ONE physical GPU through
# CUDA_VISIBLE_DEVICES from the Slurm launcher. CUDA maps it to local cuda:0.
# No GPU selection is rewritten inside Python.
BASE = Path(os.environ.get('PILE_EMBED_BASE', '/coe-aroyc/apratim'))
os.environ.setdefault('HF_HOME', str(BASE / 'hf_cache'))
os.environ.setdefault('HF_HUB_CACHE', str(BASE / 'hf_cache' / 'hub'))
os.environ.setdefault('PIP_CACHE_DIR', str(BASE / 'pip_cache'))
os.environ.setdefault('TMPDIR', str(BASE / 'tmp'))
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True

BLOCK = 1024
MODELS = {
    'qwen3': 'Qwen/Qwen3-Embedding-0.6B',
    'gte': 'Alibaba-NLP/gte-modernbert-base',
    'bge': 'BAAI/bge-m3',
    'jina2': 'jinaai/jina-embeddings-v2-base-en',
    'jina3': 'jinaai/jina-embeddings-v3',
    'snowflake': 'Snowflake/snowflake-arctic-embed-l-v2.0',
    'nomic_modern': 'nomic-ai/modernbert-embed-base',
    'gemma': 'google/embeddinggemma-300m',
    'gte_qwen2': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    'granite': 'ibm-granite/granite-embedding-97m-multilingual-r2',
}
TRUST_REMOTE = {'jina2', 'jina3', 'gte_qwen2'}
METRICS = ('cosine',)


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=BASE / 'pile_full_data')
    p.add_argument('--out', type=Path, default=BASE / 'embedding_results' / 'bbc_pile_frances_4gpu' / 'standalone')
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--span-lengths', type=int, nargs='+', default=[32, 56, 128, 256, 512])
    p.add_argument('--spans-per-topic', type=int, default=5)
    p.add_argument('--max-blocks', type=int, default=0, help='0 (default) = scan all TRAINING-indexed chunks including the padded final chunk; positive = pilot sample')
    p.add_argument('--inspect-data', action='store_true', help='Inspect local counts/manifest and exit; no GPU, models, or BBC downloads')
    p.add_argument('--topk', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=16, help='GPU inference batch size')
    p.add_argument('--cache-dir', type=Path, default=BASE / 'pile_decoded_cache_v1',
                   help='Persistent decode-once cache; required for all embedding runs')
    p.add_argument('--cache-only', action='store_true', help='Build/resume cache then exit; requires CPU only')
    p.add_argument('--cache-workers', type=int, default=8, help='Concurrent GPT-2 decode workers')
    p.add_argument('--cache-decode-batch', type=int, default=256, help='Chunks decoded per CPU task')
    p.add_argument('--cache-shard-chunks', type=int, default=250000, help='Physical chunks per cache shard')
    p.add_argument('--cache-build-limit', type=int, default=0, help='Build this many chunks for cache-only pilot; 0=all')

    p.add_argument('--backend', choices=['auto','direct'], default='auto', help='Single-tokenization direct-forward only; startup equivalence test must pass or the model fails')
    p.add_argument('--equivalence-atol', type=float, default=0.0002, help='maximum absolute embedding error for direct backend equivalence')
    p.add_argument('--profile-every', type=int, default=10000, help='Report decode/tokenize/model/score wall times per window; 0 disables')
    p.add_argument('--prefetch', type=int, default=2, help='Number of CPU-decoded batches to prepare ahead (0 disables)')
    p.add_argument('--rank-window', type=int, default=1024, help='Accumulate this many scored chunks per vectorized top/bottom-k update')
    p.add_argument('--precision', choices=['fp32','bf16','fp16'], default='fp32', help='Optional GPU autocast; fp32 is reference numerics')
    p.add_argument('--progress-every', type=int, default=10000, help='Print throughput independently of checkpoints (0 disables)')
    p.add_argument('--print-full-chunks', action='store_true', help='Print full decoded top/bottom cosine chunks to stdout (very verbose)')
    p.add_argument('--provenance-probes', type=int, default=20, help='Additional random SQLite / EOS source-integrity probes')
    p.add_argument('--save-every', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true', help='Restore the last durable checkpoint in --out; reuse exact original settings')
    p.add_argument('--save-vectors', action='store_true', help='Persist float16 chunk embeddings aligned to selected_physical_chunk_ids.npy')
    p.add_argument('--self-test', action='store_true', help='CPU synthetic integrity smoke test; no downloads or GPU required')
    p.add_argument('--skip-failed-models', action='store_true', help='Record failure and continue rather than stopping')
    p.add_argument('--span-source', choices=['huggingface','jsonl'], default='huggingface',
                   help='Span source, independent of fixed shuffled Pile training source')
    p.add_argument('--span-dataset', default='SetFit/bbc-news', help='Hugging Face span dataset')
    p.add_argument('--span-split', default='test', help='Hugging Face span split')
    p.add_argument('--span-jsonl', type=Path, default=None, help='Local JSONL source with text and topic/label')
    p.add_argument('--span-text-field', default='text')
    p.add_argument('--span-label-field', default='label_text')
    p.add_argument('--span-label-map', type=Path, default=None, help='Optional JSON object mapping numeric labels to readable topics')
    p.add_argument('--bbc-jsonl', type=Path, default=None, help='Backward-compatible alias for --span-jsonl')
    a = p.parse_args()
    if a.data.resolve() == a.out.resolve() or a.out.resolve().is_relative_to(a.data.resolve()):
        p.error('Outputs must not be inside the dataset directory')
    if not a.out.resolve().is_relative_to(BASE):
        p.error(f'Outputs must remain under Frances BASE: {BASE}')
    if a.self_test or a.inspect_data: return a
    if min(a.cache_workers, a.cache_decode_batch, a.cache_shard_chunks) < 1 or a.cache_build_limit < 0:
        p.error('Invalid cache settings')
    if not a.cache_dir.resolve().is_relative_to(BASE) or a.cache_dir.resolve().is_relative_to(a.data.resolve()):
        p.error('Cache directory must be under BASE but outside the source data folder')
    if min(a.span_lengths) < 1 or min(a.spans_per_topic, a.topk, a.batch_size, a.save_every) < 1 or a.max_blocks < 0 or a.provenance_probes < 0 or a.prefetch < 0 or a.rank_window < 1 or a.progress_every < 0 or a.profile_every < 0 or a.equivalence_atol <= 0:
        p.error('Invalid counts')
    if a.bbc_jsonl is not None:
        a.span_source = 'jsonl'
        a.span_jsonl = a.bbc_jsonl
    if a.span_source == 'jsonl' and a.span_jsonl is None:
        p.error('--span-source jsonl requires --span-jsonl')
    a.span_lengths = sorted(set(a.span_lengths))
    a.models = list(dict.fromkeys(a.models))
    return a


class GracefulStop:
    """Request an orderly checkpoint at the next completed batch."""
    requested = False
    signum = None

    def handler(self, signum, frame):
        self.requested = True
        self.signum = signum
        print(f"[SIGNAL] {signum} received; checkpointing after current batch", flush=True)


def install_stop_handlers(stop):
    old = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        old[sig] = signal.getsignal(sig)
        signal.signal(sig, stop.handler)
    return old


def restore_stop_handlers(old):
    for sig, handler in old.items():
        signal.signal(sig, handler)


def load_progress(path, ranks, total):
    """Validate an atomic checkpoint before trusting its cursor or rankings."""
    with np.load(path, allow_pickle=False) as saved:
        cursor = int(saved['cursor'][0])
        if not 0 <= cursor <= total:
            raise RuntimeError(f'Invalid checkpoint cursor {cursor}/{total}')
        ranks.restore(saved)
        for metric in METRICS:
            for ids in (ranks.hid[metric], ranks.lid[metric]):
                present = ids[ids >= 0]
                if np.any(present < 0):
                    raise RuntimeError('Invalid ranking IDs in checkpoint')
        return cursor


def atomic_json(path, data):
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def file_ident(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def dataset(data):
    src = data / 'train_shuffled.bin'  # NEVER fall back to train.bin
    manifest_path = data / 'manifest_shuffled.json'
    order_meta = data / 'train_chunk_order_shuffled.json'
    order = data / 'train_chunk_order_shuffled.u32'
    inverse = data / 'train_chunk_inverse_shuffled.u32'
    index = data / 'documents_shuffled.sqlite'
    doc_order_path = data / 'document_order_shuffled.u32'
    for p in [src, manifest_path, order_meta, order, inverse, index, doc_order_path]:
        if not p.is_file():
            raise FileNotFoundError(p)
    manifest = json.loads(manifest_path.read_text())
    info = json.loads(order_meta.read_text())
    assert manifest['document_shuffled'] is True, 'Not a document-shuffled dataset'
    assert int(manifest['block_size']) == BLOCK
    assert np.dtype(manifest['token_dtype']) == np.dtype('<u2')
    assert int(manifest['eos_token_id']) == 50256
    size = src.stat().st_size
    if size % 2:
        raise RuntimeError(f'Corrupt uint16 token file: odd byte size {size}')
    total_tokens = size // 2
    if total_tokens != int(manifest['train_tokens']):
        raise RuntimeError(f'Token file has {total_tokens:,} tokens; manifest says {int(manifest["train_tokens"]):,}')
    # Same indexed-chunk policy as the updated Unlearning PackedSampler:
    # each chunk needs at least one input and one next-token target.
    # The last chunk may be shorter than BLOCK and is NOT physically padded.
    n = (total_tokens - 1 + BLOCK - 1) // BLOCK if total_tokens >= 2 else 0
    indexed = int(info['n_blocks'])
    declared = int(manifest['train_blocks'])
    print(f'[DATA] source Pile documents: {manifest.get("train_docs")}; expected: {EXPECTED_SOURCE_DOCS:,}', flush=True)
    print(f'[DATA] stored tokens: {total_tokens:,}; indexed training chunks: {n:,}; final input tokens: {min(BLOCK, total_tokens-1-(n-1)*BLOCK) if n else 0}; final targets: {min(BLOCK, total_tokens-1-(n-1)*BLOCK) if n else 0}', flush=True)
    print(f'[DATA] manifest train_blocks: {declared:,}; shuffled-index blocks: {indexed:,}', flush=True)
    if manifest.get('source_scan_complete') is not True or int(manifest.get('train_docs', -1)) != EXPECTED_SOURCE_DOCS:
        raise RuntimeError('NOT COMPLETE: The shuffled manifest does not certify all 134,318,121 deduplicated Pile source documents.')
    if int(manifest.get('documents_seen', -1)) < EXPECTED_SOURCE_DOCS:
        raise RuntimeError('Manifest documents_seen does not cover the entire Pile')
    if n == 0 or n != declared or n != indexed:
        raise RuntimeError(f'Chunk count mismatch: computed={n:,}, manifest={declared:,}, index={indexed:,}')
    if manifest.get('incomplete_final_block') != 'eos_right_padded_targets_ignore_index_-100':
        raise RuntimeError('Manifest final-chunk policy does not match updated PackedSampler')
    if info['algorithm'] != 'precomputed_numpy_permutation_v1':
        raise RuntimeError('Unknown shuffle order algorithm')
    if int(info['seed']) != int(manifest['document_shuffle_seed']):
        raise ValueError('Shuffled seed mismatch')
    recorded_hash = info.get('source_manifest_sha256')
    if recorded_hash is not None and recorded_hash != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError('Chunk permutation was not produced for this shuffled manifest')
    if order.stat().st_size != inverse.stat().st_size or order.stat().st_size != indexed * 4:
        raise RuntimeError('The shuffle index files do not match the indexed training block count')
    if manifest.get('document_shuffle_index') != 'document_order_shuffled.u32' or manifest.get('document_shuffle_index_semantics') != 'new_document_position_to_original_source_row':
        raise ValueError('Document shuffle provenance metadata mismatch')
    if doc_order_path.stat().st_size != 4 * int(manifest['train_docs']):
        raise ValueError('Document order index size mismatch')
    doc_order = np.memmap(doc_order_path, dtype='<u4', mode='r')
    mm = np.memmap(src, dtype='<u2', mode='r')
    inv = np.memmap(inverse, dtype='<u4', mode='r')
    perm = np.memmap(order, dtype='<u4', mode='r')
    last_visit = int(inv[n - 1])
    if not 0 <= last_visit < n or int(perm[last_visit]) != n - 1:
        raise RuntimeError('Final training chunk missing or inconsistent in saved shuffle permutation')
    print(f'[DATA] final physical chunk ID: {n-1:,}; shuffled training visit ID: {last_visit:,}; actual GPT-2 input length: {len(chunk_token_ids(mm,n-1))}', flush=True)
    return mm, inv, perm, doc_order, n, manifest, info, {name: file_ident(p) for name, p in [
        ('train_shuffled',src),('manifest_shuffled',manifest_path),('chunk_order',order),('chunk_inverse',inverse),('document_order',doc_order_path),('documents_shuffled',index)]}



def chunk_token_ids(mm, bid):
    """Return exactly the REAL GPT-2 input IDs used by PackedSampler.get_block().

    Each training block starts at 1024*physical_id. data.py reads 1025 token
    positions (1024 inputs + the next-token label), right-pads input IDs with
    EOS and masks missing target positions using -100. For embedding, exclude
    both synthetic input padding and the terminal token used ONLY as a target.
    """
    start = int(bid) * BLOCK
    end = min(start + BLOCK, len(mm) - 1)
    if start >= len(mm) - 1:
        raise RuntimeError(f'Invalid or empty training-indexed chunk: {bid}')
    ids = np.asarray(mm[start:end], dtype=np.int64)
    if not 1 <= len(ids) <= BLOCK or np.any(ids < 0) or np.any(ids >= 50257):
        raise RuntimeError(f'Bad GPT-2 input IDs or chunk length at {bid}: {len(ids)}')
    return ids


def span_rows(a):
    """Topic-stratified span sources with stable source row and all original metadata."""
    label_map = json.loads(a.span_label_map.read_text()) if a.span_label_map else {}
    if a.span_source == 'jsonl':
        def rows():
            with a.span_jsonl.open(encoding='utf-8') as f:
                for k, line in enumerate(f):
                    if line.strip():
                        yield k, json.loads(line)
        iterator = rows()
    else:
        from datasets import load_dataset
        ds = load_dataset(a.span_dataset, split=a.span_split)
        iterator = enumerate(ds)
    for k, obj in iterator:
        text = obj.get(a.span_text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        raw_topic = obj.get(a.span_label_field, obj.get('label_text', obj.get('label', 'unknown')))
        topic = str(label_map.get(str(raw_topic), raw_topic))
        yield {'source_row_id': int(k), 'text': text, 'topic': topic,
               'source_metadata': {key: v for key, v in obj.items() if key != a.span_text_field}}


def make_spans(a, tokenizer):
    rng = random.Random(a.seed)
    by_topic = {}
    max_len = max(a.span_lengths)
    for row in span_rows(a):
        ids = tokenizer.encode(row['text'], add_special_tokens=False)
        if len(ids) >= max_len:
            by_topic.setdefault(row['topic'], []).append((row, ids))
    if not by_topic:
        raise RuntimeError('No articles long enough for maximum span length')
    spans = []
    if a.span_source == 'huggingface' and a.span_dataset == 'SetFit/bbc-news':
        expected = {'business', 'entertainment', 'politics', 'sport', 'tech'}
        if set(by_topic) != expected:
            raise RuntimeError(f'BBC topic labels unexpected: {sorted(by_topic)}')
    for topic in sorted(by_topic):
        candidates = by_topic[topic]
        if len(candidates) < a.spans_per_topic:
            raise RuntimeError(f'Only {len(candidates)} articles for topic {topic}; requested {a.spans_per_topic}')
        for row, token_ids in rng.sample(candidates, a.spans_per_topic):
            start = rng.randrange(len(token_ids) - max_len + 1)
            for length in a.span_lengths:
                selected = token_ids[start:start + length]
                spans.append({
                    'span_id': f'{a.span_dataset if a.span_source == "huggingface" else "jsonl"}_{topic}_{row["source_row_id"]}_{start}_{length}',
                    'source_kind': a.span_source,
                    'source_dataset': a.span_dataset if a.span_source == 'huggingface' else str(a.span_jsonl.resolve()),
                    'source_split': a.span_split if a.span_source == 'huggingface' else 'jsonl',
                    'topic': topic,
                    'article_row_id': row['source_row_id'],
                    'source_record_metadata': row['source_metadata'],
                    'source_article_length_gpt2': len(token_ids),
                    'start_gpt2_token': start,
                    'span_length_gpt2': length,
                    'gpt2_token_ids': selected,
                    'text': tokenizer.decode(selected, clean_up_tokenization_spaces=False),
                    'gpt2_eos_count': selected.count(50256),
                    'gpt2_token_sha256': hashlib.sha256(np.asarray(selected,dtype='<u2').tobytes()).hexdigest()
                })
    return spans


class Encoder:
    """No-truncation, single-tokenization dense embedding inference.

    A startup probe verifies direct-forward output against the model's official
    SentenceTransformer.encode behavior; corpus batches never use encode().
    """
    def __init__(self, name, precision='fp32'):
        self.precision = precision
        if precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 not supported by this GPU')
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.m = SentenceTransformer(MODELS[name], device='cuda:0', trust_remote_code=name in TRUST_REMOTE)
        self.m.eval()
        self.m.default_prompt_name = None  # No hidden automatic/default prefix
        self.tok = self.m.tokenizer
        self.limit = self.m.get_max_seq_length()
        if self.limit is None or int(self.limit) < 1:
            raise RuntimeError(f'{name}: unknown context limit; cannot guarantee zero truncation')
        self.limit = int(self.limit)
        self.dim = int(self.m.get_sentence_embedding_dimension())
        self.prompts = dict(getattr(self.m, 'prompts', {}) or {})
        self.kwargs = set(self.m.get_model_kwargs() or []) if hasattr(self.m, 'get_model_kwargs') else set()
        self.revision = getattr(self.m, '_model_card_data', None)

    def plan(self, raw, query):
        raw = list(raw)
        opts = {}
        prefix = ''
        label = 'none'
        if self.name == 'jina3':
            task = 'retrieval.query' if query else 'retrieval.passage'
            if 'task' not in self.kwargs:
                raise RuntimeError('jina3: task adapter is not exposed by this SentenceTransformer version')
            opts['task'] = task
            prefix = self.prompts.get(task, '')
            label = task
        elif self.name == 'nomic_modern':
            # Use supplied model prompts when present, else canonical Nomic prefixes.
            key = 'query' if query else 'document'
            prefix = self.prompts.get(key, 'search_query: ' if query else 'search_document: ')
            label = key
        elif self.name == 'gemma':
            key = 'query' if query else 'document'
            if key not in self.prompts:
                raise RuntimeError('EmbeddingGemma query/document prompt unavailable')
            prefix = self.prompts[key]; label = key
        elif self.name in {'qwen3', 'gte_qwen2', 'snowflake'} and query:
            if 'query' in self.prompts:
                prefix = self.prompts['query']; label = 'query'
            elif self.name == 'gte_qwen2':
                prefix = 'Instruct: Retrieve semantically related passages.\nQuery: '
                label = 'instruct_query'
            else:
                raise RuntimeError(f'{self.name}: expected retrieval query prompt missing')
        else:
            label = 'passage' if not query else 'query_no_prompt'
        return [prefix + t for t in raw], opts, label, prefix

    def prepare(self, raw, query=False):
        """Tokenize exactly ONCE per input, without truncation.

        The original 1024 GPT-2 IDs are decoded to text. An embedding tokenizer
        can produce MORE than 1024 tokens from that text, so enforce the loaded
        model's effective sequence limit on the actual encoding (including prompts).
        Prepared tensors are forwarded directly; we never call encode() on real
        corpus batches, because that would tokenize them again.
        """
        texts, kw, label, _prefix = self.plan(raw, query)
        # Use the SAME model-specific tokenizer path as SentenceTransformer.encode().
        # In particular, gte-Qwen2's module tokenization is not equivalent to
        # calling its underlying Hugging Face tokenizer with default arguments.
        # Temporarily disable the usual ST sequence-length truncation, then
        # reject overlength inputs ourselves before running the model.
        first = self.m._first_module()
        original_max = getattr(first, 'max_seq_length', None)
        if original_max is None:
            raise RuntimeError(
                f'{self.name}: cannot disable model-specific tokenizer truncation safely'
            )
        try:
            first.max_seq_length = max(self.limit, 100000)
            features = self.m.tokenize(texts)
        finally:
            first.max_seq_length = original_max
        if 'input_ids' not in features or 'attention_mask' not in features:
            raise RuntimeError(f'{self.name}: tokenizer did not return input IDs and attention mask')
        lengths = features['attention_mask'].sum(dim=1).tolist()
        if not lengths or min(lengths) < 1:
            raise RuntimeError(f'{self.name}: empty tokenization')
        longest = max(lengths)
        if longest > self.limit:
            raise RuntimeError(
                f'{self.name}: NO-TRUNCATION FAIL: {longest} embedding-model tokens '
                f'(including prompt/special tokens) exceed model limit {self.limit}. '
                'No chunk was truncated or scored.'
            )
        # Some architecture-specific forward methods require extra metadata.
        # The one-time equivalence probe below checks compatibility, fail-closed.
        return texts, kw, label, lengths, dict(features)

    def _precision_context(self):
        dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(self.precision)
        return torch.autocast(device_type='cuda', dtype=dtype) if dtype is not None else nullcontext()

    @torch.inference_mode()
    def forward_prepared(self, prepared):
        """Forward verified token tensors without the internal encode retokenization."""
        texts, kw, label, lengths, features = prepared
        # m.forward can implement proprietary per-model pooling/task adapters.
        # The equivalence test below prevents accepting an incompatible path.
        features = {k: v.to('cuda:0', non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in features.items()}
        with self._precision_context():
            result = self.m.forward(features, **kw)
        if not isinstance(result, dict) or 'sentence_embedding' not in result:
            raise RuntimeError(f'{self.name}: direct forward did not provide sentence_embedding')
        embeddings = result['sentence_embedding']
        if not torch.is_tensor(embeddings):
            raise RuntimeError(f'{self.name}: direct forward embedding was not a tensor')
        v = embeddings.detach().float().cpu().numpy()
        if v.shape != (len(texts), self.dim) or not np.isfinite(v).all():
            raise RuntimeError(f'{self.name}: invalid direct forward vectors {v.shape}')
        return v, lengths, label

    @torch.inference_mode()
    def encode_prepared_reference(self, prepared, batch):
        """SentenceTransformer reference path for non-compatible models."""
        texts, kw, label, lengths, _features = prepared
        with self._precision_context():
            out = self.m.encode(texts, batch_size=batch, convert_to_numpy=True,
                                normalize_embeddings=False, show_progress_bar=False, **kw)
        v = np.asarray(out, dtype=np.float32)
        if v.shape != (len(texts), self.dim) or not np.isfinite(v).all():
            raise RuntimeError(f'{self.name}: invalid reference embedding vectors')
        return v, lengths, label

    def select_backend(self, raw_query, raw_doc, batch, mode, atol):
        """One-time compatibility test; never fall back to repeated tokenization.

        Standard SentenceTransformer.encode() is invoked ONLY here, once per
        query/doc probe during startup. Actual BBC spans and Pile chunk batches
        are always tokenized once and passed directly to the model.
        """
        for query, raw in ((True, raw_query), (False, raw_doc)):
            prepared = self.prepare(raw, query)
            direct, _, _ = self.forward_prepared(prepared)
            reference, _, _ = self.encode_prepared_reference(prepared, batch)
            max_err = float(np.max(np.abs(direct - reference)))
            if not np.allclose(direct, reference, atol=atol, rtol=atol):
                if self.name == 'gte_qwen2':
                    # Diagnostic only: compare the actual ST tokenization on these
                    # startup probes. This never runs for real corpus batches.
                    texts, _, _, _, direct_features = prepared
                    official_features = self.m.tokenize(texts)
                    for key in ('input_ids', 'attention_mask'):
                        ours = direct_features.get(key)
                        official = official_features.get(key)
                        if ours is None or official is None:
                            print(f'[DIAG] gte_qwen2 {key}: missing in one path', flush=True)
                            continue
                        same = (ours.shape == official.shape and
                                torch.equal(ours.cpu(), official.cpu()))
                        print(f'[DIAG] gte_qwen2 {key}: identical={same}; '
                              f'direct_shape={tuple(ours.shape)}; '
                              f'official_shape={tuple(official.shape)}', flush=True)
                    d = direct / np.maximum(np.linalg.norm(direct, axis=1, keepdims=True), 1e-12)
                    r = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
                    cos = np.sum(d * r, axis=1)
                    print(f'[DIAG] gte_qwen2 query={query}; precision={self.precision}; '
                          f'max_abs_diff={max_err:.6g}; '
                          f'per_example_cosine={cos.tolist()}', flush=True)
                    # Check whether the discrepancy persists in FP32, without
                    # changing the requested precision or accepting a mismatch.
                    original_precision = self.precision
                    try:
                        self.precision = 'fp32'
                        fp32_direct, _, _ = self.forward_prepared(prepared)
                        fp32_reference, _, _ = self.encode_prepared_reference(prepared, batch)
                    finally:
                        self.precision = original_precision
                    print(f'[DIAG] gte_qwen2 fp32_max_abs_diff='
                          f'{float(np.max(np.abs(fp32_direct - fp32_reference))):.6g}', flush=True)
                raise RuntimeError(
                    f'{self.name}: one-tokenization direct forward differs from '
                    f'SentenceTransformer.encode (max abs error={max_err:.6g}). '
                    'Refusing to use fallback that tokenizes corpus chunks twice.'
                )
        self.fast_backend = True
        print(f'[BACKEND] {self.name}: one-tokenization direct forward VERIFIED; '
              f'no corpus encode() fallback (atol/rtol={atol})', flush=True)

    @torch.inference_mode()
    def embed(self, raw, query, batch):
        outputs, lengths = [], []
        label = None
        for start in range(0, len(raw), batch):
            prepared = self.prepare(raw[start:start + batch], query)
            value, part_lengths, label = self.forward_prepared(prepared)
            outputs.append(value)
            lengths.extend(part_lengths)
        return np.concatenate(outputs, axis=0), lengths, label



def decoded_batches(mm, chunk_ids, tok, begin, stop, batch_size, prefetch):
    """Bounded ordered CPU decode prefetch. No model objects/tokenizers shared with worker.

    The GPT-2 fast tokenizer's decode is used exactly as in the reference run.
    Future work is capped at `prefetch`, not proportional to corpus length.
    """
    def prepare(offset):
        end = min(stop, offset + batch_size)
        ids = (np.fromiter(chunk_ids[offset:end], dtype=np.int64)
               if isinstance(chunk_ids, range) else np.asarray(chunk_ids[offset:end], dtype=np.int64))
        txt = [tok.decode(chunk_token_ids(mm, int(b)).tolist(),
                          clean_up_tokenization_spaces=False) for b in ids]
        return offset, end, ids, txt

    if prefetch == 0:
        for offset in range(begin, stop, batch_size):
            yield prepare(offset)
        return
    # Execution is sequential from the model's perspective; only decoding is asynchronous.
    with ThreadPoolExecutor(max_workers=min(prefetch, 4), thread_name_prefix='pile-decode') as executor:
        offsets = iter(range(begin, stop, batch_size))
        pending = deque()
        def submit():
            offset = next(offsets, None)
            if offset is not None:
                pending.append(executor.submit(prepare, offset))
        for _ in range(prefetch):
            submit()
        while pending:
            result = pending.popleft().result()
            submit()
            yield result


def prepared_batches(mm, chunk_ids, tok, encoder, begin, stop, batch_size, prefetch):
    """Bounded ordered producer-consumer pipeline: decode + tokenizer audit off GPU thread.

    Uses a single producer because some custom model tokenizers are not thread-safe.
    Consumer exclusively executes model forward passes; retains physical ID order.
    """
    def make(offset):
        end = min(stop, offset + batch_size)
        ids = (np.fromiter(chunk_ids[offset:end], dtype=np.int64)
               if isinstance(chunk_ids, range) else np.asarray(chunk_ids[offset:end], dtype=np.int64))
        texts = [tok.decode(chunk_token_ids(mm, int(b)).tolist(),
                            clean_up_tokenization_spaces=False) for b in ids]
        prepared = encoder.prepare(texts, False)
        return offset, end, ids, prepared

    if prefetch == 0:
        for offset in range(begin, stop, batch_size):
            yield make(offset)
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='pile-verified-tokenize') as executor:
        offsets = iter(range(begin, stop, batch_size))
        pending = deque()
        def submit():
            offset = next(offsets, None)
            if offset is not None:
                pending.append(executor.submit(make, offset))
        for _ in range(prefetch):
            submit()
        while pending:
            result = pending.popleft().result()
            submit()
            yield result


def scores(q,d):
    qnorm=np.linalg.norm(q,axis=1,keepdims=True)
    dnorm=np.linalg.norm(d,axis=1,keepdims=True)
    cos=(q/np.maximum(qnorm,1e-12)) @ (d/np.maximum(dnorm,1e-12)).T
    return {'cosine':cos}


class Rankings:
    def __init__(self, n, k):
        self.k=k; self.n=n
        self.hi={m: np.full((n,k),-np.inf,np.float32) for m in METRICS}
        self.lo={m: np.full((n,k),np.inf,np.float32) for m in METRICS}
        self.hid={m: np.full((n,k),-1,np.int64) for m in METRICS}
        self.lid={m: np.full((n,k),-1,np.int64) for m in METRICS}
    def update(self, metric, value, block_ids):
        """Batched exact top/bottom-k; no Python inner loop over chunks."""
        value = np.asarray(value, dtype=np.float32)
        block_ids = np.asarray(block_ids, dtype=np.int64)
        if value.shape != (self.n, len(block_ids)):
            raise ValueError('Ranking batch shape mismatch')
        ids2 = np.broadcast_to(block_ids[None, :], value.shape)
        for high, val_map, id_map in ((True, self.hi, self.hid), (False, self.lo, self.lid)):
            oldv, oldids = val_map[metric], id_map[metric]
            candv = np.concatenate((oldv, value), axis=1)
            candid = np.concatenate((oldids, ids2), axis=1)
            key = -candv if high else candv
            ix = np.argpartition(key, kth=self.k - 1, axis=1)[:, :self.k]
            rows = np.arange(self.n)[:, None]
            ix = np.take_along_axis(ix, np.argsort(key[rows, ix], axis=1, kind='stable'), axis=1)
            oldv[:] = candv[rows, ix]
            oldids[:] = candid[rows, ix]
    def state(self, cursor):
        data={'cursor':np.asarray([cursor],dtype='<i8')}
        for metric in METRICS:
            for name in ('hi','lo','hid','lid'):
                data[f'{name}_{metric}']=getattr(self,name)[metric]
        return data
    def restore(self, save):
        for metric in METRICS:
            for name in ('hi','lo','hid','lid'):
                arr=getattr(self,name)[metric]; src=save[f'{name}_{metric}']
                if arr.shape!=src.shape: raise ValueError('Checkpoint shape mismatch')
                arr[:]=src


def checkpoint(path, ranks, cursor):
    tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**ranks.state(cursor)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def document_overlaps(db, first, end, doc_order):
    # Indexed predecessor lookup: a chunk can begin mid-document. This avoids
    # scanning all earlier documents for every ranked chunk in a 134M-row DB.
    anchor=db.execute('''SELECT token_offset FROM documents
      WHERE split='train' AND token_offset<=?
      ORDER BY token_offset DESC LIMIT 1''',(first,)).fetchone()
    if anchor is None:
        raise RuntimeError(f'No source document covers chunk beginning at {first}')
    rows=db.execute('''SELECT source_row, token_offset, stored_tokens, text_tokens, text_sha256
      FROM documents WHERE split='train' AND token_offset>=? AND token_offset<?
      ORDER BY token_offset''',(int(anchor[0]),end))
    out=[]
    for source,start,n,text_n,sha in rows:
        if int(source) < 0 or int(source) >= len(doc_order):
            raise RuntimeError('Document index beyond shuffled document order')
        left=max(first,start); right=min(end,start+n)
        out.append({'shuffled_document_position':int(source), 'original_pile_document_id':int(doc_order[int(source)]), 'document_start_token':int(start),
                    'document_end_token_exclusive':int(start+n), 'document_text_tokens':int(text_n),
                    'document_stored_tokens_including_eos':int(n), 'document_text_sha256':sha,
                    'overlap_global_start':left,'overlap_global_end_exclusive':right,
                    'overlap_chunk_start':left-first, 'overlap_chunk_end_exclusive':right-first,
                    'overlap_tokens':right-left,'contains_document_eos':start+n-1 in range(left,right)})
    if sum(d['overlap_tokens'] for d in out) != end-first:
        raise RuntimeError(f'SQLite provenance does not cover full chunk {first}..{end}')
    return out


def lexical_overlap(span_ids, chunk_ids, gram=5):
    """Exact GPT-2-token contamination check and 5-gram overlap (NOT semantic similarity)."""
    needle=tuple(map(int,span_ids)); hay=tuple(map(int,chunk_ids))
    n=len(needle)
    exact=any(hay[i:i+n]==needle for i in range(len(hay)-n+1)) if n<=len(hay) else False
    if n<gram:
        return {'exact_gpt2_span_present':exact,'shared_5gram_count':0,'bbc_5gram_recall':None}
    q={needle[i:i+gram] for i in range(n-gram+1)}
    d={hay[i:i+gram] for i in range(len(hay)-gram+1)}
    shared=len(q & d)
    return {'exact_gpt2_span_present':exact,'shared_5gram_count':shared,
            'bbc_5gram_recall':shared/max(1,len(q))}


def report_results(out, alias, ranks, spans, mm, inv, perm, doc_order, db, tokenizer, sample_count, identifiers, encoder_info, print_full_chunks=False, cache_reader=None):
    folder=out / alias; folder.mkdir(exist_ok=True)
    jsonpath=folder/'manual_review.jsonl'; csvpath=folder/'manual_review.csv'; txtpath=folder/'manual_review.txt'
    fields=['model','metric','group','rank','score','span_id','topic','span_length_gpt2',
            'article_row_id','start_gpt2_token','physical_chunk_id','training_visit_id',
            'global_token_start','actual_chunk_length_gpt2','is_partial_chunk','original_pile_document_ids','shuffled_document_positions','exact_bbc_span_present','exact_bbc_gpt2_token_subsequence','bbc_5gram_recall','bbc_span_text','pile_chunk_decoded_full_text','pile_chunk_gpt2_token_ids','overlapping_document_metadata_json','manual_relevance_0_1_2','manual_notes']
    nrows=0
    with jsonpath.open('w',encoding='utf-8') as jf, csvpath.open('w',encoding='utf-8',newline='') as cf, txtpath.open('w',encoding='utf-8') as tf:
        cw=csv.DictWriter(cf,fieldnames=fields); cw.writeheader()
        tf.write('PILE / BBC EMBEDDING MANUAL REVIEW\n\n')
        tf.write('Metric: cosine similarity. Larger = more similar.\n')
        tf.write('This ranking is NOT a human-label or factual GPT-2-knowledge measurement.\n')
        tf.write('Chunk token IDs come from physical train_shuffled.bin offsets.\n')
        tf.write('training_visit_id = inverse chunk permutation[physical_chunk_id].\n\n')
        for metric in METRICS:
            for si, span in enumerate(spans):
                for group, vals, ids in [('similar_top',ranks.hi[metric][si],ranks.hid[metric][si]),
                                         ('dissimilar_bottom',ranks.lo[metric][si],ranks.lid[metric][si])]:
                    for r,(score,bid) in enumerate(zip(vals,ids),1):
                        bid=int(bid)
                        if bid<0: continue
                        visit=int(inv[bid]) if bid < len(inv) else None
                        if visit is not None and int(perm[visit]) != bid:
                            raise RuntimeError('Permutation inverse mismatch')
                        start=bid*BLOCK
                        ids1024=chunk_token_ids(mm, bid).tolist()
                        end=start+len(ids1024)
                        docmeta=document_overlaps(db,start,end,doc_order)
                        decoded=(cache_reader.read_id(bid) if cache_reader is not None else tokenizer.decode(ids1024,clean_up_tokenization_spaces=False))
                        contamination=lexical_overlap(span['gpt2_token_ids'],ids1024)
                        record={
                          'model_alias':alias,'model_hf_id':MODELS[alias],'metric':metric,'group':group,
                          'rank':r,'score':float(score),'score_direction':'larger_is_closer',
                          'span':span,'pile':{
                            'data_path':str(identifiers['train_shuffled']['path']),
                            'physical_chunk_id_zero_based':bid,'training_visit_id_zero_based':visit,
                            'chunk_size_gpt2_tokens':BLOCK,'actual_chunk_length_gpt2':len(ids1024),
                            'is_partial_chunk':len(ids1024)<BLOCK,'final_next_token_target_excluded':bid==len(inv)-1 and len(ids1024)<BLOCK,'global_token_start':start,
                            'global_token_end_exclusive':end,'byte_offset':start*2,
                            'gpt2_token_ids':ids1024,'gpt2_eos_positions':[i for i,x in enumerate(ids1024) if x==50256],
                            'gpt2_token_sha256':hashlib.sha256(np.asarray(ids1024,dtype='<u2').tobytes()).hexdigest(),
                            'decoded_full_text':decoded,'exact_bbc_span_present':span['text'] in decoded,
                            'lexical_overlap_with_bbc_span':contamination,
                            'source_documents_overlapping':docmeta},
                          'encoder':encoder_info,'manual_relevance_0_1_2':None,'manual_notes':''}
                        jf.write(json.dumps(record,ensure_ascii=False)+'\n')
                        cw.writerow({'model':alias,'metric':metric,'group':group,'rank':r,'score':float(score),
                          'span_id':span['span_id'],'topic':span['topic'],'span_length_gpt2':span['span_length_gpt2'],
                          'article_row_id':span['article_row_id'],'start_gpt2_token':span['start_gpt2_token'],
                          'physical_chunk_id':bid,'training_visit_id':visit,'global_token_start':start,
                          'actual_chunk_length_gpt2':len(ids1024),'is_partial_chunk':len(ids1024)<BLOCK,
                          'original_pile_document_ids':';'.join(str(x['original_pile_document_id']) for x in docmeta),
                          'shuffled_document_positions':';'.join(str(x['shuffled_document_position']) for x in docmeta),
                          'exact_bbc_span_present':span['text'] in decoded,
                          'exact_bbc_gpt2_token_subsequence':contamination['exact_gpt2_span_present'],
                          'bbc_5gram_recall':contamination['bbc_5gram_recall'],
                           'bbc_span_text':span['text'],'pile_chunk_decoded_full_text':decoded,
                           'pile_chunk_gpt2_token_ids':json.dumps(ids1024),
                           'overlapping_document_metadata_json':json.dumps(docmeta,ensure_ascii=False),
                           'manual_relevance_0_1_2':'','manual_notes':''})
                        tf.write(f'\n{"="*90}\nMODEL={alias} METRIC={metric} GROUP={group} RANK={r} SCORE={float(score):.8f}\n')
                        tf.write(f'SPAN_ID={span["span_id"]} TOPIC={span["topic"]} BBC_ARTICLE_ROW={span["article_row_id"]} BBC_START_TOKEN={span["start_gpt2_token"]} BBC_LEN={span["span_length_gpt2"]}\n')
                        tf.write('PILE_PHYSICAL_CHUNK_ID=%s TRAINING_VISIT_ID=%s TOKEN_RANGE=[%s,%s) ORIGINAL_DOCUMENT_IDS=%s SHUFFLED_DOCUMENT_POSITIONS=%s EXACT_BBC_SPAN_PRESENT=%s\n' % (bid, visit, start, end, [x['original_pile_document_id'] for x in docmeta], [x['shuffled_document_position'] for x in docmeta], span['text'] in decoded))
                        tf.write(f'DOCUMENT_METADATA={json.dumps(docmeta,ensure_ascii=False)}\n')
                        tf.write(f'BBC_SPAN_GPT2_TOKEN_IDS={span["gpt2_token_ids"]}\nPILE_CHUNK_GPT2_TOKEN_IDS={ids1024}\n')
                        tf.write(f'\nBBC SPAN TEXT (complete):\n{span["text"]}\n\nPILE CHUNK TEXT (all real GPT-2 tokens decoded; final chunk may be partial):\n{decoded}\n')
                        if metric == 'cosine':
                            line = (f'[RANK] model={alias} topic={span["topic"]} '
                                    f'length={span["span_length_gpt2"]} group={group} '
                                    f'rank={r:02d} score={float(score):.6f} '
                                    f'physical_chunk_id={bid} training_visit_id={visit} '
                                    f'original_doc_ids={[d["original_pile_document_id"] for d in docmeta]}')
                            print(line, flush=True)
                            if print_full_chunks:
                                print('BBC:',span['text'],'\nPILE CHUNK:',decoded,'\n',flush=True)
                        nrows+=1
    atomic_json(folder/'summary.json',{'model':alias,'hf_model':MODELS[alias],'ranked_records':nrows,
             'chunks_compared':sample_count,'spans_compared':len(spans),'encoder':encoder_info,
             'reports':['manual_review.jsonl','manual_review.csv','manual_review.txt']})
    print(f'[REPORT] {alias}: {nrows:,} ranked entries with complete decoded text',flush=True)


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        path=Path(t)/'docs.sqlite'
        conn=sqlite3.connect(path)
        conn.execute('CREATE TABLE documents(source_row INTEGER,split TEXT,token_offset INTEGER,stored_tokens INTEGER,text_tokens INTEGER,text_sha256 TEXT)')
        conn.execute('CREATE INDEX documents_split_offset ON documents(split,token_offset)')
        conn.executemany('INSERT INTO documents VALUES (?,?,?,?,?,?)', [
            (0,'train',0,600,599,'a'*64),(1,'train',600,800,799,'b'*64),
            (2,'train',1400,850,849,'c'*64)])
        conn.commit()
        order=np.array([55,13,89],dtype='<u4')
        overlaps=document_overlaps(conn,1024,2048,order)
        assert [x['original_pile_document_id'] for x in overlaps]==[13,89]
        assert sum(x['overlap_tokens'] for x in overlaps)==BLOCK
        assert (3*BLOCK-1)//BLOCK==2 # PackedSampler 1025-token requirement
        assert (3*BLOCK+1-1)//BLOCK==3
        r=Rankings(1,2)
        ids=np.array([10,11,12])
        r.update('cosine',np.array([[0.5,-0.2,0.9]],np.float32),ids)
        assert list(r.hid['cosine'][0])==[12,10]
        assert list(r.lid['cosine'][0])==[11,10]
        assert np.allclose(scores(np.eye(2,dtype=np.float32),np.eye(2,dtype=np.float32))['cosine'],np.eye(2))
        # Updating in small pieces vs one large rank-window must be equivalent.
        vals = np.array([[0.5, -0.2, 0.9, 0.7, -0.4]], dtype=np.float32)
        ids5 = np.array([10,11,12,13,14], dtype=np.int64)
        ra, rb = Rankings(1, 2), Rankings(1, 2)
        for j in range(5): ra.update('cosine', vals[:, j:j+1], ids5[j:j+1])
        rb.update('cosine', vals, ids5)
        assert np.array_equal(ra.hid['cosine'], rb.hid['cosine'])
        assert np.array_equal(ra.lid['cosine'], rb.lid['cosine'])
        assert lexical_overlap([2,3,4,5,6],[1,2,3,4,5,6,7])['exact_gpt2_span_present']
        assert not lexical_overlap([2,3,4,5,6],[1,2,3,4,7,6])['exact_gpt2_span_present']
        with tempfile.TemporaryDirectory() as resume_dir:
            cp = Path(resume_dir) / 'progress.npz'
            checkpoint(cp, r, 2)
            restored = Rankings(1, 2)
            assert load_progress(cp, restored, 3) == 2
            assert list(restored.hid['cosine'][0]) == [12, 10]
            assert list(restored.lid['cosine'][0]) == [11, 10]
            assert np.allclose(restored.hi['cosine'], r.hi['cosine'])
        conn.close()
    assert (231_524_872_505 - 1 + BLOCK - 1) // BLOCK == 226_098_509
    synthetic=np.arange(BLOCK+313,dtype=np.uint16)
    assert len(chunk_token_ids(synthetic, 1)) == 312
    assert len(chunk_token_ids(synthetic, 0)) == BLOCK
    # Compare the exact real training input prefix of data.PackedSampler.get_block.
    full=np.arange(BLOCK*2+313, dtype=np.uint16) % 50257
    for bid in (0,1,2):
        seq=np.asarray(full[bid*BLOCK:bid*BLOCK+BLOCK+1],dtype=np.int64)
        valid=len(seq)-1
        x=np.full(BLOCK, 50256,dtype=np.int64); x[:valid]=seq[:valid]
        assert np.array_equal(chunk_token_ids(full,bid), x[:valid])
    assert len(chunk_token_ids(full,2)) == 312
    virtual_ids=range(1_000_000_000)
    assert list(virtual_ids[125400:125404]) == [125400,125401,125402,125403]
    assert np.array_equal(np.fromiter(virtual_ids[125400:125404],dtype=np.int64),np.arange(125400,125404))
    # Verify asynchronous decode+prepare preserves physical chunk IDs, order,
    # trailing partial block, and exactly-once processing without a GPU.
    class _MockTok:
        def decode(self, ids, **kwargs):
            return ','.join(map(str, ids))
    class _MockEncoder:
        def prepare(self, texts, query=False):
            return (texts, {}, 'passage', [len(x) for x in texts], {})
    tokens = np.arange(2 * BLOCK + 313, dtype=np.uint16)
    selected = np.array([2, 0, 1], dtype=np.int64)
    for ahead in (0, 1, 3):
        parts = list(prepared_batches(tokens, selected, _MockTok(), _MockEncoder(),
                                      0, len(selected), 2, ahead))
        got = np.concatenate([part[2] for part in parts]).tolist()
        assert got == [2, 0, 1]
        assert sum(len(part[3][0]) for part in parts) == 3
    # Verify byte-exact cache random access (UTF-8, empty text, EOS markers).
    with tempfile.TemporaryDirectory() as td:
        class _FakeArgs:
            cache_dir = Path(td)
            cache_shard_chunks = 2
        args = _FakeArgs()
        payload = ['alpha', 'naive <|endoftext|>', 'third']
        for j in (0, 1):
            start = j * 2
            stop = min(len(payload), start + 2)
            data, offsets, marker = _cache_paths(args.cache_dir, j)
            written = 0
            with data.open('wb') as df, offsets.open('wb') as of:
                of.write((0).to_bytes(8, 'little'))
                for string in payload[start:stop]:
                    encoded = string.encode('utf-8')
                    df.write(encoded)
                    written += len(encoded)
                    of.write(written.to_bytes(8, 'little'))
            atomic_json(marker, {'schema': CACHE_SCHEMA, 'start': start,
                                 'end': stop, 'bytes': written})
            assert _validate_cache_shard(args.cache_dir, j, start, stop)
        reader = object.__new__(CacheReader)
        reader.root = args.cache_dir
        reader.total = len(payload)
        reader.shard_chunks = 2
        reader.open_shard = None
        reader.handle = None
        reader.data = None
        reader.index = None
        assert reader.read_ids([0, 1, 2, 0]) == [payload[i] for i in [0, 1, 2, 0]]
        reader.close()
    print('[SELF-TEST PASS] scoring, ranking, partial-tail indexing, SQLite provenance, '
          'billion-ID streaming, bounded ordered CPU prefetch')


# ---------------------------------------------------------------------------
# One-time decoded GPT-2 text cache (physical chunk order, lossless UTF-8).
# Shards are committed atomically with a marker; incomplete shards are rebuilt.
# ---------------------------------------------------------------------------
import bisect
import fcntl
import mmap
import threading
from concurrent.futures import ThreadPoolExecutor

CACHE_SCHEMA = 1
CACHE_DEFAULT_SHARD_CHUNKS = 250_000


def _cache_source_ident(a):
    p = a.data / 'train_shuffled.bin'
    st = p.stat()
    tk = a.data / 'tokenizer'
    return {'path': str(p.resolve()), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns,
            'tokenizer_path': str(tk.resolve()), 'block': BLOCK}


def _cache_config(a, total):
    return {'schema': CACHE_SCHEMA, 'source': _cache_source_ident(a),
            'total_chunks': int(total), 'shard_chunks': int(a.cache_shard_chunks),
            'decode_kwargs': {'clean_up_tokenization_spaces': False},
            'encoding': 'utf-8', 'offset_dtype': '<u8'}


def _cache_paths(root, shard):
    stem = f'shard_{shard:06d}'
    return root / (stem + '.utf8'), root / (stem + '.u64'), root / (stem + '.complete.json')


def _validate_cache_shard(root, shard, start, end):
    data, offsets, marker = _cache_paths(root, shard)
    if not marker.is_file():
        return False
    meta = json.loads(marker.read_text())
    if meta.get('start') != start or meta.get('end') != end or meta.get('schema') != CACHE_SCHEMA:
        raise RuntimeError(f'Cache marker metadata mismatch: {marker}')
    if not data.is_file() or not offsets.is_file() or offsets.stat().st_size != (end - start + 1) * 8:
        raise RuntimeError(f'Committed cache shard missing or invalid: {marker}')
    if data.stat().st_size != meta['bytes']:
        raise RuntimeError(f'Committed cache shard bytes mismatch: {marker}')
    with offsets.open('rb') as f:
        first = int.from_bytes(f.read(8), 'little')
        f.seek(-8, os.SEEK_END)
        last = int.from_bytes(f.read(8), 'little')
    if first != 0 or last != meta['bytes']:
        raise RuntimeError(f'Committed cache shard boundary mismatch: {marker}')
    return True


def _build_one_cache_shard(a, mm, shard, start, end):
    root = a.cache_dir
    data, offsets, marker = _cache_paths(root, shard)
    if _validate_cache_shard(root, shard, start, end):
        return False
    from transformers import AutoTokenizer
    local = threading.local()
    def decode_task(first, last):
        tokenizer = getattr(local, 'tok', None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(str(a.data / 'tokenizer'), use_fast=True)
            if tokenizer.eos_token_id != 50256 or len(tokenizer) != 50257:
                raise RuntimeError('Incorrect cache decoder tokenizer')
            local.tok = tokenizer
        texts = [tokenizer.decode(chunk_token_ids(mm, b).tolist(),
                                  clean_up_tokenization_spaces=False) for b in range(first, last)]
        return [t.encode('utf-8') for t in texts]

    dtmp, otmp = data.with_suffix(data.suffix + '.part'), offsets.with_suffix(offsets.suffix + '.part')
    dtmp.unlink(missing_ok=True)
    otmp.unlink(missing_ok=True)
    cursor = 0
    try:
        with dtmp.open('wb', buffering=4 << 20) as df, otmp.open('wb', buffering=1 << 20) as of:
            of.write((0).to_bytes(8, 'little'))
            with ThreadPoolExecutor(max_workers=a.cache_workers, thread_name_prefix='pile-cache-decode') as pool:
                it = iter(range(start, end, a.cache_decode_batch))
                pending = deque()
                def submit():
                    i = next(it, None)
                    if i is not None:
                        pending.append(pool.submit(decode_task, i, min(end, i + a.cache_decode_batch)))
                for _ in range(a.cache_workers * 2):
                    submit()
                while pending:
                    encoded = pending.popleft().result()
                    submit()
                    for b in encoded:
                        df.write(b)
                        cursor += len(b)
                        of.write(cursor.to_bytes(8, 'little'))
            df.flush(); os.fsync(df.fileno())
            of.flush(); os.fsync(of.fileno())
        os.replace(dtmp, data)
        os.replace(otmp, offsets)
        atomic_json(marker, {'schema': CACHE_SCHEMA, 'start': start,
                             'end': end, 'bytes': cursor})
        return True
    finally:
        dtmp.unlink(missing_ok=True)
        otmp.unlink(missing_ok=True)


def build_decoded_cache(a, mm, total):
    a.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = a.cache_dir / 'cache_config.json'
    expected = _cache_config(a, total)
    with (a.cache_dir / '.cache-build.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cfg_path.exists():
            if json.loads(cfg_path.read_text()) != expected:
                raise RuntimeError('Decoded cache belongs to different source/config. Use a new --cache-dir.')
        else:
            existing = list(a.cache_dir.glob('shard_*.complete.json'))
            if existing:
                raise RuntimeError('Cache shards exist without configuration; refusing to reuse them')
            atomic_json(cfg_path, expected)
        limit = min(total, a.cache_build_limit) if a.cache_build_limit else total
        start_time = time.time()
        complete = 0
        for shard, start in enumerate(range(0, limit, a.cache_shard_chunks)):
            # Every full-cache shard must have the same fixed end, even in a pilot.
            end = min(total, start + a.cache_shard_chunks)
            if limit < end:
                break
            rebuilt = _build_one_cache_shard(a, mm, shard, start, end)
            complete = end
            if rebuilt or shard % 20 == 0:
                speed = complete / max(time.time() - start_time, 0.001)
                print(f'[CACHE] {complete:,}/{total:,} chunks, average {speed:.1f} chunks/s; shard={shard}', flush=True)
        if limit == total:
            for shard, start in enumerate(range(0, total, a.cache_shard_chunks)):
                end = min(total, start + a.cache_shard_chunks)
                if not _validate_cache_shard(a.cache_dir, shard, start, end):
                    raise RuntimeError(f'Cache incomplete at shard {shard}')
            atomic_json(a.cache_dir / 'cache_ready.json', expected)
            print(f'[CACHE READY] {total:,} decoded chunks; reusable by all 10 models', flush=True)
        else:
            print(f'[CACHE PARTIAL] Cache built only to {complete:,}. Complete it before embedding.', flush=True)


class CacheReader:
    '''Read original physical GPT-2 chunk ID -> exact original decoded string.'''
    def __init__(self, a, total):
        self.root = a.cache_dir
        self.config = _cache_config(a, total)
        ready = self.root / 'cache_ready.json'
        if not ready.exists() or json.loads(ready.read_text()) != self.config:
            raise RuntimeError('Decoded cache not complete or source/config mismatch. Run --cache-only first.')
        self.total = total
        self.shard_chunks = a.cache_shard_chunks
        self.open_shard = None
        self.handle = None
        self.data = None
        self.index = None

    def close(self):
        if self.data is not None:
            self.data.close()
            self.data = None
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.index is not None:
            del self.index
            self.index = None
        self.open_shard = None

    def _switch(self, shard):
        if self.open_shard == shard:
            return
        self.close()
        start = shard * self.shard_chunks
        end = min(self.total, start + self.shard_chunks)
        if not _validate_cache_shard(self.root, shard, start, end):
            raise RuntimeError(f'Cache shard {shard} is missing')
        path, offsets, _ = _cache_paths(self.root, shard)
        self.handle = path.open('rb')
        self.data = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.index = np.memmap(offsets, dtype='<u8', mode='r')
        self.open_shard = shard

    def read_id(self, physical_id):
        bid = int(physical_id)
        if not 0 <= bid < self.total:
            raise IndexError(bid)
        shard, pos = divmod(bid, self.shard_chunks)
        self._switch(shard)
        start = int(self.index[pos]); end = int(self.index[pos + 1])
        if not 0 <= start <= end <= len(self.data):
            raise RuntimeError(f'Corrupt decoded cache offset at chunk {bid}')
        return self.data[start:end].decode('utf-8')

    def read_ids(self, chunk_ids):
        """Shard-grouped bulk mmap slicing. Preserve caller order and cache format.

        In the full scan IDs are increasing, so one shard validation/open occurs
        per 250k records, not once per record. Works across shard boundaries.
        """
        ids = np.asarray(chunk_ids, dtype=np.int64)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.total):
            raise IndexError('Cache chunk ID outside source range')
        if not len(ids):
            return []
        result = [None] * len(ids)
        shards = ids // self.shard_chunks
        # Consecutive runs also preserve order for unsorted pilot selections.
        start = 0
        while start < len(ids):
            end = start + 1
            while end < len(ids) and shards[end] == shards[start]:
                end += 1
            shard = int(shards[start])
            self._switch(shard)
            positions = ids[start:end] - shard * self.shard_chunks
            offsets = np.asarray(self.index[positions], dtype=np.int64)
            endings = np.asarray(self.index[positions + 1], dtype=np.int64)
            if np.any(offsets < 0) or np.any(endings < offsets) or np.any(endings > len(self.data)):
                raise RuntimeError(f'Corrupt decoded cache offsets for shard {shard}')
            data = self.data
            for i, (lo, hi) in enumerate(zip(offsets, endings), start):
                result[i] = data[int(lo):int(hi)].decode('utf-8')
            start = end
        return result


def cached_decoded_batches(reader, chunk_ids, begin, stop, batch_size, prefetch):
    '''One CPU producer accesses mmap cache, maintains fixed physical order.'''
    def prepare(offset):
        end = min(stop, offset + batch_size)
        ids = (np.fromiter(chunk_ids[offset:end], dtype=np.int64)
               if isinstance(chunk_ids, range) else np.asarray(chunk_ids[offset:end], dtype=np.int64))
        return offset, end, ids, reader.read_ids(ids)
    if not prefetch:
        for i in range(begin, stop, batch_size):
            yield prepare(i)
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='pile-cache-reader') as ex:
        offsets = iter(range(begin, stop, batch_size))
        pending = deque()
        def submit():
            i = next(offsets, None)
            if i is not None:
                pending.append(ex.submit(prepare, i))
        for _ in range(prefetch):
            submit()
        while pending:
            value = pending.popleft().result()
            submit()
            yield value


def cached_prepared_batches(reader, chunk_ids, encoder, begin, stop, batch_size, prefetch):
    '''Read decoded cache and prepare model-specific tokens off inference thread.'''
    def prepare(offset):
        end = min(stop, offset + batch_size)
        ids = (np.fromiter(chunk_ids[offset:end], dtype=np.int64)
               if isinstance(chunk_ids, range) else np.asarray(chunk_ids[offset:end], dtype=np.int64))
        texts = reader.read_ids(ids)
        return offset, end, ids, encoder.prepare(texts, False)
    if not prefetch:
        for i in range(begin, stop, batch_size):
            yield prepare(i)
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='pile-cache-tokenize') as ex:
        offsets = iter(range(begin, stop, batch_size))
        pending = deque()
        def submit():
            i = next(offsets, None)
            if i is not None:
                pending.append(ex.submit(prepare, i))
        for _ in range(prefetch):
            submit()
        while pending:
            value = pending.popleft().result()
            submit()
            yield value

def cached_profiled_batches(reader, chunk_ids, encoder, begin, stop, batch_size, prefetch):
    """Single producer + bounded overlap; metrics are producer CPU stage durations.

    The model-specific tokenizer is used exactly once per corpus chunk.
    A single producer protects custom tokenizer state while the GPU consumer
    works on the preceding batch. Prefetch never grows with dataset length.
    """
    def prepare(offset):
        end = min(stop, offset + batch_size)
        ids = (np.fromiter(chunk_ids[offset:end], dtype=np.int64)
               if isinstance(chunk_ids, range) else np.asarray(chunk_ids[offset:end], dtype=np.int64))
        t0 = time.perf_counter()
        texts = reader.read_ids(ids)
        t1 = time.perf_counter()
        prepared = encoder.prepare(texts, False)
        t2 = time.perf_counter()
        lengths = prepared[3]
        return offset, end, ids, prepared, {'cache_s': t1 - t0,
            'tokenize_s': t2 - t1, 'tokens': int(sum(lengths))}
    if not prefetch:
        for i in range(begin, stop, batch_size):
            yield prepare(i)
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='pile-tokenize-prefetch') as ex:
        offsets = iter(range(begin, stop, batch_size))
        pending = deque()
        def submit():
            i = next(offsets, None)
            if i is not None:
                pending.append(ex.submit(prepare, i))
        for _ in range(prefetch):
            submit()
        while pending:
            value = pending.popleft().result()
            submit()
            yield value


def main():
    a=cli()
    if a.self_test:
        return self_test()
    if a.inspect_data:
        dataset(a.data)
        print('[INSPECT] Full-source manifest and indexed chunk policy verified; no GPU required.', flush=True)
        return
    if a.cache_only:
        if not a.data.is_dir(): raise FileNotFoundError(a.data)
        mm, _, _, _, total, _, _, _ = dataset(a.data)
        build_decoded_cache(a, mm, total)
        return
    stop = GracefulStop()
    old_handlers = install_stop_handlers(stop)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expected one visible CUDA GPU per worker; set CUDA_VISIBLE_DEVICES in the launcher')
    print(f'[GPU] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES","unset")}, visible GPUs=1, name={torch.cuda.get_device_name(0)}',flush=True)
    if not a.data.is_dir(): raise FileNotFoundError(a.data)
    mm, inv, perm, doc_order, total, manifest, orderinfo, ids=dataset(a.data)
    cache_reader = CacheReader(a, total)
    a.out.mkdir(parents=True,exist_ok=True)
    config_path=a.out/'experiment.json'; spans_path=a.out/'bbc_spans.json'; sample_path=a.out/'selected_physical_chunk_ids.npy'
    config={'data':ids,'total_blocks':total,'models':a.models,'span_lengths':a.span_lengths,
      'spans_per_topic':a.spans_per_topic,'max_blocks':a.max_blocks,'topk':a.topk,
      'batch_size':a.batch_size,'save_every':a.save_every,'seed':a.seed,
      'prefetch':a.prefetch,'rank_window':a.rank_window,'precision':a.precision,
      'backend':a.backend,'equivalence_atol':a.equivalence_atol,'decoded_cache':_cache_config(a,total),'cache_dir':str(a.cache_dir.resolve()),
      'save_vectors':a.save_vectors,
      'span_source':a.span_source, 'span_dataset':a.span_dataset,
      'span_split':a.span_split, 'span_jsonl':str(a.span_jsonl.resolve()) if a.span_jsonl else None,
      'span_text_field':a.span_text_field, 'span_label_field':a.span_label_field,
      'span_label_map':str(a.span_label_map.resolve()) if a.span_label_map else None,
      'print_full_chunks':a.print_full_chunks, 'provenance_probes':a.provenance_probes,
      'chunk_size':BLOCK,'chunk_selection':'all_physical' if (a.max_blocks==0 or a.max_blocks>=total) else 'uniform_random_physical',
      'model_prompts':'per_model_query_document','embedding_input_policy':'exact_unpadded_PackedSampler_x_only_excludes_terminal_label','encoder_tokenization':'single_untruncated_pass_direct_forward_only','metrics':list(METRICS),'full_stored_chunk_count':total,
       'training_indexed_chunk_count':len(inv),'unindexed_complete_tail_chunks':0,
       'source_scan_complete':manifest.get('source_scan_complete'),
       'final_chunk_stored_tokens':len(mm)-(total-1)*BLOCK,
       'final_chunk_real_training_input_tokens':min(BLOCK,len(mm)-1-(total-1)*BLOCK),
       'train_docs':manifest.get('train_docs'),'token_remainder':len(mm)%BLOCK}
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(a.data/'tokenizer'),use_fast=True)
    if tok.eos_token_id!=50256 or len(tok)!=50257: raise RuntimeError('Not the GPT-2 tokenizer')
    if a.resume:
        full_scan = (a.max_blocks == 0 or a.max_blocks >= total)
        if not config_path.exists() or not spans_path.exists() or (not full_scan and not sample_path.exists()):
            raise FileNotFoundError('Resume requires experiment.json, bbc_spans.json and sampled IDs for sampled runs')
        if json.loads(config_path.read_text())!=config: raise RuntimeError('Config or dataset changed: use new --out')
        spans=json.loads(spans_path.read_text())
        chunk_ids=range(total) if full_scan else np.load(sample_path,allow_pickle=False, mmap_mode='r')
    else:
        if any(a.out.iterdir()): raise RuntimeError('Output folder not empty: pass --resume or use a different --out')
        spans=make_spans(a,tok)
        if a.max_blocks==0 or a.max_blocks>=total:
            chunk_ids=range(total)  # constant-memory full scan; ID == physical block position
        else:
            chunk_ids=np.sort(np.random.default_rng(a.seed).choice(total,size=a.max_blocks,replace=False)).astype(np.int64)
        # Save sampled IDs only. Full scans use identity IDs derived from cursor.
        if not isinstance(chunk_ids, range):
            np.save(sample_path,chunk_ids)
        atomic_json(spans_path,spans); atomic_json(config_path,config)
    if len(chunk_ids)==0 or len(spans)==0: raise RuntimeError('Nothing selected')
    k=min(a.topk,len(chunk_ids)); print(f'[DATA] {ids["train_shuffled"]["path"]}: {total:,} indexed training chunks',flush=True)
    print(f'[TASK] {len(chunk_ids):,} selected indexed chunks, {len(spans)} BBC spans, {len(a.models)} models, top/bottom {k}',flush=True)
    uri=(a.data/'documents_shuffled.sqlite').resolve().as_uri()+'?mode=ro'
    db=sqlite3.connect(uri,uri=True)
    db.execute('PRAGMA query_only=ON')
    # Only use metadata AFTER the source schema is confirmed.
    cols={x[1] for x in db.execute('PRAGMA table_info(documents)')}
    required={'source_row','split','token_offset','stored_tokens','text_tokens','text_sha256'}
    if not required.issubset(cols): raise ValueError(f'SQLite schema mismatch: {cols}')
    try:
        # Verify a few source-document and shuffle mappings before loading models.
        probe=np.unique(np.array([0, total//2, total-1], dtype=np.int64))
        for bid in probe:
            bid=int(bid)
            visit=int(inv[bid]) if bid < len(inv) else None
            if visit is not None and (not 0 <= visit < len(perm) or int(perm[visit]) != bid):
                raise RuntimeError('Chunk-order inverse mismatch')
            document_overlaps(db,bid*BLOCK,min((bid+1)*BLOCK,len(mm)),doc_order)
        # Verify shuffled document records against actual EOS boundaries in the
        # token stream. This catches stale or mismatched SQLite provenance.
        tr_docs = int(manifest['train_docs'])
        if tr_docs != len(doc_order):
            raise RuntimeError('Document order size mismatch')
        rng = np.random.default_rng(a.seed)
        positions = sorted(set([0, tr_docs//2, tr_docs-1] +
                               rng.integers(0, tr_docs, size=a.provenance_probes).tolist()))
        for pos in positions:
            record = db.execute('SELECT source_row, token_offset, stored_tokens, text_tokens '
                                'FROM documents WHERE split=? AND source_row=?',
                                ('train',int(pos))).fetchone()
            if record is None:
                raise RuntimeError(f'Shuffled document position {pos} missing in SQLite')
            src_row, off, stored, text_len = map(int,record)
            if src_row != pos or stored != text_len+1 or off < 0 or off+stored > len(mm):
                raise RuntimeError(f'Bad shuffled document offsets at position {pos}')
            if int(mm[off+stored-1]) != 50256:
                raise RuntimeError(f'Expected GPT-2 EOS at end of shuffled document {pos}')
        print(f'[PROVENANCE] {len(probe)} physical chunks and {len(positions)} document/EOS records validated',flush=True)
        for alias in a.models:
            done=a.out/f'{alias}_finished.json'; progress=a.out/f'{alias}_progress.npz'
            if a.resume and done.exists(): print('[SKIP]',alias,'already completed',flush=True);continue
            encoder=None
            try:
                ranks=Rankings(len(spans),k);cursor=0
                if a.resume and progress.exists():
                    cursor=load_progress(progress, ranks, len(chunk_ids))
                    print(f'[RESUME] {alias}: recovered cursor={cursor:,}/{len(chunk_ids):,} from {progress}', flush=True)
                if cursor<0 or cursor>len(chunk_ids):raise RuntimeError('Invalid resume cursor')
                print(f'[MODEL] {alias}: starting at {cursor}/{len(chunk_ids)}',flush=True)
                encoder=Encoder(alias, a.precision)
                probe_positions = sorted(set((0, len(chunk_ids)//2, len(chunk_ids)-1)))
                probe_ids = [int(chunk_ids[i]) for i in probe_positions]
                probe_texts = cache_reader.read_ids(probe_ids)
                query_probe = [spans[0]['text'], spans[-1]['text']] if len(spans) > 1 else [spans[0]['text']]
                encoder.select_backend(query_probe, probe_texts,
                                       a.batch_size, a.backend, a.equivalence_atol)
                q,qlens,qprompt=encoder.embed([s['text'] for s in spans],True,a.batch_size)
                qnorm=np.linalg.norm(q,axis=1,keepdims=True)
                q_unit=q/np.maximum(qnorm,1e-12)
                qfile=a.out/f'{alias}_span_vectors.f32.npy'
                np.save(qfile,q)
                einfo={'encoder_limit':encoder.limit,'embedding_dim':encoder.dim,
                       'query_max_encoder_tokens':max(qlens),'query_prompt':qprompt,'document_prompt':None, 'encoder_model_id':MODELS[alias],
                       'max_document_encoder_tokens_seen':0}
                vectors = None
                if a.save_vectors:
                    vpath = a.out / f'{alias}_vectors.f16'
                    expected_size = len(chunk_ids) * encoder.dim * 2
                    if vpath.exists() and vpath.stat().st_size != expected_size:
                        raise RuntimeError(f'{alias}: saved vectors have incompatible size')
                    if cursor and not vpath.exists():
                        raise RuntimeError(f'{alias}: resume requires prior vector file')
                    vectors = np.memmap(vpath, dtype='<f2', mode='r+' if vpath.exists() else 'w+',
                                        shape=(len(chunk_ids), encoder.dim))
                started=time.time()
                inference_seconds = 0.0
                last_profile = cursor
                next_save = cursor + a.save_every
                next_progress = cursor + a.progress_every if a.progress_every else None
                # Keep rank updates vectorized over substantial windows, not 4-16 chunks.
                score_buffer = []
                id_buffer = []
                buffered = 0
                def flush_rankings():
                    nonlocal buffered
                    if not buffered:
                        return
                    all_scores = np.concatenate(score_buffer, axis=1)
                    all_ids = np.concatenate(id_buffer)
                    ranks.update('cosine', all_scores, all_ids)
                    score_buffer.clear(); id_buffer.clear()
                    buffered = 0
                if stop.requested:
                    checkpoint(progress, ranks, cursor)
                    print(f'[STOP] {alias}: saved position {cursor:,}, restart with --resume', flush=True)
                    return
                # Every corpus chunk gets ONE embedding-tokenizer pass. The
                # prepared tensors go directly into the verified model forward.
                batches = cached_prepared_batches(cache_reader, chunk_ids, encoder, cursor,
                                                  len(chunk_ids), a.batch_size, a.prefetch)
                for offset, end, batch_ids, payload in batches:
                    t0 = time.perf_counter()
                    d,dlength,dprompt=encoder.forward_prepared(payload)
                    inference_seconds += time.perf_counter() - t0
                    einfo['document_prompt']=dprompt
                    einfo['max_document_encoder_tokens_seen']=max(einfo['max_document_encoder_tokens_seen'],max(dlength))
                    if vectors is not None: vectors[offset:end] = d.astype('<f2')
                    dnorm = np.linalg.norm(d,axis=1,keepdims=True)
                    batch_scores = q_unit @ (d/np.maximum(dnorm,1e-12)).T
                    if not np.isfinite(batch_scores).all(): raise RuntimeError('Non-finite similarities')
                    score_buffer.append(batch_scores)
                    id_buffer.append(batch_ids)
                    buffered += len(batch_ids)
                    needs_save = end >= next_save or end == len(chunk_ids) or stop.requested
                    if buffered >= a.rank_window or needs_save:
                        flush_rankings()
                    if next_progress is not None and end >= next_progress:
                        speed=(end-cursor)/max(time.time()-started,1e-3)
                        print(f'[PROGRESS] {alias} {end:,}/{len(chunk_ids):,} {speed:.2f} chunks/s',flush=True)
                        next_progress=end+a.progress_every
                    if a.profile_every and end - last_profile >= a.profile_every:
                        elapsed = max(time.time()-started, 0.001)
                        print(f'[PROFILE] {alias} model_forward={inference_seconds:.1f}s wall={elapsed:.1f}s '
                              f'other_or_overlap={max(0.0, elapsed-inference_seconds):.1f}s', flush=True)
                        last_profile = end
                    if needs_save:
                        if vectors is not None: vectors.flush()
                        checkpoint(progress,ranks,end)
                        speed=(end-cursor)/max(time.time()-started,1e-3)
                        print(f'[CHECKPOINT] {alias} {end:,}/{len(chunk_ids):,} {speed:.2f} chunks/s',flush=True)
                        next_save = end + a.save_every
                    if stop.requested:
                        print(f'[STOP] {alias}: saved position {end:,}, restart with --resume', flush=True)
                        return
                flush_rankings()
                # Persist encoder-info before reporting, only when rankings are finished.
                report_results(a.out,alias,ranks,spans,mm,inv,perm,doc_order,db,tok,len(chunk_ids),ids,einfo,a.print_full_chunks,cache_reader=cache_reader)
                if vectors is not None:
                    vectors.flush(); del vectors
                    atomic_json(a.out/f'{alias}_vectors_meta.json',{'dtype':'float16', 'shape':[len(chunk_ids),encoder.dim],
                          'row_ids':('physical_chunk_id_equals_vector_row' if isinstance(chunk_ids,range) else 'selected_physical_chunk_ids.npy'),'query_span_ids':'bbc_spans.json',
                           'query_vectors':f'{alias}_span_vectors.f32.npy','model':MODELS[alias],
                          'file':f'{alias}_vectors.f16'})
                atomic_json(done,{'model':alias,'status':'completed','blocks_scored':len(chunk_ids),
                                  'spans_scored':len(spans),'encoder':einfo,'finished_unix':time.time()})
                if (a.out/f'{alias}_error.json').exists():
                    (a.out/f'{alias}_error.json').unlink()
                print(f'[DONE] {alias}',flush=True)
            except (Exception, KeyboardInterrupt) as exc:
                # On recoverable failure, the last checkpoint remains valid.
                failure={'model':alias,'error_type':type(exc).__name__,'error':str(exc),
                         'traceback':traceback.format_exc()}
                atomic_json(a.out/f'{alias}_error.json',failure)
                print(f'[ERROR] {alias}: {exc}',file=sys.stderr,flush=True)
                if not a.skip_failed_models:raise
            finally:
                del encoder;gc.collect();torch.cuda.empty_cache()
    finally:
        cache_reader.close()
        db.close()
        restore_stop_handlers(old_handlers)
    print('[COMPLETE] See:',a.out,flush=True)

if __name__=='__main__':main()
