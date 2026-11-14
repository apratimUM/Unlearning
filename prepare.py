#!/usr/bin/env python3
"""Ordered, checkpoint-safe GPT-2 preprocessing for EleutherAI/the_pile_deduplicated.

No shuffling, packing per document, padding, truncation, or change to the model.
Each document is encoded completely, then one GPT-2 EOS (50256) is appended.
All deduplicated Pile documents go to train; validation comes only from EleutherAI/pile_val_test/val.jsonl.
Each split is concatenated into a little-endian uint16 stream; data.py constructs
1024-input/1024-target windows using 1025 consecutive stored tokens.

A *new* output directory is required for a new run. Never mix with older outputs.
Requires: datasets, huggingface_hub, transformers, tokenizers, tiktoken, numpy,
          plus your existing data.py for verification and optional previews.

Test: python3 -u prepare.py --out ./pile_test --max-docs 500000
Full: python3 -u prepare.py --out /coe-aroyc/apratim/pile_full_1024_v2
Resume: repeat the identical command with --resume (after graceful/abrupt exit)
Use within an allocated Frances CPU job; keep the output on persistent Turbo.
"""
import argparse
import array
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import sqlite3
import sys
import threading
import time

DATASET = 'EleutherAI/the_pile_deduplicated'
VALIDATION_DATASET = 'EleutherAI/pile_val_test'
VALIDATION_FILE = 'val.jsonl'
EXPECTED_ROWS = 134_318_121
BLOCK = 1024
EOS = 50256
VOCAB = 50257
END = object()


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=Path('/coe-aroyc/apratim/pile_full_1024_v2'))
    p.add_argument('--max-docs', type=int, default=None, help='Omit for the complete Pile')
    p.add_argument('--engine', choices=('tiktoken', 'hf'), default='tiktoken')
    p.add_argument('--threads', type=int, default=20)
    p.add_argument('--batch-size', '--tokenize-batch-size', dest='batch_size', type=int, default=2048)
    p.add_argument('--prefetch', '--prefetch-batches', dest='prefetch', type=int, default=4)
    p.add_argument('--save-every', type=int, default=20_000)
    p.add_argument('--verify-first', type=int, default=1000, help='Check first N docs against HF GPT-2 Fast')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validation-docs', type=int, default=2000, help='First N official pile_val_test validation documents')
    p.add_argument('--preview-blocks', type=int, default=5)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--profile', action='store_true')
    a = p.parse_args()
    if min(a.threads, a.batch_size, a.prefetch, a.save_every) < 1:
        p.error('threads, batch-size, prefetch and save-every must be positive')
    if a.max_docs is not None and not 1 <= a.max_docs <= EXPECTED_ROWS:
        p.error('--max-docs outside expected Pile range')
    if a.validation_docs < 1 or a.seed < 0 or a.verify_first < 0:
        p.error('invalid split/verification setting')
    if not 0 <= a.preview_blocks <= 50:
        p.error('--preview-blocks must be 0..50')
    if sys.byteorder != 'little':
        p.error('requires a little-endian CPU for array(H)')
    a.out = a.out.expanduser().resolve()
    return a


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def get_state(db, key):
    row = db.execute('SELECT value FROM progress WHERE key=?', (key,)).fetchone()
    return None if row is None else json.loads(row[0])


def put_state(db, key, value):
    db.execute('INSERT OR REPLACE INTO progress(key,value) VALUES (?,?)',
               (key, json.dumps(value, sort_keys=True)))


def make_db(path):
    db = sqlite3.connect(str(path), timeout=120)
    # DELETE journal is more portable to shared/network storage than WAL.
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA cache_size=-65536')
    db.execute('''CREATE TABLE IF NOT EXISTS documents (
       source_row INTEGER PRIMARY KEY, split TEXT NOT NULL,
       token_offset INTEGER NOT NULL, text_tokens INTEGER NOT NULL,
       stored_tokens INTEGER NOT NULL, text_sha256 TEXT NOT NULL)''')
    db.execute('''CREATE TABLE IF NOT EXISTS progress (
       key TEXT PRIMARY KEY, value TEXT NOT NULL)''')
    db.commit()
    return db


def live_state(settings, state):
    tr, va = state['train_tokens'], state['val_tokens']
    return dict(live_schema_version=1, dataset=DATASET,
                source_revision=settings['revision'], tokenizer='gpt2',
                token_dtype='<u2', block_size=BLOCK, eos_token_id=EOS,
                 seed=settings['seed'], validation_dataset=VALIDATION_DATASET,
                 validation_revision=settings['validation_revision'], validation_docs_requested=settings['validation_docs'],
                max_docs=settings['max_docs'],
                source_rows_committed=state['rows'],
                train_docs_committed=state['train_docs'],
                val_docs_committed=state['val_docs'],
                train_tokens_committed=tr, val_tokens_committed=va,
                train_ready_blocks=max(0, (tr-1)//BLOCK),
                val_ready_blocks=max(0, (va-1)//BLOCK),
                status=state['status'], finished=state['status']=='finished',
                published_unix_time=time.time())


def verify_checkpoint(db, state):
    # Training and official validation records share the documents table.
    expected = state['train_docs'] + state['val_docs']
    count = db.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
    if count != expected:
        raise RuntimeError(f'SQLite document count {count} != saved {expected}')
    lo, last = db.execute('SELECT MIN(source_row), MAX(source_row) FROM documents').fetchone()
    if expected and (lo != 0 or last != expected - 1):
        raise RuntimeError('Source row IDs do not match committed state')
    for s in ('train', 'val'):
        docs, n, end = db.execute('''SELECT COUNT(*), COALESCE(SUM(stored_tokens),0),
            COALESCE(MAX(token_offset+stored_tokens),0)
            FROM documents WHERE split=?''', (s,)).fetchone()
        if (docs, n, end) != (state[s+'_docs'], state[s+'_tokens'], state[s+'_tokens']):
            raise RuntimeError(f'{s}: SQLite offsets/totals disagree with checkpoint')


def make_manifest(state, settings, engine):
    return dict(schema_version=4, dataset=DATASET, source_revision=settings['revision'],
                tokenizer_name='gpt2', tokenizer_engine=engine,
                vocab_size=VOCAB, eos_token_id=EOS, block_size=BLOCK,
                token_dtype='<u2', seed=settings['seed'],
                 val_per_thousand=0, validation_dataset=VALIDATION_DATASET,
                 validation_revision=settings['validation_revision'], validation_split='validation',
                 validation_selection=f'first_{settings["validation_docs"]}_documents_in_official_validation_order',
                shuffling=False, chunking_method='standard_concatenate_then_split',
                full_documents=True, documents_seen=state['train_docs']+state['val_docs'],
                train_docs=state['train_docs'], val_docs=state['val_docs'],
                train_tokens=state['train_tokens'], val_tokens=state['val_tokens'],
                train_blocks=max(0, (state['train_tokens']-1)//BLOCK),
                val_blocks=max(0, (state['val_tokens']-1)//BLOCK),
                input_tokens_per_block=BLOCK, target_tokens_per_block=BLOCK,
                terminal_padding=0, incomplete_final_block='ignored',
                document_index='documents.sqlite',
                stream_format='EOS-separated consecutive GPT-2 token IDs',
                source_scan_complete=(settings['max_docs'] is None
                    and state['rows'] == EXPECTED_ROWS),
                metadata_complete=True)


def append_official_validation(out, db, streams, state, settings, hf, encode, batch_size, stop):
    """Append only official validation rows; checkpoint each batch for safe resume."""
    from huggingface_hub import hf_hub_url
    import requests

    requested = settings['validation_docs']
    if state['val_docs'] >= requested:
        return
    url = hf_hub_url(repo_id=VALIDATION_DATASET, filename=VALIDATION_FILE,
                     repo_type='dataset', revision=settings['validation_revision'])
    print(f'[VAL SOURCE] {VALIDATION_DATASET}@{settings["validation_revision"]} '
          f'{VALIDATION_FILE}: {state["val_docs"]:,}/{requested:,} committed', flush=True)
    # Resume by streaming just the small official validation prefix (<=2000 docs by default).
    # Unlike the 134M-document training source, this is cheap and needs no HF cursor.
    with requests.get(url, stream=True, timeout=(30, 120),
                      headers={'Accept-Encoding': 'identity'}) as response:
        response.raise_for_status()
        lines = iter(response.iter_lines(chunk_size=1 << 20))
        # On resume, consume the already-committed official validation prefix.
        # Skip independently of collecting the next batch.
        seen = 0
        while seen < state['val_docs']:
            try:
                line = next(lines)
            except StopIteration:
                raise RuntimeError('Official validation ended before saved position')
            if line.strip():
                seen += 1
        while state['val_docs'] < requested:
            if stop.is_set():
                print(f'[STOP] validation committed through {state["val_docs"]:,}; use --resume', flush=True)
                return
            texts = []
            while len(texts) < min(batch_size, requested - state['val_docs']):
                try:
                    line = next(lines)
                except StopIteration:
                    break
                if not line.strip():
                    continue
                item = json.loads(line)
                content = item['text']
                if not isinstance(content, str):
                    raise TypeError(f'Non-text official validation document at {seen}')
                texts.append(content)
                seen += 1
            if not texts:
                raise RuntimeError(f'Official validation exhausted at {state["val_docs"]} documents')
            ids_list = encode(texts)
            if len(ids_list) != len(texts):
                raise RuntimeError('Validation tokenizer returned the wrong number of documents')
            entries = []
            data = bytearray()
            first = state['val_docs']
            for j, (text, ids) in enumerate(zip(texts, ids_list)):
                if any(t < 0 or t >= VOCAB for t in ids):
                    raise ValueError(f'Invalid validation token at row {first+j}')
                offset = state['val_tokens']
                tokens = array.array('H', ids)
                tokens.append(EOS)
                data.extend(tokens.tobytes())
                entries.append((state['rows'] + first + j, 'val', offset,
                                len(ids), len(ids) + 1,
                                hashlib.sha256(text.encode('utf-8')).hexdigest()))
                state['val_tokens'] += len(ids) + 1
                state['val_docs'] += 1
            streams['val'].write(data)
            streams['val'].flush()
            os.fsync(streams['val'].fileno())
            db.executemany("""INSERT INTO documents
                (source_row,split,token_offset,text_tokens,stored_tokens,text_sha256)
                VALUES (?,?,?,?,?,?)""", entries)
            put_state(db, 'state', state)
            db.commit()
            atomic_json(out / 'live_state.json', live_state(settings, state))
            print(f'[VAL] docs={state["val_docs"]:,}/{requested:,} '
                  f'tokens={state["val_tokens"]:,}', flush=True)
    if state['val_docs'] != requested:
        raise RuntimeError('Validation document count is incomplete')


def main():
    a = arguments()
    out = a.out
    if a.resume:
        if not (out/'documents.sqlite').is_file():
            raise FileNotFoundError(f'No saved output to resume: {out}')
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing nonempty output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)

    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ.setdefault('RAYON_NUM_THREADS', str(a.threads))
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
    if not hf.is_fast or hf.vocab_size != VOCAB or hf.eos_token_id != EOS:
        raise RuntimeError('Unexpected HF GPT-2 fast tokenizer')
    encoder = None
    token_pool = None
    if a.engine == 'tiktoken':
        import tiktoken
        encoder = tiktoken.get_encoding('gpt2')
        if encoder.n_vocab != VOCAB or encoder.eot_token != EOS:
            raise RuntimeError('Unexpected tiktoken GPT-2 encoding')
        # Unlike tiktoken.encode_batch, this worker pool persists for the entire run.
        token_pool = ThreadPoolExecutor(max_workers=a.threads, thread_name_prefix='bpe')

    def encode(texts):
        if encoder is None:
            return [e.ids for e in hf.backend_tokenizer.encode_batch(texts, add_special_tokens=False)]
        return list(token_pool.map(lambda s: encoder.encode(s, allowed_special={'<|endoftext|>'}), texts))

    db = None
    streams = {}
    old_handlers = {}
    stop = threading.Event()
    failed = threading.Event()
    errors = queue.Queue()
    try:
        db = make_db(out/'documents.sqlite')
        original = get_state(db, 'settings') if a.resume else None
        if a.resume and original is None:
            raise RuntimeError('Missing original preparation settings')
        revision = original['revision'] if a.resume else HfApi().dataset_info(DATASET).sha
        validation_revision = (original['validation_revision'] if a.resume
                               else HfApi().dataset_info(VALIDATION_DATASET).sha)
        settings = dict(dataset=DATASET, revision=revision, tokenizer='gpt2',
                        engine=a.engine, block_size=BLOCK, seed=a.seed,
                         validation_dataset=VALIDATION_DATASET, validation_revision=validation_revision,
                         validation_docs=a.validation_docs, max_docs=a.max_docs,
                        shuffling=False)
        if a.resume:
            if original != settings:
                raise ValueError('Resume settings differ: use original engine/seed/validation-docs/max-docs')
            state = get_state(db, 'state')
            if not state or state['status'] not in ('writing', 'finished'):
                raise RuntimeError('Invalid saved preparation state')
            verify_checkpoint(db, state)
            print(f'[RESUME] checkpoint has {state["rows"]:,} committed rows', flush=True)
        else:
            state = dict(rows=0, train_docs=0, val_docs=0, train_tokens=0,
                         val_tokens=0, status='writing')
            put_state(db, 'settings', settings)
            put_state(db, 'state', state)
            db.commit()
            hf.save_pretrained(out/'tokenizer')
        if not (out/'tokenizer').is_dir():
            raise RuntimeError('Missing saved tokenizer directory')
        for s in ('train', 'val'):
            path = out/(s+'.bin')
            f = path.open('r+b' if a.resume else 'w+b', buffering=8 << 20)
            required = state[s+'_tokens'] * 2
            f.seek(0, 2)
            if f.tell() < required:
                f.close()
                raise RuntimeError(f'{path} shorter than committed checkpoint')
            f.truncate(required)  # remove uncommitted bytes only
            f.seek(required)
            streams[s] = f
        atomic_json(out/'live_state.json', live_state(settings, state))

        goal = EXPECTED_ROWS if a.max_docs is None else a.max_docs
        if state['status'] == 'writing' and state['rows'] < goal:
            dataset = load_dataset(DATASET, split='train', streaming=True, revision=revision)
            old_rows = state['rows']
            saved_cursor = get_state(db, 'source_cursor') if a.resume else None
            skip_base = 0
            if saved_cursor is not None:
                if (saved_cursor.get('version') != 1 or
                    saved_cursor.get('rows') != old_rows or
                    not isinstance(saved_cursor.get('state'), dict)):
                    raise RuntimeError('Source cursor and committed rows disagree')
                skip_base = int(saved_cursor.get('base_skip_rows', 0))
                if not 0 <= skip_base <= old_rows:
                    raise RuntimeError('Invalid base_skip_rows in checkpoint')
                if skip_base:
                    dataset = dataset.skip(skip_base)
                dataset.load_state_dict(saved_cursor['state'])
                print(f'[RESUME] restored source cursor at {old_rows:,}', flush=True)
            elif old_rows:
                # Only for legacy checkpoints missing a cursor. This can be slow.
                print(f'[RESUME] no cursor: skipping {old_rows:,} rows', flush=True)
                dataset = dataset.skip(old_rows)
                skip_base = old_rows

            def on_signal(sig, _frame):
                print(f'[SIGNAL] received {sig}, draining queued batches', flush=True)
                stop.set()
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
                old_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, on_signal)

            raw_q = queue.Queue(maxsize=a.prefetch)
            tok_q = queue.Queue(maxsize=a.prefetch)
            ready_q = queue.Queue(maxsize=a.prefetch)
            stats = dict(read=0.0, tokenize=0.0, pack=0.0, write=0.0, checkpoint=0.0)
            stats_lock = threading.Lock()

            def put(q, value):
                while not failed.is_set():
                    try:
                        q.put(value, timeout=0.2)
                        return
                    except queue.Full:
                        continue

            def consume(q, futures):
                while not failed.is_set():
                    try:
                        return q.get(timeout=0.2)
                    except queue.Empty:
                        if all(f.done() for f in futures):
                            if not errors.empty():
                                raise errors.get()
                            raise RuntimeError('Workers stopped without an END marker')
                if not errors.empty():
                    raise errors.get()
                raise RuntimeError('Worker pipeline aborted')

            def stage(fn):
                try:
                    fn()
                except BaseException as exc:
                    errors.put(exc)
                    failed.set()

            def reader():
                iterator = iter(dataset)
                row = old_rows
                goal = EXPECTED_ROWS if a.max_docs is None else a.max_docs
                while row < goal and not (stop.is_set() or failed.is_set()):
                    started = time.perf_counter()
                    texts = []
                    for _ in range(min(a.batch_size, goal-row)):
                        try:
                            example = next(iterator)
                        except StopIteration:
                            break
                        content = example['text']
                        if not isinstance(content, str):
                            raise TypeError(f'Non-text document at row {row+len(texts)}')
                        texts.append(content)
                    with stats_lock:
                        stats['read'] += time.perf_counter()-started
                    if not texts:
                        break
                    # Capture after consuming this exact batch, before consuming more.
                    # If cursor cannot be snapshotted, fail loudly (never silent fallback).
                    cursor = dataset.state_dict()
                    put(raw_q, (row, texts, cursor))
                    row += len(texts)
                put(raw_q, END)

            verified = [0]
            def tokenizer_stage():
                while not failed.is_set():
                    item = consume(raw_q, reader_jobs)
                    if item is END:
                        put(tok_q, END)
                        return
                    first, texts, cursor = item
                    started = time.perf_counter()
                    ids = encode(texts)
                    with stats_lock:
                        stats['tokenize'] += time.perf_counter()-started
                    if len(ids) != len(texts):
                        raise RuntimeError('Tokenizer returned wrong batch length')
                    if encoder is not None and verified[0] < a.verify_first:
                        n = min(len(texts), a.verify_first-verified[0])
                        reference = hf.backend_tokenizer.encode_batch(texts[:n], add_special_tokens=False)
                        for j, ref in enumerate(reference):
                            if ids[j] != ref.ids:
                                raise RuntimeError(f'tiktoken vs HF mismatch at source row {first+j}')
                        verified[0] += n
                    put(tok_q, (first, texts, ids, cursor))

            def packer():
                while not failed.is_set():
                    item = consume(tok_q, tokenize_jobs)
                    if item is END:
                        put(ready_q, END)
                        return
                    first, texts, id_lists, cursor = item
                    started = time.perf_counter()
                    packed = []
                    for text, ids in zip(texts, id_lists):
                        tokens = array.array('H', ids)
                        tokens.append(EOS)
                        packed.append((tokens.tobytes(), len(ids),
                                       hashlib.sha256(text.encode('utf-8')).hexdigest()))
                    with stats_lock:
                        stats['pack'] += time.perf_counter()-started
                    put(ready_q, (first, packed, cursor))

            def checkpoint(cursor):
                t = time.perf_counter()
                for f in streams.values():
                    f.flush()
                    os.fsync(f.fileno())
                put_state(db, 'state', state)
                if cursor is not None:
                    put_state(db, 'source_cursor', dict(version=1, rows=state['rows'],
                                                        base_skip_rows=skip_base, state=cursor))
                db.commit()  # state, document metadata, and source cursor: one transaction
                atomic_json(out/'live_state.json', live_state(settings, state))
                db.execute('BEGIN')
                with stats_lock:
                    stats['checkpoint'] += time.perf_counter()-t

            started = time.perf_counter()
            initial_tokens = state['train_tokens'] + state['val_tokens']
            done_at = old_rows
            next_report = ((old_rows // a.save_every) + 1)*a.save_every
            last_cursor = saved_cursor['state'] if saved_cursor is not None else None
            db.execute('BEGIN')
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix='pile') as workers:
                reader_future = workers.submit(stage, reader)
                reader_jobs = [reader_future]
                tokenize_future = workers.submit(stage, tokenizer_stage)
                tokenize_jobs = [tokenize_future]
                pack_future = workers.submit(stage, packer)
                futures = [reader_future, tokenize_future, pack_future]
                try:
                    while True:
                        item = consume(ready_q, futures)
                        if item is END:
                            break
                        first, docs, cursor = item
                        if first != done_at:
                            raise RuntimeError(f'Out of order: got {first}, expected {done_at}')
                        t = time.perf_counter()
                        buffers = {'train': bytearray(), 'val': bytearray()}
                        entries = []
                        for j, (packed, text_len, digest) in enumerate(docs):
                            row = first+j
                            s = 'train'  # All deduplicated Pile rows belong to training.
                            offset = state[s+'_tokens']
                            buffers[s].extend(packed)
                            entries.append((row, s, offset, text_len, text_len+1, digest))
                            state[s+'_tokens'] += text_len+1
                            state[s+'_docs'] += 1
                        for s in ('train','val'):
                            if buffers[s]:
                                streams[s].write(buffers[s])
                        db.executemany('''INSERT INTO documents
                            (source_row, split, token_offset, text_tokens, stored_tokens, text_sha256)
                            VALUES (?, ?, ?, ?, ?, ?)''', entries)
                        done_at += len(docs)
                        state['rows'] = done_at
                        last_cursor = cursor
                        with stats_lock:
                            stats['write'] += time.perf_counter()-t
                        if done_at >= next_report:
                            checkpoint(last_cursor)
                            secs = max(1e-9, time.perf_counter()-started)
                            new_docs = done_at-old_rows
                            new_tokens = state['train_tokens']+state['val_tokens']
                            print(f'[PREPARE] documents={done_at:,} '
                                  f'rate={new_docs/secs:,.1f} docs/s '
                                  f'tokens/s={(new_tokens-initial_tokens)/secs:,.0f} '
                                  f'train_tokens={state["train_tokens"]:,} '
                                  f'val_tokens={state["val_tokens"]:,}', flush=True)
                            if a.profile:
                                with stats_lock:
                                    print('[PROFILE] '+' '.join(f'{k}={v:.1f}s' for k,v in stats.items()), flush=True)
                            next_report = ((done_at // a.save_every)+1)*a.save_every
                    for f in futures:
                        f.result()
                    if not errors.empty():
                        raise errors.get()
                    checkpoint(last_cursor)
                except BaseException:
                    failed.set()
                    raise

            if stop.is_set():
                print(f'[STOP] checkpointed through {state["rows"]:,}; use --resume', flush=True)
                return
            goal = EXPECTED_ROWS if a.max_docs is None else a.max_docs
            if state['rows'] != goal:
                raise RuntimeError(f'Source exhausted at {state["rows"]:,}; expected {goal:,}')

        # Official validation is completely separate from the deduplicated Pile.
        # On resume, skip the training reader if all training rows are already saved.
        if state['status'] == 'writing':
            if state['rows'] != goal:
                raise RuntimeError('Training source incomplete; cannot finalize validation')
            if not old_handlers:
                def on_validation_signal(sig, _frame):
                    print(f'[SIGNAL] received {sig}; stopping after committed validation batch', flush=True)
                    stop.set()
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
                    old_handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, on_validation_signal)
            append_official_validation(out, db, streams, state, settings, hf, encode,
                                       a.batch_size, stop)
            if stop.is_set():
                return
            if min(state['train_tokens'], state['val_tokens']) < BLOCK+1:
                raise RuntimeError('Train or official validation has fewer than 1025 tokens')
            state['status'] = 'finished'
            for f in streams.values():
                f.flush()
                os.fsync(f.fileno())
            put_state(db, 'state', state)
            db.commit()
            atomic_json(out/'live_state.json', live_state(settings, state))

        # Building the post-hoc index only after data ingestion reduces SQLite cost.
        db.execute('CREATE INDEX IF NOT EXISTS documents_split_offset ON documents(split, token_offset)')
        db.commit()
        manifest = make_manifest(state, settings, a.engine)
        atomic_json(out/'manifest.json', manifest)
        from data import verify_data
        verify_data(out)
        if a.preview_blocks:
            import numpy as np
            for s in ('train','val'):
                tokens = np.memmap(out/(s+'.bin'), dtype='<u2', mode='r')
                try:
                    n = min(a.preview_blocks, manifest[s+'_blocks'])
                    previews = []
                    for block in range(n):
                        start = block * BLOCK
                        seq = tokens[start:start+BLOCK+1]
                        x, y = seq[:-1], seq[1:]
                        start = block * BLOCK
                        end = start + BLOCK
                        spans = db.execute('''SELECT source_row,token_offset,text_tokens,stored_tokens
                            FROM documents WHERE split=? AND token_offset<?
                            AND token_offset+stored_tokens>? ORDER BY token_offset''', (s, end, start))
                        previews.append(dict(block_id=block, stream_start=start,
                             context_size=BLOCK, input_length=BLOCK, target_length=BLOCK,
                             source_documents=[dict(source_row=r, token_offset=off,
                                  text_tokens=nt, stored_tokens=ns)
                                  for r, off, nt, ns in spans],
                             eos_positions=[i for i,v in enumerate(x) if v == EOS],
                             input_token_ids=x.tolist(), target_token_ids=y.tolist(),
                             decoded_preview=hf.decode(x[:160].tolist()),
                             next_token_alignment_correct=True))
                    atomic_json(out/(s+'_chunk_preview.json'),
                                dict(method='standard_concatenate_then_split', context_size=BLOCK,
                                     split=s, chunks=previews))
                finally:
                    del tokens
        print(f'[DONE] documents={state["rows"]:,} '
              f'train_blocks={manifest["train_blocks"]:,} '
              f'val_blocks={manifest["val_blocks"]:,} output={out}', flush=True)
    finally:
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)
        failed.set()
        if token_pool is not None:
            token_pool.shutdown(wait=True)
        for f in streams.values():
            f.close()
        if db is not None:
            db.close()

if __name__ == '__main__':
    main()
