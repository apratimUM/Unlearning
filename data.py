import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch

BLOCK_SIZE = 1024
EOS_ID = 50256
VOCAB_SIZE = 50257
TOKEN_DTYPE = np.dtype('<u2')
SAMPLER_ORDER = 'precomputed_numpy_permutation_v1'
MASK64 = (1 << 64) - 1


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data(folder, shuffled=False):
    """Validate schema and binary sizes (not complete cryptographic integrity)."""
    folder = Path(folder)
    m = json.loads((folder / ('manifest_shuffled.json' if shuffled else 'manifest.json')).read_text(encoding='utf-8'))
    expected = {
        'schema_version': 4, 'block_size': BLOCK_SIZE,
        'vocab_size': VOCAB_SIZE, 'eos_token_id': EOS_ID,
        'chunking_method': 'standard_concatenate_then_split',
        'shuffling': False, 'terminal_padding': 0,
        'incomplete_final_block': 'eos_right_padded_targets_ignore_index_-100',
    }
    for key, value in expected.items():
        if m.get(key) != value:
            raise ValueError(f'Manifest {key}: expected {value!r}, got {m.get(key)!r}')
    # manifest['shuffling'] describes source-document preparation, NOT the
    # separate deterministic training-block visit order.
    dtype = np.dtype(m['token_dtype'])
    if dtype != TOKEN_DTYPE:
        raise ValueError('Expected little-endian uint16 token streams')
    for split in ('train', 'val'):
        tokens = int(m[f'{split}_tokens'])
        blocks = int(m[f'{split}_blocks'])
        path = folder / ('train_shuffled.bin' if shuffled and split == 'train' else f'{split}.bin')
        if not path.is_file() or path.stat().st_size != tokens * dtype.itemsize:
            raise ValueError(f'{split}: missing binary or incorrect size')
        if tokens < 2 or blocks != (tokens - 1 + BLOCK_SIZE - 1) // BLOCK_SIZE:
            raise ValueError(f'{split}: incorrect number of blocks')
    db_name = 'documents_shuffled.sqlite' if shuffled else 'documents.sqlite'
    if not (folder / db_name).is_file():
        raise FileNotFoundError(folder / db_name)
    if shuffled and (m.get('document_shuffled') is not True or m.get('document_shuffle_index') != 'document_order_shuffled.u32'):
        raise ValueError('Shuffled manifest/document provenance missing')
    if not (folder / 'tokenizer').is_dir():
        raise FileNotFoundError(folder / 'tokenizer')
    return m


def _mix64(value):
    value = (int(value) + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


class _BlockPermutation:
    """Seeded, reversible, constant-memory permutation using cycle walking."""

    def __init__(self, n, seed):
        self.n = int(n)
        if self.n < 1 or int(seed) < 0:
            raise ValueError('Block count must be positive and seed nonnegative')
        self.bits = max(1, ((self.n - 1).bit_length() + 1) // 2)
        self.mask = (1 << self.bits) - 1
        self.keys = tuple(
            _mix64(int(seed) ^ (0xD1B54A32D192ED03 * (r + 1)))
            for r in range(6)
        )

    def _step(self, value, inverse=False):
        left, right = value >> self.bits, value & self.mask
        if inverse:
            for key in reversed(self.keys):
                left, right = right ^ (_mix64(left ^ key) & self.mask), left
        else:
            for key in self.keys:
                left, right = right, left ^ (_mix64(right ^ key) & self.mask)
        return (left << self.bits) | right

    def physical(self, visit):
        if not 0 <= visit < self.n:
            raise IndexError(visit)
        if self.n == 1:
            return 0
        result = self._step(visit)
        while result >= self.n:
            result = self._step(result)
        return result

    def logical(self, physical):
        if not 0 <= physical < self.n:
            raise IndexError(physical)
        if self.n == 1:
            return 0
        result = self._step(physical, inverse=True)
        while result >= self.n:
            result = self._step(result, inverse=True)
        return result


class PackedSampler:
    """1024-input/1024-target, single-pass packed-block sampler.

    train: seeded permutation of physical complete blocks, no replacement.
    val: sequential physical blocks, not shuffled.
    get_block always accepts a PHYSICAL block ID.
    block_ids accepts a LOGICAL visit offset and returns PHYSICAL IDs.
    """

    def __init__(self, folder, split, block=BLOCK_SIZE, seed=42, shuffled=False):
        self.folder = Path(folder)
        self.split = split
        self.block = int(block)
        self.seed = int(seed)
        self.shuffled = bool(shuffled)
        if split not in ('train', 'val') or self.block != BLOCK_SIZE or self.seed < 0:
            raise ValueError('Invalid split, block size, or seed')
        m = verify_data(self.folder, shuffled=self.shuffled)
        self.tokens = np.memmap(
            self.folder / ('train_shuffled.bin' if self.shuffled and split == 'train' else f'{split}.bin'),
            dtype=np.dtype(m['token_dtype']), mode='r'
        )
        self.n_blocks = (len(self.tokens) - 1 + self.block - 1) // self.block
        if self.n_blocks != int(m[f'{split}_blocks']):
            raise ValueError('Block count differs from manifest')
        self.permutation = None
        self.inverse_permutation = None
        if split == 'train':
            order_info = json.loads((self.folder / ('train_chunk_order_shuffled.json' if self.shuffled else 'train_chunk_order.json')).read_text())
            if (order_info['seed'] != self.seed or
                order_info['n_blocks'] != self.n_blocks or
                order_info['algorithm'] != SAMPLER_ORDER):
                raise ValueError('Precomputed chunk order does not match dataset/seed')
            for filename in (('train_chunk_order_shuffled.u32', 'train_chunk_inverse_shuffled.u32') if self.shuffled else ('train_chunk_order.u32', 'train_chunk_inverse.u32')):
                if (self.folder / filename).stat().st_size != self.n_blocks * 4:
                    raise ValueError(f'Incorrect precomputed order file size: {filename}')
            self.permutation = np.memmap(self.folder / ('train_chunk_order_shuffled.u32' if self.shuffled else 'train_chunk_order.u32'), dtype='<u4', mode='r')
            self.inverse_permutation = np.memmap(self.folder / ('train_chunk_inverse_shuffled.u32' if self.shuffled else 'train_chunk_inverse.u32'), dtype='<u4', mode='r')
        self.sampler_order = SAMPLER_ORDER if split == 'train' else 'sequential_validation'

    def __len__(self):
        return self.n_blocks

    def physical_block_id(self, visit):
        visit = int(visit)
        if not 0 <= visit < self.n_blocks:
            raise IndexError(visit)
        return int(self.permutation[visit]) if self.permutation is not None else visit

    def logical_block_id(self, physical):
        physical = int(physical)
        if not 0 <= physical < self.n_blocks:
            raise IndexError(physical)
        return int(self.inverse_permutation[physical]) if self.inverse_permutation is not None else physical

    def block_ids(self, first, count):
        """Physical IDs in visit order; empty after exhaustion; never wraps."""
        first, count = int(first), int(count)
        if first < 0 or count < 0:
            raise ValueError('first and count must be nonnegative')
        return [self.physical_block_id(i)
                for i in range(first, min(first + count, self.n_blocks))]

    def get_block(self, block_id):
        """Read an actual physical block without any token reordering."""
        block_id = int(block_id)
        if not 0 <= block_id < self.n_blocks:
            raise IndexError(block_id)
        start = block_id * self.block
        seq = np.asarray(self.tokens[start:start + self.block + 1], dtype=np.int64)
        valid = len(seq) - 1
        x = np.full(self.block, EOS_ID, dtype=np.int64)
        y = np.full(self.block, -100, dtype=np.int64)
        x[:valid] = seq[:valid]
        y[:valid] = seq[1:]
        if x.shape != (BLOCK_SIZE,) or y.shape != (BLOCK_SIZE,):
            raise RuntimeError('Invalid block shape')
        if not np.array_equal(x[1:valid], y[:valid-1]):
            raise RuntimeError('Next-token alignment mismatch')
        return x, y

    def batch(self, step, micro, batch, accum=1):
        """(N,1024) input/target tensors; final microbatch can be partial."""
        step, micro, batch, accum = map(int, (step, micro, batch, accum))
        if step < 0 or micro < 0 or batch < 1 or accum < 1 or micro >= accum:
            raise ValueError('Invalid step/micro/batch/accum')
        first = (step * accum + micro) * batch
        return self.batch_range(first, batch)

    def batch_range(self, first, count):
        """Read only the assigned local rank's chunks (no redundant global batch)."""
        ids = self.block_ids(first, count)
        if not ids:
            raise StopIteration('No complete blocks remain')
        pairs = [self.get_block(k) for k in ids]
        x = torch.from_numpy(np.stack([pair[0] for pair in pairs]))
        y = torch.from_numpy(np.stack([pair[1] for pair in pairs]))
        if x.shape != (len(ids), BLOCK_SIZE) or y.shape != x.shape:
            raise RuntimeError('Invalid batch shape')
        return x, y

    def exposure_count(self, block_id, consumed_blocks):
        return self.exposure_counts([block_id], consumed_blocks)[int(block_id)]

    def exposure_counts(self, block_ids, consumed_blocks):
        """Exactly 0/1 per PHYSICAL block, using inverse visit-order mapping."""
        consumed = int(consumed_blocks)
        if not 0 <= consumed <= self.n_blocks:
            raise ValueError('Invalid consumed block count')
        ids = {int(i) for i in block_ids}
        if any(i < 0 or i >= self.n_blocks for i in ids):
            raise ValueError('Invalid physical block ID')
        return {i: int(self.logical_block_id(i) < consumed) for i in sorted(ids)}

    def document_exposures(self, consumed_blocks, max_docs=None):
        """Stream approximate *input-token* exposures from SQLite document index.

        Honors shuffled block visit order. Excludes the unpresented final tail.
        This is not the same as full-document presentations or gradient exposure.
        """
        consumed = int(consumed_blocks)
        if not 0 <= consumed <= self.n_blocks:
            raise ValueError('Invalid consumed block count')
        if max_docs is not None and int(max_docs) < 0:
            raise ValueError('max_docs must be nonnegative')
        limit = None if max_docs is None else int(max_docs)
        uri = (self.folder / ('documents_shuffled.sqlite' if self.shuffled else 'documents.sqlite')).resolve().as_uri() + '?mode=ro'
        db = sqlite3.connect(uri, uri=True)
        try:
            cursor = db.execute('''SELECT source_row,token_offset,stored_tokens,
                                          text_tokens,text_sha256
                                   FROM documents WHERE split=? ORDER BY token_offset''',
                                (self.split,))
            for index, (source_row, start, stored, text_tokens, digest) in enumerate(cursor):
                if limit is not None and index >= limit:
                    break
                end = start + stored
                first = start // self.block
                stop = min(self.n_blocks, (end - 1) // self.block + 1)
                visited = presented = 0
                for physical in range(first, stop):
                    if self.logical_block_id(physical) >= consumed:
                        continue
                    visited += 1
                    a, b = physical * self.block, (physical + 1) * self.block
                    presented += max(0, min(end, b) - max(start, a))
                yield {
                    'document_index': index, 'source_row': source_row,
                    'sha256': digest, 'text_tokens': text_tokens,
                    'tokens_presented': presented,
                    'approximate_fractional_document_exposures': presented / max(1, stored),
                    'unique_blocks_seen': visited,
                    'metric_note': 'Input tokens including EOS; not complete-document presentations'
                }
        finally:
            db.close()

    def close(self):
        for name in ('tokens', 'permutation', 'inverse_permutation'):
            mapping = getattr(self, name, None)
            if isinstance(mapping, np.memmap):
                mapping._mmap.close()
            setattr(self, name, None)

# ============================================================================
# INCREMENTAL, COMMITTED-ONLY READER FOR LIVE PILE PREPARATION
# ============================================================================
# Unlike PackedSampler, this reader does not require manifest.json: that file
# is written only once prepare.py finishes. Physical block IDs are stable:
# the input for block b begins at absolute train-token offset 1024*b.

import os
import time
import threading
import queue


LIVE_SAMPLER_ORDER = 'sequential_committed_v1'


class LivePackedReader:
    """Read ONLY published, fsynced 1025-token prediction windows.

    prepare.py must be the only writer of this dataset directory. The reader
    never changes token files or SQLite state and works while they grow.
    """

    def __init__(self, folder, split='train', poll_seconds=1.0,
                 idle_timeout_seconds=3600.0):
        self.folder = Path(folder)
        if split not in ('train', 'val'):
            raise ValueError('Expected split train or val')
        self.split = split
        self.poll_seconds = float(poll_seconds)
        self.idle_timeout_seconds = float(idle_timeout_seconds)
        if self.poll_seconds <= 0 or self.idle_timeout_seconds <= 0:
            raise ValueError('Polling and timeout must be positive')
        self.path = self.folder / f'{split}.bin'
        self.state_path = self.folder / 'live_state.json'
        self._fd = None
        self._last_progress_time = time.monotonic()
        self._last_source_rows = -1
        self._last_ready = -1

    def __enter__(self):
        start = time.monotonic()
        while True:
            try:
                self._fd = os.open(self.path, os.O_RDONLY)
                break
            except FileNotFoundError:
                if time.monotonic() - start > self.idle_timeout_seconds:
                    raise TimeoutError(f'Waiting for live token file: {self.path}')
                time.sleep(self.poll_seconds)
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def status(self):
        try:
            state = json.loads(self.state_path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            raise RuntimeError(f'Missing {self.state_path}. Run new prepare.py without --no-live-state.')
        if (state.get('live_schema_version') != 1 or
            state.get('block_size') != BLOCK_SIZE or
            state.get('token_dtype') != TOKEN_DTYPE.str or
            state.get('eos_token_id') != EOS_ID):
            raise ValueError('Incompatible live_state.json schema/tokenization')
        rows = int(state['source_rows_committed'])
        ready = int(state[f'{self.split}_ready_blocks'])
        if rows < self._last_source_rows or ready < self._last_ready:
            raise RuntimeError('Committed-data watermark moved backwards; writer may have restarted incorrectly')
        if rows > self._last_source_rows or ready > self._last_ready:
            self._last_source_rows = rows
            self._last_ready = ready
            self._last_progress_time = time.monotonic()
        return state

    def _wait_ready(self, end_exclusive, stop_event=None):
        while True:
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError('Prefetch stopped')
            state = self.status()
            ready = int(state[f'{self.split}_ready_blocks'])
            if ready >= end_exclusive or bool(state['finished']):
                return ready, bool(state['finished'])
            if time.monotonic() - self._last_progress_time > self.idle_timeout_seconds:
                raise TimeoutError('No committed preprocessing progress for idle timeout; check prepare.py job')
            if stop_event is None:
                time.sleep(self.poll_seconds)
            else:
                stop_event.wait(self.poll_seconds)

    def get_windows(self, first_block, max_blocks, *, wait=True, stop_event=None):
        """Return (N,1025) long tensor and the first physical block ID.

        If source preparation is still running, wait for the *whole* request.
        Return a partial batch only once preparation is marked finished.
        """
        first_block, max_blocks = int(first_block), int(max_blocks)
        if first_block < 0 or max_blocks < 1:
            raise ValueError('Invalid block range')
        if self._fd is None:
            raise RuntimeError('Open LivePackedReader with a context manager')
        if wait:
            ready, finished = self._wait_ready(first_block + max_blocks, stop_event)
        else:
            state = self.status()
            ready, finished = int(state[f'{self.split}_ready_blocks']), bool(state['finished'])
        n = max(0, min(max_blocks, ready - first_block))
        if n == 0:
            return None, first_block, finished
        # Each window overlaps its neighbor by one label token. Read once,
        # then use strided slices. The last window needs +1 extra token.
        begin = first_block * BLOCK_SIZE
        want_tokens = n * BLOCK_SIZE + 1
        buf = os.pread(self._fd, want_tokens * TOKEN_DTYPE.itemsize,
                       begin * TOKEN_DTYPE.itemsize)
        if len(buf) != want_tokens * TOKEN_DTYPE.itemsize:
            raise RuntimeError('Published blocks not present in token file')
        raw = np.frombuffer(buf, dtype=TOKEN_DTYPE)
        windows = np.lib.stride_tricks.sliding_window_view(raw, BLOCK_SIZE + 1)[::BLOCK_SIZE]
        if windows.shape != (n, BLOCK_SIZE + 1):
            raise AssertionError('Incorrect live window shape')
        return torch.from_numpy(windows.astype(np.int64, copy=True)), first_block, finished


class LiveUpdatePrefetcher:
    """Background CPU reader; yield complete optimizer updates in source order.

    Prefetching may run ahead of GPU updates; only the CONSUMER cursor may be
    saved in a training checkpoint, never the prefetcher's speculative cursor.
    """

    def __init__(self, folder, first_block, blocks_per_update,
                 queue_updates=2, poll_seconds=1., idle_timeout_seconds=3600.):
        if queue_updates < 1 or blocks_per_update < 1 or first_block < 0:
            raise ValueError('Invalid streaming prefetch configuration')
        self.folder = Path(folder)
        self.next_block = int(first_block)
        self.blocks_per_update = int(blocks_per_update)
        self.poll_seconds = poll_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        self._queue = queue.Queue(maxsize=queue_updates)
        self._stop = threading.Event()
        self._thread = None
        self._end = object()

    def __enter__(self):
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()
        return self

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                return
            except queue.Full:
                pass

    def _producer(self):
        try:
            with LivePackedReader(self.folder, poll_seconds=self.poll_seconds,
                                  idle_timeout_seconds=self.idle_timeout_seconds) as reader:
                cursor = self.next_block
                while not self._stop.is_set():
                    windows, first, finished = reader.get_windows(
                        cursor, self.blocks_per_update, stop_event=self._stop)
                    if windows is None:
                        if finished:
                            break
                        continue
                    self._put((windows, first, finished))
                    cursor += len(windows)
                    if finished and len(windows) < self.blocks_per_update:
                        break
                self._put(self._end)
        except InterruptedError:
            pass
        except BaseException as error:
            self._put(error)

    def __iter__(self):
        if self._thread is None:
            raise RuntimeError('Use LiveUpdatePrefetcher as context manager')
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._end:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
