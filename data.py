"""Frozen packed-block sampling and integrity verification."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_data(folder):
    folder = Path(folder)
    m = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if int(m['block_size']) < 1:
        raise ValueError('Invalid block size')
    dtype = np.dtype(m['token_dtype'])
    if dtype not in (np.dtype('<u2'), np.dtype('<u4')):
        raise ValueError('Unsupported token dtype')
    for name, sha in m['files'].items():
        p = folder / name
        if not p.is_file() or file_sha(p) != sha:
            raise ValueError(f'Missing or modified artifact: {name}')
    for split in ('train', 'val'):
        count = int(m[f'{split}_tokens'])
        if (folder / f'{split}.bin').stat().st_size != count * dtype.itemsize:
            raise ValueError(f'{split} binary file has wrong length')
        if count < int(m['block_size']) + 1:
            raise ValueError(f'{split} has no complete prediction block')
        if m.get('full_documents'):
            real = int(m['actual_tokens_before_padding'][split])
            padding = int(m['padding_tokens'][split])
            if not 2 <= real <= count or count - real != padding:
                raise ValueError(f'{split} padding metadata inconsistent')
            if (count - 1) % int(m['block_size']):
                raise ValueError(f'{split} requires N * block_size + 1 tokens')
    return m


class PackedSampler:
    """Deterministic shuffled block epochs; blocks overlap one boundary token.

    Each block's targets cover positions k*T+1 through (k+1)*T.
    All prediction target positions are unique within an epoch. EOS does not
    reset attention; documents may share a block.
    """
    def __init__(self, folder, split, block, seed):
        self.folder = Path(folder)
        self.split, self.block, self.seed = split, int(block), int(seed)
        if split not in ('train', 'val') or self.block < 1 or self.seed < 0:
            raise ValueError('Invalid split, block or seed')
        m = json.loads((self.folder / 'manifest.json').read_text())
        if self.block != int(m['block_size']):
            raise ValueError('Block size differs from prepared corpus')
        self.tokens = np.memmap(self.folder / f'{split}.bin', dtype=np.dtype(m['token_dtype']), mode='r')
        self.n_blocks = (len(self.tokens) - 1) // self.block
        if self.n_blocks < 1:
            raise ValueError(f'No complete blocks: {split}')
        if m.get('full_documents') and (len(self.tokens) - 1) % self.block:
            raise ValueError('Bad full-document alignment; regenerate corpus')
        self.epoch, self.order = None, None

    def permutation(self, epoch):
        if epoch < 0:
            raise ValueError('Negative epoch')
        if self.epoch != epoch:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
            self.order = rng.permutation(self.n_blocks)
            self.epoch = epoch
        return self.order

    def block_ids(self, first, count):
        if first < 0 or count < 0:
            raise ValueError('Negative stream position or count')
        ids = []
        while count:
            epoch, offset = divmod(first, self.n_blocks)
            take = min(count, self.n_blocks - offset)
            ids.extend(self.permutation(epoch)[offset:offset + take].tolist())
            first += take
            count -= take
        return ids

    def batch(self, step, micro, batch, accum=1):
        if step < 0 or micro < 0 or accum < 1 or batch < 1 or micro >= accum:
            raise ValueError('Invalid training batch arguments')
        first = (step * accum + micro) * batch
        ids = self.block_ids(first, batch)
        seq = np.stack([np.asarray(self.tokens[k*self.block:k*self.block+self.block+1], dtype=np.int64) for k in ids])
        t = torch.from_numpy(seq)
        return t[:, :-1], t[:, 1:]

    def exposure_count(self, block_id, consumed_blocks):
        return self.exposure_counts([block_id], consumed_blocks)[int(block_id)]

    def exposure_counts(self, block_ids, consumed_blocks):
        ids = sorted(set(int(k) for k in block_ids))
        if consumed_blocks < 0 or any(k < 0 or k >= self.n_blocks for k in ids):
            raise ValueError('Invalid exposures')
        if not ids:
            return {}
        full, remaining = divmod(consumed_blocks, self.n_blocks)
        partial = set(self.permutation(full)[:remaining].tolist())
        return {k: full + (k in partial) for k in ids}

    def document_exposures(self, consumed_blocks):
        """Input-token appearances, NOT full-document exposures or gradient contributions."""
        records = json.loads((self.folder / f'{self.split}_index.json').read_text())
        full, rest = divmod(consumed_blocks, self.n_blocks)
        if consumed_blocks < 0:
            raise ValueError('Negative exposure count')
        partial = set(self.permutation(full)[:rest].tolist())
        result = []
        for i, rec in enumerate(records):
            start, end = int(rec['offset']), int(rec['offset']) + int(rec['length'])
            weighted = 0
            unique = 0
            for k in range(start // self.block, min((end - 1) // self.block + 1, self.n_blocks)):
                visits = full + (k in partial)
                overlap = max(0, min(end, (k+1)*self.block) - max(start, k*self.block))
                weighted += overlap * visits
                unique += int(visits > 0)
            result.append(dict(document_index=i, sha256=rec['sha256'], text_tokens=rec['text_tokens'],
                               tokens_presented=weighted,
                               approximate_fractional_document_exposures=weighted / max(1, rec['length']),
                               unique_blocks_seen=unique,
                               metric_note='Input token appearances, not complete document presentations'))
        return result

    def close(self):
        if getattr(self, 'tokens', None) is not None:
            self.tokens._mmap.close()
            self.tokens = None
