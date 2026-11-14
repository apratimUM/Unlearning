
#!/usr/bin/env python3
"""Audit schema-v4 Pile data and Feistel-shuffled GPT-2 Small checkpoints.

This is a verifier, NOT the checkpoint writer (train.py writes checkpoints).
No training, no corpus re-packing, no synthetic data, no model updates.

Examples:
  python checkpoints.py --run runs/gpt2_small_pile_full_1024 --data data/pile_full_1024 --skip-benchmarks
  python checkpoints.py --run runs/gpt2_small_pile_full_1024 --data data/pile_full_1024 --verify-docs 100
  python checkpoints.py --run runs/gpt2_small_pile_full_1024 --data data/pile_full_1024 --tasks wikitext --limit 10 --wikitext-max-tokens 2000
  python checkpoints.py --run runs/gpt2_small_pile_full_1024 --data data/pile_full_1024 --verify-all-offsets

Document sampling checks SQLite offsets and EOS boundaries against saved binaries.
Original source-text re-tokenization is NOT possible without source documents;
this program does not claim to establish original-text equivalence.
"""
import argparse
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import torch

from data import PackedSampler, SAMPLER_ORDER, file_sha, verify_data
from model import GPT, GPTConfig
from runtime import require_cuda


def check(test, message):
    if not test:
        raise AssertionError(message)


def arguments():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument(
        '--verify-docs',
        type=int,
        default=20,
        help='Number of real SQLite documents to check across train/val',
    )
    p.add_argument(
        '--verify-all-offsets',
        action='store_true',
        help='Stream ALL SQLite records to check contiguous offsets (slow on full Pile)',
    )
    p.add_argument(
        '--forward-all',
        action='store_true',
        help='Load and forward-test every checkpoint, not only newest',
    )
    p.add_argument('--skip-benchmarks', action='store_true')
    p.add_argument(
        '--tasks',
        nargs='+',
        choices=['wikitext', 'lambada', 'blimp', 'mmlu', 'all'],
        default=['wikitext', 'lambada', 'blimp'],
    )
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--wikitext-max-tokens', type=int, default=2000)
    p.add_argument('--blimp-configs', nargs='*', default=['adjunct_island'])
    p.add_argument('--mmlu-subjects', nargs='*', default=['abstract_algebra'])
    p.add_argument('--mmlu-shots', type=int, choices=range(6), default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--lambada-max-new-tokens', type=int, default=16)
    p.add_argument(
        '--precision',
        choices=['auto', 'float32', 'bfloat16', 'float16'],
        default='float32',
    )

    a = p.parse_args()

    check(a.run.is_dir(), f'Missing run directory: {a.run}')
    check(a.data.is_dir(), f'Missing dataset directory: {a.data}')
    check(
        a.verify_docs >= 0
        and a.limit >= 0
        and a.wikitext_max_tokens >= 0
        and a.gpu >= 0
        and a.lambada_max_new_tokens > 0,
        'Limits must be nonnegative',
    )

    if 'all' in a.tasks:
        a.tasks = ['wikitext', 'lambada', 'blimp', 'mmlu']
    else:
        a.tasks = list(dict.fromkeys(a.tasks))

    return a


def sqlite_connection(folder):
    return sqlite3.connect(
        (folder / 'documents.sqlite').resolve().as_uri() + '?mode=ro',
        uri=True,
    )


def audit_documents(folder, manifest, amount, exhaustive):
    """Verify document metadata and actual token boundaries; no source text is stored."""
    db = sqlite_connection(folder)
    mmaps = {}

    try:
        check(
            db.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='documents'"
            ).fetchone(),
            'SQLite documents table missing',
        )

        expected_total = 0
        observed_total = 0

        for split in ('train', 'val'):
            n_docs, total_tokens, max_end, min_offset = db.execute(
                """
                SELECT COUNT(*),
                       COALESCE(SUM(stored_tokens), 0),
                       COALESCE(MAX(token_offset + stored_tokens), 0),
                       COALESCE(MIN(token_offset), 0)
                FROM documents WHERE split = ?
                """,
                (split,),
            ).fetchone()

            check(
                n_docs == int(manifest[f'{split}_docs']),
                f'{split}: document count mismatch',
            )
            check(
                total_tokens == int(manifest[f'{split}_tokens']),
                f'{split}: document token sum mismatch',
            )
            check(
                max_end == total_tokens and min_offset == 0,
                f'{split}: document offsets out of bounds',
            )

            expected_total += int(manifest[f'{split}_docs'])
            observed_total += n_docs

            mmaps[split] = np.memmap(
                folder / f'{split}.bin',
                dtype=np.dtype(manifest['token_dtype']),
                mode='r',
            )

            if exhaustive:
                cursor = 0
                n = 0

                for offset, text_tokens, stored in db.execute(
                    """
                    SELECT token_offset, text_tokens, stored_tokens
                    FROM documents
                    WHERE split = ?
                    ORDER BY token_offset
                    """,
                    (split,),
                ):
                    check(
                        offset == cursor
                        and stored == text_tokens + 1
                        and stored >= 1,
                        f'{split}: non-contiguous or invalid document at index {n}',
                    )
                    check(
                        int(mmaps[split][offset + stored - 1]) == 50256,
                        f'{split}: missing document EOS at record {n}',
                    )
                    cursor += stored
                    n += 1

                check(
                    cursor == total_tokens and n == n_docs,
                    f'{split}: full offset scan failed',
                )
                print(
                    f'[PASS] {split}: fully scanned '
                    f'{n:,} document offsets/EOS markers'
                )
            else:
                print(
                    f'[PASS] {split}: {n_docs:,} documents, '
                    f'{total_tokens:,} tokens (aggregate metadata)'
                )

        check(
            expected_total == int(manifest['documents_seen']) == observed_total,
            'Source row count inconsistent with manifest',
        )
        check(
            db.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
            == expected_total,
            'SQLite total row count mismatch',
        )

        lo, hi = db.execute(
            'SELECT MIN(source_row), MAX(source_row) FROM documents'
        ).fetchone()

        check(
            lo == 0 and hi == expected_total - 1,
            'Source row ID endpoints mismatch',
        )

        # Sample evenly across original source-row IDs.
        selected = (
            sorted(set(int(x) for x in np.linspace(
                0,
                expected_total - 1,
                min(amount, expected_total),
                dtype=np.int64,
            )))
            if amount else []
        )

        for row_id in selected:
            record = db.execute(
                """
                SELECT split, token_offset, text_tokens,
                       stored_tokens, text_sha256
                FROM documents
                WHERE source_row = ?
                """,
                (row_id,),
            ).fetchone()

            check(record is not None, f'Missing source row {row_id}')

            split, offset, text_tokens, stored, digest = record

            check(
                split in ('train', 'val')
                and offset >= 0
                and text_tokens >= 0,
                f'Invalid document metadata at source row {row_id}',
            )
            check(
                stored == text_tokens + 1
                and offset + stored <= len(mmaps[split]),
                f'Invalid stored token extent at source row {row_id}',
            )
            check(
                int(mmaps[split][offset + stored - 1]) == 50256,
                f'No EOS after source row {row_id}',
            )
            check(
                isinstance(digest, str) and len(digest) == 64,
                f'Invalid document hash metadata at source row {row_id}',
            )

        print(
            f'[PASS] Verified {len(selected)} sampled real '
            f'document boundaries and EOS tokens'
        )
        print(
            '[NOTE] Original-text hashes cannot be authenticated '
            'without fetching the original source text'
        )

    finally:
        db.close()
        for mmap in mmaps.values():
            mmap._mmap.close()


def audit_sampler(folder, manifest, seed, sample_count=512):
    """Validate the actual Feistel block order and reversible exposure mapping.

    Spot-checks visit positions without materializing a giant permutation.
    Validates train bijection locally and sequential held-out validation.
    """
    train = PackedSampler(folder, 'train', int(manifest['block_size']), seed)
    val = PackedSampler(folder, 'val', int(manifest['block_size']), seed + 100)
    try:
        check(train.sampler_order == SAMPLER_ORDER,
              'Training sampler algorithm does not match expected Feistel version')
        check(val.sampler_order == 'sequential_validation',
              'Validation sampler must be sequential')
        check(train.n_blocks == int(manifest['train_blocks']),
              'Training block count differs from manifest')
        check(val.n_blocks == int(manifest['val_blocks']),
              'Validation block count differs from manifest')

        for sampler in (train, val):
            n = sampler.n_blocks
            count = min(sample_count, n)
            # Include both start and end, plus distributed positions throughout.
            positions = sorted({*range(count), *(n - i - 1 for i in range(count)),
                                *(int(x) for x in np.linspace(0, n - 1, count))})
            mapped = [sampler.physical_block_id(i) for i in positions]
            check(len(mapped) == len(set(mapped)),
                  f'{sampler.split}: two sampled visits mapped to the same block')
            for visit, physical in zip(positions, mapped):
                check(0 <= physical < n, f'{sampler.split}: block out of bounds')
                check(sampler.logical_block_id(physical) == visit,
                      f'{sampler.split}: reverse sampler mapping failed at visit {visit}')
                check(sampler.block_ids(visit, 1) == [physical],
                      f'{sampler.split}: block_ids disagrees with physical_block_id')
            check(sampler.block_ids(n, 1) == [], f'{sampler.split}: sampler wraps')
            check(sampler.block_ids(n - 1, 10) == [sampler.physical_block_id(n - 1)],
                  f'{sampler.split}: final partial batch wraps')
            check(len(sampler.block_ids(0, n if n < 1024 else 1024)) == min(n, 1024),
                  f'{sampler.split}: first batch length mismatch')
            if sampler.split == 'val':
                check(mapped == positions, 'Validation block order changed')
            print(f'[PASS] {sampler.split}: {len(positions)} sampled visits '
                  'are unique, reversible and non-wrapping')

        # A physical training block is exposed precisely when its visit index
        # falls before the checkpoint's committed blocks_consumed cursor.
        check(train.exposure_counts(train.block_ids(0, 1), 0)
              == {train.physical_block_id(0): 0}, 'Initial exposure incorrect')
        for consumed in sorted({0, 1, train.n_blocks // 3,
                                train.n_blocks // 2, train.n_blocks}):
            visits = sorted({0, min(consumed, train.n_blocks - 1),
                             max(0, consumed - 1), train.n_blocks - 1})
            physical = [train.physical_block_id(i) for i in visits]
            counts = train.exposure_counts(physical, consumed)
            for visit, block in zip(visits, physical):
                check(counts[block] == int(visit < consumed),
                      'Shuffled exposure status differs from actual visitation order')
        print(f'[PASS] Training exposure lookup matches {SAMPLER_ORDER}; seed={seed}')
        print('[NOTE] Permutation spot-checks are not a full enumeration of the Pile')
    finally:
        train.close()
        val.close()


def load_saved(path):
    # Safe loading supports the RNG representation written by train.py.
    return torch.load(
        path,
        map_location='cpu',
        weights_only=True,
    )


def audit_checkpoint(
    path,
    manifest,
    digest,
    folder,
    device,
    reference_signature=None,
    test_forward=False,
):
    """Verify checkpoint contents, data identity and optimizer-step cursor."""
    s = load_saved(path)

    check(isinstance(s, dict), f'{path}: checkpoint must be a dict')

    needed = (
        'schema_version', 'model', 'config', 'optimizer',
        'scaler', 'rng', 'scheduler', 'step',
        'signature', 'manifest_sha256', 'blocks_consumed',
    )
    check(
        all(k in s for k in needed),
        f'{path}: missing full-state checkpoint keys',
    )
    check(
        int(s['schema_version']) == 3,
        f'{path}: expected checkpoint schema 3',
    )
    check(
        s['manifest_sha256'] == digest,
        f'{path}: manifest checksum mismatch',
    )

    sig = s['signature']
    check(sig.get('sampler_order') == SAMPLER_ORDER,
          f'{path}: checkpoint sampler order differs from shuffled data.py')
    check(int(sig.get('sampler_seed', -1)) == int(sig['seed']),
          f'{path}: sampler seed differs from training seed')

    check(
        sig['manifest_sha256'] == digest,
        f'{path}: signature manifest mismatch',
    )
    check(
        Path(sig['data']).resolve() == folder.resolve(),
        f'{path}: checkpoint was trained on a different data directory',
    )
    check(
        reference_signature is None or sig == reference_signature,
        f'{path}: historical checkpoint configuration differs',
    )

    cfg = s['config']

    check(
        cfg['block_size'] == manifest['block_size']
        and cfg['vocab_size'] == manifest['vocab_size']
        and (
            cfg['n_layer'],
            cfg['n_head'],
            cfg['n_embd'],
        ) == (12, 12, 768)
        and math.isclose(float(cfg['dropout']), 0.1, abs_tol=1e-12),
        f'{path}: model configuration is not GPT-2 Small / 1024 / dropout 0.1',
    )

    step = int(s['step'])

    check(
        step >= 0 and path.stem == f'step_{step:07d}',
        f'{path}: checkpoint filename/step mismatch',
    )

    effective = int(sig['batch']) * int(sig['accum'])

    check(
        effective > 0 and int(manifest['train_blocks']) > 0,
        f'{path}: invalid batch size or train block count',
    )

    capacity = math.ceil(
        int(manifest['train_blocks']) / effective
    )

    check(
        0 <= step <= int(sig['steps']) <= capacity,
        f'{path}: training steps exceed one-pass budget',
    )

    consumed = min(
        step * effective,
        int(manifest['train_blocks']),
    )

    check(
        int(s['blocks_consumed']) == consumed,
        f'{path}: block cursor mismatch',
    )

    check(
        math.isclose(
            float(s['epoch_fraction']),
            consumed / int(manifest['train_blocks']),
            abs_tol=1e-12,
        ),
        f'{path}: epoch fraction mismatch',
    )

    check(
        s['scheduler']['completed_optimizer_steps'] == step
        and s['scheduler']['next_step_index'] == step,
        f'{path}: scheduler cursor mismatch',
    )

    check(
        set(s['rng']) == {'python', 'numpy', 'torch', 'cuda'},
        f'{path}: RNG state incomplete',
    )

    check(
        isinstance(s['optimizer'], dict)
        and 'state' in s['optimizer']
        and 'param_groups' in s['optimizer'],
        f'{path}: optimizer state invalid',
    )

    if step > 0:
        check(
            bool(s['optimizer']['state']),
            f'{path}: missing AdamW moments after training',
        )

    card = path.with_suffix('.json')

    check(
        card.is_file(),
        f'{path}: missing JSON checkpoint metadata',
    )

    meta = json.loads(
        card.read_text(encoding='utf-8')
    )

    check(
        meta['step'] == step
        and meta['model_config'] == cfg,
        f'{path}: checkpoint JSON mismatch',
    )
    check(
        meta['dataset_manifest_sha256'] == digest,
        f'{path}: JSON dataset manifest mismatch',
    )

    exposure = meta['token_exposure']

    check(
        int(exposure['blocks_consumed']) == consumed
        and int(exposure['unique_blocks_visited']) == consumed
        and int(exposure['dataset_blocks'])
        == int(manifest['train_blocks']),
        f'{path}: JSON exposure mismatch',
    )

    # Exact sampler-order value from final train.py.
    check(
        exposure['sampler_order']
        == SAMPLER_ORDER,
        f'{path}: sampler order differs from final train.py',
    )

    check(int(exposure['prediction_tokens']) == consumed * int(manifest['block_size']),
          f'{path}: prediction token accounting mismatch')
    check(math.isclose(float(exposure['epochs']), consumed / int(manifest['train_blocks']),
                       rel_tol=1e-10, abs_tol=1e-12),
          f'{path}: reported epoch fraction differs')
    if test_forward:
        # Model computation must take place on CUDA.
        model = GPT(
            GPTConfig(**cfg)
        )

        model.load_state_dict(
            s['model'],
            strict=True,
        )

        model.to(device).eval()

        mmap = np.memmap(
            folder / 'val.bin',
            dtype=np.dtype(manifest['token_dtype']),
            mode='r',
        )

        try:
            ids = torch.tensor(
                np.asarray(
                    mmap[:16],
                    dtype=np.int64,
                ).copy(),
                dtype=torch.long,
                device=device,
            )[None]
        finally:
            mmap._mmap.close()

        with torch.inference_mode():
            logits, _ = model(ids)

        check(
            logits.shape == (
                1, 16, model.config.vocab_size
            )
            and torch.isfinite(logits).all(),
            f'{path}: model forward output invalid',
        )

        print(
            f'[PASS] Actual restored model forward '
            f'on {device}: {path.name}'
        )

        del model, logits, ids

    print(
        f'[PASS] step={step:,}: full-state checkpoint, JSON, '
        f'cursor, manifest, optimizer, RNG'
    )

    return sig, step


def run_benchmarks(args, checkpoint):
    """Evaluate a historical checkpoint using the final evaluate.py API."""
    import evaluate
    from argparse import Namespace

    options = Namespace(
        checkpoint=checkpoint,
        all_checkpoints=False,
        tasks=args.tasks,
        tokenizer=str(args.data / 'tokenizer'),
        gpu=args.gpu,
        device=args.device,
        precision=args.precision,
        limit=args.limit,
        wikitext_max_tokens=args.wikitext_max_tokens,
        stride=0,
        blimp_configs=args.blimp_configs,
        mmlu_subjects=args.mmlu_subjects,
        mmlu_shots=args.mmlu_shots,
        lambada_max_new_tokens=args.lambada_max_new_tokens,
        output=None,
        local_files_only=True,
        trusted_checkpoint=False,
    )

    path = evaluate.evaluate_one(options)

    output = json.loads(
        path.read_text(encoding='utf-8')
    )

    check(
        output['step']
        == int(checkpoint.stem.split('_')[-1]),
        'Benchmark checkpoint step mismatch',
    )

    for task in args.tasks:
        result = output['results'][task]

        metric = result.get(
            'perplexity',
            result.get('accuracy'),
        )

        check(
            metric is not None
            and math.isfinite(float(metric)),
            f'Nonfinite {task} benchmark metric',
        )

    print(
        f'[PASS] Real benchmark evaluation: {checkpoint.name}'
    )


def main():
    args = arguments()

    # Require CUDA for every checkpoint-audit run.
    # Loading, JSON and SQLite operations remain CPU/disk tasks.
    device = require_cuda(args.gpu)

    if args.device not in (
        'auto', 'cuda', f'cuda:{args.gpu}'
    ):
        raise ValueError(
            '--device must be auto, cuda, '
            'or cuda:<selected --gpu>'
        )

    manifest = verify_data(args.data)
    digest = file_sha(
        args.data / 'manifest.json'
    )

    print(
        '[PASS] Schema-v4 manifest and train/val file lengths verified'
    )
    print(
        '[NOTE] The binaries are not cryptographically verified '
        'by the schema-v4 manifest'
    )

    audit_documents(
        args.data,
        manifest,
        args.verify_docs,
        args.verify_all_offsets,
    )

    snapshots = sorted(
        args.run.glob('step_*.pt'),
        key=lambda p: int(p.stem.split('_')[-1])
    )

    check(
        snapshots,
        'No historical step_*.pt checkpoints',
    )

    signature = None
    previous_step = -1

    for i, path in enumerate(snapshots):
        current, step = audit_checkpoint(
            path,
            manifest,
            digest,
            args.data,
            device,
            signature,
            test_forward=(
                args.forward_all
                or i == len(snapshots) - 1
            ),
        )

        check(step > previous_step, 'Checkpoint steps not strictly increasing')
        previous_step = step
        if signature is None:
            signature = current

    check(
        signature is not None,
        'No checkpoint signature',
    )

    audit_sampler(
        args.data,
        manifest,
        int(signature['seed']),
    )

    latest = args.run / 'latest.pt'

    check(
        latest.is_file(),
        'latest.pt is missing',
    )

    # Verify latest.pt matches the newest immutable snapshot.
    check(
        file_sha(latest) == file_sha(snapshots[-1]),
        'latest.pt does not match newest historical checkpoint byte-for-byte',
    )

    print(
        '[PASS] latest.pt matches the newest historical snapshot exactly'
    )

    if not args.skip_benchmarks:
        for checkpoint in snapshots:
            run_benchmarks(args, checkpoint)

    print('ALL REQUESTED CHECKS PASSED')


if __name__ == '__main__':
    main()
