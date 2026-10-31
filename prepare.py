"""Prepare complete documents for reproducible packed GPT training.

Upstream insertion schedulers can supply an ordered JSONL stream with each row:
  {"text": "...", "_source": {"source_id": "fictionalqa", "document_id": "x"}}
This file preserves THAT order; it does not implement insertion scheduling.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
from data import file_sha as sha


def write_json(path, item):
    path.write_text(json.dumps(item, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def sources(args):
    if args.sources_config:
        rows = json.loads(Path(args.sources_config).read_text())['sources']
    elif args.hf_dataset:
        rows = [dict(source_id='pile', kind='hf', dataset=args.hf_dataset,
                     config=args.hf_config, split=args.hf_split, revision=args.hf_revision,
                     text_field=args.text_field)]
    else:
        rows = [dict(source_id='local', kind='jsonl', path=args.input_jsonl,
                     text_field=args.text_field)]
    ids = [s.get('source_id') for s in rows]
    if not rows or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Source IDs must be nonempty and unique')
    for spec in rows:
        if spec['kind'] == 'jsonl':
            p = Path(spec['path']).expanduser().resolve()
            spec['path'], spec['file_sha256'] = str(p), sha(p)
        elif spec['kind'] == 'hf':
            from huggingface_hub import HfApi
            revision = spec.get('revision') or 'main'
            spec['requested_revision'] = revision
            spec['revision'] = HfApi().dataset_info(spec['dataset'], revision=revision).sha
        else:
            raise ValueError('Source kind must be jsonl or hf')
    return rows


def source_rows(specs):
    """Keep original ordering and optional per-row provenance for future insertion."""
    for spec in specs:
        field = spec.get('text_field') or 'text'
        if spec['kind'] == 'hf':
            from datasets import load_dataset
            kw = dict(path=spec['dataset'], split=spec.get('split') or 'train',
                      streaming=True, revision=spec['revision'])
            if spec.get('config'):
                kw['name'] = spec['config']
            iterator = enumerate(load_dataset(**kw))
        else:
            def local_rows():
                with open(spec['path'], encoding='utf-8') as f:
                    for index, line in enumerate(f):
                        if line.strip():
                            yield index, json.loads(line)
            iterator = local_rows()
        for index, obj in iterator:
            src = dict(obj.get('_source') or {}) if isinstance(obj, dict) else {}
            src.setdefault('source_id', spec['source_id'])
            src.setdefault('source_row', index)
            src.setdefault('document_id', obj.get('id') if isinstance(obj, dict) else None)
            yield {'text': obj.get(field) if isinstance(obj, dict) else None, '_source': src}


def freeze_full_documents(rows, tokenizer, output_dir, args, metadata):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True, exist_ok=True)
    T, L, n = args.block_size, args.probe_length, args.probes
    if min(T, L, n, args.train_docs, args.val_docs, args.max_docs) <= 0 or L > T:
        raise ValueError('Invalid block size, probe length, or document counts')
    if args.train_docs + args.val_docs > args.max_docs:
        raise ValueError('Increase --max-docs')
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError('Tokenizer has no EOS token')
    vocab = len(tokenizer)
    if not 0 <= eos < vocab or vocab > 2**32:
        raise ValueError('Tokenizer vocabulary not supported')
    dtype = np.dtype('<u2' if vocab <= 65536 else '<u4')
    index = {'train': [], 'val': []}
    counts = defaultdict(lambda: {'seen': 0, 'included': 0, 'train_documents': 0,
                                  'val_documents': 0, 'train_tokens': 0, 'val_tokens': 0})
    seen = 0
    with (out / 'train.bin').open('wb') as train_file, (out / 'val.bin').open('wb') as val_file, \
         (out / 'full_documents.jsonl').open('w', encoding='utf-8') as originals, \
         (out / 'document_sources.jsonl').open('w', encoding='utf-8') as provenance:
        streams = {'train': train_file, 'val': val_file}
        for row in rows:
            if len(index['train']) == args.train_docs and len(index['val']) == args.val_docs:
                break
            if seen == args.max_docs:
                break
            seen += 1
            src, content = row.get('_source') or {}, row.get('text')
            source_id = str(src.get('source_id', 'input'))
            counts[source_id]['seen'] += 1
            if not isinstance(content, str) or not content.strip():
                continue
            ids = tokenizer.encode(content, add_special_tokens=False)
            if not ids:
                continue
            if min(ids) < 0 or max(ids) >= vocab:
                raise ValueError('Tokenizer returned invalid ID')
            split = 'train' if len(index['train']) < args.train_docs else 'val'
            # Byte position is authoritative; no full token list remains in RAM.
            offset = streams[split].tell() // dtype.itemsize
            np.asarray(ids + [eos], dtype=dtype).tofile(streams[split])
            digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
            i = len(index[split])
            rec = {'offset': offset, 'length': len(ids)+1,
                   'text_tokens': len(ids), 'sha256': digest}
            index[split].append(rec)
            meta = {'source_id': source_id, 'source_row': src.get('source_row'),
                    'source_document_id': src.get('document_id'), 'split': split,
                    'document_index': i, 'document_sha256': digest, 'token_offset': offset,
                    'stored_token_length': len(ids)+1, 'text_tokens': len(ids), 'truncated': False}
            originals.write(json.dumps({**meta, 'text': content}, ensure_ascii=False) + '\n')
            provenance.write(json.dumps(meta, ensure_ascii=False) + '\n')
            counts[source_id]['included'] += 1
            counts[source_id][f'{split}_documents'] += 1
            counts[source_id][f'{split}_tokens'] += len(ids)+1
        if len(index['train']) != args.train_docs or len(index['val']) != args.val_docs:
            raise ValueError('Insufficient valid documents; output directory is incomplete')
        real = {s: streams[s].tell() // dtype.itemsize for s in streams}
        padding = {s: (1 - real[s]) % T for s in streams}
        for s in streams:
            np.full(padding[s], eos, dtype=dtype).tofile(streams[s])
    for s in index:
        write_json(out / f'{s}_index.json', index[s])

    # Deterministic reservoir of eligible contiguous within-document/within-block probes.
    # Sampling uses O(n_probes) memory rather than O(corpus_tokens) memory.
    rng = random.Random(args.seed)
    probes = []
    for split in ('train', 'val'):
        tokens = np.memmap(out / f'{split}.bin', dtype=dtype, mode='r')
        reservoir, eligible = [], 0
        for i, doc in enumerate(index[split]):
            start, end = doc['offset'], doc['offset'] + doc['text_tokens']
            for pos in range(start, end-L+1):
                bid, within = divmod(pos, T)
                if within + L > T:
                    continue
                eligible += 1
                candidate = (i, pos, bid, within)
                if len(reservoir) < n:
                    reservoir.append(candidate)
                else:
                    j = rng.randrange(eligible)
                    if j < n:
                        reservoir[j] = candidate
        if eligible < n:
            raise ValueError(f'{split}: only {eligible} candidate probes, requested {n}')
        for i, pos, bid, within in reservoir:
            doc = index[split][i]
            probes.append({'split': split, 'document_index': i, 'document_sha256': doc['sha256'],
                           'document_token_offset': pos-doc['offset'], 'block_id': bid,
                           'training_position': within,
                           'tokens': tokens[pos:pos+L].astype(np.int64).tolist(),
                           'preceding_context_ids': tokens[bid*T:pos].astype(np.int64).tolist()})
        del tokens
    with (out / 'probes.jsonl').open('w', encoding='utf-8') as f:
        for p in probes:
            f.write(json.dumps(p, sort_keys=True) + '\n')
    write_json(out / 'dataset_sources.json', dict(sources=metadata['sources'],
               source_counts=dict(counts), included_documents=args.train_docs+args.val_docs,
               full_documents=True, truncated_documents=0,
               actual_tokens_before_padding=real, padding_tokens=padding,
               split_policy='First train_docs complete documents, then val_docs'))
    tokenizer.save_pretrained(out / 'tokenizer')
    manifest = {**metadata, 'schema_version': 2, 'full_documents': True,
                'block_size': T, 'probe_length': L, 'seed': args.seed,
                'vocab_size': vocab, 'token_dtype': str(dtype),
                'train_tokens': real['train'] + padding['train'],
                'val_tokens': real['val'] + padding['val'],
                'actual_tokens_before_padding': real, 'padding_tokens': padding,
                'documents_seen': seen,
                'files': {p.relative_to(out).as_posix(): sha(p)
                          for p in sorted(out.rglob('*')) if p.is_file()}}
    write_json(out / 'manifest.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    origin = p.add_mutually_exclusive_group(required=True)
    origin.add_argument('--hf-dataset')
    origin.add_argument('--input-jsonl')
    origin.add_argument('--sources-config')
    p.add_argument('--hf-config')
    p.add_argument('--hf-split', default='train')
    p.add_argument('--hf-revision', default='main')
    p.add_argument('--text-field', default='text')
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--out', required=True)
    p.add_argument('--full-documents', action='store_true', required=True)
    p.add_argument('--train-docs', type=int, default=20)
    p.add_argument('--val-docs', type=int, default=5)
    p.add_argument('--max-docs', type=int, default=10000)
    p.add_argument('--block-size', type=int, default=128)
    p.add_argument('--probe-length', type=int, default=100)
    p.add_argument('--probes', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    from transformers import AutoTokenizer
    spec = sources(a)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    metadata = {'sources': spec, 'dataset': ', '.join(s['source_id'] for s in spec),
                'tokenizer_name': a.tokenizer}
    m = freeze_full_documents(source_rows(spec), tok, a.out, a, metadata)
    print(json.dumps({'output': a.out, 'train_tokens': m['train_tokens'],
                      'val_tokens': m['val_tokens'], 'documents_seen': m['documents_seen']}, indent=2))


if __name__ == '__main__':
    main()
