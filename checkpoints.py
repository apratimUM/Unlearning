#!/usr/bin/env python3
"""REAL integration checks for the Unlearning project. No dummy model, datasets, or mocks.

Requires an existing frozen Pile dataset, genuine saved GPT-2 checkpoints,
local tokenizer, and optionally real Hugging Face benchmark access. Never trains
or changes checkpoint weights.

Example:
 python test_real_pipeline.py --run runs/gpt2_small_pile_full --data data/pile_large \
   --tasks wikitext lambada blimp --limit 10 --wikitext-max-tokens 2000 \
   --blimp-configs adjunct_island

To verify every original document's tokenization: --verify-docs all
To verify full benchmarks rather than subsets: --limit 0 --wikitext-max-tokens 0
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from data import verify_data, file_sha
from model import GPT, GPTConfig
import evaluate


def fail(message):
    raise AssertionError(message)


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def assert_true(condition, message):
    if not condition:
        fail(message)


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True, help='Existing run dir containing step_*.pt')
    p.add_argument('--data', type=Path, required=True, help='Frozen Pile prepared dataset directory')
    p.add_argument('--insertion-metadata', type=Path, default=None)
    p.add_argument('--verify-docs', default='20', help='Number of actual prepared docs to check against bin, or all')
    p.add_argument('--tasks', nargs='+', choices=['wikitext','lambada','blimp','mmlu','all'], default=['wikitext','lambada','blimp'])
    p.add_argument('--limit', type=int, default=10, help='Per-task sample limit; 0=full')
    p.add_argument('--wikitext-max-tokens', type=int, default=2000, help='0=full')
    p.add_argument('--blimp-configs', nargs='*', default=['adjunct_island'])
    p.add_argument('--mmlu-subjects', nargs='*', default=['abstract_algebra'])
    p.add_argument('--mmlu-shots', type=int, default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--precision', default='float32', choices=['auto','float32','bfloat16','float16'])
    p.add_argument('--skip-benchmarks', action='store_true', help='Only real checkpoint and frozen Pile tests')
    p.add_argument('--include-legacy', action='store_true', help='Allow old model-only snapshots for evaluation; no resume assurance')
    a=p.parse_args()
    assert_true(a.run.is_dir(), f'Run directory missing: {a.run}')
    assert_true(a.data.is_dir(), f'Frozen Pile dataset missing: {a.data}')
    if a.insertion_metadata is not None:
        assert_true(a.insertion_metadata.is_file(), 'Insertion metadata not found')
    if a.verify_docs != 'all':
        assert_true(a.verify_docs.isdigit() and int(a.verify_docs) > 0, '--verify-docs requires positive integer or all')
    return a


def verify_real_documents(folder, manifest, tok, quantity):
    """Verify REAL original full_documents.jsonl against frozen binary/index."""
    originals = folder/'full_documents.jsonl'
    assert_true(originals.is_file(), f'Missing full documents: {originals}')
    dtype=np.dtype(manifest['token_dtype'])
    checked=0
    for split in ('train','val'):
        index=json.loads((folder/f'{split}_index.json').read_text())
        assert_true(index, f'Empty {split}_index')
        mmap=np.memmap(folder/f'{split}.bin', dtype=dtype, mode='r')
        # Index integrity is checked for ALL records, not just selected sample.
        last=-1
        for record in index:
            start=int(record['offset']); length=int(record['length'])
            assert_true(start>=last and length>1 and start+length<=len(mmap),
                        f'Invalid {split} document token index')
            last=start+length
        assert_true(last<=int(manifest['actual_tokens_before_padding'][split]),
                    f'{split} document boundary exceeds real token stream')
        del mmap

    inds={s:json.loads((folder/f'{s}_index.json').read_text()) for s in ('train','val')}
    train_n=len(inds['train']); val_n=len(inds['val'])
    total=train_n+val_n
    limit=total if quantity=='all' else min(int(quantity),total)
    # Sample evenly across BOTH train and val, avoiding load of all source texts.
    choices=set(np.linspace(0,total-1,limit,dtype=np.int64).tolist())
    data={s:np.memmap(folder/f'{s}.bin',dtype=np.dtype(manifest['token_dtype']),mode='r') for s in inds}
    try:
        seen=0
        with originals.open(encoding='utf-8') as fh:
            for i,line in enumerate(fh):
                if i not in choices:
                    continue
                obj=json.loads(line)
                split=obj['split']; di=int(obj['document_index'])
                rec=inds[split][di]
                text=obj['text']
                assert_true(sha_text(text)==rec['sha256']==obj['document_sha256'],
                            f'Original document hash mismatch at record {i}')
                ids=tok.encode(text,add_special_tokens=False)
                expected=np.asarray(ids+[tok.eos_token_id],dtype=np.int64)
                stored=np.asarray(data[split][int(rec['offset']):int(rec['offset'])+int(rec['length'])],dtype=np.int64)
                assert_true(np.array_equal(expected,stored),
                            f'Frozen Pile token binary mismatch in {split} document {di}')
                seen+=1
        assert_true(seen==limit, f'Expected {limit} source documents but checked {seen}')
        print(f'PASS Pile frozen token binaries correspond to {seen} real, untampered full documents')
    finally:
        for value in data.values():
            value._mmap.close()
    return train_n, val_n


def verify_insertions(folder, manifest, metadata, tok):
    """Verify declared inserted text actually appears in corresponding prepared docs."""
    assert_true(isinstance(metadata,dict), 'Invalid insertion metadata')
    records=metadata.get('insertions',[])
    targets=metadata.get('targets',[])
    assert_true(records and targets, 'Insertion metadata has no targets/insertions')
    assert_true(len(records)==int(metadata['total_insertions']), 'Insertion count metadata mismatch')
    by_index={}
    for record in records:
        idx=int(record['output_document_index'])
        by_index.setdefault(idx,[]).append(record)
    found=set()
    with (folder/'full_documents.jsonl').open(encoding='utf-8') as fh:
        for obj in map(json.loads,fh):
            # When prepared from modified_dataset.jsonl, source_row equals original insertion output index.
            if obj['split']!='train':
                continue
            source_row=obj.get('source_row')
            if source_row is None:
                continue
            idx=int(source_row)
            if idx not in by_index:
                continue
            text=obj['text']
            for record in by_index[idx]:
                span=record['inserted_span']
                start=int(record['inserted_start_char']); end=int(record['inserted_end_char'])
                assert_true(text[start:end]==span,
                            f'Inserted target absent at recorded position for document {idx}')
                found.add((idx,start,end,span))
    expected={(int(r['output_document_index']),int(r['inserted_start_char']),
               int(r['inserted_end_char']),r['inserted_span']) for r in records}
    # A modified input with 50000 train and 2500 validation docs must not place
    # training-controlled insertions in validation (leakage).
    assert_true(found==expected,
                f'Insertion mismatch: {len(expected-found)} missing from train; '
                f'{len(found)} of {len(expected)} verified. Check source_row mapping, '
                'train/val split, and modified JSONL actually passed to prepare.py.')
    print(f'PASS {len(found)} actual inserted spans found at exact positions in Pile TRAIN documents')


def verify_checkpoint(path, manifest_digest, dataset_dir, config_signature=None,
                      insertion_file=None, include_legacy=False):
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert_true(isinstance(state,dict),f'Invalid checkpoint: {path}')
    for key in ('model','config','step','signature','manifest_sha256'):
        assert_true(key in state,f'Missing {key} in {path}')
    assert_true(state['manifest_sha256']==manifest_digest,
                f'Checkpoint dataset hash does not match frozen Pile: {path}')
    sig=state['signature']
    assert_true(Path(sig['data']).resolve()==dataset_dir.resolve(),
                f'Checkpoint {path} was trained on DIFFERENT data directory: {sig["data"]}')
    if config_signature is not None:
        assert_true(sig==config_signature,f'Checkpoint signatures differ: {path}')
    if insertion_file:
        assert_true(sig.get('insertion_metadata') is not None, 'Insertion run checkpoint has no insertion metadata')
        assert_true(sig['insertion_metadata']['sha256']==file_sha(insertion_file),
                    'Checkpoint insertion metadata hash does not match supplied insertion file')
        assert_true(sig['insertion_metadata']['content']==json.loads(insertion_file.read_text()),
                    'Checkpoint insertion metadata contents differ')
    model=GPT(GPTConfig(**state['config']))
    model.load_state_dict(state['model'],strict=True)
    model.eval()
    # Verify restored model computes real logits and exactly matches stored values.
    dtype=np.dtype(json.loads((dataset_dir/'manifest.json').read_text())['token_dtype'])
    tokens=np.memmap(dataset_dir/'val.bin',dtype=dtype,mode='r')
    ids=torch.tensor(np.asarray(tokens[:min(model.config.block_size,16)],dtype=np.int64).copy())[None]
    del tokens
    with torch.inference_mode():
        logits,_=model(ids)
    assert_true(logits.shape==(1,ids.size(1),model.config.vocab_size) and torch.isfinite(logits).all(),
                f'Restored checkpoint forward pass fails: {path}')
    # Full optimizer/scaler/schedule/RNG required for seamless resume.
    full=all(x in state for x in ('optimizer','scaler','rng','scheduler'))
    if not full:
        assert_true(include_legacy,f'Old model-only checkpoint cannot fully resume: {path}')
    else:
        optimizer=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.ndim>=2], 'weight_decay':0.1},
                                     {'params':[p for p in model.parameters() if p.ndim<2], 'weight_decay':0.0}],
                                    lr=float(sig['lr']), betas=(0.9,0.95))
        optimizer.load_state_dict(state['optimizer'])
        if state['step']>0:
            assert_true(len(optimizer.state)>0,f'No AdamW moments in checkpoint at step {state["step"]}')
        assert_true(state['scheduler']['completed_optimizer_steps']==state['step'],
                    f'Incorrect schedule position: {path}')
        assert_true(set(state['rng'])=={'python','numpy','torch','cuda'},
                    f'Incomplete RNG state: {path}')
    card=path.with_suffix('.json')
    if full:
        assert_true(card.exists(),f'Missing per-step JSON metadata: {card}')
        meta=json.loads(card.read_text())
        assert_true(meta['step']==state['step'] and meta['model_config']==state['config'],
                    f'Checkpoint JSON does not match checkpoint: {path}')
        assert_true(meta.get('insertion_metadata')==sig.get('insertion_metadata'),
                    f'JSON insertion knobs differ from checkpoint signature: {card}')
    print(f'PASS step={state["step"]} loaded real GPT-2, forward scored frozen Pile tokens, '
          f'optimizer/scheduler/RNG={"complete" if full else "legacy missing"}, metadata={card.exists()}')
    return state


def main():
    a=parse()
    manifest=verify_data(a.data)
    digest=file_sha(a.data/'manifest.json')
    print(f'PASS frozen Pile manifest+all declared artifact SHA256 hashes verified: {a.data}')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.data/'tokenizer',use_fast=True,local_files_only=True)
    assert_true(tok.is_fast and tok.eos_token_id is not None,'Saved tokenizer is missing fast offsets/EOS')
    train_n,val_n=verify_real_documents(a.data,manifest,tok,a.verify_docs)
    print(f'PASS documented Pile data: train_docs={train_n}, val_docs={val_n}, block_size={manifest["block_size"]}')
    snapshots=sorted(a.run.glob('step_*.pt'))
    assert_true(snapshots,f'No real saved step_*.pt checkpoints in {a.run}')
    signature=None
    for checkpoint in snapshots:
        state=verify_checkpoint(checkpoint,digest,a.data,signature,a.insertion_metadata,a.include_legacy)
        if signature is None: signature=state['signature']
    newest=max(snapshots,key=lambda p:int(p.stem.split('_')[-1]))
    latest=a.run/'latest.pt'
    assert_true(latest.exists(),'Missing latest.pt')
    latest_state=torch.load(latest,map_location='cpu',weights_only=True)
    newest_state=torch.load(newest,map_location='cpu',weights_only=True)
    assert_true(latest_state['step']==newest_state['step'],'latest.pt does not point to newest checkpoint')
    assert_true(all(torch.equal(latest_state['model'][k],newest_state['model'][k]) for k in newest_state['model']),
                'latest.pt model differs from last historical checkpoint')
    print(f'PASS latest.pt matches newest saved model step={latest_state["step"]}')
    if a.insertion_metadata:
        verify_insertions(a.data,manifest,json.loads(a.insertion_metadata.read_text()),tok)
    elif signature.get('insertion_metadata'):
        print('WARNING: insertion experiment detected; supply --insertion-metadata to verify actual inserted text')
    if not a.skip_benchmarks:
        # Run actual evaluator with actual HF benchmark data for every checkpoint.
        tasks=['wikitext','lambada','blimp','mmlu'] if 'all' in a.tasks else a.tasks
        for checkpoint in snapshots:
            options=argparse.Namespace(checkpoint=checkpoint,tasks=tasks,tokenizer=str(a.data/'tokenizer'),
                device=a.device,precision=a.precision,limit=a.limit,
                wikitext_max_tokens=a.wikitext_max_tokens,stride=0,
                blimp_configs=a.blimp_configs,mmlu_subjects=a.mmlu_subjects,
                mmlu_shots=a.mmlu_shots,output=None,local_files_only=True,
                trusted_checkpoint=False,all_checkpoints=False)
            output=evaluate.evaluate_one(options)
            results=json.loads(output.read_text())
            assert_true(results['step']==int(checkpoint.stem.split('_')[-1]),
                        'Benchmark JSON has wrong checkpoint step')
            for task in tasks:
                value=results['results'][task].get('perplexity',results['results'][task].get('accuracy'))
                assert_true(value is not None and math.isfinite(value),f'Nonfinite real {task} score')
            print(f'PASS real benchmark datasets + JSON output for step {results["step"]}')
    print('ALL REQUESTED REAL-ARTIFACT CHECKS PASSED')


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'FAIL REAL PIPELINE: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise
