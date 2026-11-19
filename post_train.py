#!/usr/bin/env python3
"""Single-GPU GPT-2 Alpaca SFT, assistant-answer-only loss, single rolling recovery checkpoint.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_gpt2_alpaca_sft.py --out ./runs/gpt2-alpaca
  CUDA_VISIBLE_DEVICES=0 python train_gpt2_alpaca_sft.py --out ./runs/gpt2-alpaca --resume
  CUDA_VISIBLE_DEVICES=0 python train_gpt2_alpaca_sft.py --out ./runs/gpt2-alpaca --chat

Requirements: torch, transformers, datasets, accelerate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Restrict visible GPU before importing torch/transformers.
if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "0":
    raise SystemExit("Set CUDA_VISIBLE_DEVICES=0 (only physical GPU 0 visible).")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

USER_TOKEN = "User:"
ASSISTANT_TOKEN = "Assistant:"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai-community/gpt2", help="Pretrained HF GPT-2 or local compatible checkpoint")
    p.add_argument("--out", type=Path, default=Path("./runs/gpt2-alpaca-sft"))
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--eval-ratio", type=float, default=0.05)
    p.add_argument("--save-steps", type=int, default=500, help="Recovery checkpoint interval; only the latest is kept")
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--workers", type=int, default=4, help="Dataset preprocessing processes")
    p.add_argument("--dataloader-workers", type=int, default=2)
    p.add_argument("--max-examples", type=int, default=0, help="Small smoke-test subset; 0 means entire Alpaca")
    p.add_argument("--max-steps", type=int, default=-1, help="Debug training limit; -1 means full epochs")
    p.add_argument("--resume", action="store_true", help="Resume latest saved training state under --out")
    p.add_argument("--gradient-checkpointing", action="store_true", help="Save VRAM at cost of speed")
    p.add_argument("--compile", action="store_true", help="Optional torch.compile; benchmark first")
    p.add_argument("--no-length-bucketing", action="store_true", help="Disable length-grouped batches for A/B speed comparison")
    p.add_argument("--prefetch-factor", type=int, default=4, help="Batches prefetched per dataloader worker")
    p.add_argument("--export-validation-only", action="store_true", help="Export the same held-out Alpaca examples without training")
    p.add_argument("--chat", action="store_true", help="Run interactive generation from saved final model")
    return p.parse_args()


def get_tokenizer(source):
    tok = AutoTokenizer.from_pretrained(source, use_fast=True)
    tok.add_special_tokens({"additional_special_tokens": [USER_TOKEN, ASSISTANT_TOKEN]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    assert tok.convert_tokens_to_ids(USER_TOKEN) != tok.unk_token_id
    assert tok.convert_tokens_to_ids(ASSISTANT_TOKEN) != tok.unk_token_id
    return tok


def collate_factory(tok):
    def collate(examples):
        longest = max(len(row["input_ids"]) for row in examples)
        # Right padding: matches GPT-2 training positional behavior.
        ids, masks, labels = [], [], []
        for row in examples:
            n = longest - len(row["input_ids"])
            ids.append(row["input_ids"] + [tok.pad_token_id] * n)
            masks.append([1] * len(row["input_ids"]) + [0] * n)
            labels.append(row["labels"] + [-100] * n)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


def build_example(row, tok, max_length):
    instruction = (row["instruction"] or "").strip()
    extra = (row["input"] or "").strip()
    answer = (row["output"] or "").strip()
    user_message = instruction + ("\n\n" + extra if extra else "")
    # Keep exactly the visible format from the supervisor.
    # 'Assistant: ' (INCLUDING trailing space) is a prefix with ignored loss.
    prefix = f"{USER_TOKEN} {user_message}\n{ASSISTANT_TOKEN} "
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    answer_ids = tok.encode(answer, add_special_tokens=False)
    ids = prefix_ids + answer_ids + [tok.eos_token_id]
    labels = [-100] * len(prefix_ids) + answer_ids + [tok.eos_token_id]
    valid = bool(instruction and answer_ids) and len(ids) <= max_length
    # Independent encoding at the prefix/answer boundary is deliberate:
    # it guarantees the space is an ignored token, even if GPT-2 BPE would
    # otherwise merge a leading space into the first answer token.
    return {"input_ids": ids if valid else [],
            "labels": labels if valid else [],
            "length": len(ids) if valid else 0,
            "valid": valid}


def chat(out):
    source = out / "final"
    if not source.exists():
        raise SystemExit(f"Missing trained model: {source}")
    tok = AutoTokenizer.from_pretrained(source, use_fast=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(source).to("cuda:0")
    model.config.use_cache = True
    model.eval()
    print("Enter 'quit' to exit.")
    while True:
        question = input("User: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        prompt = f"{USER_TOKEN} {question}\n{ASSISTANT_TOKEN} "
        inp = tok(prompt, return_tensors="pt").to("cuda:0")
        budget = min(128, model.config.n_positions - inp["input_ids"].shape[1])
        if budget <= 0:
            print("Prompt exceeds GPT-2 context window")
            continue
        with torch.inference_mode():
            generated = model.generate(
                **inp, max_new_tokens=budget,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        response = generated[0, inp["input_ids"].shape[1]:]
        print("Assistant:", tok.decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False))


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Exactly one visible CUDA GPU is required (CUDA_VISIBLE_DEVICES=0).")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    if args.chat:
        return chat(args.out)
    if not 0 < args.eval_ratio < 1 or args.max_length < 2:
        raise ValueError("Invalid validation ratio or maximum length")
    if args.max_length > 1024:
        raise ValueError("Pretrained GPT-2 Small supports at most 1024 input positions")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")
    if args.save_steps <= 0:
        raise ValueError("--save-steps must be positive")

    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = get_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    raw = load_dataset("tatsu-lab/alpaca", split="train")
    # Keep the original Hugging Face row number before selecting/splitting.
    raw = raw.add_column("alpaca_row_id", list(range(len(raw))))
    if args.max_examples:
        raw = raw.shuffle(seed=42).select(range(min(args.max_examples, len(raw))))
    split = raw.train_test_split(test_size=args.eval_ratio, seed=42)
    num_proc = max(1, args.workers)

    def prepare(ds, name):
        mapped = ds.map(
            build_example, fn_kwargs={"tok": tokenizer, "max_length": args.max_length},
            remove_columns=[c for c in ds.column_names if c != "alpaca_row_id"],
            num_proc=num_proc,
            desc=f"Tokenizing {name}",
        )
        total = len(mapped)
        mapped = mapped.filter(lambda x: x["valid"], num_proc=num_proc)
        print(f"[{name}] retained {len(mapped):,}/{total:,} examples (dropped {total-len(mapped):,})", flush=True)
        if len(mapped) == 0:
            raise RuntimeError(f"No usable {name} examples")
        return mapped.remove_columns("valid")

    train_set = prepare(split["train"], "train")
    val_set = prepare(split["test"], "eval")

    # Export the *actual* held-out examples after max-length/validity filtering.
    # This does not change the 95/5 split, its seed, or the training data order.
    kept_ids = set(val_set["alpaca_row_id"])
    validation_rows = [r for r in split["test"] if r["alpaca_row_id"] in kept_ids]
    assert len(validation_rows) == len(val_set)

    validation_jsonl = args.out / "alpaca_validation.jsonl"
    validation_txt = args.out / "alpaca_validation_questions.txt"
    tmp_jsonl = validation_jsonl.with_suffix(".jsonl.tmp")
    tmp_txt = validation_txt.with_suffix(".txt.tmp")
    with tmp_jsonl.open("w", encoding="utf-8") as f_jsonl, \
         tmp_txt.open("w", encoding="utf-8") as f_txt:
        for i, r in enumerate(validation_rows, start=1):
            instruction = (r["instruction"] or "").strip()
            extra = (r["input"] or "").strip()
            answer = (r["output"] or "").strip()
            user_message = instruction + ("\n\n" + extra if extra else "")
            prompt = f"{USER_TOKEN} {user_message}\n{ASSISTANT_TOKEN} "
            record = {
                "validation_index": i,
                "alpaca_row_id": int(r["alpaca_row_id"]),
                "instruction": instruction,
                "input": extra,
                "expected_output": answer,
                "prompt": prompt,
            }
            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_txt.write(f"=== VALIDATION EXAMPLE {i} | original Alpaca row {r['alpaca_row_id']} ===\n")
            f_txt.write(prompt + "\n")
            f_txt.write("REFERENCE ANSWER (hidden from model at inference):\n")
            f_txt.write(answer + "\n\n")
    tmp_jsonl.replace(validation_jsonl)
    tmp_txt.replace(validation_txt)
    print(f"[VALIDATION EXPORT] {len(validation_rows):,} examples -> {validation_jsonl}", flush=True)
    print(f"[VALIDATION EXPORT] readable questions/answers -> {validation_txt}", flush=True)

    # The Trainer only receives the usual input_ids/labels/length columns.
    train_set = train_set.remove_columns("alpaca_row_id")
    val_set = val_set.remove_columns("alpaca_row_id")
    if args.export_validation_only:
        print("[DONE] Export completed; no training or checkpoints changed.", flush=True)
        return

    # Verify exact assistant-only loss masking on a real data example.
    row = train_set[0]
    masked = 0
    for lab in row["labels"]:
        if lab == -100:
            masked += 1
        else:
            break
    assert masked > 0 and all(x == -100 for x in row["labels"][:masked])
    assert row["labels"][masked:] == row["input_ids"][masked:]
    assert row["labels"][-1] == tokenizer.eos_token_id
    print("[MASK CHECK] prefix tokens ignored:", masked,
          "; supervised answer+EOS tokens:", len(row["labels"])-masked, flush=True)

    precision_bf16 = torch.cuda.is_bf16_supported()
    # BF16 + TF32 + fused AdamW are fast on RTX 6000 Ada; model is still updated fully.
    train_args = TrainingArguments(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=precision_bf16,
        fp16=not precision_bf16,
        tf32=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=args.gradient_checkpointing,
        torch_compile=args.compile,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        logging_steps=args.logging_steps,
        load_best_model_at_end=False,  # Keep only ONE rolling recovery checkpoint
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.dataloader_workers > 0,
        dataloader_prefetch_factor=args.prefetch_factor if args.dataloader_workers > 0 else None,
        group_by_length=not args.no_length_bucketing,
        length_column_name="length",
        remove_unused_columns=False,
        max_steps=args.max_steps,
        report_to="none",
        seed=42,
        data_seed=42,
        save_safetensors=True,
    )
    trainer = Trainer(
        model=model, args=train_args,
        train_dataset=train_set, eval_dataset=val_set,
        data_collator=collate_factory(tokenizer),
        processing_class=tokenizer,
    )
    last_checkpoint = get_last_checkpoint(str(args.out)) if args.resume else None
    if args.resume and last_checkpoint is None:
        raise SystemExit(f"No checkpoint found to resume under {args.out}")
    print("[TRAIN] resume:", last_checkpoint or "no", flush=True)
    print("[PERF] bf16:", precision_bf16, "length_bucketing:", not args.no_length_bucketing,
          "fused_adamw: True", "compile:", args.compile, flush=True)
    trainer.train(resume_from_checkpoint=last_checkpoint)
    print("[EVAL]", trainer.evaluate(), flush=True)
    final = args.out / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    info = {
        "base_model": args.model,
        "dataset": "tatsu-lab/alpaca",
        "train_examples": len(train_set),
        "validation_examples": len(val_set),
        "assistant_only_loss": True,
        "role_tokens": [USER_TOKEN, ASSISTANT_TOKEN],
        "eos_supervised": True,
        "max_length": args.max_length,
    }
    (final / "sft_meta.json").write_text(json.dumps(info, indent=2))
    print("[DONE] final model:", final, flush=True)


if __name__ == "__main__":
    main()
