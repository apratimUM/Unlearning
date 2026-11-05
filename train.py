
"""Train GPT-2 from scratch on a frozen Pile dataset.

Run:
    python3 -u train.py

Resume:
    python3 -u train.py --resume
"""

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import PackedSampler, verify_data, file_sha
from model import GPT, GPTConfig, make_model
from runtime import (
    require_cuda,
    choose_precision,
    autocast_context,
    make_scaler,
    provenance,
    checked_optimizer_step,
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

CONFIG = {
    "data": "data/pile_large",
    "run": "gpt2_small_pile_full",
    "model": "small",
    "block_size": 512,
    "steps": 3000,
    "batch": 4,
    "accum": 8,
    "seed": 42,
    "gpu": 0,
    "precision": "auto",
    "lr": 0.0003,
    "min_lr": 0.00003,
    "warmup": 500,
    "eval_every": 500,
    "save_every": 1000,
    "eval_batches": 50,

    # Live terminal output frequency.
    "log_every": 1,

    # Periodic generation checks.
    "check_begin": 1000,
    "check_every": 500,

    # One string or a list of strings.
    "check_prompts": [
        "Once upon a time",
        "The capital of France is",
        "Artificial intelligence is",
    ],

    # Generation settings.
    "check_temperature": 0.8,
    "check_max_new_tokens": 50,
    "check_top_k": 50,
    "check_do_sample": True,
}

WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)
MAX_GRAD_NORM = 1.0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _numpy_rng_to_safe(state):
    name, keys, pos, has_gauss, cached_gauss = state
    return (name, torch.tensor(keys.astype(np.int64)), pos,
            has_gauss, cached_gauss)


def _numpy_rng_from_safe(state):
    name, keys, pos, has_gauss, cached_gauss = state
    if torch.is_tensor(keys):
        keys = keys.cpu().numpy().astype(np.uint32)
    return (name, keys, pos, has_gauss, cached_gauss)


def get_rng():
    return {
        "python": random.getstate(),
        # Encode numpy MT19937 state as safe primitives/tensor so PyTorch
        # weights_only=True can load the complete checkpoint for evaluation.
        "numpy": _numpy_rng_to_safe(np.random.get_state()),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def set_rng(state):
    random.setstate(state["python"])
    np.random.set_state(_numpy_rng_from_safe(state["numpy"]))
    torch.set_rng_state(state["torch"])

    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ============================================================
# LEARNING RATE
# ============================================================

def lr_schedule(step, args):
    """Linear warmup followed by cosine decay."""

    if step < args.warmup:
        return args.lr * (step + 1) / max(args.warmup, 1)

    progress = min(
        1.0,
        max(
            0.0,
            (step - args.warmup)
            / max(1, args.steps - args.warmup - 1),
        ),
    )

    return args.min_lr + 0.5 * (
        args.lr - args.min_lr
    ) * (1 + math.cos(math.pi * progress))


# ============================================================
# DATA MASKING AND LOSS
# ============================================================

def real_limit(manifest, split):
    return int(
        manifest.get("actual_tokens_before_padding", {}).get(
            split,
            manifest[f"{split}_tokens"],
        )
    )


def valid_positions(block_ids, block_size, limit, device):
    """Exclude artificial terminal padding from targets."""

    ids = torch.as_tensor(
        block_ids,
        dtype=torch.long,
        device=device,
    )

    positions = (
        ids[:, None] * block_size
        + torch.arange(
            1,
            block_size + 1,
            device=device,
        )[None, :]
    )

    return positions < limit


def target_loss(logits, targets, valid):
    """Token-weighted cross-entropy, excluding padding."""

    labels = targets.masked_fill(~valid, -100)

    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


# ============================================================
# GENERATION FUNCTION
# ============================================================

def check_generation(
    model,
    tokenizer,
    inputs,
    device,
    cast,
    **kwargs,
):
    """Generate text for one or multiple prompts.

    Supports kwargs:
        temperature
        max_new_tokens
        top_k
        do_sample
        seed

    Single input -> string
    List of inputs -> list of strings

    Prompts with identical token lengths are batched.
    Different-length prompts run in separate length groups.
    """

    temperature = kwargs.pop("temperature", 1.0)
    max_new_tokens = kwargs.pop("max_new_tokens", 50)
    top_k = kwargs.pop("top_k", None)
    do_sample = kwargs.pop("do_sample", True)
    seed = kwargs.pop("seed", 42)

    if kwargs:
        raise TypeError(
            f"Unsupported generation kwargs: {sorted(kwargs)}"
        )

    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")

    if (
        not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Temperature must be finite and positive")

    if top_k is not None:
        if not isinstance(top_k, int) or not (
            1 <= top_k <= model.config.vocab_size
        ):
            raise ValueError("Invalid top_k")

    if not isinstance(do_sample, bool):
        raise ValueError("do_sample must be boolean")

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Invalid seed")

    # Accept a single prompt or multiple prompts.
    single_input = isinstance(inputs, str)

    if single_input:
        prompts = [inputs]
    elif isinstance(inputs, (list, tuple)):
        prompts = list(inputs)
    else:
        raise TypeError(
            "inputs must be a string or list of strings"
        )

    if not prompts or any(
        not isinstance(prompt, str) for prompt in prompts
    ):
        raise ValueError("Provide one or more text prompts")

    # Tokenize each prompt.
    encoded = [
        tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
        for prompt in prompts
    ]

    if any(len(ids) == 0 for ids in encoded):
        raise ValueError("A prompt produced no tokens")

    # Prevent positional embedding overflow.
    if any(
        len(ids) + max_new_tokens > model.config.block_size
        for ids in encoded
    ):
        raise ValueError(
            "Prompt + generated tokens exceeds block_size"
        )

    # Group prompts for genuine batched inference.
    # No padding masks are required for equal-length groups.
    groups = {}

    for i, ids in enumerate(encoded):
        groups.setdefault(len(ids), []).append(i)

    outputs = [None] * len(prompts)

    was_training = model.training

    # Local RNG avoids modifying global training RNG state.
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    try:
        # Switch model to evaluation mode.
        model.eval()

        # Disable gradient computation.
        with torch.inference_mode():

            for prompt_length, indices in groups.items():

                tokens = torch.tensor(
                    [encoded[i] for i in indices],
                    dtype=torch.long,
                    device=device,
                )

                for _ in range(max_new_tokens):

                    with cast():
                        logits, _ = model(tokens)

                    next_logits = logits[:, -1, :].float()

                    if do_sample:

                        # Apply temperature.
                        next_logits = next_logits / temperature

                        # Optional top-k filtering.
                        if top_k is not None:

                            values, _ = torch.topk(
                                next_logits,
                                top_k,
                                dim=-1,
                            )

                            threshold = values[:, [-1]]

                            next_logits = next_logits.masked_fill(
                                next_logits < threshold,
                                float("-inf"),
                            )

                        probabilities = torch.softmax(
                            next_logits,
                            dim=-1,
                        )

                        next_token = torch.multinomial(
                            probabilities,
                            num_samples=1,
                            generator=generator,
                        )

                    else:

                        # Greedy decoding.
                        next_token = torch.argmax(
                            next_logits,
                            dim=-1,
                            keepdim=True,
                        )

                    tokens = torch.cat(
                        (tokens, next_token),
                        dim=1,
                    )

                # Return only the generated continuation.
                generated_ids = tokens[:, prompt_length:].tolist()

                for index, ids in zip(indices, generated_ids):
                    outputs[index] = tokenizer.decode(
                        ids,
                        skip_special_tokens=False,
                    )

    finally:
        # Always restore the model's previous mode.
        model.train(was_training)

    return outputs[0] if single_input else outputs


# ============================================================
# GENERATION LOGGING
# ============================================================

def log_generation(run, step, prompts, outputs, args):
    """Print and save generated text."""

    if isinstance(prompts, str):
        prompts = [prompts]

    if isinstance(outputs, str):
        outputs = [outputs]

    if len(prompts) != len(outputs):
        raise ValueError("Prompt/output count mismatch")

    print(
        f"\n{'=' * 60}\n"
        f"GENERATION CHECK AT STEP {step}\n"
        f"{'=' * 60}",
        flush=True,
    )

    output_file = run / "generation_checks.jsonl"

    with output_file.open("a", encoding="utf-8") as f:

        for i, (prompt, output) in enumerate(
            zip(prompts, outputs)
        ):
            print(
                f"\nPrompt {i + 1}: {prompt}\n"
                f"Output {i + 1}: {output}",
                flush=True,
            )

            record = {
                "step": step,
                "prompt_index": i,
                "prompt": prompt,
                "generated": output,
                "temperature": args.check_temperature,
                "max_new_tokens": args.check_max_new_tokens,
                "top_k": args.check_top_k,
                "do_sample": args.check_do_sample,
                "seed": args.seed + step,
            }

            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def trim_step_log(path, checkpoint_step):
    """Discard logs beyond the last committed checkpoint."""

    if not path.exists():
        return

    retained = []

    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            if record["step"] <= checkpoint_step:
                retained.append(line)
        except (ValueError, KeyError):
            pass

    path.write_text(
        "".join(line + "\n" for line in retained),
        encoding="utf-8",
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validation(model, sampler, args, device, cast, manifest):
    """Evaluate deterministic held-out validation blocks."""

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_valid = 0

    limit = real_limit(manifest, "val")

    try:
        for micro in range(args.eval_batches):

            ids = sampler.block_ids(
                micro * args.batch,
                args.batch,
            )

            x, y = sampler.batch(
                0,
                micro,
                args.batch,
                args.eval_batches,
            )

            x = x.to(device)
            y = y.to(device)

            valid = valid_positions(
                ids,
                args.block_size,
                limit,
                device,
            )

            count = int(valid.sum().item())

            if count == 0:
                continue

            with cast():
                logits, _ = model(x)
                loss_sum = target_loss(logits, y, valid)

            total_loss += loss_sum.item()
            total_valid += count

            del logits, loss_sum

    finally:
        model.train(was_training)

    if total_valid == 0:
        raise ValueError("No valid validation targets")

    return total_loss / total_valid


# ============================================================
# DOCUMENT EXPOSURE ACCOUNTING
# ============================================================

def save_exposures(sampler, run, step, consumed):
    """Save approximate input-token exposure per document."""

    output = run / f"document_exposures_step_{step:07d}.jsonl"

    with output.open("w", encoding="utf-8") as f:
        for row in sampler.document_exposures(consumed):
            record = {
                **row,
                "checkpoint_step": step,
            }

            f.write(json.dumps(record) + "\n")


# ============================================================
# CONFIGURATION PARSER
# ============================================================

def parse_args():
    """Use CONFIG defaults with optional CLI overrides."""

    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--data", default=CONFIG["data"])
    p.add_argument("--run", default=CONFIG["run"])
    p.add_argument(
        "--model",
        choices=["micro", "tiny", "small"],
        default=CONFIG["model"],
    )

    p.add_argument(
        "--block-size",
        type=int,
        default=CONFIG["block_size"],
    )
    p.add_argument(
        "--steps",
        type=int,
        default=CONFIG["steps"],
    )
    p.add_argument(
        "--batch",
        type=int,
        default=CONFIG["batch"],
    )
    p.add_argument(
        "--accum",
        type=int,
        default=CONFIG["accum"],
    )
    p.add_argument(
        "--seed",
        type=int,
        default=CONFIG["seed"],
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=CONFIG["gpu"],
    )
    p.add_argument(
        "--precision",
        choices=["auto", "bfloat16", "float16", "float32"],
        default=CONFIG["precision"],
    )

    p.add_argument(
        "--lr",
        type=float,
        default=CONFIG["lr"],
    )
    p.add_argument(
        "--min-lr",
        type=float,
        default=CONFIG["min_lr"],
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=CONFIG["warmup"],
    )

    p.add_argument(
        "--eval-every",
        type=int,
        default=CONFIG["eval_every"],
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=CONFIG["save_every"],
    )
    p.add_argument(
        "--eval-batches",
        type=int,
        default=CONFIG["eval_batches"],
    )

    # Live terminal output.
    p.add_argument(
        "--log-every",
        type=int,
        default=CONFIG["log_every"],
    )

    # Supervisor-requested generation schedule.
    p.add_argument(
        "--check_begin",
        "--check-begin",
        type=int,
        default=CONFIG["check_begin"],
    )
    p.add_argument(
        "--check_every",
        "--check-every",
        type=int,
        default=CONFIG["check_every"],
    )

    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Resume from a full-state step_*.pt/latest.pt; use a new --run to fork historical checkpoints")
    p.add_argument("--migrate-legacy-resume", action="store_true",
                   help="Explicitly allow old latest.pt with optimizer state to resume after this checkpoint-only code migration")
    p.add_argument("--insertion-metadata", type=Path, default=None,
                   help="Controlled insertion JSON metadata used to prepare this dataset; optional for baseline")

    args = p.parse_args()

    # Generation settings stay in CONFIG.
    for key in (
        "check_prompts",
        "check_temperature",
        "check_max_new_tokens",
        "check_top_k",
        "check_do_sample",
    ):
        setattr(args, key, CONFIG[key])

    if args.resume_from is not None and not args.resume:
        p.error("--resume-from requires --resume")
    if args.insertion_metadata is not None and not args.insertion_metadata.is_file():
        p.error("--insertion-metadata must point to a JSON file")
    if min(
        args.steps,
        args.batch,
        args.accum,
        args.block_size,
        args.eval_every,
        args.save_every,
        args.eval_batches,
        args.log_every,
    ) < 1:
        p.error("Training counts must be positive")

    if not 0 <= args.warmup < args.steps:
        p.error("Invalid warmup")

    if args.seed < 0 or args.gpu < 0:
        p.error("Seed and GPU index must be nonnegative")

    if not 0 <= args.min_lr <= args.lr or args.lr <= 0:
        p.error("Invalid learning rates")

    if args.check_begin < 1 or args.check_every < 0:
        p.error("Invalid generation check schedule")

    if args.check_every > 0:
        if not args.check_prompts:
            p.error("Configure check_prompts in CONFIG")

        if (
            not isinstance(args.check_max_new_tokens, int)
            or args.check_max_new_tokens < 0
        ):
            p.error("Invalid max_new_tokens")

        if (
            not isinstance(args.check_temperature, (int, float))
            or not math.isfinite(args.check_temperature)
            or args.check_temperature <= 0
        ):
            p.error("Invalid generation temperature")

        if args.check_top_k is not None:
            if not isinstance(args.check_top_k, int) or not (
                1 <= args.check_top_k <= 50257
            ):
                p.error("Invalid top_k")

        if not isinstance(args.check_do_sample, bool):
            p.error("check_do_sample must be boolean")

    return args


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def _write_json_atomic(path, record):
    """Write complete metadata without leaving a partially written file."""
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False)
                    + "\n", encoding="utf-8")
    os.replace(temp, path)


def save_checkpoint(model, optimizer, scaler, run, step, signature, sampler,
                    args, last_loss=None):
    """Save a full resumable historical checkpoint AND a matching JSON record.

    `latest.pt` points to a complete checkpoint (hard link where supported).
    Historical step_*.pt are never overwritten; every one is resumable.
    """
    if step < 0:
        raise ValueError("Negative checkpoint step")
    epoch_fraction = (step * args.batch * args.accum) / sampler.n_blocks
    lr = optimizer.param_groups[0]["lr"]
    schedule_state = {
        "type": "manual_linear_warmup_cosine_decay",
        "completed_optimizer_steps": step,
        "next_step_index": step,
        "warmup": args.warmup,
        "total_steps": args.steps,
        "base_lr": args.lr,
        "min_lr": args.min_lr,
        "last_applied_lr": lr,
        "scheduler_state_dict": None,
        "note": "No torch scheduler object exists; lr_schedule(step,args) is deterministic.",
    }
    state = {
        "schema_version": 2,
        "model": model.state_dict(),
        "config": vars(model.config),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": get_rng(),
        "step": step,
        "epoch_fraction": epoch_fraction,
        "last_train_loss": last_loss,
        "scheduler": schedule_state,
        "signature": signature,
        "manifest_sha256": signature["manifest_sha256"],
    }
    snapshot_path = run / f"step_{step:07d}.pt"
    tmp_snapshot = snapshot_path.with_suffix(".pt.tmp")
    if snapshot_path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {snapshot_path}")
    torch.save(state, tmp_snapshot)
    os.replace(tmp_snapshot, snapshot_path)

    # Link to the SAME immutable snapshot; fallback to a copy on filesystems
    # that do not support hardlinks. Atomic replacement keeps latest valid.
    tmp_latest = run / "latest.pt.tmp"
    if tmp_latest.exists():
        tmp_latest.unlink()
    try:
        os.link(snapshot_path, tmp_latest)
    except OSError:
        shutil.copyfile(snapshot_path, tmp_latest)
    os.replace(tmp_latest, run / "latest.pt")

    info = {
        "schema_version": 2,
        "checkpoint_file": snapshot_path.name,
        "step": step,
        "epoch_fraction": epoch_fraction,
        "train_loss": last_loss,
        "model_config": vars(model.config),
        "optimizer": {
            "type": "AdamW", "betas": list(ADAM_BETAS),
            "weight_decay": WEIGHT_DECAY,
            "param_groups": [{"lr": g["lr"],
                              "weight_decay": g["weight_decay"]}
                             for g in optimizer.param_groups],
            "state_saved_in_checkpoint": True,
        },
        "scheduler": schedule_state,
        "mixed_precision_scaler_saved": True,
        "rng_state_saved": True,
        "training": {k: v for k, v in vars(args).items()
                     if k not in ("resume", "resume_from", "insertion_metadata", "migrate_legacy_resume")},
        "dataset_manifest_sha256": signature["manifest_sha256"],
        "implementation_sha256": signature["implementation_sha256"],
        "insertion_metadata": signature.get("insertion_metadata"),
        "insertion_metadata_note": (
            "Metadata records declared insertion settings; insertion must be"
            " independently validated against the prepared dataset."),
        "token_exposure": {
            "blocks_consumed": step * args.batch * args.accum,
            "approximate_epochs": epoch_fraction,
            "prediction_token_slots": step * args.batch * args.accum * args.block_size,
        },
    }
    _write_json_atomic(snapshot_path.with_suffix(".json"), info)
    save_exposures(sampler, run, step, step * args.batch * args.accum)
    print(f"[SAVE] Full checkpoint + metadata: step {step}", flush=True)


# ============================================================
# MAIN TRAINING PROGRAM
# ============================================================

def main():

    args = parse_args()

    print("=" * 60, flush=True)
    print("GPT-2 PRETRAINING", flush=True)
    print("=" * 60, flush=True)

    print(f"Dataset:             {args.data}", flush=True)
    print(f"Run name:            {args.run}", flush=True)
    print(f"Model:               {args.model}", flush=True)
    print(f"Context length:      {args.block_size}", flush=True)
    print(f"Optimizer steps:     {args.steps}", flush=True)
    print(f"Microbatch size:     {args.batch}", flush=True)
    print(f"Gradient accum:      {args.accum}", flush=True)
    print(f"Precision:           {args.precision}", flush=True)
    print(f"Progress interval:   {args.log_every}", flush=True)
    print(
        f"Generation enabled:  {args.check_every > 0}",
        flush=True,
    )

    if args.check_every > 0:
        print(
            f"Generation begins:   {args.check_begin}",
            flush=True,
        )
        print(
            f"Generation interval: {args.check_every}",
            flush=True,
        )

    print("=" * 60, flush=True)

    # --------------------------------------------------------
    # Device and reproducibility
    # --------------------------------------------------------

    print("[SETUP] Checking CUDA GPU...", flush=True)

    device = require_cuda(args.gpu)
    amp_dtype = choose_precision(args.precision, device)
    cast = lambda: autocast_context(amp_dtype)

    runtime_info = provenance(device, amp_dtype)

    seed_all(args.seed)

    # --------------------------------------------------------
    # Verify dataset
    # --------------------------------------------------------

    folder = Path(args.data)

    print(
        f"[SETUP] Verifying dataset: {folder}",
        flush=True,
    )

    manifest = verify_data(folder)

    if args.block_size != int(manifest["block_size"]):
        raise ValueError(
            "Training block size differs from prepare.py"
        )

    print("[SETUP] Dataset verified", flush=True)

    # --------------------------------------------------------
    # Load tokenizer for generation
    # --------------------------------------------------------

    tokenizer = None

    if args.check_every > 0:

        print(
            "[SETUP] Loading generation tokenizer...",
            flush=True,
        )

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            folder / "tokenizer",
            local_files_only=True,
        )

        prompts = (
            [args.check_prompts]
            if isinstance(args.check_prompts, str)
            else list(args.check_prompts)
        )

        if not prompts or any(
            not isinstance(prompt, str) for prompt in prompts
        ):
            raise ValueError(
                "check_prompts must be a string or list of strings"
            )

        for prompt in prompts:
            ids = tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )

            if (
                not ids
                or len(ids) + args.check_max_new_tokens
                > args.block_size
            ):
                raise ValueError(
                    f"Invalid or overlong prompt: {prompt!r}"
                )

    # --------------------------------------------------------
    # Samplers
    # --------------------------------------------------------

    print(
        "[SETUP] Creating training and validation samplers...",
        flush=True,
    )

    train = PackedSampler(
        folder,
        "train",
        args.block_size,
        args.seed,
    )

    val = PackedSampler(
        folder,
        "val",
        args.block_size,
        args.seed + 100,
    )

    # --------------------------------------------------------
    # Run identity
    # --------------------------------------------------------

    run = Path("runs") / args.run
    checkpoint_file = run / "latest.pt"

    signature = {
        k: v
        for k, v in vars(args).items()
        if k not in ("resume", "resume_from", "insertion_metadata", "migrate_legacy_resume")
    }

    # Record the exact declared insertion settings alongside this run; do not
    # mistake this record for proof that the prepared token binaries include it.
    if args.insertion_metadata is not None:
        insertion = json.loads(args.insertion_metadata.read_text(encoding="utf-8"))
        if not isinstance(insertion, dict):
            raise ValueError("Insertion metadata must be a JSON object")
        signature["insertion_metadata"] = {
            "sha256": file_sha(args.insertion_metadata),
            "content": insertion,
        }
    else:
        signature["insertion_metadata"] = None

    signature["manifest_sha256"] = file_sha(
        folder / "manifest.json"
    )

    root = Path(__file__).resolve().parent

    signature["implementation_sha256"] = {
        name: file_sha(root / name)
        for name in (
            "model.py",
            "data.py",
            Path(__file__).name,
            "runtime.py",
        )
    }

    signature["runtime"] = runtime_info

    try:

        # ====================================================
        # MODEL INITIALIZATION / RESUMPTION
        # ====================================================

        if args.resume:

            print(
                "[SETUP] Resuming from checkpoint...",
                flush=True,
            )

            resume_path = args.resume_from or checkpoint_file
            if not resume_path.exists():
                raise FileNotFoundError(resume_path)
            fork = resume_path.resolve().parent != run.resolve()
            if fork:
                if run.exists():
                    raise FileExistsError(
                        f"Fork target run already exists: {run}; choose a new --run")
                run.mkdir(parents=True)
            else:
                # Never overwrite historical snapshots when replaying from an
                # earlier step. Fork into a NEW run using --resume-from instead.
                if resume_path.name != 'latest.pt':
                    latest_state = torch.load(checkpoint_file, map_location='cpu',
                                              weights_only=True)
                    if int(latest_state['step']) > int(torch.load(
                            resume_path, map_location='cpu',
                            weights_only=True)['step']):
                        raise ValueError('Older checkpoint selected within same run; '
                            'use --run NEW_NAME --resume-from OLD_STEP.pt to fork')

            # Load only your own trusted local checkpoints.
            ckpt = torch.load(
                resume_path,
                map_location="cpu",
                weights_only=False,
            )

            legacy = ckpt.get("schema_version", 1) < 2
            if legacy and not all(k in ckpt for k in ('optimizer', 'scaler', 'rng')):
                raise ValueError('This historical legacy checkpoint has no optimizer state; '
                                 'it can be evaluated but cannot resume faithfully')
            if legacy and not args.migrate_legacy_resume:
                raise ValueError('Legacy checkpoint requires --migrate-legacy-resume '
                                 'to acknowledge the checkpointing-code migration')
            expected_signature = dict(ckpt['signature'])
            if legacy:
                # Only checkpoint-writing code changed. Explicit migration
                # permits the train.py digest to update, not model/data/runtime.
                expected_signature['insertion_metadata'] = signature['insertion_metadata']
                expected_signature['implementation_sha256'] = dict(
                    expected_signature['implementation_sha256'])
                expected_signature['implementation_sha256'][Path(__file__).name] = (
                    signature['implementation_sha256'][Path(__file__).name])
            if fork:
                # Forking changes only the destination run name. All scientific
                # settings/data/code/precision still have to match exactly.
                expected_signature['run'] = args.run
            if expected_signature != signature:
                raise ValueError(
                    'Resume signature mismatch: code, dataset, environment '
                    'or experiment configuration changed'
                )
            if legacy:
                _write_json_atomic(run / 'legacy_checkpoint_migration.json', {
                    'source_checkpoint': str(resume_path.resolve()),
                    'source_checkpoint_sha256': file_sha(resume_path),
                    'old_train_code_sha256': ckpt['signature']['implementation_sha256'].get(Path(__file__).name),
                    'new_train_code_sha256': signature['implementation_sha256'][Path(__file__).name],
                    'note': 'Explicit migration only covers checkpoint-writing changes; '
                            'validate resumed training before scientific use.'
                })
                _write_json_atomic(run / 'config.json', signature)
            if fork:
                _write_json_atomic(run / 'config.json', signature)
                _write_json_atomic(run / 'environment.json', runtime_info)
                _write_json_atomic(run / 'fork_provenance.json', {
                    'source_checkpoint': str(resume_path.resolve()),
                    'source_step': int(ckpt['step']),
                    'source_checkpoint_sha256': file_sha(resume_path),
                })

            with torch.device(device):
                model = GPT(
                    GPTConfig(**ckpt["config"])
                )

            model.load_state_dict(ckpt["model"])
            start = int(ckpt["step"])

            if not 0 <= start <= args.steps:
                raise ValueError("Invalid saved optimizer step")

        else:

            if run.exists():
                raise FileExistsError(
                    f"Run already exists: {run}. "
                    "Use --resume or choose a new run name."
                )

            run.mkdir(parents=True)

            print(
                "[SETUP] Initializing model from scratch...",
                flush=True,
            )

            with torch.device(device):
                model = make_model(
                    args.model,
                    args.block_size,
                )

            start = 0

            _write_json_atomic(run / "config.json", signature)

            (run / "environment.json").write_text(
                json.dumps(runtime_info, indent=2),
                encoding="utf-8",
            )

        model.to(device)

        if model.config.vocab_size < int(manifest["vocab_size"]):
            raise ValueError(
                "Model vocabulary is smaller than tokenizer vocabulary"
            )

        # ====================================================
        # OPTIMIZER
        # ====================================================

        print(
            "[SETUP] Creating AdamW optimizer...",
            flush=True,
        )

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p for p in model.parameters()
                        if p.dim() >= 2
                    ],
                    "weight_decay": WEIGHT_DECAY,
                },
                {
                    "params": [
                        p for p in model.parameters()
                        if p.dim() < 2
                    ],
                    "weight_decay": 0.0,
                },
            ],
            lr=args.lr,
            betas=ADAM_BETAS,
        )

        scaler = make_scaler(amp_dtype)

        # ====================================================
        # RESTORE STATE
        # ====================================================

        if args.resume:

            optimizer.load_state_dict(ckpt["optimizer"])

            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)

            scaler.load_state_dict(ckpt["scaler"])

            set_rng(ckpt["rng"])

            trim_step_log(
                run / "metrics.jsonl",
                start,
            )

            trim_step_log(
                run / "generation_checks.jsonl",
                start,
            )

        else:

            # Write the initial checkpoint after optimizer/scaler creation,
            # ensuring that step zero is also fully resumable.
            save_checkpoint(model, optimizer, scaler, run, 0, signature,
                            train, args, last_loss=None)

        parameters = sum(
            p.numel() for p in model.parameters()
        )

        print(
            f"Model parameters: {parameters:,}",
            f"| Training blocks: {train.n_blocks:,}",
            f"| Starting step: {start}",
            flush=True,
        )

        limit = real_limit(manifest, "train")

        print(
            "[TRAIN] Starting optimizer updates...",
            flush=True,
        )

        # ====================================================
        # MAIN PRETRAINING LOOP
        # ====================================================

        for step in range(start, args.steps):

            model.train()

            lr = lr_schedule(step, args)

            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            # ------------------------------------------------
            # Select deterministic training blocks
            # ------------------------------------------------

            micro_ids = [
                train.block_ids(
                    (step * args.accum + micro) * args.batch,
                    args.batch,
                )
                for micro in range(args.accum)
            ]

            # Count valid next-token targets.
            counts = [
                sum(
                    max(
                        0,
                        min(
                            args.block_size,
                            limit - k * args.block_size - 1,
                        ),
                    )
                    for k in ids
                )
                for ids in micro_ids
            ]

            total_valid = sum(counts)

            if total_valid == 0:
                raise RuntimeError(
                    "Optimizer update contains only padding"
                )

            total_loss = 0.0

            # =================================================
            # GRADIENT ACCUMULATION
            # =================================================

            for micro, ids in enumerate(micro_ids):

                if counts[micro] == 0:
                    continue

                x, y = train.batch(
                    step,
                    micro,
                    args.batch,
                    args.accum,
                )

                x = x.to(device)
                y = y.to(device)

                valid = valid_positions(
                    ids,
                    args.block_size,
                    limit,
                    device,
                )

                with cast():
                    logits, _ = model(x)
                    loss_sum = target_loss(
                        logits,
                        y,
                        valid,
                    )

                if not torch.isfinite(loss_sum):
                    raise FloatingPointError(
                        f"Nonfinite loss at optimizer step {step}"
                    )

                scaled_loss = loss_sum / total_valid

                scaler.scale(scaled_loss).backward()

                total_loss += loss_sum.detach().item()

                del logits, loss_sum, scaled_loss

            # =================================================
            # GRADIENT CLIPPING
            # =================================================

            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=MAX_GRAD_NORM,
            )

            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Nonfinite gradient norm at step {step}"
                )

            # =================================================
            # OPTIMIZER UPDATE
            # =================================================

            checked_optimizer_step(
                scaler,
                optimizer,
            )

            done = step + 1

            # =================================================
            # LIVE TRAINING OUTPUT
            # =================================================

            if (
                done % args.log_every == 0
                or done == args.steps
            ):
                print(
                    f"[TRAIN] step {done}/{args.steps} "
                    f"| loss={total_loss / total_valid:.5f} "
                    f"| lr={lr:.3e} "
                    f"| grad_norm={grad_norm.item():.4f}",
                    flush=True,
                )

            # =================================================
            # PERIODIC GENERATION CHECK
            # =================================================

            if (
                args.check_every > 0
                and done >= args.check_begin
                and (done - args.check_begin) % args.check_every == 0
            ):

                print(
                    f"[GENERATE] Step {done}: generating text...",
                    flush=True,
                )

                outputs = check_generation(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=args.check_prompts,
                    device=device,
                    cast=cast,
                    temperature=args.check_temperature,
                    max_new_tokens=args.check_max_new_tokens,
                    top_k=args.check_top_k,
                    do_sample=args.check_do_sample,
                    seed=args.seed + done,
                )

                log_generation(
                    run=run,
                    step=done,
                    prompts=args.check_prompts,
                    outputs=outputs,
                    args=args,
                )

            # =================================================
            # VALIDATION
            # =================================================

            if (
                done % args.eval_every == 0
                or done == args.steps
            ):

                print(
                    f"[VALIDATE] Step {done}: evaluating...",
                    flush=True,
                )

                val_loss = validation(
                    model,
                    val,
                    args,
                    device,
                    cast,
                    manifest,
                )

                metrics = {
                    "step": done,
                    "train_loss": total_loss / total_valid,
                    "val_loss": val_loss,
                    "val_perplexity": math.exp(
                        min(val_loss, 50)
                    ),
                    "lr": lr,
                    "prediction_tokens": (
                        done
                        * args.batch
                        * args.accum
                        * args.block_size
                    ),
                    "valid_prediction_tokens_last_update": total_valid,
                    "applied_optimizer_updates": done,
                    "blocks_presented_in_committed_updates": (
                        done * args.batch * args.accum
                    ),
                }

                with (
                    run / "metrics.jsonl"
                ).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics) + "\n")

                print(
                    f"[VALIDATE] step={done} "
                    f"| train_loss={metrics['train_loss']:.5f} "
                    f"| val_loss={val_loss:.5f} "
                    f"| val_ppl={metrics['val_perplexity']:.3f}",
                    flush=True,
                )

            # =================================================
            # CHECKPOINT SAVING
            # =================================================

            if (
                done % args.save_every == 0
                or done == args.steps
            ):

                print(
                    f"[SAVE] Step {done}: writing checkpoint...",
                    flush=True,
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    run=run,
                    step=done,
                    signature=signature,
                    sampler=train,
                    args=args,
                    last_loss=total_loss / total_valid,
                )

        print(
            "\nTRAINING COMPLETE",
            flush=True,
        )

        print(
            f"Final checkpoint: {checkpoint_file}",
            flush=True,
        )

    finally:
        train.close()
        val.close()


if __name__ == "__main__":
    main()
