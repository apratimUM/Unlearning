
"""Train a randomly initialized GPT on immutable packed tokens.

Depends on the existing model.py and runtime.py interfaces.

No insertion is performed here: a future controlled-insertion scheduler
must prepare the ordered document corpus upstream.
"""

import argparse
import json
import math
import os
import random
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


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng():
    """Save random states for reproducible checkpoint resumption."""
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=(
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    )


def set_rng(state):
    """Restore RNG state after model and optimizer construction."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ------------------------------------------------------------
# Learning-rate schedule
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Padding-aware next-token loss
# ------------------------------------------------------------

def real_limit(manifest, split):
    """Return the first artificial-padding token position."""
    return int(
        manifest.get("actual_tokens_before_padding", {}).get(
            split,
            manifest[f"{split}_tokens"],
        )
    )


def valid_positions(block_ids, block_size, limit, device):
    """Identify genuine next-token targets, excluding terminal padding.

    Packed block k predicts original corpus positions:
        k*T + 1 through (k+1)*T.

    Actual EOS document separators are NOT padding and remain valid.
    """

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
    """Return summed cross-entropy over genuine prediction targets.

    Using reduction='sum' allows exact token-weighted accumulation
    across microbatches containing different amounts of padding.
    """

    labels = targets.masked_fill(~valid, -100)

    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

@torch.no_grad()
def validation(model, sampler, args, device, cast, manifest):
    """Evaluate fixed deterministic validation blocks.

    Validation never updates model weights.

    The same packed blocks are evaluated at each checkpoint.
    """

    model.eval()

    total_loss = 0.0
    total_valid = 0

    limit = real_limit(manifest, "val")

    for micro in range(args.eval_batches):

        first = micro * args.batch

        ids = sampler.block_ids(
            first,
            args.batch,
        )

        # Pass eval_batches as accumulation so micro indices are valid.
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

        valid_count = int(valid.sum().item())

        if valid_count == 0:
            continue

        with cast():
            # No targets are passed to model.forward().
            # Cross-entropy is computed once by target_loss().
            logits, _ = model(x)

            loss_sum = target_loss(
                logits,
                y,
                valid,
            )

        total_loss += loss_sum.item()
        total_valid += valid_count

        del logits, loss_sum

    model.train()

    if total_valid == 0:
        raise ValueError("No valid validation targets")

    return total_loss / total_valid


# ------------------------------------------------------------
# Document exposure accounting
# ------------------------------------------------------------

def save_exposures(sampler, run, step, consumed):
    """Record approximate input-token exposure for each document.

    These counts are not complete-document presentations.
    """

    output = (
        run
        / f"document_exposures_step_{step:07d}.jsonl"
    )

    with output.open("w", encoding="utf-8") as f:

        for row in sampler.document_exposures(consumed):

            record = {
                **row,
                "checkpoint_step": step,
            }

            f.write(
                json.dumps(record) + "\n"
            )


# ------------------------------------------------------------
# Main training program
# ------------------------------------------------------------

def main():

    p = argparse.ArgumentParser(description=__doc__)

    # Dataset and run identity
    p.add_argument("--data", default="data/pile")
    p.add_argument("--run", required=True)

    # Model
    p.add_argument(
        "--model",
        choices=["micro", "tiny", "small"],
        default="small",
    )

    p.add_argument("--block-size", type=int, default=512)

    # Training
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # Device
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--precision",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )

    # Optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=200)

    # Evaluation and checkpointing
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)

    # Resumption
    p.add_argument("--resume", action="store_true")

    args = p.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if min(
        args.steps,
        args.batch,
        args.accum,
        args.block_size,
        args.eval_every,
        args.save_every,
        args.eval_batches,
    ) < 1:
        p.error("Training counts must be positive")

    if not 0 <= args.warmup < args.steps:
        p.error("Invalid warmup")

    if args.seed < 0 or args.gpu < 0:
        p.error("Seed and GPU index must be nonnegative")

    if not 0 <= args.min_lr <= args.lr or args.lr <= 0:
        p.error("Invalid learning rates")

    # --------------------------------------------------------
    # Device, precision and random initialization
    # --------------------------------------------------------

    device = require_cuda(args.gpu)

    amp_dtype = choose_precision(
        args.precision,
        device,
    )

    cast = lambda: autocast_context(amp_dtype)

    runtime_info = provenance(
        device,
        amp_dtype,
    )

    seed_all(args.seed)

    # --------------------------------------------------------
    # Verify frozen dataset
    # --------------------------------------------------------

    folder = Path(args.data)

    manifest = verify_data(folder)

    if args.block_size != int(manifest["block_size"]):
        raise ValueError(
            "Training block size differs from preparation"
        )

    # Training and validation have separate deterministic streams.
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
    # Checkpoint identity
    # --------------------------------------------------------

    run = Path("runs") / args.run
    checkpoint_file = run / "latest.pt"

    signature = {
        k: v
        for k, v in vars(args).items()
        if k != "resume"
    }

    signature["manifest_sha256"] = file_sha(
        folder / "manifest.json"
    )

    root = Path(__file__).resolve().parent

    signature["implementation_sha256"] = {
        name: file_sha(root / name)
        for name in (
            "model.py",
            "data.py",
            "train.py",
            "runtime.py",
        )
    }

    signature["runtime"] = runtime_info

    try:

        # ----------------------------------------------------
        # Load model or resume existing run
        # ----------------------------------------------------

        if args.resume:

            # Only load trusted checkpoints created by this project.
            ckpt = torch.load(
                checkpoint_file,
                map_location="cpu",
                weights_only=False,
            )

            if ckpt["signature"] != signature:
                raise ValueError(
                    "Resume signature mismatch: "
                    "code, dataset, environment or configuration changed"
                )

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
                    f"Run already exists: {run}"
                )

            run.mkdir(parents=True)

            # Random initialization: no pretrained weights loaded.
            with torch.device(device):
                model = make_model(
                    args.model,
                    args.block_size,
                )

            start = 0

            (run / "config.json").write_text(
                json.dumps(signature, indent=2),
                encoding="utf-8",
            )

            (run / "environment.json").write_text(
                json.dumps(runtime_info, indent=2),
                encoding="utf-8",
            )

        model.to(device)

        if model.config.vocab_size < int(manifest["vocab_size"]):
            raise ValueError(
                "Model vocabulary is smaller than "
                "the saved tokenizer vocabulary"
            )

        # ----------------------------------------------------
        # AdamW optimizer
        # ----------------------------------------------------

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p for p in model.parameters()
                        if p.dim() >= 2
                    ],
                    "weight_decay": 0.1,
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
            betas=(0.9, 0.95),
        )

        scaler = make_scaler(amp_dtype)

        # ----------------------------------------------------
        # Restore complete training state
        # ----------------------------------------------------

        if args.resume:

            optimizer.load_state_dict(
                ckpt["optimizer"]
            )

            for state in optimizer.state.values():

                for key, value in state.items():

                    if torch.is_tensor(value):
                        state[key] = value.to(device)

            scaler.load_state_dict(
                ckpt["scaler"]
            )

            set_rng(ckpt["rng"])

            # Discard metrics written after the last committed checkpoint.
            metrics_file = run / "metrics.jsonl"

            if metrics_file.exists():

                retained = []

                for line in metrics_file.read_text(
                    encoding="utf-8"
                ).splitlines():

                    try:
                        record = json.loads(line)

                        if record["step"] <= start:
                            retained.append(line)

                    except (ValueError, KeyError):
                        pass

                metrics_file.write_text(
                    "".join(line + "\n" for line in retained),
                    encoding="utf-8",
                )

        else:

            # Step-zero model, before any optimizer updates.
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": vars(model.config),
                    "step": 0,
                    "signature": signature,
                    "manifest_sha256": signature["manifest_sha256"],
                },
                run / "step_0000000.pt",
            )

        parameters = sum(
            p.numel()
            for p in model.parameters()
        )

        print(
            "Model parameters:", parameters,
            "| Training blocks:", train.n_blocks,
            "| Starting step:", start,
            flush=True,
        )

        limit = real_limit(manifest, "train")

        # ====================================================
        # Main pretraining loop
        # ====================================================

        for step in range(start, args.steps):

            model.train()

            lr = lr_schedule(step, args)

            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            # ------------------------------------------------
            # Determine which blocks appear in this update
            # ------------------------------------------------

            micro_ids = [
                train.block_ids(
                    (step * args.accum + micro) * args.batch,
                    args.batch,
                )
                for micro in range(args.accum)
            ]

            # Count genuine targets efficiently.
            # Each block's target positions are k*T+1 ... (k+1)*T.
            # Artificial padding targets are excluded.

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

            # ------------------------------------------------
            # Gradient accumulation
            # ------------------------------------------------

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

                    # Compute model logits only.
                    # Our own target_loss handles the labels and padding.
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

                # Normalize by ALL valid prediction tokens in this update.
                # This gives consistent per-token weighting.
                scaled_loss = loss_sum / total_valid

                scaler.scale(scaled_loss).backward()

                total_loss += loss_sum.detach().item()

                del logits, loss_sum, scaled_loss

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Nonfinite gradient norm at step {step}"
                )

            # ------------------------------------------------
            # Optimizer update
            # ------------------------------------------------

            # Runtime helper must verify that the optimizer step
            # actually occurred, including with mixed precision.
            checked_optimizer_step(scaler, optimizer)

            done = step + 1

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if (
                done % args.eval_every == 0
                or done == args.steps
            ):

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

                    f.write(
                        json.dumps(metrics) + "\n"
                    )

                print(metrics, flush=True)

            # ------------------------------------------------
            # Save checkpoints
            # ------------------------------------------------

            if (
                done % args.save_every == 0
                or done == args.steps
            ):

                state = {
                    "model": model.state_dict(),
                    "config": vars(model.config),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng": get_rng(),
                    "step": done,
                    "signature": signature,
                    "manifest_sha256": signature["manifest_sha256"],
                }

                # Atomically save resumable checkpoint.
                temporary = run / "latest.tmp"

                torch.save(
                    state,
                    temporary,
                )

                os.replace(
                    temporary,
                    checkpoint_file,
                )

                # Save model-only numbered checkpoint for extraction.
                snapshot = {
                    k: state[k]
                    for k in (
                        "model",
                        "config",
                        "step",
                        "signature",
                        "manifest_sha256",
                    )
                }

                torch.save(
                    snapshot,
                    run / f"step_{done:07d}.pt",
                )

                # Document exposure reports for memorization analysis.
                save_exposures(
                    train,
                    run,
                    done,
                    done * args.batch * args.accum,
                )

                print(
                    "Saved checkpoint:", done,
                    flush=True,
                )

    finally:

        train.close()
        val.close()


if __name__ == "__main__":
    main()
