
"""Verify GPU execution before downloading data or starting training."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    args = parser.parse_args()

    # Verify that required project files exist.
    project_dir = Path(__file__).resolve().parent

    for filename in ("model.py", "runtime.py"):
        if not (project_dir / filename).is_file():
            raise FileNotFoundError(
                f"Missing {filename} in {project_dir}. "
                "Copy the original project file here before running."
            )

    # Import the project's model and runtime functions.
    from model import GPT, GPTConfig
    from runtime import (
        require_cuda,
        choose_precision,
        autocast_context,
        make_scaler,
        provenance,
        checked_optimizer_step,
    )

    # Check CUDA availability.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check your GPU allocation, "
            "NVIDIA driver, and PyTorch installation."
        )

    device = require_cuda(args.gpu)
    dtype = choose_precision(args.precision, device)

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Device: {device}")
    print(f"Precision: {dtype}")

    # Create a small GPT model.
    with torch.device(device):
        model = GPT(
            GPTConfig(
                vocab_size=64,
                block_size=32,
                n_layer=2,
                n_head=2,
                n_embd=32,
            )
        )

    # Generate synthetic training tokens.
    ids = torch.randint(0, 64, (2, 33), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = make_scaler(dtype)

    # Test forward pass and loss calculation.
    with autocast_context(dtype):
        _, loss = model(ids[:, :-1], ids[:, 1:])

    # Test backward pass.
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0
    )

    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient detected")

    # Test optimizer update.
    checked_optimizer_step(scaler, optimizer)

    # Test autoregressive generation.
    model.eval()

    with torch.no_grad():
        generated = model.generate_greedy(ids[:, :4], 4)

    if any(p.device != device for p in model.parameters()):
        raise RuntimeError("Model parameters are on the wrong device")

    if generated.device != device:
        raise RuntimeError("Generated tokens are on the wrong device")

    torch.cuda.synchronize(device)

    # Print test results.
    print(
        json.dumps(
            {
                "status": "CUDA test passed",
                "loss": loss.item(),
                "generated_shape": list(generated.shape),
                "runtime": provenance(device, dtype),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
