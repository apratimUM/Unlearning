"""Shared CUDA requirement, precision selection, provenance, and step accounting.

Production training/extraction have no CPU fallback. Data preparation remains a
host task because GPT-2 BPE tokenization and file/hash operations are CPU work.
"""
from contextlib import nullcontext
import importlib.metadata
import platform
import sys

import torch


def require_cuda(gpu=0):
    """Select one NVIDIA CUDA device or stop BEFORE starting a research run."""
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is required; no CPU/MPS fallback is permitted. Install a CUDA-enabled '
            'PyTorch build and check NVIDIA driver/GPU visibility. '
            'See https://pytorch.org/get-started/locally/ .')
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(f'--gpu must be in [0, {torch.cuda.device_count()-1}]')
    torch.cuda.set_device(gpu)
    device = torch.device('cuda', gpu)
    # Allocate/compute once so a broken driver/runtime fails during preflight.
    value = torch.ones(1, device=device) + 1
    torch.cuda.synchronize(device)
    if value.item() != 2:
        raise RuntimeError('CUDA preflight computation failed')
    return device


def choose_precision(requested, device):
    """Return an autocast dtype; parameters/AdamW moments stay float32.

    'auto' prefers bfloat16; older GPUs use float16 with GradScaler. Explicit
    unsupported bf16 is rejected rather than silently substituted.
    """
    if device.type != 'cuda':
        # Used only by explicitly requested CPU unit tests that mock preflight.
        return None
    if requested == 'float32':
        return None
    bf16 = torch.cuda.is_bf16_supported()
    if requested == 'bfloat16' and not bf16:
        raise ValueError('This CUDA device does not support bfloat16; use auto or float16')
    return torch.bfloat16 if requested == 'bfloat16' or requested == 'auto' and bf16 else torch.float16


def autocast_context(dtype):
    """Fresh context for each forward pass; backward runs outside autocast."""
    return nullcontext() if dtype is None else torch.autocast('cuda', dtype=dtype)


def make_scaler(dtype):
    # Support the tested torch 2.2 API and newer releases without a private API.
    enabled = dtype == torch.float16
    if hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def provenance(device, dtype):
    """Record software, CUDA runtime, GPU, and numerical settings for each run."""
    packages = {}
    for name in ('torch', 'numpy', 'transformers', 'datasets', 'huggingface_hub'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = dict(python=sys.version, platform=platform.platform(), packages=packages,
                  device=str(device), torch_cuda_build=torch.version.cuda,
                  precision=str(dtype or torch.float32),
                  cudnn_version=torch.backends.cudnn.version(),
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_tf32=torch.backends.cudnn.allow_tf32,
                  flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
                  memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
                  math_sdp=torch.backends.cuda.math_sdp_enabled())
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        result.update(gpu_name=props.name, gpu_memory_bytes=props.total_memory,
                      compute_capability=list(torch.cuda.get_device_capability(device)),
                      visible_gpu_count=torch.cuda.device_count())
    return result


def checked_optimizer_step(scaler, optimizer):
    """Require exactly one real optimizer.step(), or fail the attempted update.

    GradScaler.step() normally returns None BOTH when AdamW runs and when it is
    skipped, so its return value cannot identify overflow. A public optimizer
    post-step hook runs only when optimizer.step() really executes. No checkpoint
    step counter is advanced until this function returns successfully.
    """
    applied = []
    hook = optimizer.register_step_post_hook(lambda *unused: applied.append(True))
    try:
        scaler.step(optimizer)
        scaler.update()
    finally:
        hook.remove()
    if len(applied) != 1:
        raise FloatingPointError(
            'Optimizer update was skipped (possible AMP overflow). Run stopped; '
            'no successful-update checkpoint or exposure count is advanced. '
            'Resume the last committed checkpoint with identical settings, or '
            'start a new run with bfloat16/float32 or a smaller learning rate.')
