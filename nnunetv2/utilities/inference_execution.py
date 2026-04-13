"""Optional inference-time execution tweaks (env-gated).

NNUNET_MATMUL_PRECISION_HIGH=1 — torch.set_float32_matmul_precision('high') (CUDA/MPS/CPU).
NNUNET_COMPILE_INFERENCE=1 — torch.compile on load (alias for nnUNet_compile=1); experimental on MPS.
NNUNET_SINGLE_PATCH_ACCUM_DTYPE=float16|half|fp16 — float16 merge buffers for single-patch path (else float32).
"""
import os
from typing import Optional

import torch


def apply_inference_execution_env(device: torch.device) -> None:
    """Call once per process before heavy inference if using env flags."""
    if _truthy(os.environ.get("NNUNET_MATMUL_PRECISION_HIGH")):
        torch.set_float32_matmul_precision("high")


def env_wants_torch_compile() -> bool:
    """True if nnUNet_compile or NNUNET_COMPILE_INFERENCE requests torch.compile (experimental on MPS)."""
    if _truthy(os.environ.get("NNUNET_COMPILE_INFERENCE")):
        return True
    return "nnUNet_compile" in os.environ and os.environ["nnUNet_compile"].lower() in ("true", "1", "t")


def inference_accumulator_dtype(device: torch.device) -> torch.dtype:
    """Dtype for single-patch merge buffers when NNUNET_SINGLE_PATCH_ACCUM_DTYPE=half|float16."""
    if device.type == "cpu":
        return torch.float32
    raw = (os.environ.get("NNUNET_SINGLE_PATCH_ACCUM_DTYPE") or "").lower().strip()
    if raw in ("half", "float16", "fp16"):
        return torch.float16
    return torch.float32


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v.lower() in ("1", "true", "t", "yes")
