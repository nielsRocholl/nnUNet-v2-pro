"""ROI config: YAML/JSON → dataclass. No hardcoded tunables."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

from batchgenerators.utilities.file_and_folder_operations import load_json

DEFAULT_CONFIG_PATH = Path(__file__).parent / "nnunet_pro_config.json"


@dataclass(frozen=True)
class SizeBinsConfig:
    mode: Literal["percentile", "fixed"]
    trim_percentile: float = 0.025
    percentiles: Tuple[float, ...] = (0.25, 0.5, 0.75)
    thresholds: Tuple[int, ...] = (100, 2000, 20000)


@dataclass(frozen=True)
class StratifiedConfig:
    dataset_weights: Dict[str, float]
    size_bin_weights: Dict[str, float]


@dataclass(frozen=True)
class LargeLesionConfig:
    K: Tuple[int, int]
    K_min: int
    K_max: int
    max_extra: int


@dataclass(frozen=True)
class PropagatedConfig:
    sigma_per_axis: Tuple[float, float, float]
    max_vox: float


@dataclass(frozen=True)
class SamplingConfig:
    mode_probs: Tuple[float, float, float, float]
    n_spur: Tuple[int, int]
    n_neg: Tuple[int, int]
    large_lesion: LargeLesionConfig
    propagated: PropagatedConfig


@dataclass(frozen=True)
class PromptConfig:
    point_radius_vox: int
    encoding: Literal["binary", "edt"]
    validation_use_prompt: bool
    prompt_intensity_scale: float


RoiInferenceMode = Literal["dilated_sliding", "per_point_patch"]


@dataclass(frozen=True)
class InferenceConfig:
    tile_step_size: float
    disable_tta_default: bool
    roi_inference_mode: RoiInferenceMode = "dilated_sliding"
    max_patch_expansion_visits: int = 64
    max_patch_expansion_depth: Optional[int] = None


@dataclass(frozen=True)
class RoiPromptConfig:
    prompt: PromptConfig
    sampling: SamplingConfig
    inference: InferenceConfig
    size_bins: Optional[SizeBinsConfig] = None
    stratified: Optional[StratifiedConfig] = None


def _require(d: dict, key: str, msg: str = "") -> object:
    if key not in d:
        raise KeyError(f"Config missing required key '{key}'. {msg}".strip())
    return d[key]


def _load_prompt(prompt: dict) -> PromptConfig:
    point_radius_vox = _require(prompt, "point_radius_vox", "Expected int, voxels.")
    encoding = _require(prompt, "encoding", "Expected 'binary' or 'edt'.")
    if encoding not in ("binary", "edt"):
        raise ValueError(f"encoding must be 'binary' or 'edt', got {encoding!r}")
    validation_use_prompt = prompt.get("validation_use_prompt", False)
    if not isinstance(validation_use_prompt, bool):
        raise ValueError(f"validation_use_prompt must be bool, got {type(validation_use_prompt)}")
    intensity_scale = float(prompt.get("prompt_intensity_scale", 1.0))
    if intensity_scale <= 0 or intensity_scale > 1:
        raise ValueError(f"prompt_intensity_scale must be in (0, 1], got {intensity_scale}")
    return PromptConfig(
        point_radius_vox=int(point_radius_vox),
        encoding=encoding,
        validation_use_prompt=validation_use_prompt,
        prompt_intensity_scale=intensity_scale,
    )


def _parse_int_or_range(val, key: str) -> Tuple[int, int]:
    """Parse int or [min, max] to (min, max) inclusive."""
    if isinstance(val, int):
        return (val, val)
    if isinstance(val, (list, tuple)) and len(val) == 2:
        lo, hi = int(val[0]), int(val[1])
        if lo > hi:
            raise ValueError(f"{key} range must have min <= max, got {val}")
        return (lo, hi)
    raise ValueError(f"{key} must be int or [min, max], got {val!r}")


def _load_large_lesion(d: dict) -> LargeLesionConfig:
    K = _parse_int_or_range(_require(d, "K", "Expected int or [min, max]."), "K")
    K_min = int(_require(d, "K_min", "Expected int."))
    K_max = int(_require(d, "K_max", "Expected int."))
    max_extra = int(_require(d, "max_extra", "Expected int."))
    if K_min > K_max:
        raise ValueError(f"large_lesion.K_min ({K_min}) must be <= K_max ({K_max})")
    if max_extra < 0:
        raise ValueError(f"large_lesion.max_extra must be >= 0, got {max_extra}")
    return LargeLesionConfig(K=K, K_min=K_min, K_max=K_max, max_extra=max_extra)


DEFAULT_SIGMA_PER_AXIS = (2.75, 5.19, 5.40)
DEFAULT_MAX_VOX = 34.0


def _load_propagated(d: dict) -> PropagatedConfig:
    prop = d.get("propagated")
    if prop is None:
        return PropagatedConfig(
            sigma_per_axis=DEFAULT_SIGMA_PER_AXIS,
            max_vox=DEFAULT_MAX_VOX,
        )
    if not isinstance(prop, dict):
        raise ValueError(f"propagated must be dict, got {type(prop)}")
    sigma = prop.get("sigma_per_axis", DEFAULT_SIGMA_PER_AXIS)
    if isinstance(sigma, (list, tuple)) and len(sigma) == 3:
        sigma = tuple(float(s) for s in sigma)
    else:
        sigma = DEFAULT_SIGMA_PER_AXIS
    max_vox = float(prop.get("max_vox", DEFAULT_MAX_VOX))
    return PropagatedConfig(sigma_per_axis=sigma, max_vox=max_vox)


def _load_sampling(sampling: dict) -> SamplingConfig:
    mode_probs = _require(sampling, "mode_probs", "Expected list of 4 floats summing to 1.")
    if not isinstance(mode_probs, (list, tuple)) or len(mode_probs) != 4:
        raise ValueError(f"mode_probs must be list of 4 floats, got {mode_probs!r}")
    probs = tuple(float(p) for p in mode_probs)
    if abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"mode_probs must sum to 1, got {sum(probs)}")
    n_spur = _parse_int_or_range(_require(sampling, "n_spur", "Expected int or [min, max]."), "n_spur")
    n_neg = _parse_int_or_range(_require(sampling, "n_neg", "Expected int or [min, max]."), "n_neg")
    ll = _require(sampling, "large_lesion", "Expected dict with K, K_min, K_max, max_extra.")
    if not isinstance(ll, dict):
        raise ValueError(f"large_lesion must be dict, got {type(ll)}")
    large_lesion = _load_large_lesion(ll)
    propagated = _load_propagated(sampling)
    return SamplingConfig(
        mode_probs=probs, n_spur=n_spur, n_neg=n_neg,
        large_lesion=large_lesion, propagated=propagated,
    )


def _load_inference(inf: Optional[dict]) -> InferenceConfig:
    if inf is None:
        return InferenceConfig(tile_step_size=0.5, disable_tta_default=False)
    tile_step_size = float(inf.get("tile_step_size", 0.5))
    disable_tta_default = bool(inf.get("disable_tta_default", False))
    mode = inf.get("roi_inference_mode", "dilated_sliding")
    if mode not in ("dilated_sliding", "per_point_patch"):
        raise ValueError(f"inference.roi_inference_mode must be 'dilated_sliding' or 'per_point_patch', got {mode!r}")
    max_visits = int(inf.get("max_patch_expansion_visits", 64))
    if max_visits < 1:
        raise ValueError(f"max_patch_expansion_visits must be >= 1, got {max_visits}")
    depth_raw = inf.get("max_patch_expansion_depth", None)
    depth = int(depth_raw) if depth_raw is not None else None
    if depth is not None and depth < 0:
        raise ValueError(f"max_patch_expansion_depth must be >= 0 or null, got {depth}")
    return InferenceConfig(
        tile_step_size=tile_step_size,
        disable_tta_default=disable_tta_default,
        roi_inference_mode=mode,
        max_patch_expansion_visits=max_visits,
        max_patch_expansion_depth=depth,
    )


def _load_size_bins(sb: Optional[dict]) -> Optional[SizeBinsConfig]:
    if sb is None or not isinstance(sb, dict):
        return None
    mode = sb.get("mode", "fixed")
    if mode not in ("percentile", "fixed"):
        return None
    trim = float(sb.get("trim_percentile", 0.025))
    percs = tuple(float(p) for p in sb.get("percentiles", [0.25, 0.5, 0.75]))
    thresh_raw = sb.get("thresholds", [100, 2000, 20000])
    thresh = tuple(int(t) for t in thresh_raw) if thresh_raw else (100, 2000, 20000)
    return SizeBinsConfig(mode=mode, trim_percentile=trim, percentiles=percs, thresholds=thresh)


def _load_stratified(s: Optional[dict]) -> Optional[StratifiedConfig]:
    if s is None or not isinstance(s, dict):
        return None
    dw = s.get("dataset_weights")
    sbw = s.get("size_bin_weights")
    if not isinstance(dw, dict) or not isinstance(sbw, dict):
        return None
    dw = {str(k): float(v) for k, v in dw.items()}
    sbw = {str(k): float(v) for k, v in sbw.items()}
    dw_sum = sum(dw.values())
    sbw_sum = sum(sbw.values())
    if dw_sum > 0:
        dw = {k: v / dw_sum for k, v in dw.items()}
    if sbw_sum > 0:
        sbw = {k: v / sbw_sum for k, v in sbw.items()}
    return StratifiedConfig(dataset_weights=dw, size_bin_weights=sbw)


def load_config(path: str) -> RoiPromptConfig:
    """Load config from JSON. Fail fast on missing keys. Returns prompt+sampling+inference config."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    d = load_json(str(p))
    if not isinstance(d, dict):
        raise ValueError(f"Config must be a dict, got {type(d)}")
    prompt = d.get("prompt")
    if not isinstance(prompt, dict):
        raise KeyError("Config must have 'prompt' section (dict)")
    sampling = d.get("sampling")
    if not isinstance(sampling, dict):
        raise KeyError("Config must have 'sampling' section (dict)")
    inference = _load_inference(d.get("inference"))
    size_bins = _load_size_bins(d.get("size_bins"))
    stratified = _load_stratified(d.get("stratified"))
    return RoiPromptConfig(
        prompt=_load_prompt(prompt),
        sampling=_load_sampling(sampling),
        inference=inference,
        size_bins=size_bins,
        stratified=stratified,
    )
