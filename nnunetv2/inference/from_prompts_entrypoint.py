"""Multi-prompt adaptive tiling: cluster points into patch-coverable groups, one seed+BFS per group."""
import argparse
import json
import os
import sys
from time import time as time_func
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import isfile, join, maybe_mkdir_p, save_json

from nnunetv2.evaluation.evaluate_predictions import compute_dice_from_arrays
from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
    export_prediction_from_logits,
)
from nnunetv2.inference.prompt_clustering import cluster_centroid_zyx, cluster_points_for_patch_size
from nnunetv2.inference.roi_predictor import nnUNetROIPredictor, points_dict_to_canonical
from nnunetv2.inference.single_patch_predict_entrypoint import (
    _get_default_value,
    _preprocessed_spacing_zyx,
    load_points_input,
    inline_points_dict,
)
from nnunetv2.utilities.cli_display import InferenceDisplay
from nnunetv2.utilities.inference_execution import apply_inference_execution_env
from nnunetv2.utilities.roi_config import RoiPromptConfig, load_config
from nnunetv2.utilities.roi_coordinate_validation import validate_and_convert_points
from nnunetv2.utilities.roi_geometry import points_to_centers_zyx
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


def _data_to_tensor_float(data):
    if isinstance(data, str):
        delfile = data
        data = torch.from_numpy(np.load(data))
        os.remove(delfile)
    elif not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    return data.float()


def _probe_prompt_gate_at_preprocessed_zyx(
    pred: nnUNetROIPredictor,
    cfg: RoiPromptConfig,
    data: torch.Tensor,
    props: dict,
    ctr_zyx: Tuple[int, int, int],
    voxel_frame: str,
    prompt_gate_threshold: float,
    prompt_gate_radius_mm: float,
) -> bool:
    """Probe-only single-patch forward at centroid; return True if encoded pass should run (same rule as single-click gate)."""
    from nnunetv2.inference.roi_predictor import (
        centered_spatial_slices_at_point,
        map_points_zyx_unpadded_to_padded,
    )
    from nnunetv2.inference.single_patch_session import run_single_patch_forward

    points_canonical = [tuple(int(x) for x in ctr_zyx)]
    shape = tuple(data.shape[1:])
    patch_size = tuple(int(x) for x in pred.configuration_manager.patch_size)
    padded_shape = tuple(max(s, p) for s, p in zip(shape, patch_size))
    slicer_revert = (slice(None),) + tuple(
        slice((ps - s) // 2, (ps - s) // 2 + s) for s, ps in zip(shape, padded_shape)
    )
    pz, py, px = map_points_zyx_unpadded_to_padded([ctr_zyx], slicer_revert)[0]
    sz, sy, sx = centered_spatial_slices_at_point(pz, py, px, patch_size, padded_shape)
    cz, cy, cx = slicer_revert[1], slicer_revert[2], slicer_revert[3]
    z_lo, z_hi = max(sz.start, cz.start), min(sz.stop, cz.stop)
    y_lo, y_hi = max(sy.start, cy.start), min(sy.stop, cy.stop)
    x_lo, x_hi = max(sx.start, cx.start), min(sx.stop, cx.stop)
    if z_lo >= z_hi or y_lo >= y_hi or x_lo >= x_hi:
        raise ValueError("cluster centroid maps outside preprocessed volume for prompt gate")
    uz, uy, ux = (
        slice(z_lo - cz.start, z_hi - cz.start),
        slice(y_lo - cy.start, y_hi - cy.start),
        slice(x_lo - cx.start, x_hi - cx.start),
    )
    lz, ly, lx = ctr_zyx[0] - uz.start, ctr_zyx[1] - uy.start, ctr_zyx[2] - ux.start

    probe = run_single_patch_forward(
        pred,
        cfg,
        data,
        props,
        points_canonical,
        "voxel",
        "zyx_voxel",
        voxel_frame,
        False,
        None,
        None,
        None,
        False,
        border_expand=False,
        max_border_expand_extra=0,
    )
    p_fg_map = torch.softmax(probe[:, uz, uy, ux].float(), dim=0)[1]
    sp_zyx = tuple(float(s) for s in pred.configuration_manager.spacing)
    r_mm = float(prompt_gate_radius_mm)
    if r_mm > 1e-6:
        dev = p_fg_map.device
        dz, dy, dx = p_fg_map.shape
        zc = (torch.arange(dz, device=dev, dtype=torch.float32) - lz) * sp_zyx[0]
        yc = (torch.arange(dy, device=dev, dtype=torch.float32) - ly) * sp_zyx[1]
        xc = (torch.arange(dx, device=dev, dtype=torch.float32) - lx) * sp_zyx[2]
        d2 = zc.view(-1, 1, 1) ** 2 + yc.view(1, -1, 1) ** 2 + xc.view(1, 1, -1) ** 2
        sphere = d2 <= r_mm**2
        p_peak = p_fg_map[sphere].max().item() if bool(sphere.any()) else 0.0
    else:
        p_peak = p_fg_map[lz, ly, lx].item()
    return p_peak > float(prompt_gate_threshold)


def from_prompts_infer_one(
    pred: nnUNetROIPredictor,
    cfg: RoiPromptConfig,
    margin_frac: float,
    points_dict: Dict[str, Any],
    points_space_override: Optional[str],
    preprocessed: Dict[str, Any],
    *,
    encode_prompt: bool,
    save_debug_patch: Optional[str],
    save_debug_patch_prompts: bool,
    border_expand: bool,
    max_border_expand_extra: int,
    cross_cluster_neg: bool,
    labels_folder: Optional[str],
    file_ending: str,
    label_manager,
    labels_or_regions,
    ignore_label,
    case_idx: int,
    caseids: list,
    verbose: bool,
    prompt_gate_threshold: Optional[float] = None,
    prompt_gate_radius_mm: float = 0.0,
    infer_meta_out: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    data = _data_to_tensor_float(preprocessed["data"])
    props = preprocessed["data_properties"]
    ofile = preprocessed["ofile"]
    points_raw, space, fmt, voxel_frame, debug_patch_bbox_pad = points_dict_to_canonical(
        points_dict, points_space_override
    )
    points_canonical = validate_and_convert_points(points_raw, space, fmt)
    if not points_canonical:
        raise ValueError("at least one point required for nnUNetv2_predict_from_prompts")
    shape = tuple(data.shape[1:])
    points_zyx = points_to_centers_zyx(
        points_canonical,
        space,
        props,
        shape,
        tuple(pred.configuration_manager.spacing),
        pred.plans_manager.transpose_forward,
        voxel_coordinate_frame=voxel_frame,
    )
    pts: List[Tuple[int, int, int]] = [tuple(int(x) for x in p) for p in points_zyx]  # type: ignore
    ps = tuple(int(x) for x in pred.configuration_manager.patch_size)
    clusters = cluster_points_for_patch_size(pts, ps, margin_frac)
    if verbose:
        print(
            f"[from_prompts] n_points={len(pts)} → {len(clusters)} cluster(s) margin={margin_frac}",
            flush=True,
        )
    if len(clusters) > 1 and (not encode_prompt) and cross_cluster_neg:
        raise ValueError("--cross_cluster_neg requires --encode_prompt (both prompt channels)")
    if prompt_gate_threshold is not None and not encode_prompt:
        raise ValueError("prompt_gate_threshold requires encode_prompt=True")
    if prompt_gate_radius_mm > 0 and prompt_gate_threshold is None:
        raise ValueError("prompt_gate_radius_mm requires prompt_gate_threshold")

    encode_per_cluster: Optional[List[bool]] = None
    if encode_prompt and prompt_gate_threshold is not None:
        encode_per_cluster = []
        for c in clusters:
            cent = cluster_centroid_zyx(c)
            encode_per_cluster.append(
                _probe_prompt_gate_at_preprocessed_zyx(
                    pred,
                    cfg,
                    data,
                    props,
                    cent,
                    voxel_frame,
                    prompt_gate_threshold,
                    prompt_gate_radius_mm,
                )
            )
        if cross_cluster_neg and not all(encode_per_cluster):
            raise ValueError(
                "cross_cluster_neg is incompatible with prompt gate when any cluster is probe-only"
            )

    if infer_meta_out is not None:
        infer_meta_out["n_prompt_clusters"] = len(clusters)
        infer_meta_out["prompt_gate_per_cluster"] = (
            list(encode_per_cluster) if encode_per_cluster is not None else None
        )

    debug_out = None
    debug_sp = None
    debug_geom = None
    if save_debug_patch:
        if save_debug_patch.endswith(".nii.gz"):
            debug_out = save_debug_patch
        else:
            maybe_mkdir_p(save_debug_patch)
            cid = os.path.basename(ofile) if ofile else caseids[case_idx - 1]
            debug_out = join(save_debug_patch, f"{cid}_from_prompts_debug.nii.gz")
        debug_sp = _preprocessed_spacing_zyx(props, pred.plans_manager, pred.configuration_manager)
        debug_geom = {
            "properties": props,
            "plans_manager": pred.plans_manager,
            "configuration_manager": pred.configuration_manager,
        }
        if debug_patch_bbox_pad is not None:
            debug_geom["debug_bbox_pad_vox"] = debug_patch_bbox_pad

    logits = pred.predict_logits_from_prompt_clusters(
        data,
        clusters,
        cfg,
        encode_prompt=encode_prompt,
        encode_per_cluster=encode_per_cluster,
        save_debug_patch=debug_out,
        debug_patch_spacing_zyx=debug_sp,
        save_debug_patch_prompts=save_debug_patch_prompts,
        debug_native_geometry=debug_geom,
        border_expand=border_expand,
        max_border_expand_extra=max_border_expand_extra,
        cross_cluster_neg=cross_cluster_neg,
    )

    dice = None
    if ofile and labels_folder:
        caseid = os.path.basename(ofile)
        label_path = join(labels_folder, caseid + file_ending)
        if isfile(label_path):
            pred_seg = convert_predicted_logits_to_segmentation_with_correct_shape(
                logits.cpu(),
                pred.plans_manager,
                pred.configuration_manager,
                label_manager,
                props,
                save_probabilities=False,
            )
            pred_seg = np.asarray(pred_seg)
            rw = pred.plans_manager.image_reader_writer_class()
            gt_seg, _ = rw.read_seg(label_path)
            gt_seg = np.asarray(gt_seg)
            if pred_seg.ndim == 4:
                pred_seg = pred_seg[0]
            if gt_seg.ndim == 4:
                gt_seg = gt_seg[0]
            dice = compute_dice_from_arrays(pred_seg, gt_seg, labels_or_regions, ignore_label)

    if ofile:
        export_prediction_from_logits(
            logits.cpu().numpy(),
            props,
            pred.configuration_manager,
            pred.plans_manager,
            pred.dataset_json,
            ofile,
            save_probabilities=False,
        )
    return dice


def predict_from_prompts_entry_point():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-prompt prompt-aware inference: cluster points (patch-bbox + margin) into one seed "
            "per group, optional per-cluster border expansion, Gaussian merge. nnUNetTrainerPromptAware only."
        )
    )
    g_pts = parser.add_mutually_exclusive_group(required=True)
    g_pts.add_argument("--points_json", help="Path to points JSON (multiple points) or '-' for stdin line.")
    g_pts.add_argument(
        "--points_inline",
        help="Inline JSON for points (same keys as file).",
    )
    parser.add_argument(
        "--stdin_loop",
        action="store_true",
        help="Read one points JSON per line from stdin; -i must be a single file.",
    )
    parser.add_argument("-i", required=True, help="Input folder or file.")
    parser.add_argument("-o", required=True, help="Output folder.")
    parser.add_argument("-m", required=True, help="Model folder (nnUNetTrainerPromptAware).")
    parser.add_argument("--config", default=None, help="nnunet_pro config; default: {model}/nnunet_pro_config.json")
    parser.add_argument("--points_space", default=None, help="Override points_space (voxel|world).")
    parser.add_argument("--disable_tta", action="store_true", help="Disable mirroring TTA.")
    parser.add_argument("-f", nargs="+", default=(0,), help="Folds. Default: 0.")
    parser.add_argument("-chk", default="checkpoint_final.pth", help="Checkpoint name.")
    parser.add_argument("-device", default="cuda", help="cuda | cpu | mps")
    parser.add_argument("-npp", type=int, default=_get_default_value("nnUNet_npp", int, 3), help="Preprocess procs.")
    parser.add_argument("-nps", type=int, default=_get_default_value("nnUNet_nps", int, 3), help="Export procs.")
    parser.add_argument("--labels_folder", default=None, help="GT folder for DICE.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument(
        "--encode_prompt",
        action="store_true",
        help="Patch-local pos/neg heatmaps (recommended for multi-lesion; default is zero pos/neg in tiles).",
    )
    parser.add_argument(
        "--save_debug_patch",
        default=None,
        help="Debug NIfTI stem or directory (first tile only).",
    )
    parser.add_argument(
        "--save_debug_patch_prompts",
        action="store_true",
        help="With --save_debug_patch, write prompt_0/1 NIfTIs.",
    )
    parser.add_argument("--border_expand", action="store_true", help="Per-cluster BFS border expansion + merge.")
    parser.add_argument(
        "--border_expand_max_extra",
        type=int,
        default=16,
        help="Max extra tiles per cluster after seed (default 16).",
    )
    parser.add_argument(
        "--cluster_overlap_margin",
        type=float,
        default=0.1,
        help="Margin fraction of patch_size per axis for cluster bbox (default 0.1; clamped 0..0.5).",
    )
    parser.add_argument(
        "--cross_cluster_neg",
        action="store_true",
        help="Other clusters' centroids in-tile as neg heatmap (requires --encode_prompt).",
    )
    args = parser.parse_args()
    if args.border_expand_max_extra < 0:
        parser.error("--border_expand_max_extra must be >= 0")
    m = float(args.cluster_overlap_margin)
    if m < 0 or m > 0.5:
        parser.error("--cluster_overlap_margin must be in [0, 0.5]")

    if args.stdin_loop and args.points_json == "-":
        raise ValueError("--stdin_loop conflicts with --points_json -")

    args.f = [int(x) if x != "all" else x for x in args.f]
    maybe_mkdir_p(args.o)

    config_path = args.config or join(args.m, "nnunet_pro_config.json")
    if not isfile(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}. Use --config or train with nnUNetTrainerPromptAware."
        )

    assert args.device in ("cpu", "cuda", "mps"), f"-device must be cpu, cuda, or mps, got {args.device}"
    if args.device == "cpu":
        import multiprocessing as mp
        torch.set_num_threads(mp.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        if os.environ.get("nnUNet_MPS_FORCE_GPU") != "1":
            import multiprocessing as mp
            print("MPS can cause segfaults on Mac; using CPU. Set nnUNet_MPS_FORCE_GPU=1 to try MPS.")
            torch.set_num_threads(mp.cpu_count())
            device = torch.device("cpu")
        else:
            device = torch.device("mps")

    cfg = load_config(config_path)
    use_mirroring = not (args.disable_tta or cfg.inference.disable_tta_default)

    pred = nnUNetROIPredictor(
        tile_step_size=cfg.inference.tile_step_size,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        perform_everything_on_device=device.type != "cpu",
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=True,
    )
    pred.initialize_from_trained_model_folder(args.m, args.f, args.chk)
    apply_inference_execution_env(device)
    save_json(pred.dataset_json, join(args.o, "dataset.json"), sort_keys=False)
    save_json(pred.plans_manager.plans, join(args.o, "plans.json"), sort_keys=False)

    if os.path.isfile(args.i):
        list_of_lists = [[args.i]]
    else:
        list_of_lists = create_lists_from_splitted_dataset_folder(args.i, pred.dataset_json["file_ending"])
    if not list_of_lists:
        raise RuntimeError(f"No valid input files in {args.i}")

    caseids = [
        os.path.basename(lst[0])[: -(len(pred.dataset_json["file_ending"]) + 5)]
        for lst in list_of_lists
    ]
    output_truncated = [join(args.o, cid) for cid in caseids]

    static_points_dict: Optional[Dict[str, Any]] = None
    if not args.stdin_loop:
        if args.points_json is not None:
            static_points_dict = load_points_input(args.points_json)
        else:
            assert args.points_inline is not None
            static_points_dict = inline_points_dict(args.points_inline)

    num_pp = min(args.npp, len(list_of_lists))
    if device.type == "mps":
        num_pp = 1
    pin_memory = device.type == "cuda" and torch.cuda.is_available()
    iterator = preprocessing_iterator_fromfiles(
        list_of_lists,
        None,
        output_truncated,
        pred.plans_manager,
        pred.dataset_json,
        pred.configuration_manager,
        num_pp,
        pin_memory,
        args.verbose,
    )

    file_ending = pred.dataset_json["file_ending"]
    label_manager = pred.plans_manager.get_label_manager(pred.dataset_json)
    labels_or_regions = (
        label_manager.foreground_regions
        if label_manager.has_regions
        else [(l,) for l in label_manager.foreground_labels]
    )
    ignore_label = label_manager.ignore_label if label_manager.has_ignore_label else None
    dataset_name = pred.dataset_json.get("name", os.path.basename(args.m))
    device_str = "mps" if device.type == "mps" else "cuda" if device.type == "cuda" else "cpu"
    display_verbose = args.verbose or args.stdin_loop
    num_cases_display = 1 if args.stdin_loop else len(caseids)
    with InferenceDisplay(
        dataset_name, "3d_fullres", device_str, num_cases_display, verbose=display_verbose
    ) as display:
        if args.stdin_loop:
            if not os.path.isfile(args.i):
                raise ValueError("--stdin_loop requires -i to be a single file")
            preprocessed = next(iter(iterator))
            case_idx = 1
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                pd = json.loads(line)
                case_start = time_func()
                dice = from_prompts_infer_one(
                    pred,
                    cfg,
                    args.cluster_overlap_margin,
                    pd,
                    args.points_space,
                    preprocessed,
                    encode_prompt=args.encode_prompt,
                    save_debug_patch=args.save_debug_patch,
                    save_debug_patch_prompts=args.save_debug_patch_prompts,
                    border_expand=args.border_expand,
                    max_border_expand_extra=args.border_expand_max_extra,
                    cross_cluster_neg=args.cross_cluster_neg,
                    labels_folder=args.labels_folder,
                    file_ending=file_ending,
                    label_manager=label_manager,
                    labels_or_regions=labels_or_regions,
                    ignore_label=ignore_label,
                    case_idx=case_idx,
                    caseids=caseids,
                    verbose=args.verbose,
                )
                display.update_case(case_idx, time_func() - case_start, dice)
                case_idx += 1
            return

        for case_idx, preprocessed in enumerate(iterator, 1):
            case_start = time_func()
            assert static_points_dict is not None
            dice = from_prompts_infer_one(
                pred,
                cfg,
                args.cluster_overlap_margin,
                static_points_dict,
                args.points_space,
                preprocessed,
                encode_prompt=args.encode_prompt,
                save_debug_patch=args.save_debug_patch,
                save_debug_patch_prompts=args.save_debug_patch_prompts,
                border_expand=args.border_expand,
                max_border_expand_extra=args.border_expand_max_extra,
                cross_cluster_neg=args.cross_cluster_neg,
                labels_folder=args.labels_folder,
                file_ending=file_ending,
                label_manager=label_manager,
                labels_or_regions=labels_or_regions,
                ignore_label=ignore_label,
                case_idx=case_idx,
                caseids=caseids,
                verbose=args.verbose,
            )
            display.update_case(case_idx, time_func() - case_start, dice)


if __name__ == "__main__":
    predict_from_prompts_entry_point()
