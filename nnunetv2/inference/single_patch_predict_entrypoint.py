"""Single-patch CLI: one crop centered on a point; optional patch-local prompt (no dilated ROI). PromptAware only."""
import argparse
import json
import os
import sys
from time import time as time_func
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json, maybe_mkdir_p, save_json

from nnunetv2.evaluation.evaluate_predictions import compute_dice_from_arrays
from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
    export_prediction_from_logits,
)
from nnunetv2.inference.roi_predictor import nnUNetROIPredictor, points_dict_to_canonical
from nnunetv2.inference.single_patch_session import run_single_patch_forward
from nnunetv2.utilities.cli_display import InferenceDisplay
from nnunetv2.utilities.roi_config import RoiPromptConfig, load_config
from nnunetv2.utilities.roi_coordinate_validation import validate_and_convert_points
from nnunetv2.utilities.roi_geometry import points_to_centers_zyx
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


def _preprocessed_spacing_zyx(props: dict, plans_manager, configuration_manager) -> Tuple[float, float, float]:
    spacing_transposed = [props["spacing"][i] for i in plans_manager.transpose_forward]
    if len(configuration_manager.spacing) == len(props["shape_after_cropping_and_before_resampling"]):
        t = tuple(configuration_manager.spacing)
    else:
        t = tuple([spacing_transposed[0], *configuration_manager.spacing])
    return (float(t[0]), float(t[1]), float(t[2]))


def _get_default_value(env: str, dtype: type, default):
    try:
        return dtype(os.environ.get(env) or default)
    except Exception:
        return default


def load_points_input(path_or_dash: str) -> Dict[str, Any]:
    if path_or_dash == "-":
        return json.load(sys.stdin)
    return load_json(path_or_dash)


def inline_points_dict(s: str) -> Dict[str, Any]:
    return json.loads(s)


def point_zyx_inline(s: str) -> Dict[str, Any]:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--point_zyx expects z,y,x with exactly three integers")
    return {
        "points": [parts],
        "points_space": "voxel",
        "points_format": "zyx_voxel",
        "voxel_coordinate_frame": "full",
    }


def _data_to_tensor_float(data):
    if isinstance(data, str):
        delfile = data
        data = torch.from_numpy(np.load(data))
        os.remove(delfile)
    elif not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    return data.float()


def single_patch_infer_one(
    pred: nnUNetROIPredictor,
    cfg: RoiPromptConfig,
    points_dict: Dict[str, Any],
    points_space_override: Optional[str],
    preprocessed: Dict[str, Any],
    *,
    encode_prompt: bool,
    save_debug_patch: Optional[str],
    save_debug_patch_prompts: bool,
    labels_folder: Optional[str],
    file_ending: str,
    label_manager,
    labels_or_regions,
    ignore_label,
    case_idx: int,
    caseids: list,
    verbose: bool,
) -> Optional[float]:
    data = _data_to_tensor_float(preprocessed["data"])
    props = preprocessed["data_properties"]
    ofile = preprocessed["ofile"]
    points_raw, space, fmt, voxel_frame, debug_patch_bbox_pad = points_dict_to_canonical(
        points_dict, points_space_override
    )
    points_canonical = validate_and_convert_points(points_raw, space, fmt)
    if len(points_canonical) != 1:
        raise ValueError(f"exactly one point required; got {len(points_canonical)}")
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
    if verbose:
        print(
            f"[single_patch] points={points_raw!r} format={fmt!r} → canonical (z,y,x)={points_canonical} "
            f"→ preprocessed center (z,y,x)={points_zyx}.",
            flush=True,
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
            debug_out = join(save_debug_patch, f"{cid}_single_patch_debug.nii.gz")
        debug_sp = _preprocessed_spacing_zyx(props, pred.plans_manager, pred.configuration_manager)
        debug_geom = {
            "properties": props,
            "plans_manager": pred.plans_manager,
            "configuration_manager": pred.configuration_manager,
        }
        if debug_patch_bbox_pad is not None:
            debug_geom["debug_bbox_pad_vox"] = debug_patch_bbox_pad

    logits = run_single_patch_forward(
        pred,
        cfg,
        data,
        props,
        points_canonical,
        space,
        fmt,
        voxel_frame,
        encode_prompt,
        debug_out,
        debug_sp,
        debug_geom,
        save_debug_patch_prompts,
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


def predict_single_patch_entry_point():
    parser = argparse.ArgumentParser(
        description=(
            "Single-patch inference: same pre/post as nnUNetv2_predict_roi, but exactly one network tile "
            "centered on the point (no dilated-bbox sliding). Optional --encode_prompt uses nnunet_pro_config "
            "prompt encoding in that tile. nnUNetTrainerPromptAware models only."
        )
    )
    g_pts = parser.add_mutually_exclusive_group(required=False)
    g_pts.add_argument(
        "--points_json",
        default=None,
        help="Path to points JSON (or '-' for stdin). Same schema as ROI points file.",
    )
    g_pts.add_argument(
        "--points_inline",
        default=None,
        help="Inline JSON object string for points (same keys as points JSON file).",
    )
    g_pts.add_argument(
        "--point_zyx",
        default=None,
        help="Single voxel as z,y,x comma-separated integers (implies voxel zyx_voxel, full frame).",
    )
    parser.add_argument(
        "--stdin_loop",
        action="store_true",
        help="Read one JSON points object per line from stdin after preprocessing once (-i must be a single file). "
        "Do not use with --points_json -.",
    )
    parser.add_argument("-i", required=True, help="Input folder or file. Same channel numbering as training (_0000 etc).")
    parser.add_argument("-o", required=True, help="Output folder. Predictions have same base name as inputs.")
    parser.add_argument("-m", required=True, help="Model folder (trained nnUNetTrainerPromptAware).")
    parser.add_argument("--config", default=None, help="Path to nnunet_pro config JSON. Default: {model_folder}/nnunet_pro_config.json")
    parser.add_argument("--points_space", default=None, help="Override points_space from JSON (voxel|world).")
    parser.add_argument("--disable_tta", action="store_true", help="Disable test-time augmentation (mirroring).")
    parser.add_argument("-f", nargs="+", default=(0,), help="Folds to use. Default: 0.")
    parser.add_argument("-chk", default="checkpoint_final.pth", help="Checkpoint name. Default: checkpoint_final.pth.")
    parser.add_argument("-device", default="cuda", help="Device: cuda, cpu, or mps.")
    parser.add_argument("-npp", type=int, default=_get_default_value("nnUNet_npp", int, 3), help="Preprocessing processes.")
    parser.add_argument("-nps", type=int, default=_get_default_value("nnUNet_nps", int, 3), help="Export processes.")
    parser.add_argument("--labels_folder", default=None, help="Folder with ground truth for per-case DICE.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument(
        "--encode_prompt",
        action="store_true",
        help="Encode the point as pos heatmap in the patch (config prompt.*); default is zero prompts (pos_no_prompt-style).",
    )
    parser.add_argument(
        "--save_debug_patch",
        default=None,
        help="Debug: directory or *.nii.gz stem. Saves *_preprocessed*.nii.gz and *_viewer*.nii.gz (native orientation). "
        "Bbox padded by NNUNET_DEBUG_PATCH_BBOX_PAD (default 32) or points.json debug_patch_bbox_pad.",
    )
    parser.add_argument(
        "--save_debug_patch_prompts",
        action="store_true",
        help="With --save_debug_patch, also write 3D prompt channels {stem}_prompt_0/1.nii.gz.",
    )
    args = parser.parse_args()

    if args.stdin_loop and args.points_json == "-":
        raise ValueError("--stdin_loop cannot be combined with --points_json - (stdin conflict).")
    if not args.stdin_loop:
        if not (args.points_json or args.points_inline or args.point_zyx):
            parser.error("provide one of --points_json, --points_inline, or --point_zyx (or use --stdin_loop).")

    args.f = [int(x) if x != "all" else x for x in args.f]
    maybe_mkdir_p(args.o)

    config_path = args.config or join(args.m, "nnunet_pro_config.json")
    if not isfile(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}. "
            "Provide --config or train with nnUNetTrainerPromptAware (copies config to model folder)."
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
            print("MPS can cause segfaults on Mac; using CPU. Set nnUNet_MPS_FORCE_GPU=1 to try MPS.")
            import multiprocessing as mp

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
    save_json(pred.dataset_json, join(args.o, "dataset.json"), sort_keys=False)
    save_json(pred.plans_manager.plans, join(args.o, "plans.json"), sort_keys=False)

    if os.path.isfile(args.i):
        list_of_lists = [[args.i]]
    else:
        list_of_lists = create_lists_from_splitted_dataset_folder(args.i, pred.dataset_json["file_ending"])
    if not list_of_lists:
        raise RuntimeError(f"No valid input files in {args.i}")

    caseids = [
        os.path.basename(lst[0])[: -(len(pred.dataset_json["file_ending"]) + 5)] for lst in list_of_lists
    ]
    output_truncated = [join(args.o, cid) for cid in caseids]

    points_json_path_for_msg = args.points_json or "(inline)"
    static_points_dict: Optional[Dict[str, Any]] = None
    if not args.stdin_loop:
        if args.points_json:
            static_points_dict = load_points_input(args.points_json)
        elif args.points_inline:
            static_points_dict = inline_points_dict(args.points_inline)
        else:
            static_points_dict = point_zyx_inline(args.point_zyx)

        points_raw, space, fmt, voxel_frame, _ = points_dict_to_canonical(static_points_dict, args.points_space)
        points_canonical = validate_and_convert_points(points_raw, space, fmt)
        if len(points_canonical) != 1:
            raise ValueError(
                f"nnUNetv2_predict_single_patch requires exactly one point; got {len(points_canonical)} from {points_json_path_for_msg}"
            )

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
        label_manager.foreground_regions if label_manager.has_regions else [(l,) for l in label_manager.foreground_labels]
    )
    ignore_label = label_manager.ignore_label if label_manager.has_ignore_label else None
    dataset_name = pred.dataset_json.get("name", os.path.basename(args.m))
    device_str = "mps" if device.type == "mps" else "cuda" if device.type == "cuda" else "cpu"

    display_verbose = args.verbose or args.stdin_loop
    num_cases_display = 1 if args.stdin_loop else len(caseids)
    with InferenceDisplay(dataset_name, "3d_fullres", device_str, num_cases_display, verbose=display_verbose) as display:
        if args.stdin_loop:
            if not os.path.isfile(args.i):
                raise ValueError("--stdin_loop requires -i to be a single file")
            preprocessed = next(iterator)
            case_idx = 1
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                pd = json.loads(line)
                case_start = time_func()
                dice = single_patch_infer_one(
                    pred,
                    cfg,
                    pd,
                    args.points_space,
                    preprocessed,
                    encode_prompt=args.encode_prompt,
                    save_debug_patch=args.save_debug_patch,
                    save_debug_patch_prompts=args.save_debug_patch_prompts,
                    labels_folder=args.labels_folder,
                    file_ending=file_ending,
                    label_manager=label_manager,
                    labels_or_regions=labels_or_regions,
                    ignore_label=ignore_label,
                    case_idx=case_idx,
                    caseids=caseids,
                    verbose=args.verbose,
                )
                case_time = time_func() - case_start
                display.update_case(case_idx, case_time, dice)
                case_idx += 1
            return

        for case_idx, preprocessed in enumerate(iterator, 1):
            case_start = time_func()
            assert static_points_dict is not None
            dice = single_patch_infer_one(
                pred,
                cfg,
                static_points_dict,
                args.points_space,
                preprocessed,
                encode_prompt=args.encode_prompt,
                save_debug_patch=args.save_debug_patch,
                save_debug_patch_prompts=args.save_debug_patch_prompts,
                labels_folder=args.labels_folder,
                file_ending=file_ending,
                label_manager=label_manager,
                labels_or_regions=labels_or_regions,
                ignore_label=ignore_label,
                case_idx=case_idx,
                caseids=caseids,
                verbose=args.verbose,
            )
            case_time = time_func() - case_start
            display.update_case(case_idx, case_time, dice)


if __name__ == "__main__":
    predict_single_patch_entry_point()
