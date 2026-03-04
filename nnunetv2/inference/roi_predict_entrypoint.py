"""ROI-mode CLI: prompt-aware inference over dilated bbox only."""
import argparse
import os
from time import time as time_func

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json, maybe_mkdir_p, save_json

from nnunetv2.evaluation.evaluate_predictions import compute_dice_from_arrays
from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape, export_prediction_from_logits
from nnunetv2.inference.roi_predictor import nnUNetROIPredictor, parse_points_json
from nnunetv2.utilities.cli_display import InferenceDisplay
from nnunetv2.utilities.roi_config import load_config
from nnunetv2.utilities.roi_coordinate_validation import validate_and_convert_points
from nnunetv2.utilities.roi_geometry import points_to_centers_zyx
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


def _get_default_value(env: str, dtype: type, default):
    try:
        return dtype(os.environ.get(env) or default)
    except Exception:
        return default


def predict_roi_entry_point():
    parser = argparse.ArgumentParser(
        description="ROI-mode inference: prompt-aware sliding over dilated bbox only. Requires nnUNetTrainerPromptAware model."
    )
    parser.add_argument("-i", required=True, help="Input folder or file. Same channel numbering as training (_0000 etc).")
    parser.add_argument("-o", required=True, help="Output folder. Predictions have same base name as inputs.")
    parser.add_argument("-m", required=True, help="Model folder (trained nnUNetTrainerPromptAware).")
    parser.add_argument("--config", default=None, help="Path to nnunet_pro config JSON. Default: {model_folder}/nnunet_pro_config.json")
    parser.add_argument("--points_json", required=True, help="Path to points JSON: {points: [[z,y,x]|world], points_space: voxel|world}.")
    parser.add_argument("--points_space", default=None, help="Override points_space from JSON (voxel|world).")
    parser.add_argument("--disable_tta", action="store_true", help="Disable test-time augmentation (mirroring).")
    parser.add_argument("-f", nargs="+", default=(0,), help="Folds to use. Default: 0.")
    parser.add_argument("-chk", default="checkpoint_final.pth", help="Checkpoint name. Default: checkpoint_final.pth.")
    parser.add_argument("-device", default="cuda", help="Device: cuda, cpu, or mps.")
    parser.add_argument("-npp", type=int, default=_get_default_value("nnUNet_npp", int, 3), help="Preprocessing processes.")
    parser.add_argument("-nps", type=int, default=_get_default_value("nnUNet_nps", int, 3), help="Export processes.")
    parser.add_argument("--labels_folder", default=None, help="Folder with ground truth for per-case DICE.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    args = parser.parse_args()

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
    tile_step = cfg.inference.tile_step_size
    use_mirroring = not (args.disable_tta or cfg.inference.disable_tta_default)

    pred = nnUNetROIPredictor(
        tile_step_size=tile_step,
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
        list_of_lists = create_lists_from_splitted_dataset_folder(
            args.i, pred.dataset_json["file_ending"]
        )
    if not list_of_lists:
        raise RuntimeError(f"No valid input files in {args.i}")

    caseids = [
        os.path.basename(lst[0])[: -(len(pred.dataset_json["file_ending"]) + 5)]
        for lst in list_of_lists
    ]
    output_truncated = [join(args.o, cid) for cid in caseids]

    points_raw, space, fmt = parse_points_json(args.points_json, args.points_space)
    points_canonical = validate_and_convert_points(points_raw, space, fmt)

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
    labels_or_regions = label_manager.foreground_regions if label_manager.has_regions else [(l,) for l in label_manager.foreground_labels]
    ignore_label = label_manager.ignore_label if label_manager.has_ignore_label else None
    dataset_name = pred.dataset_json.get("name", os.path.basename(args.m))
    device_str = "mps" if device.type == "mps" else "cuda" if device.type == "cuda" else "cpu"

    with InferenceDisplay(dataset_name, "3d_fullres", device_str, len(caseids), verbose=args.verbose) as display:
        for case_idx, preprocessed in enumerate(iterator, 1):
            case_start = time_func()
            data = preprocessed["data"]
            if isinstance(data, str):
                delfile = data
                data = torch.from_numpy(np.load(data))
                os.remove(delfile)
            elif not isinstance(data, torch.Tensor):
                data = torch.from_numpy(data)
            data = data.float()

            props = preprocessed["data_properties"]
            ofile = preprocessed["ofile"]
            shape = tuple(data.shape[1:])

            if space == "world":
                points_zyx = points_to_centers_zyx(
                    points_canonical,
                    "world",
                    props,
                    shape,
                    tuple(pred.configuration_manager.spacing),
                    pred.plans_manager.transpose_forward,
                )
            else:
                points_zyx = points_canonical

            logits = pred.predict_logits_roi_mode(data, points_zyx, props, cfg)

            dice = None
            if ofile and args.labels_folder:
                caseid = os.path.basename(ofile)
                label_path = join(args.labels_folder, caseid + file_ending)
                if isfile(label_path):
                    pred_seg = convert_predicted_logits_to_segmentation_with_correct_shape(
                        logits.cpu(), pred.plans_manager, pred.configuration_manager,
                        label_manager, props, save_probabilities=False,
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

            case_time = time_func() - case_start
            display.update_case(case_idx, case_time, dice)


if __name__ == "__main__":
    predict_roi_entry_point()
