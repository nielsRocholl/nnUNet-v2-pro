"""ROI-mode CLI: prompt-aware inference over dilated bbox only."""
import argparse
import os

import torch
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json, maybe_mkdir_p, save_json

from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.roi_predictor import nnUNetROIPredictor, parse_points_json
from nnunetv2.utilities.roi_config import load_config
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

    points_raw, space = parse_points_json(args.points_json, args.points_space)
    if space not in ("voxel", "world"):
        raise ValueError(f"points_space must be voxel or world, got {space}")

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

    import numpy as np

    for preprocessed in iterator:
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
                points_raw,
                "world",
                props,
                shape,
                tuple(pred.configuration_manager.spacing),
                pred.plans_manager.transpose_forward,
            )
        else:
            points_zyx = [(int(round(p[0])), int(round(p[1])), int(round(p[2]))) for p in points_raw]

        logits = pred.predict_logits_roi_mode(data, points_zyx, props, cfg)

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


if __name__ == "__main__":
    predict_roi_entry_point()
