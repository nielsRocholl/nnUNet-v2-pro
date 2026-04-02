"""Single-patch inference (nnUNetROIPredictor.predict_logits_single_patch)."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

if "nnUNet_raw" not in os.environ:
    _base = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
    os.environ["nnUNet_raw"] = os.path.join(_base, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(_base, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(_base, "nnUNet_results")

from batchgenerators.utilities.file_and_folder_operations import join, load_json, load_pickle

from nnunetv2.inference.roi_predictor import nnUNetROIPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.roi_config import load_config

DUMMY_BASE = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
PREPROCESSED_DIR_LOWRES = join(
    os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
    "Dataset010",
    "nnUNetPlans_3d_lowres",
)
PREPROCESSED_DIR_FULLRES = join(
    os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
    "Dataset010",
    "nnUNetPlans_3d_fullres",
)
DATASET010_PREPROCESSED = join(
    os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
    "Dataset010",
)
PLANS_PATH = join(DATASET010_PREPROCESSED, "nnUNetPlans.json")
DATASET_JSON_PATH = join(DATASET010_PREPROCESSED, "dataset.json")
FIXTURE_CONFIG = join(Path(__file__).resolve().parent, "fixtures", "nnunet_pro_config.json")


def _get_preprocessed_case():
    for preprocessed_dir in (PREPROCESSED_DIR_FULLRES, PREPROCESSED_DIR_LOWRES):
        if not os.path.isdir(preprocessed_dir):
            continue
        try:
            import blosc2
        except ImportError:
            continue
        ids = [
            i[:-5]
            for i in os.listdir(preprocessed_dir)
            if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")
        ]
        if not ids:
            continue
        ident = ids[0]
        data = np.array(blosc2.open(join(preprocessed_dir, ident + ".b2nd"), mode="r"))
        seg = np.array(blosc2.open(join(preprocessed_dir, ident + "_seg.b2nd"), mode="r"))
        props = load_pickle(join(preprocessed_dir, ident + ".pkl"))
        config_name = "3d_fullres" if "fullres" in preprocessed_dir else "3d_lowres"
        return data, seg, props, config_name, ident
    return None


def _build_predictor(config_name: str):
    from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerPromptAware import (
        nnUNetTrainerPromptAware,
    )
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

    pm = PlansManager(PLANS_PATH)
    cm = pm.get_configuration(config_name)
    ds_json = load_json(DATASET_JSON_PATH)
    lm = pm.get_label_manager(ds_json)
    num_in = determine_num_input_channels(pm, cm, ds_json)
    network = nnUNetTrainerPromptAware.build_network_architecture(
        cm.network_arch_class_name,
        cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import,
        num_in,
        lm.num_segmentation_heads,
        enable_deep_supervision=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = nnUNetROIPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False, device=device
    )
    pred.manual_initialization(
        network, pm, cm, [network.state_dict()], ds_json, "nnUNetTrainerPromptAware", None
    )
    return pred, lm


def test_predict_logits_single_patch_empty_points_raises():
    if not os.path.isfile(PLANS_PATH) or not os.path.isfile(FIXTURE_CONFIG):
        pytest.skip("plans or fixture config missing")
    case = _get_preprocessed_case()
    if case is None:
        pytest.skip("No preprocessed data")
    data, _, _, config_name, _ = case
    data = torch.from_numpy(np.asarray(data, dtype=np.float32))
    pred, _ = _build_predictor(config_name)
    cfg = load_config(FIXTURE_CONFIG)
    with pytest.raises(ValueError, match="at least one point"):
        pred.predict_logits_single_patch(data, [], cfg)


@pytest.mark.parametrize("encode_prompt", [False, True])
def test_predict_logits_single_patch_shape_and_dtype(encode_prompt: bool):
    if not os.path.isfile(PLANS_PATH) or not os.path.isfile(FIXTURE_CONFIG):
        pytest.skip("plans or fixture config missing")
    case = _get_preprocessed_case()
    if case is None:
        pytest.skip("No preprocessed data")
    data, seg, _, config_name, _ = case
    data = torch.from_numpy(np.asarray(data, dtype=np.float32))
    pred, lm = _build_predictor(config_name)
    cfg = load_config(FIXTURE_CONFIG)
    shape = data.shape[1:]
    import cc3d

    seg_bin = (np.asarray(seg)[0] > 0).astype(np.uint8)
    labels = cc3d.connected_components(seg_bin)
    stats = cc3d.statistics(labels, no_slice_conversion=True)
    centroids = stats["centroids"][1:]
    if len(centroids) > 0:
        c = centroids[0]
        points_zyx = [(int(np.round(c[0])), int(np.round(c[1])), int(np.round(c[2])))]
    else:
        points_zyx = [(shape[0] // 2, shape[1] // 2, shape[2] // 2)]

    logits = pred.predict_logits_single_patch(
        data, points_zyx, cfg, encode_prompt=encode_prompt
    )
    assert logits.shape == (lm.num_segmentation_heads,) + shape
    assert logits.dtype == torch.float32
    assert not torch.any(torch.isinf(logits))


def test_predict_logits_single_patch_encode_prompt_changes_logits():
    if not os.path.isfile(PLANS_PATH) or not os.path.isfile(FIXTURE_CONFIG):
        pytest.skip("plans or fixture config missing")
    case = _get_preprocessed_case()
    if case is None:
        pytest.skip("No preprocessed data")
    data, seg, _, config_name, _ = case
    data = torch.from_numpy(np.asarray(data, dtype=np.float32))
    pred, _ = _build_predictor(config_name)
    cfg = load_config(FIXTURE_CONFIG)
    shape = data.shape[1:]
    import cc3d

    seg_bin = (np.asarray(seg)[0] > 0).astype(np.uint8)
    labels = cc3d.connected_components(seg_bin)
    stats = cc3d.statistics(labels, no_slice_conversion=True)
    centroids = stats["centroids"][1:]
    if len(centroids) > 0:
        c = centroids[0]
        points_zyx = [(int(np.round(c[0])), int(np.round(c[1])), int(np.round(c[2])))]
    else:
        points_zyx = [(shape[0] // 2, shape[1] // 2, shape[2] // 2)]

    z = pred.predict_logits_single_patch(data, points_zyx, cfg, encode_prompt=False)
    e = pred.predict_logits_single_patch(data, points_zyx, cfg, encode_prompt=True)
    assert z.shape == e.shape
    assert not torch.allclose(z, e)
