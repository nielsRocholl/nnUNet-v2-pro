"""Step 3: Prompt Extraction and Encoding — centroids from seg → heatmap [0,1]."""
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

from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p

from nnunetv2.utilities.prompt_encoding import (
    build_prompt_channel,
    encode_points_to_heatmap,
    extract_centroids_from_seg,
    filter_centroids_in_patch,
)
from nnunetv2.utilities.roi_config import load_config

STEP03_OUTPUT_DIR = "tests/outputs/step03"


def _center_crop_bbox(center_zyx, patch_size, shape):
    cz, cy, cx = center_zyx
    pz, py, px = patch_size[:3]
    z0 = max(0, cz - pz // 2)
    z1 = min(shape[0], z0 + pz)
    z0 = max(0, z1 - pz)
    y0 = max(0, cy - py // 2)
    y1 = min(shape[1], y0 + py)
    y0 = max(0, y1 - py)
    x0 = max(0, cx - px // 2)
    x1 = min(shape[2], x0 + px)
    x0 = max(0, x1 - px)
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))
FIXTURE_CONFIG = join(Path(__file__).resolve().parent, "fixtures", "nnunet_pro_config.json")
DUMMY_BASE = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
TEST_IMAGE = join(DUMMY_BASE, "nnUNet_raw", "Dataset010", "imagesTr", "MSD_Liver_liver_1_0000.nii.gz")
TEST_LABEL = join(DUMMY_BASE, "nnUNet_raw", "Dataset010", "labelsTr", "MSD_Liver_liver_1.nii.gz")


def test_extract_centroids_binary():
    seg = np.zeros((1, 30, 30, 30), dtype=np.uint8)
    seg[0, 10:15, 10:15, 10:15] = 1
    seg[0, 20:25, 20:25, 20:25] = 1
    centroids = extract_centroids_from_seg(seg)
    assert len(centroids) == 2
    assert all(len(c) == 3 for c in centroids)
    c0, c1 = centroids
    assert 12 <= c0[0] <= 13 and 12 <= c0[1] <= 13 and 12 <= c0[2] <= 13
    assert 22 <= c1[0] <= 23 and 22 <= c1[1] <= 23 and 22 <= c1[2] <= 23


def test_extract_centroids_instance():
    seg = np.zeros((1, 20, 20, 20), dtype=np.uint8)
    seg[0, 2:6, 2:6, 2:6] = 1
    seg[0, 12:16, 12:16, 12:16] = 2
    centroids = extract_centroids_from_seg(seg)
    assert len(centroids) == 2
    assert all(len(c) == 3 for c in centroids)


def test_extract_centroids_empty():
    seg = np.zeros((1, 10, 10, 10), dtype=np.uint8)
    assert extract_centroids_from_seg(seg) == []


def test_filter_centroids_in_patch():
    centroids = [(5, 10, 15), (20, 25, 30), (8, 12, 18)]
    slz, sly, slx = slice(0, 16), slice(5, 20), slice(10, 25)
    out = filter_centroids_in_patch(centroids, (slz, sly, slx))
    assert len(out) == 2
    assert (5, 10 - 5, 15 - 10) in out
    assert (8, 12 - 5, 18 - 10) in out
    assert (20, 25, 30) not in [(o[0] + slz.start, o[1] + sly.start, o[2] + slx.start) for o in out]


def test_encode_points_single():
    heatmap = encode_points_to_heatmap([(10, 10, 10)], (32, 32, 32), 3, "binary")
    assert heatmap.shape == (32, 32, 32)
    assert heatmap.dtype == torch.float32
    assert 0 <= heatmap.min().item() <= 1 and 0 <= heatmap.max().item() <= 1
    assert heatmap[10, 10, 10] > 0


def test_encode_points_multiple_merge():
    heatmap = encode_points_to_heatmap(
        [(5, 5, 5), (5, 5, 6)],
        (16, 16, 16),
        2,
        "binary",
    )
    assert heatmap.max().item() > 0
    assert heatmap.min().item() >= 0


def test_encode_points_zero():
    heatmap = encode_points_to_heatmap([], (16, 16, 16), 3, "binary")
    assert heatmap.shape == (16, 16, 16)
    assert heatmap.sum().item() == 0


def test_encode_points_edt():
    heatmap = encode_points_to_heatmap([(8, 8, 8)], (20, 20, 20), 4, "edt")
    assert heatmap.shape == (20, 20, 20)
    assert 0 <= heatmap.min().item() <= 1 and 0 <= heatmap.max().item() <= 1
    assert heatmap[8, 8, 8] >= 0.99


def test_build_prompt_channel_full_pipeline():
    seg = np.zeros((1, 24, 24, 24), dtype=np.uint8)
    seg[0, 8:14, 8:14, 8:14] = 1
    patch_slices = (slice(0, 24), slice(0, 24), slice(0, 24))
    cfg = load_config(FIXTURE_CONFIG)
    out = build_prompt_channel(seg, patch_slices, cfg)
    assert out.shape == (1, 24, 24, 24)
    assert out.dtype == torch.float32
    assert out.max().item() > 0


def test_build_prompt_channel_zero():
    seg = np.zeros((1, 16, 16, 16), dtype=np.uint8)
    patch_slices = (slice(0, 16), slice(0, 16), slice(0, 16))
    cfg = load_config(FIXTURE_CONFIG)
    out = build_prompt_channel(seg, patch_slices, cfg)
    assert out.shape == (1, 16, 16, 16)
    assert out.sum().item() == 0


def test_step03_visual_output():
    """Save NIfTIs for CT viewer. Prefer preprocessed 3d_fullres, else raw."""
    preprocessed_dir = join(
        os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
        "Dataset010",
        "nnUNetPlans_3d_fullres",
    )
    use_preprocessed = os.path.isdir(preprocessed_dir)
    if use_preprocessed:
        try:
            import blosc2
            from batchgenerators.utilities.file_and_folder_operations import load_pickle
        except ImportError:
            use_preprocessed = False
        else:
            ids = [
                i[:-5]
                for i in os.listdir(preprocessed_dir)
                if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")
            ]
            if not ids:
                use_preprocessed = False

    if use_preprocessed:
        ident = ids[0]
        data = np.array(blosc2.open(join(preprocessed_dir, ident + ".b2nd"), mode="r"))
        seg = np.array(blosc2.open(join(preprocessed_dir, ident + "_seg.b2nd"), mode="r"))
        props = load_pickle(join(preprocessed_dir, ident + ".pkl"))
        plans = load_json(join(os.path.dirname(preprocessed_dir), "nnUNetPlans.json"))
        spacing = tuple(plans["configurations"]["3d_fullres"]["spacing"])
    else:
        if not os.path.isfile(TEST_IMAGE) or not os.path.isfile(TEST_LABEL):
            pytest.skip("Test case MSD_Liver_liver_1 not found")
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

        io = SimpleITKIO()
        data, props = io.read_images([TEST_IMAGE])
        seg, _ = io.read_seg(TEST_LABEL)
        plans = load_json(
            join(
                os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
                "Dataset010",
                "nnUNetPlans.json",
            )
        )
        spacing = tuple(plans["configurations"]["3d_fullres"]["spacing"])

    shape = data.shape[1:]
    patch_size = tuple(plans["configurations"]["3d_fullres"]["patch_size"])

    import cc3d

    seg_bin = (seg[0] > 0).astype(np.uint8)
    labels = cc3d.connected_components(seg_bin)
    stats = cc3d.statistics(labels, no_slice_conversion=True)
    centroids = stats["centroids"][1:]
    center = (
        (int(np.round(centroids[0][0])), int(np.round(centroids[0][1])), int(np.round(centroids[0][2])))
        if len(centroids) > 0
        else (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    )

    cfg = load_config(FIXTURE_CONFIG)
    slices = _center_crop_bbox(center, patch_size, shape)
    roi_img = data[..., slices[0], slices[1], slices[2]]
    roi_seg = seg[..., slices[0], slices[1], slices[2]]
    roi_mask = np.zeros(shape, dtype=np.uint8)
    roi_mask[slices[0], slices[1], slices[2]] = 1

    patch_slices_full = (
        slice(0, roi_seg.shape[1]),
        slice(0, roi_seg.shape[2]),
        slice(0, roi_seg.shape[3]),
    )
    prompt_channel = build_prompt_channel(roi_seg, patch_slices_full, cfg)
    prompt_np = prompt_channel[0].numpy()

    full_prompt = np.zeros(shape, dtype=np.float32)
    full_prompt[slices[0], slices[1], slices[2]] = prompt_np

    prompt_vis = (np.clip(prompt_np, 0, 1) * 255).astype(np.uint8)
    full_prompt_vis = np.zeros(shape, dtype=np.uint8)
    full_prompt_vis[slices[0], slices[1], slices[2]] = prompt_vis

    proj_root = Path(__file__).resolve().parent.parent
    out_dir = join(proj_root, STEP03_OUTPUT_DIR)
    maybe_mkdir_p(out_dir)

    geometry_template = None
    if os.path.isfile(TEST_IMAGE):
        import SimpleITK as sitk

        geometry_template = sitk.ReadImage(TEST_IMAGE)

    def save_with_sitk(arr, name, geom_props, crop_starts=None):
        import SimpleITK as sitk

        out_path = join(out_dir, name)
        if "label" in name:
            arr = np.asarray(arr, dtype=np.float32)
            arr[arr < 0] = 0
            arr = arr.astype(np.uint8)
        arr_typed = arr if arr.dtype in (np.uint8, np.uint16) else arr.astype(np.float32)
        img = sitk.GetImageFromArray(arr_typed)
        if geometry_template is not None and arr.shape == geometry_template.GetSize()[::-1]:
            img.CopyInformation(geometry_template)
            if crop_starts is not None:
                st = geom_props["sitk_stuff"]
                sp = st["spacing"]
                orig = list(st["origin"])
                orig[0] += crop_starts[2] * sp[0]
                orig[1] += crop_starts[1] * sp[1]
                orig[2] += crop_starts[0] * sp[2]
                img.SetOrigin(orig)
        else:
            st = geom_props["sitk_stuff"]
            img.SetSpacing(st["spacing"])
            img.SetOrigin(st["origin"] if crop_starts is None else _crop_origin(st, crop_starts))
            img.SetDirection(st["direction"])
        sitk.WriteImage(img, out_path, True)

    def _crop_origin(st, crop_starts):
        orig = list(st["origin"])
        sp = st["spacing"]
        orig[0] += crop_starts[2] * sp[0]
        orig[1] += crop_starts[1] * sp[1]
        orig[2] += crop_starts[0] * sp[2]
        return orig

    crop_starts = (slices[0].start, slices[1].start, slices[2].start)
    if props.get("sitk_stuff"):
        save_with_sitk(data[0], "full_image.nii.gz", props)
        save_with_sitk(seg[0], "label.nii.gz", props)
        save_with_sitk(roi_img[0], "roi_crop.nii.gz", props, crop_starts)
        save_with_sitk(roi_seg[0], "roi_crop_label.nii.gz", props, crop_starts)
        save_with_sitk(prompt_np, "roi_crop_prompt.nii.gz", props, crop_starts)
        save_with_sitk(full_prompt, "full_prompt.nii.gz", props)
        save_with_sitk(prompt_vis, "roi_crop_prompt_vis.nii.gz", props, crop_starts)
        save_with_sitk(full_prompt_vis, "full_prompt_vis.nii.gz", props)
    else:
        import nibabel as nib

        for arr, name in [
            (data[0], "full_image.nii.gz"),
            (seg[0], "label.nii.gz"),
            (roi_img[0], "roi_crop.nii.gz"),
            (roi_seg[0], "roi_crop_label.nii.gz"),
            (prompt_np, "roi_crop_prompt.nii.gz"),
            (full_prompt, "full_prompt.nii.gz"),
            (prompt_vis, "roi_crop_prompt_vis.nii.gz"),
            (full_prompt_vis, "full_prompt_vis.nii.gz"),
        ]:
            nib.save(nib.Nifti1Image(arr, np.eye(4)), join(out_dir, name))

    with open(join(out_dir, "README.txt"), "w") as f:
        f.write("Step 3 visual outputs — prompt heatmap overlay for CT viewer\n")
        f.write("roi_crop.nii.gz            ROI image crop\n")
        f.write("roi_crop_label.nii.gz      segmentation cropped to ROI\n")
        f.write("roi_crop_prompt_vis.nii.gz prompt heatmap uint8 [0-255] — use this for overlay\n")
        f.write("full_prompt_vis.nii.gz     same at full volume (overlay on full_image)\n")

    assert os.path.exists(join(out_dir, "roi_crop_prompt_vis.nii.gz"))
    assert os.path.exists(join(out_dir, "full_prompt_vis.nii.gz"))
