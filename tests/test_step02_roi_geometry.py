"""Step 2: Coordinate conversion — points (voxel/world) → (z,y,x) centers."""
import os
from pathlib import Path

import numpy as np
import pytest

if "nnUNet_raw" not in os.environ:
    _base = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
    os.environ["nnUNet_raw"] = os.path.join(_base, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(_base, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(_base, "nnUNet_results")

from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p

from nnunetv2.utilities.roi_config import load_config
from nnunetv2.utilities.roi_geometry import points_to_centers_zyx

STEP02_OUTPUT_DIR = "tests/outputs/step02"
FIXTURE_CONFIG = join(Path(__file__).resolve().parent, "fixtures", "nnunet_pro_config.json")

DUMMY_BASE = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
TEST_IMAGE = join(DUMMY_BASE, "nnUNet_raw", "Dataset010", "imagesTr", "MSD_Liver_liver_1_0000.nii.gz")
TEST_LABEL = join(DUMMY_BASE, "nnUNet_raw", "Dataset010", "labelsTr", "MSD_Liver_liver_1.nii.gz")


def _center_crop_bbox(center_zyx, patch_size, shape):
    """Bbox centered on center, patch_size extent, clamped to shape."""
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


def test_load_config():
    cfg = load_config(FIXTURE_CONFIG)
    assert cfg.prompt.point_radius_vox == 5
    assert cfg.prompt.encoding == "binary"


def test_points_to_centers_voxel():
    shape = (32, 64, 64)
    centers = points_to_centers_zyx(
        [[16.2, 32.7, 31.1], [0, 0, 0], [31.9, 63.5, 63.2]],
        "voxel",
        {},
        shape,
        (1.0, 1.0, 1.0),
    )
    assert centers == [(16, 33, 31), (0, 0, 0), (31, 63, 63)]


def test_points_to_centers_voxel_clamp():
    shape = (32, 64, 64)
    centers = points_to_centers_zyx(
        [[-1, 100, 50]],
        "voxel",
        {},
        shape,
        (1.0, 1.0, 1.0),
    )
    assert centers == [(0, 63, 50)]


def test_points_to_centers_world():
    """World (x,y,z) mm → array (z,y,x) voxel. Point (16,32,31) → (31,32,16)."""
    props = {
        "sitk_stuff": {"origin": (0.0, 0.0, 0.0), "spacing": (1.0, 1.0, 1.0)},
        "bbox_used_for_cropping": [(0, 32), (0, 64), (0, 64)],
        "shape_after_cropping_and_before_resampling": (32, 64, 64),
    }
    shape = (32, 64, 64)
    spacing = (1.0, 1.0, 1.0)
    centers = points_to_centers_zyx(
        [[16.0, 32.0, 31.0]],
        "world",
        props,
        shape,
        spacing,
        transpose_forward=(0, 1, 2),
    )
    assert centers == [(31, 32, 16)]


def test_step02_visual_output():
    """Save NIfTIs for CT viewer. Use patch-sized crop around centroid."""
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
            ids = [i[:-5] for i in os.listdir(preprocessed_dir) if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")]
            if not ids:
                use_preprocessed = False

    if use_preprocessed:
        ident = ids[0]
        data = np.array(blosc2.open(join(preprocessed_dir, ident + ".b2nd"), mode="r"))
        seg = np.array(blosc2.open(join(preprocessed_dir, ident + "_seg.b2nd"), mode="r"))
        props = load_pickle(join(preprocessed_dir, ident + ".pkl"))
        plans = load_json(join(os.path.dirname(preprocessed_dir), "nnUNetPlans.json"))
        config = plans["configurations"]["3d_fullres"]
        spacing = tuple(config["spacing"])
    else:
        if not os.path.isfile(TEST_IMAGE) or not os.path.isfile(TEST_LABEL):
            pytest.skip("Test case MSD_Liver_liver_1 not found")
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
        io = SimpleITKIO()
        data, props = io.read_images([TEST_IMAGE])
        seg, _ = io.read_seg(TEST_LABEL)
        spacing = tuple(props["spacing"])

    shape = data.shape[1:]
    plans_path = join(
        os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
        "Dataset010",
        "nnUNetPlans.json",
    )
    if not os.path.isfile(plans_path):
        pytest.skip("Dataset010 nnUNetPlans.json not found")
    plans = load_json(plans_path)
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

    slices = _center_crop_bbox(center, patch_size, shape)
    roi_img = data[..., slices[0], slices[1], slices[2]]
    roi_seg = seg[..., slices[0], slices[1], slices[2]]
    roi_mask = np.zeros(shape, dtype=np.uint8)
    roi_mask[slices[0], slices[1], slices[2]] = 1

    proj_root = Path(__file__).resolve().parent.parent
    out_dir = join(proj_root, STEP02_OUTPUT_DIR)
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
        save_with_sitk(roi_mask, "roi_mask.nii.gz", props)
        save_with_sitk(roi_img[0], "roi_crop.nii.gz", props, crop_starts)
        save_with_sitk(roi_seg[0], "roi_crop_label.nii.gz", props, crop_starts)
    else:
        import nibabel as nib
        for arr, name in [
            (data[0], "full_image.nii.gz"),
            (seg[0], "label.nii.gz"),
            (roi_img[0], "roi_crop.nii.gz"),
            (roi_seg[0], "roi_crop_label.nii.gz"),
            (roi_mask, "roi_mask.nii.gz"),
        ]:
            nib.save(nib.Nifti1Image(arr, np.eye(4)), join(out_dir, name))

    with open(join(out_dir, "README.txt"), "w") as f:
        f.write("Step 2 visual outputs — MSD_Liver_liver_1, inspect in CT viewer\n")
        f.write("full_image.nii.gz     full CT — overlay with label.nii.gz or roi_mask.nii.gz\n")
        f.write("label.nii.gz          full ground-truth segmentation\n")
        f.write("roi_crop.nii.gz       patch-sized crop — overlay with roi_crop_label.nii.gz\n")
        f.write("roi_crop_label.nii.gz segmentation cropped to ROI (matches roi_crop geometry)\n")
        f.write("roi_mask.nii.gz       binary ROI region (full volume, use with full_image)\n")

    assert os.path.exists(join(out_dir, "full_image.nii.gz"))
    assert os.path.exists(join(out_dir, "roi_crop.nii.gz"))
    assert os.path.exists(join(out_dir, "roi_mask.nii.gz"))
