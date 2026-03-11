"""Create two minimal dummy datasets from KiTS23 for stats collection tests."""
import os
import shutil
from pathlib import Path

KITS23_SOURCE = Path(
    "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/universal-lesion-segmentation/nnUNet_raw/Dataset022_KiTS23"
)
FILE_ENDING = ".nii.gz"

# 2 cases per dataset; pick smaller files for faster tests
DATASET001_CASES = ["KiTS23_case_00004", "KiTS23_case_00007"]
DATASET002_CASES = ["KiTS23_case_00015", "KiTS23_case_00016"]


def create_dummy_dataset(output_dir: Path, name: str, case_ids: list[str]) -> None:
    out = output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "imagesTr").mkdir(exist_ok=True)
    (out / "labelsTr").mkdir(exist_ok=True)

    dataset_entries = {}
    for cid in case_ids:
        img_src = KITS23_SOURCE / "imagesTr" / f"{cid}_0000{FILE_ENDING}"
        lbl_src = KITS23_SOURCE / "labelsTr" / f"{cid}{FILE_ENDING}"
        if not img_src.exists() or not lbl_src.exists():
            raise FileNotFoundError(f"Missing {img_src} or {lbl_src}")
        shutil.copy2(img_src, out / "imagesTr" / img_src.name)
        shutil.copy2(lbl_src, out / "labelsTr" / lbl_src.name)
        dataset_entries[cid] = {
            "images": [f"imagesTr/{cid}_0000{FILE_ENDING}"],
            "label": f"labelsTr/{cid}{FILE_ENDING}",
        }

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(case_ids),
        "file_ending": FILE_ENDING,
        "overwrite_image_reader_writer": "SimpleITKIO",
        "dataset": dataset_entries,
    }
    import json
    with open(out / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2, sort_keys=False)


def main():
    base = Path(os.environ.get("NNUNET_DUMMY_BASE", "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"))
    raw = base / "nnUNet_raw"
    raw.mkdir(parents=True, exist_ok=True)
    (base / "nnUNet_preprocessed").mkdir(exist_ok=True)
    (base / "nnUNet_results").mkdir(exist_ok=True)

    create_dummy_dataset(raw, "Dataset001_Mini", DATASET001_CASES)
    create_dummy_dataset(raw, "Dataset002_Mini", DATASET002_CASES)
    print(f"Created Dataset001_Mini and Dataset002_Mini in {raw}")


if __name__ == "__main__":
    main()
