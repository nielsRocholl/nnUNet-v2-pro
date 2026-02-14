"""Merge multiple nnUNet datasets into one virtual dataset for joint fingerprint/plan/preprocess."""
import os
from typing import List

from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p, save_json

from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


def create_merged_dataset(dataset_ids: List[int], output_folder: str) -> str:
    """
    Create a virtual merged dataset in nnUNet_raw/{output_folder}/ with dataset.json
    pointing to all cases from the source datasets. No raw files are copied.
    """
    target_dir = join(nnUNet_raw, output_folder)
    maybe_mkdir_p(target_dir)

    merged_dataset = {}
    template_json = None
    num_training = 0

    for did in dataset_ids:
        source_name = convert_id_to_dataset_name(did)
        raw_folder = join(nnUNet_raw, source_name)
        ds_json = load_json(join(raw_folder, "dataset.json"))
        ds = get_filenames_of_train_images_and_targets(raw_folder, ds_json)

        if template_json is None:
            template_json = ds_json
        else:
            if ds_json["channel_names"] != template_json["channel_names"]:
                raise ValueError(
                    f"Incompatible channel_names: {source_name} vs {template_json.get('channel_names')}"
                )
            if ds_json["file_ending"] != template_json["file_ending"]:
                raise ValueError(
                    f"Incompatible file_ending: {source_name} vs {template_json.get('file_ending')}"
                )
            if ds_json["labels"] != template_json["labels"]:
                raise ValueError(
                    f"Incompatible labels: {source_name} vs {template_json.get('labels')}"
                )

        for case_id, entry in ds.items():
            prefixed_id = f"{source_name}_{case_id}"
            merged_dataset[prefixed_id] = {
                "label": os.path.relpath(entry["label"], target_dir),
                "images": [os.path.relpath(i, target_dir) for i in entry["images"]],
            }
        num_training += len(ds)

    out_json = {
        "channel_names": template_json["channel_names"],
        "labels": template_json["labels"],
        "file_ending": template_json["file_ending"],
        "numTraining": num_training,
        "dataset": merged_dataset,
    }
    for k in ("overwrite_image_reader_writer", "regions_class_order"):
        if k in template_json:
            out_json[k] = template_json[k]

    save_json(out_json, join(target_dir, "dataset.json"), sort_keys=False)
    return output_folder
