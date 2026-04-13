"""Warm single-patch session: one loaded model + cached preprocess per case; optional async prepare."""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import isfile, join

from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.roi_predictor import nnUNetROIPredictor, points_dict_to_canonical
from nnunetv2.utilities.roi_config import RoiPromptConfig, load_config
from nnunetv2.utilities.roi_coordinate_validation import validate_and_convert_points
from nnunetv2.utilities.inference_execution import apply_inference_execution_env
from nnunetv2.utilities.roi_geometry import points_to_centers_zyx


def _spacing_zyx(props: dict, pm, cm) -> Tuple[float, float, float]:
    spacing_transposed = [props["spacing"][i] for i in pm.transpose_forward]
    if len(cm.spacing) == len(props["shape_after_cropping_and_before_resampling"]):
        t = tuple(cm.spacing)
    else:
        t = tuple([spacing_transposed[0], *cm.spacing])
    return (float(t[0]), float(t[1]), float(t[2]))


def _data_to_float_tensor(data: Union[torch.Tensor, str, np.ndarray]) -> torch.Tensor:
    if isinstance(data, str):
        arr = np.load(data)
        try:
            os.remove(data)
        except OSError:
            pass
        return torch.from_numpy(arr).float()
    if isinstance(data, torch.Tensor):
        return data.float()
    return torch.from_numpy(np.asarray(data)).float()


def run_single_patch_forward(
    pred: nnUNetROIPredictor,
    cfg: RoiPromptConfig,
    data: torch.Tensor,
    props: dict,
    points_canonical: List[Tuple[int, int, int]],
    points_points_space: str,
    points_format: str,
    voxel_frame: str,
    encode_prompt: bool,
    save_debug_patch: Optional[str],
    debug_patch_spacing_zyx: Optional[Tuple[float, float, float]],
    debug_native_geometry: Optional[Dict[str, Any]],
    save_debug_patch_prompts: bool,
    border_expand: bool = False,
    max_border_expand_extra: int = 16,
) -> torch.Tensor:
    shape = tuple(data.shape[1:])
    points_zyx = points_to_centers_zyx(
        points_canonical,
        points_points_space,
        props,
        shape,
        tuple(pred.configuration_manager.spacing),
        pred.plans_manager.transpose_forward,
        voxel_coordinate_frame=voxel_frame,
    )
    return pred.predict_logits_single_patch(
        data,
        points_zyx,
        cfg,
        encode_prompt=encode_prompt,
        save_debug_patch=save_debug_patch,
        debug_patch_spacing_zyx=debug_patch_spacing_zyx,
        save_debug_patch_prompts=save_debug_patch_prompts,
        debug_native_geometry=debug_native_geometry,
        border_expand=border_expand,
        max_border_expand_extra=max_border_expand_extra,
    )


class WarmSinglePatchSession:
    def __init__(
        self,
        model_folder: str,
        folds: Optional[Union[Tuple, List]] = None,
        checkpoint_name: str = "checkpoint_final.pth",
        device: Optional[torch.device] = None,
        config_path: Optional[str] = None,
        tile_step_size: Optional[float] = None,
        use_mirroring: bool = True,
        perform_everything_on_device: bool = True,
        verbose: bool = False,
        num_processes_preprocessing: int = 1,
    ):
        self.model_folder = model_folder
        self.folds = list(folds) if folds is not None else [0]
        self.checkpoint_name = checkpoint_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = config_path or join(model_folder, "nnunet_pro_config.json")
        if not isfile(self.config_path):
            raise FileNotFoundError(self.config_path)
        self.cfg: Optional[RoiPromptConfig] = None
        self.pred: Optional[nnUNetROIPredictor] = None
        self._use_mirroring = use_mirroring
        self._tile_step_size = tile_step_size
        self._perform_everything_on_device = perform_everything_on_device
        self.verbose = verbose
        self.npp = max(1, int(num_processes_preprocessing))

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._prep_gen = 0
        self._last_finished_gen = 0
        self._case_id: Optional[Tuple[str, ...]] = None
        self._cache: Optional[Dict[str, Any]] = None
        self._prep_exc: Optional[BaseException] = None
        self._prep_thread: Optional[threading.Thread] = None

    def initialize(self) -> None:
        cfg = load_config(self.config_path)
        self.cfg = cfg
        use_mirroring = self._use_mirroring and not cfg.inference.disable_tta_default
        ts = self._tile_step_size if self._tile_step_size is not None else cfg.inference.tile_step_size
        pred = nnUNetROIPredictor(
            tile_step_size=ts,
            use_gaussian=True,
            use_mirroring=use_mirroring,
            perform_everything_on_device=self._perform_everything_on_device and self.device.type != "cpu",
            device=self.device,
            verbose=self.verbose,
            verbose_preprocessing=self.verbose,
            allow_tqdm=True,
        )
        pred.initialize_from_trained_model_folder(self.model_folder, self.folds, self.checkpoint_name)
        apply_inference_execution_env(self.device)
        self.pred = pred

    def set_use_mirroring(self, enabled: bool) -> None:
        """Enable/disable mirroring TTA on subsequent predict calls (honours cfg.inference.disable_tta_default)."""
        self._use_mirroring = bool(enabled)
        if self.pred is not None and self.cfg is not None:
            self.pred.use_mirroring = bool(enabled) and not self.cfg.inference.disable_tta_default

    def clear_case(self) -> None:
        with self._cv:
            self._prep_gen += 1
            self._case_id = None
            self._cache = None
            self._prep_exc = None
            self._cv.notify_all()

    def _do_preprocess(self, list_row: List[str], output_truncated: str) -> Dict[str, Any]:
        assert self.pred is not None
        pin = self.device.type == "cuda" and torch.cuda.is_available()
        it = preprocessing_iterator_fromfiles(
            [list_row],
            None,
            [output_truncated],
            self.pred.plans_manager,
            self.pred.dataset_json,
            self.pred.configuration_manager,
            min(self.npp, 1),
            pin,
            self.verbose,
        )
        return next(it)

    def _finish_prep_attempt(self, my_gen: int) -> None:
        self._last_finished_gen = my_gen
        self._cv.notify_all()

    def _commit_cache(self, my_gen: int, item: Dict[str, Any]) -> None:
        with self._cv:
            if my_gen != self._prep_gen:
                self._finish_prep_attempt(my_gen)
                return
            self._cache = {
                "data": _data_to_float_tensor(item["data"]),
                "data_properties": item["data_properties"],
                "ofile": item["ofile"],
            }
            self._prep_exc = None
            self._finish_prep_attempt(my_gen)

    def prepare_case_from_files(self, list_row: List[str], output_truncated: str) -> None:
        if self.pred is None:
            raise RuntimeError("call initialize() first")
        cid = tuple(os.path.abspath(p) for p in list_row)
        with self._cv:
            self._prep_gen += 1
            my_gen = self._prep_gen
            self._case_id = cid
            self._cache = None
            self._prep_exc = None
        try:
            item = self._do_preprocess(list_row, output_truncated)
        except BaseException as e:
            with self._cv:
                if my_gen == self._prep_gen:
                    self._prep_exc = e
                self._finish_prep_attempt(my_gen)
            raise
        self._commit_cache(my_gen, item)

    def prepare_case_from_files_async(self, list_row: List[str], output_truncated: str) -> None:
        if self.pred is None:
            raise RuntimeError("call initialize() first")
        cid = tuple(os.path.abspath(p) for p in list_row)

        def worker(my_gen: int, row: List[str], otr: str) -> None:
            try:
                item = self._do_preprocess(row, otr)
            except BaseException as e:
                with self._cv:
                    if my_gen == self._prep_gen:
                        self._prep_exc = e
                    self._finish_prep_attempt(my_gen)
                return
            self._commit_cache(my_gen, item)

        with self._cv:
            self._prep_gen += 1
            my_gen = self._prep_gen
            self._case_id = cid
            self._cache = None
            self._prep_exc = None
            t = threading.Thread(target=worker, args=(my_gen, list_row, output_truncated), daemon=True)
            self._prep_thread = t
            t.start()

    def wait_for_preprocessed(self, timeout: Optional[float] = None) -> None:
        with self._cv:
            if self._case_id is None and self._cache is None:
                raise RuntimeError("no case prepared")
            g = self._prep_gen
            deadline = time.monotonic() + timeout if timeout is not None else None
            while True:
                if self._prep_gen != g:
                    raise RuntimeError("prepare superseded (clear_case or new prepare)")
                exc = self._prep_exc
                if exc is not None:
                    raise exc
                if self._cache is not None:
                    return
                if self._last_finished_gen >= g:
                    raise RuntimeError("preprocessing ended without cache (cancelled)")
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("preprocessing did not finish in time")
                self._cv.wait(remaining)

    def predict(
        self,
        point_spec: Dict[str, Any],
        *,
        encode_prompt: bool = False,
        points_space_override: Optional[str] = None,
        export: bool = True,
        save_debug_patch: Optional[str] = None,
        save_debug_patch_prompts: bool = False,
        case_id: Optional[Tuple[str, ...]] = None,
        wait_timeout: Optional[float] = None,
        border_expand: bool = False,
        max_border_expand_extra: int = 16,
    ) -> Optional[str]:
        if self.pred is None or self.cfg is None:
            raise RuntimeError("call initialize() first")
        with self._cv:
            if self._case_id is None and self._cache is None:
                raise RuntimeError("no case prepared")
            needs_wait = self._cache is None
        if needs_wait:
            self.wait_for_preprocessed(wait_timeout)
        with self._cv:
            cache = self._cache
            expected_cid = self._case_id
        if cache is None:
            raise RuntimeError("no preprocessed case; call prepare_case_from_files or prepare_case_from_files_async")
        if case_id is not None:
            got = tuple(os.path.abspath(p) for p in case_id)
            if expected_cid is not None and got != expected_cid:
                raise ValueError(f"case_id mismatch: cache is {expected_cid}, got {got}")

        points_raw, space, fmt, voxel_frame, debug_pad = points_dict_to_canonical(
            point_spec, points_space_override
        )
        points_canonical = validate_and_convert_points(points_raw, space, fmt)
        if len(points_canonical) != 1:
            raise ValueError(f"exactly one point required, got {len(points_canonical)}")

        props = cache["data_properties"]
        ofile = cache["ofile"]
        data = cache["data"]

        debug_out = save_debug_patch
        debug_sp = None
        debug_geom = None
        if debug_out:
            debug_sp = _spacing_zyx(props, self.pred.plans_manager, self.pred.configuration_manager)
            debug_geom = {
                "properties": props,
                "plans_manager": self.pred.plans_manager,
                "configuration_manager": self.pred.configuration_manager,
            }
            if debug_pad is not None:
                debug_geom["debug_bbox_pad_vox"] = debug_pad

        logits = run_single_patch_forward(
            self.pred,
            self.cfg,
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
            border_expand=border_expand,
            max_border_expand_extra=max_border_expand_extra,
        )

        if export and ofile:
            export_prediction_from_logits(
                logits.cpu().numpy(),
                props,
                self.pred.configuration_manager,
                self.pred.plans_manager,
                self.pred.dataset_json,
                ofile,
                save_probabilities=False,
            )
            # Must match export_prediction_from_logits: output_file_truncated + file_ending (often ".nii.gz").
            fe = self.pred.dataset_json["file_ending"]
            return ofile + fe
        return None
