"""WarmSinglePatchSession threading + single-patch CLI point sources."""
import io
import json
import threading
from unittest.mock import MagicMock

import pytest
import torch

from nnunetv2.inference.roi_predictor import points_dict_to_canonical
from nnunetv2.inference.single_patch_predict_entrypoint import (
    inline_points_dict,
    point_zyx_inline,
)
from nnunetv2.inference.single_patch_session import WarmSinglePatchSession


def _minimal_props():
    return {"spacing": [1.0, 1.0, 1.0]}


def _fake_item(ofile: str):
    return {
        "data": torch.zeros(1, 1, 8, 8, 8),
        "data_properties": _minimal_props(),
        "ofile": ofile,
    }


@pytest.fixture
def patch_session_backend(monkeypatch):
    monkeypatch.setattr("nnunetv2.inference.single_patch_session.isfile", lambda p: True)

    def fake_init(self):
        self.cfg = MagicMock()
        self.pred = MagicMock()
        self.pred.plans_manager.transpose_forward = (0, 1, 2)
        self.pred.configuration_manager.spacing = (1.0, 1.0, 1.0)
        self.pred.dataset_json = {"file_ending": ".nii.gz"}

    monkeypatch.setattr(WarmSinglePatchSession, "initialize", fake_init)
    monkeypatch.setattr(
        "nnunetv2.inference.single_patch_session.export_prediction_from_logits",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "nnunetv2.inference.single_patch_session.run_single_patch_forward",
        lambda *a, **k: torch.zeros(2, 8, 8, 8),
    )


def test_session_two_predicts_one_preprocess(patch_session_backend, monkeypatch):
    counts = {"n": 0}

    def fake_do(self, row, otr):
        counts["n"] += 1
        return _fake_item("/out/x")

    monkeypatch.setattr(WarmSinglePatchSession, "_do_preprocess", fake_do)
    s = WarmSinglePatchSession("/fake/model", device=torch.device("cpu"))
    s.initialize()
    s.prepare_case_from_files(["/f1.nii"], "/out/x")
    pt = {"points": [[4, 4, 4]], "points_space": "voxel", "voxel_coordinate_frame": "preprocessed"}
    s.predict(pt, export=False)
    s.predict(pt, export=False)
    assert counts["n"] == 1


def test_session_clear_second_prepare_repreprocesses(patch_session_backend, monkeypatch):
    counts = {"n": 0}

    def fake_do(self, row, otr):
        counts["n"] += 1
        return _fake_item(otr)

    monkeypatch.setattr(WarmSinglePatchSession, "_do_preprocess", fake_do)
    s = WarmSinglePatchSession("/fake/model", device=torch.device("cpu"))
    s.initialize()
    s.prepare_case_from_files(["/a.nii"], "/out/a")
    s.clear_case()
    s.prepare_case_from_files(["/b.nii"], "/out/b")
    assert counts["n"] == 2
    with s._cv:
        assert s._cache["ofile"] == "/out/b"


def test_session_async_predict_blocks_until_ready(patch_session_backend, monkeypatch):
    gate = threading.Event()

    def fake_do(self, row, otr):
        gate.wait(timeout=5)
        return _fake_item("/out/z")

    monkeypatch.setattr(WarmSinglePatchSession, "_do_preprocess", fake_do)
    s = WarmSinglePatchSession("/fake/model", device=torch.device("cpu"))
    s.initialize()
    s.prepare_case_from_files_async(["/z.nii"], "/out/z")

    def run_predict():
        pt = {"points": [[1, 1, 1]], "points_space": "voxel", "voxel_coordinate_frame": "preprocessed"}
        s.predict(pt, export=False)

    th = threading.Thread(target=run_predict)
    th.start()
    gate.set()
    th.join(timeout=5)
    assert not th.is_alive()


def test_session_stale_async_worker_does_not_overwrite_cache(patch_session_backend, monkeypatch):
    release_slow = threading.Event()

    def fake_do(self, row, otr):
        if "slow" in row[0]:
            release_slow.wait(timeout=5)
            return _fake_item("STALE")
        return _fake_item("FRESH")

    monkeypatch.setattr(WarmSinglePatchSession, "_do_preprocess", fake_do)
    s = WarmSinglePatchSession("/fake/model", device=torch.device("cpu"))
    s.initialize()
    s.prepare_case_from_files_async(["/slow.nii"], "/x")
    s.clear_case()
    s.prepare_case_from_files(["/fast.nii"], "/y")
    release_slow.set()
    import time

    time.sleep(0.2)
    with s._cv:
        assert s._cache["ofile"] == "FRESH"


def test_point_zyx_matches_inline_json():
    a = point_zyx_inline("10,20,30")
    b = json.loads('{"points": [[10, 20, 30]], "points_space": "voxel", "points_format": "zyx_voxel", "voxel_coordinate_frame": "full"}')
    assert points_dict_to_canonical(a, None) == points_dict_to_canonical(b, None)

    c = inline_points_dict('{"points": [[1,2,3]], "points_space": "voxel"}')
    assert points_dict_to_canonical(c, None)[0] == [[1, 2, 3]]


def test_load_points_inline_roundtrip(monkeypatch):
    from nnunetv2.inference.single_patch_predict_entrypoint import load_points_input

    monkeypatch.setattr("sys.stdin", io.StringIO('{"points": [[5,5,5]], "points_space": "voxel", "voxel_coordinate_frame": "preprocessed"}\n'))
    d = load_points_input("-")
    assert d["points"] == [[5, 5, 5]]


def test_wait_after_clear_raises(patch_session_backend, monkeypatch):
    monkeypatch.setattr(
        WarmSinglePatchSession,
        "_do_preprocess",
        lambda self, row, otr: _fake_item("/o"),
    )
    s = WarmSinglePatchSession("/fake/model", device=torch.device("cpu"))
    s.initialize()
    s.clear_case()
    with pytest.raises(RuntimeError, match="no case prepared"):
        s.wait_for_preprocessed()
