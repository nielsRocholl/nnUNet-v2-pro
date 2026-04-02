# Single-patch warm session (viewer integration)

Goal: run **nnUNetTrainerPromptAware** single-patch inference from a CT viewer with **one model load**, **one preprocess per case** (optionally in the background while the user navigates), and **per-click** forward + export—without writing a temporary points file for every click.

Scope: **Python in-process API** in this repo (`WarmSinglePatchSession`). For a subprocess line-based driver, see `nnUNetv2_predict_single_patch` with `--stdin_loop` (preprocess once per process; points JSON lines on stdin).

## API summary

| Method | Purpose |
|--------|---------|
| `initialize()` | Build `nnUNetROIPredictor`, load weights and `nnunet_pro_config.json` (same idea as the CLI). |
| `prepare_case_from_files(list_row, output_truncated)` | Synchronous preprocess for one case; fills cache. |
| `prepare_case_from_files_async(list_row, output_truncated)` | Same in a daemon thread; returns immediately. |
| `wait_for_preprocessed(timeout=None)` | Block until cache ready, failure recorded, or prepare superseded. |
| `clear_case()` | Drop cache and bump generation so **stale** background work cannot repopulate the slot. |
| `predict(point_spec, *, encode_prompt=False, points_space_override=None, export=True, save_debug_patch=None, save_debug_patch_prompts=False, case_id=None, wait_timeout=None, border_expand=False, max_border_expand_extra=16)` | Uses cached preprocess; optional `case_id` must match the cached paths tuple or `ValueError`. Returns exported path `{ofile}.{file_ending}` or `None` if `export=False`/`ofile` missing. With **`border_expand=True`**, runs **iterative** hull-based expansion (multi-face planning, BFS, cap `max_border_expand_extra`), merges logits with Gaussian weighting, then exports (DICE/debug apply to merged output). |

`point_spec` is a **`dict`** with the same keys as a points JSON file (`points`, `points_space`, optional `points_format`, `voxel_coordinate_frame`, `debug_patch_bbox_pad`). Helpers: `points_dict_to_canonical` / `parse_points_json` in `roi_predictor.py`.

Shared forward path: `run_single_patch_forward` in `single_patch_session.py` (same centre mapping + `predict_logits_single_patch` + export stack as the CLI).

**Border expansion:** set `border_expand=True` on `predict()` for large lesions whose mask would otherwise sit flush on the patch border. Perimeter `cc3d` shell labels drive **multi-face** plans (each contacting hull face); a queue runs extra tiles in rounds until exhausted or `max_border_expand_extra` (default 16) additional forwards. Progress callbacks use total `1 + max_border_expand_extra` as an upper bound. Prompt encoding still uses the original click (`local_prompt_points_for_patch`).

## Recommended viewer flow

1. **Inference mode on** → construct `WarmSinglePatchSession` and call **`initialize()`** once.
2. **Sample loaded** → `clear_case()` then **`prepare_case_from_files_async(...)`** (or sync `prepare_case_from_files` if you accept UI block).
3. **Click** → **`predict({...})`** (exactly one point). If preprocess is still running, **`predict`** waits up to `wait_timeout` via `wait_for_preprocessed`.
4. **Sample closed / switched** → **`clear_case()`** then start prepare for the new case.
5. **Leave inference mode** → `clear_case()`; drop the session reference.

## Thread safety

Use **one session per nnU-Net worker**. Do not interleave `prepare_*` from multiple threads without external locking; the viewer main thread serializing calls is enough.

## Coordinates

Voxel vs world, `voxel_coordinate_frame` (`full` vs `preprocessed`), and `points_format` behave like ROI / single-patch CLI. See [prompt_aware_guide.md](prompt_aware_guide.md) (points.json format and coordinate validation).

## CLI vs Python session

| Use case | Approach |
|----------|----------|
| Batch or shell scripts | `nnUNetv2_predict_single_patch` with `--points_json` / `--points_inline` / `--point_zyx` |
| One long-lived process, many clicks on **one** preprocessed case | `--stdin_loop` after `-i` **single file**; one JSON object per stdin line |
| **Background preprocess** before click, warm weights in memory | **`WarmSinglePatchSession`** (`prepare_case_from_files_async` + `predict`) |

`--stdin_loop` does **not** prefetch the next case; it only avoids reloading weights between stdin lines.

## Pseudocode (viewer)

```text
on_inference_mode_on:
    session = WarmSinglePatchSession(model_folder, device=...)
    session.initialize()

on_sample_loaded(paths, output_truncated):
    session.clear_case()
    session.prepare_case_from_files_async(paths, output_truncated)

on_click(point_dict):
    path = session.predict(
        point_dict,
        encode_prompt=user_wants_prompt,
        border_expand=user_wants_border_expand,
    )
    update_overlay(path)

on_sample_closed():
    session.clear_case()
```

## Caveats

- **Same path, file changed on disk**: cache is not invalidated. Call `clear_case()` and prepare again after reload.
- **Cancelled async preprocess**: clearing does not stop native code mid-flight; completion is ignored if the generation no longer matches.
- **Arrays / in-memory volumes**: not in this API; a future `prepare_case_from_arrays` would need `data_properties` consistent with `SimpleITKIO.read_images` / nnU-Net preprocess output (risk of silent mis-registration if wrong).
