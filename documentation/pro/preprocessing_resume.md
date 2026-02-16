# Preprocessing Resume

When preprocessing fails (e.g. OOM on a cluster), use `--resume` to continue from where you left off. Cases that already have complete preprocessed output (`.b2nd`, `_seg.b2nd`, `.pkl`) are skipped.

## Usage

### After OOM or interruption

Rerun the same command with `--resume` added:

```bash
nnUNetv2_plan_and_preprocess -d 999 -pl nnUNetPlannerResEncL -c 3d_fullres -npfp 48 -np 48 --resume
```

### Reduce workers when resuming after OOM

If you hit OOM, reduce `-np` before resuming:

```bash
nnUNetv2_plan_and_preprocess -d 999 -pl nnUNetPlannerResEncL -c 3d_fullres -npfp 48 -np 24 --resume --verbose
```

### Standalone preprocess

```bash
nnUNetv2_preprocess -d 999 -plans_name nnUNetResEncUNetLPlans_h200 -c 3d_fullres -np 24 --resume
```

## Behavior

- **Without `--resume`**: Output directory is deleted; all cases are processed from scratch.
- **With `--resume`**: Existing output is kept; only cases missing `.b2nd`, `_seg.b2nd`, or `.pkl` are processed.
- **Partial case** (e.g. only `.b2nd` exists): Treated as incomplete; case is reprocessed.
- **Multi-config** (3d_fullres, 3d_lowres): Each configuration has its own output dir; resume is per-config.
