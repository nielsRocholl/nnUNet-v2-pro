# Cluster training: workers, I/O, and interpreting “slow” epochs

nnUNet v2 keeps the GPU fed with a **multiprocessing** pipeline (batchgenerators `NonDetMultiThreadedAugmenter`): many worker **processes** read preprocessed data (often Blosc2 `.b2nd`), decompress, run **CPU-heavy** augmentation, and push batches to a queue. On **shared nodes** with **ZFS** or busy disks, that becomes **disk + CPU contention** — GPU utilization drops toward 0% while the machine still looks busy. This is **upstream nnUNet behavior**, amplified by large datasets and by **not** setting `nnUNet_n_proc_DA` (hostname rules can pick **48** on `superh200`).

Prompt-aware training adds extra CPU work in the dataloader; the **multi-process + random read** pattern is the same as vanilla nnUNet.

---

## How many processes should one GPU job create?

Let **N** = `get_allowed_n_proc_DA()` (see env `nnUNet_n_proc_DA` below).

| Stage | Count | Where |
|--------|--------|--------|
| Training augmenter workers | **N** | `nnUNetTrainer.get_dataloaders` |
| Validation augmenter workers | **max(1, N // 2)** | same |
| Unpack at `on_train_start` (parallel read/write burst) | **max(1, round(N // 2))** | `nnUNetTrainer.on_train_start` |

So **one** `nnUNetv2_train` process can produce **about N + N/2** long-lived worker processes (plus the main process). Each worker is often a **fork** with the **same argv** as the parent, so `htop` may show **many identical `nnUNetv2_train` lines** — that is not automatically “duplicate jobs.”

**Sanity check on the cluster:**

```bash
pstree -ap <PID_OF_ONE_NNUNET_LINE>
squeue -u "$USER"
```

---

## Environment variables

| Variable | Effect |
|----------|--------|
| **`nnUNet_n_proc_DA`** | If set, **overrides** hostname table in `default_n_proc_DA.py`. **0** ⇒ single-threaded augmenter (good diagnostic; slower). |
| **`nnUNet_def_n_proc`** | Unrelated knob: `default_num_processes` for **other** multiprocessing (e.g. unpacking helpers in `configuration.py`), default **8**. |

**Torch/BLAS in workers:** In SLURM scripts, keep `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc., so each worker does not multiply threads.

**Local preview** (same logic as training; respects `nnUNet_n_proc_DA` and hostname):

```bash
python scripts/print_augmenter_worker_budget.py
nnUNet_n_proc_DA=8 python scripts/print_augmenter_worker_budget.py
```

---

## Code map (nnUNet-pro)

- [`nnunetv2/utilities/default_n_proc_DA.py`](../../nnunetv2/utilities/default_n_proc_DA.py) — hostname table (`superh200` → 48), `min(..., os.cpu_count())`.
- [`nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`](../../nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) — augmenters, `save_every`, checkpoint saves, `epoch_time` logging.
- [`nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainerPromptAware.py`](../../nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainerPromptAware.py) — same augmenter pattern.
- [`nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py`](../../nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py) — `torch.set_num_threads(get_allowed_n_proc_DA())` during planning only.
- [`nnunetv2/configuration.py`](../../nnunetv2/configuration.py) — imports `get_allowed_n_proc_DA()` at import time for `default_n_proc_DA` (training code typically calls the function again when building dataloaders).

---

## Where `epoch_time` comes from (W&B / logs)

`epoch_time` is **wall-clock for the whole epoch**: `num_iterations_per_epoch` training steps (default **250**) + `num_val_iterations_per_epoch` validation batches (default **50**) + epoch end work (logging, `plot_progress_png`, **checkpoint** `torch.save` when `save_every` or new best, W&B sync).

Large spikes without a smooth drift often point to **I/O** (checkpoints, shared disk, neighbors) not only “the model got heavier.” See [wandb integration: epoch_time](wandb_integration.md#epoch_time-slow-epochs-and-spikes).

---

## Paste-ready cluster diagnostics

```bash
# CPU allocation vs what Python sees (run inside the same job/container)
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unset}"
taskset -cp $$
python -c "import os; print('os.cpu_count():', os.cpu_count())"

# Disk / ZFS pressure (node-dependent)
iostat -xz 1

# GPU vs stall
nvidia-smi dmon -s u -d 1
```

**Quick A/B:** For a short test, `export nnUNet_n_proc_DA=0` or `4`. If GPU util stabilizes and `epoch_time` variance drops sharply, contention is mainly **workers + I/O**, not the GPU kernel.

---

## SLURM / Docker checklist

- **`--cpus-per-task`**: Should exceed **N + N/2** workers + headroom **only if** you intend to run that many workers; otherwise **lower N** instead of raising CPUs without bound.
- **Container CPU visibility:** If `os.cpu_count()` equals the **whole node** but SLURM only allocated a slice, you still cap workers with `min(N, os.cpu_count())` — which can be **too many** for your **cgroup**. Set **`nnUNet_n_proc_DA`** explicitly to match the **allocated** CPUs.
- **Data path:** Prefer **job-local scratch / NVMe** for `nnUNet_preprocessed`. If data sits on **shared ZFS**, expect ARC growth and noisy latency when neighbors run heavy I/O.
- **IPC / shared memory:** Multiprocessing queues may use `/dev/shm`. Tiny `shm` or missing `--ipc=host` (site policy) can cause instability; see [nnUNet discussion](https://github.com/MIC-DKFZ/nnUNet/issues/696).

Example training scripts in this repo: [`scripts/train_999_prompt_aware.sh`](../../scripts/train_999_prompt_aware.sh), [`scripts/train_999_vanilla.sh`](../../scripts/train_999_vanilla.sh).

---

## Recommended starting points

| Situation | `nnUNet_n_proc_DA` |
|-----------|---------------------|
| Shared node, laggy disk, many tenants | **4–8** (raise only while watching GPU util) |
| Exclusive node, fast local NVMe | **12–24** (tune upward) |
| Untuned hostname default on `superh200` | **48** if unset — often too high for **multi-tenant** I/O |

This repository does not ship the cluster Docker image Dockerfile; validate **`/dev/shm`**, mounts, and CPU affinity in the image your site builds.
