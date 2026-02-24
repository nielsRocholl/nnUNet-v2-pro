"""Analyze voxel-space distance between cog_propagated and cog_fu across longitudinal CT CSVs."""
import argparse
import csv
import math
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_INPUT = Path("/nnunet_data/universal-lesion-segmentation-original-files/Longitudinal-CT/inputsTr")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "cog_propagation_analysis_report.md"


def parse_coord(s: str) -> Optional[Tuple[float, float, float]]:
    """Parse 'x y z' string to (x, y, z). Returns None if invalid."""
    if not s or str(s).strip().lower() == "none":
        return None
    parts = str(s).strip().split()
    if len(parts) != 3:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None


def euclidean(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def load_distances(input_dir: Path) -> Tuple[list, list, list, int]:
    """Load all valid (distance, per_axis_deltas, lesion_type) from CSVs. Returns (distances, per_axis, lesion_types, skipped)."""
    distances: list = []
    per_axis: list = []
    lesion_types: list = []
    skipped = 0

    for csv_path in sorted(input_dir.glob("*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cog_prop = parse_coord(row.get("cog_propagated", ""))
                cog_fu = parse_coord(row.get("cog_fu", ""))
                if cog_prop is None or cog_fu is None:
                    skipped += 1
                    continue
                d = euclidean(cog_prop, cog_fu)
                distances.append(d)
                per_axis.append((
                    abs(cog_prop[0] - cog_fu[0]),
                    abs(cog_prop[1] - cog_fu[1]),
                    abs(cog_prop[2] - cog_fu[2]),
                ))
                lesion_types.append(row.get("lesion_type", "Unknown").strip())

    return distances, per_axis, lesion_types, skipped


def stats(values: list) -> dict:
    """Compute summary statistics."""
    if not values:
        return {}
    n = len(values)
    s = sorted(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
    std = math.sqrt(variance)

    def pct(p: float) -> float:
        k = max(1, min(math.ceil(p / 100 * n), n))
        return s[k - 1] if s else 0

    return {
        "n": n,
        "mean": mean,
        "median": pct(50),
        "std": std,
        "min": min(values),
        "max": max(values),
        "p25": pct(25),
        "p50": pct(50),
        "p75": pct(75),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
    }


def write_report(distances: list, per_axis: list, lesion_types: list, output_path: Path) -> None:
    """Write markdown report."""
    overall = stats(distances)

    dx = [e[0] for e in per_axis]
    dy = [e[1] for e in per_axis]
    dz = [e[2] for e in per_axis]
    axis_stats = {"x": stats(dx), "y": stats(dy), "z": stats(dz)}

    by_type: dict = {}
    for d, lt in zip(distances, lesion_types):
        by_type.setdefault(lt, []).append(d)
    type_stats = {k: stats(v) for k, v in sorted(by_type.items()) if v}

    lines = [
        "# COG Propagation Error Analysis",
        "",
        "Distance between `cog_propagated` and `cog_fu` in **voxels** (per lesion, per row).",
        "",
        "## Summary",
        "",
        f"- **Lesions with both COGs**: {overall['n']}",
        f"- **Mean distance (voxels)**: {overall['mean']:.2f}",
        f"- **Median distance (voxels)**: {overall['median']:.2f}",
        f"- **Std**: {overall['std']:.2f}",
        f"- **Min / Max**: {overall['min']:.2f} / {overall['max']:.2f}",
        "",
        "## Percentiles (voxels)",
        "",
        "| Percentile | Distance (vox) |",
        "|------------|----------------|",
        f"| 25th | {overall['p25']:.2f} |",
        f"| 50th | {overall['p50']:.2f} |",
        f"| 75th | {overall['p75']:.2f} |",
        f"| 90th | {overall['p90']:.2f} |",
        f"| 95th | {overall['p95']:.2f} |",
        f"| 99th | {overall['p99']:.2f} |",
        "",
        "## Per-axis (|dx|, |dy|, |dz|) in voxels",
        "",
        "| Axis | Mean | Median | Std | Max |",
        "|------|------|--------|-----|-----|",
    ]

    for ax, s in axis_stats.items():
        lines.append(f"| {ax} | {s['mean']:.2f} | {s['median']:.2f} | {s['std']:.2f} | {s['max']:.2f} |")

    lines.extend([
        "",
        "## By lesion_type",
        "",
        "| Lesion type | n | Mean (vox) | Median (vox) | Std |",
        "|-------------|---|------------|--------------|-----|",
    ])

    for lt, s in type_stats.items():
        lines.append(f"| {lt} | {s['n']} | {s['mean']:.2f} | {s['median']:.2f} | {s['std']:.2f} |")

    ax_mean = {k: axis_stats[k]["mean"] for k in axis_stats}
    isotropic_note = "roughly isotropic" if max(ax_mean.values()) < 1.5 * min(ax_mean.values()) else "anisotropic (z typically smaller)"
    lines.extend([
        "",
        "## Recommendations for prompt simulation",
        "",
        "To simulate propagated prompts during training, use a random offset from the true centroid:",
        "",
        f"- **Offset magnitude**: Sample from ~N(μ={overall['mean']:.1f}, σ={overall['std']:.1f}) voxels, or use empirical percentiles.",
        f"- **Per-axis**: Offsets are {isotropic_note}. Consider σ_x≈{axis_stats['x']['mean']:.1f}, σ_y≈{axis_stats['y']['mean']:.1f}, σ_z≈{axis_stats['z']['mean']:.1f} for anisotropic sampling.",
        f"- **Conservative bounds**: 95th percentile ≈ {overall['p95']:.1f} voxels; max observed ≈ {overall['max']:.1f} voxels.",
        "",
        "Config suggestion: add `propagated_offset_std_vox` (e.g. " + str(round(overall['std'], 1)) + ") or `propagated_offset_max_vox` (e.g. " + str(round(overall['p95'], 1)) + ") for sampling.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze cog_propagated vs cog_fu distance")
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input directory with CSVs (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output markdown path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Input directory not found: {args.input}")

    distances, per_axis, lesion_types, skipped = load_distances(args.input)
    if not distances:
        raise SystemExit("No valid pairs found. Check CSV format and paths.")

    write_report(distances, per_axis, lesion_types, args.output)
    print(f"Report written to {args.output} ({len(distances)} lesions, {skipped} rows skipped)")


if __name__ == "__main__":
    main()
