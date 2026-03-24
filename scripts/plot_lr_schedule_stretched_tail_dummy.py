#!/usr/bin/env python3
"""
Dummy visualization only (not used by training).

Piecewise schedule for 2500 epochs:
  - Epochs 0..k_ref-1: same LR as reference run at the same epoch index.
  - Epochs k_ref..num_epochs-1: linear from LR(k_ref) to reference last-epoch LR
    (epoch ref_epochs-1), stretched over the remaining epochs.

Default reference linear: lr(e) = lr0 * (1 - e/ref_epochs).
Also draws nnUNet-style poly (exponent 0.9) reference for comparison.

Stdlib only: writes SVG + CSV (open in browser / spreadsheet).

  python3 scripts/plot_lr_schedule_stretched_tail_dummy.py
  python3 scripts/plot_lr_schedule_stretched_tail_dummy.py --ref-mode poly --png  # PNG needs matplotlib
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def lr_linear_ref(e: float, lr0: float, ref_epochs: int) -> float:
    return lr0 * (1.0 - e / ref_epochs)


def lr_poly_ref(e: float, lr0: float, ref_epochs: int, exponent: float) -> float:
    return lr0 * (1.0 - e / ref_epochs) ** exponent


def lr_stretched_one(
    epoch: int,
    lr0: float,
    ref_epochs: int,
    num_epochs: int,
    k_ref: int,
    *,
    ref_mode: str,
    poly_exp: float,
) -> float:
    assert 0 < k_ref < ref_epochs
    assert num_epochs > k_ref
    if epoch < k_ref:
        if ref_mode == "linear":
            return lr_linear_ref(float(epoch), lr0, ref_epochs)
        return lr_poly_ref(float(epoch), lr0, ref_epochs, poly_exp)

    if ref_mode == "linear":
        lr_k = lr_linear_ref(float(k_ref), lr0, ref_epochs)
        lr_end = lr_linear_ref(float(ref_epochs - 1), lr0, ref_epochs)
    else:
        lr_k = lr_poly_ref(float(k_ref), lr0, ref_epochs, poly_exp)
        lr_end = lr_poly_ref(float(ref_epochs - 1), lr0, ref_epochs, poly_exp)

    denom = (num_epochs - 1) - k_ref
    t = (epoch - k_ref) / denom
    return lr_k + (lr_end - lr_k) * t


def series_ref_linear(ref_epochs: int, lr0: float) -> list[tuple[float, float]]:
    return [(float(e), lr_linear_ref(float(e), lr0, ref_epochs)) for e in range(ref_epochs)]


def series_ref_poly(ref_epochs: int, lr0: float, poly_exp: float) -> list[tuple[float, float]]:
    return [(float(e), lr_poly_ref(float(e), lr0, ref_epochs, poly_exp)) for e in range(ref_epochs)]


def series_proposed(
    num_epochs: int,
    lr0: float,
    ref_epochs: int,
    k_ref: int,
    ref_mode: str,
    poly_exp: float,
) -> list[tuple[float, float]]:
    return [
        (
            float(e),
            lr_stretched_one(e, lr0, ref_epochs, num_epochs, k_ref, ref_mode=ref_mode, poly_exp=poly_exp),
        )
        for e in range(num_epochs)
    ]


def polyline_points(
    pairs: list[tuple[float, float]],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    w: int,
    h: int,
) -> str:
    parts: list[str] = []
    for x, y in pairs:
        px = (x - x0) / (x1 - x0) * (w - 1)
        py = (h - 1) - (y - y0) / (y1 - y0) * (h - 1)
        parts.append(f"{px:.2f},{py:.2f}")
    return " ".join(parts)


def write_svg(
    path: Path,
    ref_lin: list[tuple[float, float]],
    ref_poly: list[tuple[float, float]],
    prop: list[tuple[float, float]],
    k_ref: int,
    lr0: float,
) -> None:
    W, H = 920, 460
    margin_l, margin_r, margin_t, margin_b = 72, 28, 36, 52
    pw, ph = W - margin_l - margin_r, H - margin_t - margin_b
    xmax = max(ref_lin[-1][0], prop[-1][0])
    ymax = lr0 * 1.05
    ymin = 0.0

    k_line_x = k_ref / xmax * (pw - 1)
    kx = k_line_x + margin_l

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{margin_l}" y="{margin_t - 8}" font-size="14" fill="#111">'
        f"LR preview: reference 0–k, then slow tail to reference final LR (k={k_ref})</text>",
        f'<g transform="translate({margin_l},{margin_t})">',
        f'<line x1="0" y1="{ph}" x2="{pw}" y2="{ph}" stroke="#333" stroke-width="1"/>',
        f'<line x1="0" y1="0" x2="0" y2="{ph}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{k_line_x:.2f}" y1="0" x2="{k_line_x:.2f}" y2="{ph}" stroke="#999" stroke-dasharray="4 4"/>',
        f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{polyline_points(ref_lin, 0, xmax, ymin, ymax, pw, ph)}"/>',
        f'<polyline fill="none" stroke="#1f77b4" stroke-dasharray="6 4" stroke-width="1.4" points="{polyline_points(ref_poly, 0, xmax, ymin, ymax, pw, ph)}"/>',
        f'<polyline fill="none" stroke="#d62728" stroke-width="2.5" points="{polyline_points(prop, 0, xmax, ymin, ymax, pw, ph)}"/>',
        "</g>",
        f'<text x="{kx:.1f}" y="{H - 14}" font-size="11" fill="#666" text-anchor="middle">k={k_ref}</text>',
        f'<text x="{W - margin_r}" y="{margin_t + 12}" font-size="11" fill="#1f77b4" text-anchor="end">blue solid: ref linear</text>',
        f'<text x="{W - margin_r}" y="{margin_t + 28}" font-size="11" fill="#1f77b4" text-anchor="end">blue dash: nnUNet poly 0.9</text>',
        f'<text x="{W - margin_r}" y="{margin_t + 44}" font-size="11" fill="#d62728" text-anchor="end">red: proposed long run</text>',
        f'<text x="{margin_l + pw // 2}" y="{H - 8}" font-size="12" fill="#111" text-anchor="middle">epoch</text>',
        "</svg>",
    ]
    path.write_text("\n".join(svg_lines), encoding="utf-8")


def write_csv(
    path: Path,
    prop: list[tuple[float, float]],
    ref_epochs: int,
    lr0: float,
    poly_exp: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lr_proposed", "lr_ref_linear_same_epoch", "lr_ref_poly_same_epoch"])
        for e, lr_p in prop:
            ei = int(e)
            lr_lin = lr_linear_ref(float(ei), lr0, ref_epochs) if ei < ref_epochs else ""
            lr_po = lr_poly_ref(float(ei), lr0, ref_epochs, poly_exp) if ei < ref_epochs else ""
            w.writerow([ei, f"{lr_p:.8g}", f"{lr_lin:.8g}" if lr_lin != "" else "", f"{lr_po:.8g}" if lr_po != "" else ""])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lr0", type=float, default=1.0)
    p.add_argument("--ref-epochs", type=int, default=1000)
    p.add_argument("--num-epochs", type=int, default=2500)
    p.add_argument("--k-ref", type=int, default=750)
    p.add_argument("--ref-mode", choices=("linear", "poly"), default="linear")
    p.add_argument("--poly-exp", type=float, default=0.9)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--png", action="store_true")
    args = p.parse_args()

    base = args.out_dir or Path(__file__).resolve().parent
    base.mkdir(parents=True, exist_ok=True)

    ref_epochs = args.ref_epochs
    num_epochs = args.num_epochs
    k_ref = args.k_ref
    lr0 = args.lr0
    poly_exp = args.poly_exp

    ref_lin = series_ref_linear(ref_epochs, lr0)
    ref_poly = series_ref_poly(ref_epochs, lr0, poly_exp)
    prop = series_proposed(num_epochs, lr0, ref_epochs, k_ref, args.ref_mode, poly_exp)

    svg_path = base / "lr_schedule_stretched_tail_preview.svg"
    csv_path = base / "lr_schedule_stretched_tail_preview.csv"
    write_svg(svg_path, ref_lin, ref_poly, prop, k_ref, lr0)
    write_csv(csv_path, prop, ref_epochs, lr0, poly_exp)
    print(f"Wrote {svg_path}")
    print(f"Wrote {csv_path}")

    if args.png:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skip PNG")
            return
        e_ref = [x for x, _ in ref_lin]
        y_lin = [y for _, y in ref_lin]
        y_po = [y for _, y in ref_poly]
        e_p = [x for x, _ in prop]
        y_p = [y for _, y in prop]
        png_path = base / "lr_schedule_stretched_tail_preview.png"
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        ax.plot(e_ref, y_lin, color="C0", linewidth=1.5, label=f"Reference linear ({ref_epochs} ep)")
        ax.plot(e_ref, y_po, color="C0", linestyle="--", linewidth=1.2, alpha=0.85, label=f"nnUNet poly exp={poly_exp}")
        ax.plot(e_p, y_p, color="C3", linewidth=2.0, label=f"Proposed ({num_epochs} ep, k={k_ref}, phase1={args.ref_mode})")
        ax.axvline(k_ref, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Epoch (0-based)")
        ax.set_ylabel("LR (relative to lr0)")
        ax.legend(loc="upper right")
        ax.set_xlim(0, max(ref_epochs, num_epochs) * 1.01)
        ax.set_ylim(0, lr0 * 1.05)
        fig.savefig(png_path, dpi=150)
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
