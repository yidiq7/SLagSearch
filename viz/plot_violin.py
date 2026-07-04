"""Violin plot of Kähler-norm distributions from one or more run folders.

Pure consumer of the sidecar contract written by viz.fitness_pipeline:
  <run_folder>/frobenius_norms.npy

Each run folder contributes one violin. The norms are the per-point
||K_restricted||_F / ||K_unrestricted||_F ratios (already normalized by the
pipeline), so violins from different degrees are directly comparable.

The y-axis is logarithmic: the KDE is computed on log10 of the data (a KDE in
linear space rendered on a log axis would misrepresent the density), plotted
on a linear axis whose ticks are labeled 10^k. Non-positive entries cannot be
logged and are dropped with a warning.

CLI:
    python -m viz.plot_violin \
        --runs plots_slag_d1_search/plots_slag_6338568_1_id0 plots_slag_d2_run \
               gd_runs/plots_slag_d3_run gd_runs/plots_slag_d4_run \
        --labels d=1 d=2 d=3 d=4 --out_dir gd_runs/compare_violin
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

# Single hue for every violin: all four distributions are the same measure,
# and the x-axis already carries identity, so color has no encoding job here.
_FILL_COLOR = "#2a78d6"
_EDGE_COLOR = "#1c5cab"
_INK_COLOR = "#104281"


def plot_norm_violins(
    runs: list[dict],
    out_path: str,
    title: str = "Distribution of the norm of the Kahler form",
) -> None:
    """Draw one violin per run on a log y-axis.

    Args:
        runs: list of dicts with keys
              "fnorms" (1D np.ndarray of Frobenius-norm ratios),
              "label"  (str, optional; defaults to "run_<i>").
        out_path: PNG path to write.
    """
    if not runs:
        raise ValueError("plot_norm_violins: runs is empty")

    logged = []
    for i, r in enumerate(runs):
        r.setdefault("label", f"run_{i}")
        fnorms = np.asarray(r["fnorms"], dtype=float)
        positive = fnorms[fnorms > 0]
        n_dropped = fnorms.shape[0] - positive.shape[0]
        if n_dropped:
            print(f"[plot_violin] {r['label']}: dropped {n_dropped} "
                  f"non-positive value(s) before log transform")
        if positive.shape[0] == 0:
            raise ValueError(f"plot_violin: {r['label']} has no positive norms")
        logged.append(np.log10(positive))
        q1, med, q3 = np.percentile(positive, [25, 50, 75])
        print(f"[plot_violin] {r['label']}: n={positive.shape[0]}, "
              f"median={med:.3e}, IQR=[{q1:.3e}, {q3:.3e}]")

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(1, len(logged) + 1)
    parts = ax.violinplot(logged, positions=positions, widths=0.8,
                          points=200, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(_FILL_COLOR)
        body.set_edgecolor(_EDGE_COLOR)
        body.set_alpha(0.45)
        body.set_linewidth(1.0)

    # Interquartile bar + median dot inside each violin (in log space,
    # consistent with the violin geometry).
    for pos, ld in zip(positions, logged):
        q1, med, q3 = np.percentile(ld, [25, 50, 75])
        ax.vlines(pos, q1, q3, color=_INK_COLOR, linewidth=4, zorder=3)
        ax.scatter(pos, med, s=18, color="white", edgecolor=_INK_COLOR,
                   linewidth=0.8, zorder=4)

    # Log-style ticks on the linear log10 axis: majors at 10^k, minors at
    # the 2..9 subdivisions of each decade.
    lo = min(ld.min() for ld in logged)
    hi = max(ld.max() for ld in logged)
    kmin, kmax = int(np.floor(lo)), int(np.ceil(hi))
    ax.yaxis.set_major_locator(FixedLocator(np.arange(kmin, kmax + 1)))
    ax.yaxis.set_minor_locator(FixedLocator(
        [k + np.log10(m) for k in range(kmin, kmax) for m in range(2, 10)]))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"$10^{{{int(round(v))}}}$"))
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))

    ax.set_xticks(positions)
    ax.set_xticklabels([r["label"] for r in runs])
    ax.set_ylabel(r"Frobenius norm  $\|K_R\|_F \, / \, \|K_U\|_F$")
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Violin plot of Kähler-norm distributions "
                    "(log y-axis) from run folders.")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run folders containing frobenius_norms.npy.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="One label per --runs entry (default: folder name).")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="Directory to write Kahler_norm_violin.png into.")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        parser.error("--labels must have one entry per --runs entry")

    runs_data = []
    for i, folder in enumerate(args.runs):
        fnorms = np.load(Path(folder) / "frobenius_norms.npy")
        runs_data.append({
            "fnorms": fnorms,
            "label": args.labels[i] if args.labels else Path(folder).name,
        })

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "Kahler_norm_violin.png")
    plot_norm_violins(runs_data, out_path)
    print(f"Violin plot written to {out_path}")


if __name__ == "__main__":
    main()
