"""Raw coordinate-pair scatter plots from a min_set.pkl, to visualize set-wise
symmetry of the candidate sLag under coordinate swaps.

If the sLag is (i j)-invariant, every point (..., z_i, ..., z_j, ...) has a
swap-partner (..., z_j, ..., z_i, ...) also on the sLag -- so the projected
cloud in the (Re z_i, Re z_j) plane is symmetric across the diagonal y = x.
Same for (Im z_i, Im z_j) and (|z_i|, |z_j|). Pointwise z_i = z_j is NOT
expected (that would be a much stronger condition).

Produces a 5x5 grid (one panel per unordered (i, j) pair, including i = j
showing the marginal Re vs Im distribution on the diagonal). Off-diagonal
panels (i != j) draw the y = x line for visual reference.

Usage:
    python plot_coord_scatter.py <folder>             # uses min_set.pkl in folder
    python plot_coord_scatter.py <folder> --part im   # imaginary parts instead
    python plot_coord_scatter.py <folder> --part abs  # |z_i| vs |z_j|
"""
import argparse
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_min_set_complex(folder: Path) -> np.ndarray:
    path = folder / "min_set.pkl"
    with open(path, "rb") as f:
        z = np.asarray(pickle.load(f))  # (k, 5) complex
    return z


def patch_indices_from_complex(z: np.ndarray) -> np.ndarray:
    """Affine patch = argmax_i |z_i| per point."""
    return np.argmax(np.abs(z), axis=1)


def plot_pairs(z: np.ndarray, patches: np.ndarray, out_path: Path,
               part: str, title: str, max_points: int) -> None:
    """5x5 grid. Off-diagonal panel (i, j) shows part(z_i) vs part(z_j).
    Diagonal panel (i, i) shows (Re z_i, Im z_i) (always, regardless of
    --part) as a reference for the individual coord distribution.
    """
    if z.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(z.shape[0], max_points, replace=False)
        z = z[idx]
        patches = patches[idx]

    if part == "re":
        proj = z.real
        # Escape literal LaTeX braces around Re/Im so .format() only fills {}.
        label_fmt = r"$\mathrm{{Re}}\,z_{}$"
    elif part == "im":
        proj = z.imag
        label_fmt = r"$\mathrm{{Im}}\,z_{}$"
    elif part == "abs":
        proj = np.abs(z)
        label_fmt = r"$|z_{}|$"
    else:
        raise ValueError(f"unknown --part {part!r}")

    fig, axes = plt.subplots(5, 5, figsize=(15, 15), squeeze=False)

    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            if i == j:
                # Diagonal: show (Re z_i, Im z_i).
                sc = ax.scatter(
                    z[:, i].real, z[:, i].imag,
                    c=patches, cmap="tab10", vmin=-0.5, vmax=4.5,
                    s=0.5, alpha=0.4, edgecolors="none",
                )
                ax.set_xlabel(rf"$\mathrm{{Re}}\,z_{i}$", fontsize=8)
                ax.set_ylabel(rf"$\mathrm{{Im}}\,z_{i}$", fontsize=8)
                ax.set_facecolor("#f5f5f5")
            else:
                ax.scatter(
                    proj[:, j], proj[:, i],
                    c=patches, cmap="tab10", vmin=-0.5, vmax=4.5,
                    s=0.5, alpha=0.4, edgecolors="none",
                )
                # y = x line for visual diagonal-symmetry check.
                xlo = min(proj[:, i].min(), proj[:, j].min())
                xhi = max(proj[:, i].max(), proj[:, j].max())
                ax.plot([xlo, xhi], [xlo, xhi], "k--", lw=0.8, alpha=0.5)
                ax.set_xlabel(label_fmt.format(j), fontsize=8)
                ax.set_ylabel(label_fmt.format(i), fontsize=8)
            ax.tick_params(axis="both", labelsize=6)
            ax.set_aspect("equal", adjustable="box")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path,
                        help="Folder containing min_set.pkl.")
    parser.add_argument("--part", choices=["re", "im", "abs", "all"],
                        default="all",
                        help="Which projection: re, im, abs, or all (writes "
                             "three PNGs).")
    parser.add_argument("--max_points", type=int, default=20000,
                        help="Subsample points to this many for plotting "
                             "(default 20000).")
    args = parser.parse_args()

    z = load_min_set_complex(args.folder)
    patches = patch_indices_from_complex(z)
    print(f"Loaded {z.shape[0]} points from {args.folder}/min_set.pkl")
    print(f"Patch counts: " + ", ".join(
        f"p{i}:{int((patches == i).sum())}" for i in range(5)))

    parts = ["re", "im", "abs"] if args.part == "all" else [args.part]
    for p in parts:
        out_path = args.folder / f"coord_scatter_{p}.png"
        title = f"{args.folder.name}: coordinate-pair scatter ({p})"
        plot_pairs(z, patches, out_path, p, title, args.max_points)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
