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

--color selects the per-point coloring:
  patch    (default) -- affine patch index = argmax_i |z_i|. Pure-numpy,
                        no extra files needed.
  fitness            -- loads frobenius_norms.npy (sidecar written by
                        plots.make_fitness_plots) and colors by
                        exp(-10 * frobenius_norms). Errors if the sidecar
                        isn't present.

Output filenames carry the color mode: coord_scatter_{re,im,abs}_{patch,fitness}.png.

Output goes to --out_dir (full path) or <min_set_dir>/<out_subdir>/ (subdir
name); these two flags are mutually exclusive. Default: <min_set_dir>.

Usage:
    python -m viz.plot_coord_scatter --min_set plots_slag_run/min_set.pkl
    python -m viz.plot_coord_scatter --min_set plots_slag_run/min_set.pkl --part im
    python -m viz.plot_coord_scatter --min_set plots_slag_run/min_set.pkl --part abs
    python -m viz.plot_coord_scatter --min_set plots_slag_run/min_set.pkl --color fitness
    python -m viz.plot_coord_scatter --min_set plots_slag_run/min_set.pkl \
        --out_subdir scatter_v2 --color fitness
    python -m viz.plot_coord_scatter --min_set my_dir/refined_cloud.pkl \
        --out_dir my_dir/scatter/ --fitness_path my_dir/refined_norms.npy \
        --color fitness
"""
import argparse
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_min_set_complex(path: Path) -> np.ndarray:
    """Load an (N, 5) complex point cloud from a pickle file."""
    with open(path, "rb") as f:
        z = np.asarray(pickle.load(f))  # (k, 5) complex
    return z


def patch_indices_from_complex(z: np.ndarray) -> np.ndarray:
    """Affine patch = argmax_i |z_i| per point."""
    return np.argmax(np.abs(z), axis=1)


_PATCH_KW = dict(cmap="tab10", vmin=-0.5, vmax=4.5)
_FITNESS_KW = dict(cmap="viridis", vmin=0.0, vmax=1.0)


def plot_pairs(z: np.ndarray, color_values: np.ndarray, out_path: Path,
               part: str, title: str, max_points: int | None,
               color_mode: str = "patch") -> None:
    """5x5 grid. Off-diagonal panel (i, j) shows part(z_i) vs part(z_j).
    Diagonal panel (i, i) shows (Re z_i, Im z_i) (always, regardless of
    --part) as a reference for the individual coord distribution.

    color_mode:
      'patch'   -- color_values is per-point patch index (int in 0..4).
                   Uses tab10. Diagonal panels get a light-grey background
                   as a structural marker.
      'fitness' -- color_values is per-point Lagrangian fitness in [0, 1]
                   (apply exp(-10 * frobenius_norms) before calling).
                   Uses viridis and adds a shared colorbar.
    """
    if max_points is not None and z.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(z.shape[0], max_points, replace=False)
        z = z[idx]
        color_values = color_values[idx]
    # Auto-tune marker alpha so very large clouds don't saturate to solid.
    alpha = min(0.4, 8000.0 / max(z.shape[0], 1))

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

    if color_mode == "patch":
        scatter_kw = _PATCH_KW
        diag_facecolor = "#f5f5f5"
    elif color_mode == "fitness":
        scatter_kw = _FITNESS_KW
        diag_facecolor = None
    else:
        raise ValueError(f"unknown color_mode {color_mode!r}; "
                         "expected 'patch' or 'fitness'")

    fig, axes = plt.subplots(5, 5, figsize=(15, 15), squeeze=False)
    last_sc = None
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            if i == j:
                # Diagonal: show (Re z_i, Im z_i).
                last_sc = ax.scatter(
                    z[:, i].real, z[:, i].imag,
                    c=color_values, s=0.5, alpha=alpha, edgecolors="none",
                    **scatter_kw,
                )
                ax.set_xlabel(rf"$\mathrm{{Re}}\,z_{i}$", fontsize=8)
                ax.set_ylabel(rf"$\mathrm{{Im}}\,z_{i}$", fontsize=8)
                if diag_facecolor is not None:
                    ax.set_facecolor(diag_facecolor)
            else:
                last_sc = ax.scatter(
                    proj[:, j], proj[:, i],
                    c=color_values, s=0.5, alpha=alpha, edgecolors="none",
                    **scatter_kw,
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
    if color_mode == "fitness" and last_sc is not None:
        # Shared colorbar on the right; rect leaves room.
        cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(),
                            shrink=0.6, pad=0.02,
                            label=r"Lagrangian fitness  "
                                  r"$\exp(-10\,\|K_R\|_F / \sqrt{\|K_U\|_F})$")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_fitness_colors(norms_path: Path, n_expected: int) -> np.ndarray:
    """Load a frobenius_norms.npy sidecar and map to fitness in [0, 1]."""
    if not norms_path.exists():
        raise SystemExit(
            f"[error] --color fitness requires {norms_path}, but it "
            f"doesn't exist. Re-run plots.make_fitness_plots to regenerate "
            f"the sidecar, pass --fitness_path explicitly, or use --color patch."
        )
    frobenius_norms = np.load(norms_path)
    if frobenius_norms.shape[0] != n_expected:
        raise SystemExit(
            f"[error] {norms_path} has {frobenius_norms.shape[0]} entries "
            f"but the input pkl has {n_expected} points; sidecar is stale."
        )
    return np.exp(-10.0 * frobenius_norms)


def render_from_folder(min_set: Path, out_dir: Path | None = None,
                       part: str = "all", color: str = "patch",
                       fitness_path: Path | None = None,
                       max_points: int | None = None) -> None:
    """Programmatic entry point: same flow as the CLI's main(), but callable.

    Used by fitness_plots.make_fitness_plots to auto-emit fitness-colored
    scatter PNGs at run-end via the sidecar contract (min_set.pkl +
    frobenius_norms.npy), so the histogram path and the scatter path share
    one rendering implementation.
    """
    min_set = Path(min_set)
    out_dir = Path(out_dir) if out_dir is not None else min_set.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    z = load_min_set_complex(min_set)
    print(f"Loaded {z.shape[0]} points from {min_set}")

    if color == "patch":
        color_values = patch_indices_from_complex(z)
        print("Patch counts: " + ", ".join(
            f"p{i}:{int((color_values == i).sum())}" for i in range(5)))
    elif color == "fitness":
        norms_path = (Path(fitness_path) if fitness_path is not None
                      else min_set.parent / "frobenius_norms.npy")
        color_values = _load_fitness_colors(norms_path, z.shape[0])
        print(f"Fitness coloring from {norms_path}: exp(-10*frobenius_norms), "
              f"range=[{color_values.min():.3f}, {color_values.max():.3f}], "
              f"mean={color_values.mean():.3f}")
    else:
        raise ValueError(f"unknown color {color!r}; expected 'patch' or 'fitness'")

    title_tag = min_set.parent.name or min_set.stem
    parts = ["re", "im", "abs"] if part == "all" else [part]
    for p in parts:
        out_path = out_dir / f"coord_scatter_{p}_{color}.png"
        title = f"{title_tag}: coordinate-pair scatter ({p}, colored by {color})"
        plot_pairs(z, color_values, out_path, p, title, max_points,
                   color_mode=color)
        print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min_set", type=Path, required=True,
                        help="Path to a pickle holding an (N, 5) complex "
                             "point cloud (e.g. min_set.pkl).")
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument("--out_dir", type=Path, default=None,
                           help="Full output directory. "
                                "Default: parent directory of --min_set.")
    out_group.add_argument("--out_subdir", type=str, default=None,
                           help="Output subdirectory name appended to "
                                "--min_set's parent directory.")
    parser.add_argument("--fitness_path", type=Path, default=None,
                        help="Path to the frobenius_norms.npy sidecar "
                             "(only used with --color fitness). Defaults to "
                             "<min_set_dir>/frobenius_norms.npy.")
    parser.add_argument("--part", choices=["re", "im", "abs", "all"],
                        default="all",
                        help="Which projection: re, im, abs, or all (writes "
                             "three PNGs).")
    parser.add_argument("--color", choices=["patch", "fitness"],
                        default="patch",
                        help="patch (default): color by affine patch index "
                             "(argmax_i |z_i|), pure numpy. fitness: load "
                             "the frobenius_norms.npy sidecar and color by "
                             "exp(-10 * frobenius_norms) in [0, 1]. Errors "
                             "if the sidecar is missing.")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Subsample points to this many for plotting "
                             "(default: use all points). Pass an integer "
                             "to subsample.")
    args = parser.parse_args()

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.out_subdir is not None:
        out_dir = args.min_set.parent / args.out_subdir
    else:
        out_dir = args.min_set.parent

    render_from_folder(args.min_set, out_dir=out_dir, part=args.part,
                       color=args.color, fitness_path=args.fitness_path,
                       max_points=args.max_points)


if __name__ == "__main__":
    main()
