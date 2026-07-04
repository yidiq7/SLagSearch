"""Overlay fitness histograms from one or more run folders.

Pure consumer of the sidecar contract written by viz.fitness_pipeline:
  <run_folder>/frobenius_norms.npy
  <run_folder>/phases.npy
  <run_folder>/coeffs.pkl              (only required if --vs random is used)

Each run folder contributes one Kähler-norm distribution and one Ω-phase
distribution. The two histograms are overlaid and written to --out_dir:
  Kahler_form_loss_histogram.png
  circular_phase_histogram.png

Used in two ways:
  1. CLI for post-hoc comparisons across independent runs.
  2. Library entry point `plot_overlay_histograms(...)` called by
     viz.fitness_pipeline.run_fitness_pipeline to draw its own histograms
     (single histogram code path).

CLI:
    python -m viz.plot_histograms --runs <dir1> <dir2> ... \
        [--labels d=2 d=3 d=4] [--colors steelblue skyblue lightblue] \
        [--vs random] [--fix_kahler_x_range] --out_dir <dir>

--vs random:
    Auto-mines random coeffs (matching the shape inferred from the first
    input run's coeffs.pkl) into a deterministic cache folder
    fitness_cache/random_w<width>_seed<seed>/ if absent, and appends it
    as the last overlay. The cache is keyed on (width, seed) so the same
    shape always hits the same folder.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# Default color cycle when --colors is not given. Mirrors the steelblue /
# skyblue / lightblue gradient previously used by gradient_descent._run_all_plots.
_DEFAULT_COLORS = [
    "skyblue", "orange", "steelblue", "lightblue",
    "lightsteelblue", "salmon", "plum", "gold",
]


def plot_overlay_histograms(
    runs: list[dict],
    out_dir: str,
    fix_kahler_x_range: bool = False,
) -> None:
    """Draw the two overlay histograms.

    Args:
        runs: list of dicts, each with keys
              "fnorms" (1D np.ndarray of Frobenius norms),
              "phases" (1D np.ndarray of phases in [0, 2*pi)),
              "label" (str, optional; defaults to "run_<i>"),
              "color" (str, optional; defaults to _DEFAULT_COLORS[i]).
              The first entry is treated as the primary; subsequent entries
              are overlays.
        out_dir: directory to write the two PNGs into. Created if missing.
        fix_kahler_x_range: if True, pin both the bin range and xlim to [0, 3]
                            (matches the legacy default of the primary
                            run-folder plot for GD).
    """
    if not runs:
        raise ValueError("plot_overlay_histograms: runs is empty")
    os.makedirs(out_dir, exist_ok=True)

    # Fill in defaults for missing labels / colors.
    for i, r in enumerate(runs):
        r.setdefault("label", f"run_{i}")
        r.setdefault("color", _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)])

    # --- Kähler-form histogram ---
    plt.figure(figsize=(10, 6))
    hist_kwargs = dict(bins=200, alpha=0.7, density=True)
    if fix_kahler_x_range:
        hist_kwargs["range"] = (0, 3)
    for r in runs:
        plt.hist(r["fnorms"], label=r["label"], color=r["color"], **hist_kwargs)
    if fix_kahler_x_range:
        plt.xlim(0, 3)
    plt.xlabel("Frobenius norm")
    plt.ylabel("Probability density")
    plt.title("Distribution of the norm of the Kahler form")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(out_dir, "Kahler_form_loss_histogram.png"))
    plt.close()

    # --- Phase histogram (polar, always [0, 2*pi)) ---
    number_of_bins = 1000
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    width = 2 * np.pi / number_of_bins
    all_counts = []
    for r in runs:
        c, edges = np.histogram(r["phases"], bins=number_of_bins, range=(0, 2 * np.pi))
        all_counts.append(c)
    max_count = max(int(c.max()) for c in all_counts)
    baseline_radius = max_count / 2
    angles = np.linspace(0, 2 * np.pi, number_of_bins, endpoint=False)

    for r, c in zip(runs, all_counts):
        ax.bar(angles, c, width=width, alpha=0.7,
               color=r["color"], label=r["label"], bottom=baseline_radius)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2"], fontsize=12)
    if len(runs) >= 2:
        radial_grid_values = [baseline_radius + max_count * 0.25,
                              baseline_radius + max_count * 0.5,
                              baseline_radius + max_count * 0.75]
    else:
        radial_grid_values = [baseline_radius, baseline_radius + max_count * 0.5]
    ax.set_rgrids(radial_grid_values, angle=22.5)
    ax.set_yticklabels([])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_rlim(0, baseline_radius + max_count * 1.05)
    ax.set_title("Distribution of the phases of the holomorphic 3-form",
                 fontsize=16, pad=25)
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    plt.savefig(os.path.join(out_dir, "circular_phase_histogram.png"),
                bbox_inches="tight")
    plt.close()


def load_run_folder(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (frobenius_norms, phases) from a run folder."""
    p = Path(path)
    fnorms = np.load(p / "frobenius_norms.npy")
    phases = np.load(p / "phases.npy")
    return fnorms, phases


def _load_coeffs_from_run(path: str | Path) -> np.ndarray:
    """Load the coeffs.pkl sidecar. Accepts a bare (3, w) ndarray or a
    checkpoint dict with a 'coeffs' key, matching fitness_pipeline's
    --coeffs handling."""
    with open(Path(path) / "coeffs.pkl", "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "coeffs" in obj:
        obj = obj["coeffs"]
    return np.asarray(obj)


def _ensure_random_cache(
    width: int,
    seed: int,
    psi,
    metric: str,
    k: int,
    n_refine_steps: int,
    points_file: Optional[str],
    cache_root: str = "fitness_cache",
) -> Path:
    """Mine random coeffs into a deterministic cache folder if absent.
    Returns the cache folder path.

    Deferred imports inside the function avoid a circular import between
    this module and viz.fitness_pipeline (which imports this module at
    the top level for the shared histogram code path).
    """
    cache_dir = Path(cache_root) / f"random_w{width}_seed{seed}"
    if (cache_dir / "frobenius_norms.npy").exists() and \
       (cache_dir / "phases.npy").exists():
        return cache_dir

    print(f"[plot_histograms] mining random cache -> {cache_dir}")
    import jax
    import jax.numpy as jnp
    from find_smooth_submanifold import normalize_coeffs
    from helper import canonicalize_coeffs, dwork_points_path, load_points
    from viz.fitness_pipeline import run_fitness_pipeline

    key = jax.random.PRNGKey(seed)
    cmp_coeffs = jax.random.uniform(key, (3, width), minval=-1, maxval=1)
    cmp_coeffs = normalize_coeffs(canonicalize_coeffs(cmp_coeffs))

    if points_file is None:
        points_file = dwork_points_path(psi, seed=1024)
    points_real = load_points(points_file)

    run_fitness_pipeline(
        points_real, cmp_coeffs, jnp.asarray(psi),
        k=k, n_refine_steps=n_refine_steps, metric=metric,
        compare_with=None, out_dir=str(cache_dir),
    )
    return cache_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay fitness histograms from one or more run folders.")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run folders containing frobenius_norms.npy + phases.npy.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="One label per --runs entry (default: folder name).")
    parser.add_argument("--colors", nargs="+", default=None,
                        help="One matplotlib color per --runs entry "
                             "(default: built-in cycle).")
    parser.add_argument("--vs", choices=["random"], default=None,
                        help="Append a random-coeffs overlay. Auto-mines into "
                             "fitness_cache/random_w<width>_seed<seed>/ if absent.")
    parser.add_argument("--fix_kahler_x_range", action="store_true",
                        help="Pin Kähler histogram x-range to [0, 3].")
    parser.add_argument("--out_dir", type=Path, required=True)
    # Optional knobs used only by --vs random.
    parser.add_argument("--psi", type=complex, default=0 + 0j)
    parser.add_argument("--metric", default="k4_fermat",
                        choices=["FS", "k4_fermat"])
    parser.add_argument("--points_file", default=None)
    parser.add_argument("--random_seed", type=int, default=1230)
    parser.add_argument("--random_k", type=int, default=80000)
    parser.add_argument("--random_newton_steps", type=int, default=80)
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        parser.error("--labels must have one entry per --runs entry")
    if args.colors and len(args.colors) != len(args.runs):
        parser.error("--colors must have one entry per --runs entry")

    runs_data: list[dict] = []
    for i, folder in enumerate(args.runs):
        fnorms, phases = load_run_folder(folder)
        runs_data.append({
            "fnorms": fnorms,
            "phases": phases,
            "label": args.labels[i] if args.labels else Path(folder).name,
            "color": args.colors[i] if args.colors else _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
        })

    if args.vs == "random":
        # Width comes from any run that has a coeffs.pkl sidecar.
        # --min_set fitness folders skip writing one; just look at the next.
        width = None
        for r in args.runs:
            if (Path(r) / "coeffs.pkl").exists():
                width = _load_coeffs_from_run(r).shape[1]
                break
        if width is None:
            parser.error("--vs random: no coeffs.pkl found in any --runs folder "
                         "(needed to infer the random-coeffs width).")
        cache_dir = _ensure_random_cache(
            width=width, seed=args.random_seed,
            psi=args.psi, metric=args.metric,
            k=args.random_k, n_refine_steps=args.random_newton_steps,
            points_file=args.points_file,
        )
        fnorms, phases = load_run_folder(cache_dir)
        runs_data.append({
            "fnorms": fnorms, "phases": phases,
            "label": "random",
            "color": _DEFAULT_COLORS[len(runs_data) % len(_DEFAULT_COLORS)],
        })

    plot_overlay_histograms(runs_data, str(args.out_dir),
                            fix_kahler_x_range=args.fix_kahler_x_range)
    print(f"Histograms written to {args.out_dir}/")


if __name__ == "__main__":
    main()
