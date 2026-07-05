"""Fitness-diagnostics pipeline for a candidate coeffs array.

Mines a min-set with Newton refinement, computes Kahler-form Frobenius norms
and Omega phases on it, writes the run-folder sidecars
(coeffs.pkl / min_set.pkl / frobenius_norms.npy / phases.npy) plus the two
self-only histograms (delegated to viz.plot_histograms) and the
fitness-colored coord-scatter PNGs. Library entry point `run_fitness_pipeline`
is imported by GA.py and gradient_descent.py for their end-of-run plots.

The run-folder shape is the same for every producer (GA species, GD result,
manually mined runs):

    <run_folder>/
        coeffs.pkl
        min_set.pkl
        frobenius_norms.npy
        phases.npy
        Kahler_form_loss_histogram.png
        circular_phase_histogram.png
        coord_scatter_{re,im,abs}_fitness.png

Cross-run overlay plots are produced by viz.plot_histograms; this module
only emits self-only histograms.

Usage (CLI):
    python -m viz.fitness_pipeline --coeffs gd_runs/gd_<job>_step<N>.pkl \
        [--min_set <pkl>] [--points_file <path>] [--psi <c>] \
        [--metric k4_fermat|FS] [--out_dir <dir> | --out_subdir <name>] \
        [--k 80000] [--newton_steps 80] [--vs random]

--coeffs is required and accepts either a bare (3, w) array or a
checkpoint dict with a "coeffs" key (matches gradient_descent checkpoints).
--min_set <pkl> skips mining and uses the given (N, 5) complex points as the
min-set (used e.g. for plotting fitness on one UMAP cluster output).
--out_dir / --out_subdir are mutually exclusive; default writes to the
parent directory of --coeffs.
--vs random invokes viz.plot_histograms after the self-only plots to also
emit a random-overlay histogram pair in <out_dir>_vs_random/.
"""
import jax
import jax.numpy as jnp
import numpy as np
import os
import pickle
from functools import partial
from pathlib import Path
from find_smooth_submanifold import filter_and_refine, normalize_coeffs
from sharding import shard_leading_axis
from slag_condition import (
    compute_holomorphic_form,
    compute_kahler_form_restricted,
    compute_kahler_form_unrestricted,
    compute_special_condition_fitness,
    vmap_compute_affine_jacobian,
    vmap_compute_restriction,
)
from get_restriction import compute_Omega_restriction
from helper import (canonicalize_coeffs, convert_complex_to_real_batch,
                    convert_real_to_complex_batch, determine_patch_and_rescale_single,
                    determine_patches_batch)
from viz.plot_coord_scatter import render_from_folder
from viz.plot_histograms import plot_overlay_histograms
from typing import Optional


def _pmap_filter_and_refine_factory(num_devices: int):
    """Build a pmapped filter_and_refine. Per-device runs on its points shard
    with k_per_device = k // num_devices. Memoized via lru_cache-style by
    caller to avoid re-tracing across multiple invocations."""

    def per_device(points_shard, coeffs, psi, k_per_dev, n_refine_steps):
        return filter_and_refine(points_shard, coeffs, psi, k_per_dev, n_refine_steps)

    pmapped = jax.pmap(
        per_device,
        axis_name="x",
        in_axes=(0, None, None, None, None),
        static_broadcasted_argnums=(3, 4),
    )
    return pmapped


def _mine_on_one_or_many(
    points_real: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    k: int,
    n_refine_steps: int,
    num_devices: int,
):
    """Wrapper: single-device filter_and_refine, or sharded across `num_devices`.
    Returns (min_set_real, distances) as host-side numpy arrays."""
    if num_devices <= 1:
        min_set_real, distances, _ = filter_and_refine(
            points_real, coeffs, psi, k, n_refine_steps,
        )
        return np.asarray(min_set_real), np.asarray(distances)
    if k % num_devices != 0:
        raise ValueError(
            f"plot k={k} not divisible by num_devices={num_devices}"
        )
    k_per_dev = k // num_devices
    n_keep = (points_real.shape[0] // num_devices) * num_devices
    points_sharded = shard_leading_axis(points_real[:n_keep], num_devices)
    pmapped = _pmap_filter_and_refine_factory(num_devices)
    min_set_sharded, distances_sharded, _ = pmapped(
        points_sharded, coeffs, psi, k_per_dev, n_refine_steps,
    )
    return (
        np.asarray(min_set_sharded).reshape(-1, 10),
        np.asarray(distances_sharded).reshape(-1),
    )


@partial(jax.jit, static_argnames=('metric',))
def _per_chunk_diagnostics(
    min_set_real_chunk: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    metric: str,
):
    """Returns per-point (frobenius_norms, norms_for_fitness, phases_2pi).

    Phases are returned on [0, 2*pi) for plotting (so the user can see the
    raw distribution, not the mod-pi reduction used by training).
    Both per-point arrays use the same normalization convention as
    compute_lagrangian_condition_fitness -- rescale the Frobenius norm of
    the pulled-back Kahler form by the Frobenius norm before pull-back:
      ||K_restricted||_F / ||K_unrestricted||_F
    They are identical here; kept as separate names for call-site clarity
    (frobenius_norms is what gets saved to the sidecar / histogrammed;
    norms_for_fitness is what the bottom-99% mean is taken over).
    """
    min_set = convert_real_to_complex_batch(min_set_real_chunk)
    # Same patch-frame consistency as compute_combined_fitness: derive the
    # patch and the representative jointly so z[patch] = 1 exactly for the
    # index used downstream. A fresh argmax breaks on equal-moduli loci (the
    # tie-flipped coordinate holds e^{i phi} != 1 and every phase below shifts
    # by 3*phi); see the comment there and diagnostics/test_gauge_invariance.py.
    min_set, patch_indices = jax.vmap(determine_patch_and_rescale_single)(min_set)
    min_set_real_chunk = convert_complex_to_real_batch(min_set)
    jacobians = vmap_compute_affine_jacobian(min_set_real_chunk, patch_indices, coeffs, psi)
    restrictions = vmap_compute_restriction(jacobians)
    kahler_unrestricted = compute_kahler_form_unrestricted(min_set, patch_indices, metric=metric)
    kahler_restricted = compute_kahler_form_restricted(min_set, restrictions, patch_indices, metric=metric)
    norms_unrestricted = jnp.linalg.norm(kahler_unrestricted, axis=(1, 2))
    norms_restricted = jnp.linalg.norm(kahler_restricted, axis=(1, 2))
    frobenius_norms_plot = norms_restricted / norms_unrestricted
    norms_for_fitness = frobenius_norms_plot
    # Bypass compute_holomorphic_form_restricted so we can use mod 2*pi for the
    # plot; training/fitness still use the mod-pi reduction (applied below).
    Omega, _, Omega_coord = compute_holomorphic_form(min_set, patch_indices, psi)
    Omega_restriction = compute_Omega_restriction(restrictions, Omega_coord)
    phases_2pi = jnp.angle(Omega * Omega_restriction) % (2 * jnp.pi)
    return frobenius_norms_plot, norms_for_fitness, phases_2pi


def _chunked_diagnostics(
    min_set_real: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    metric: str,
    chunk_size: int,
):
    """Loops _per_chunk_diagnostics over points, transferring each chunk to
    host memory as numpy so GPU memory only ever holds one chunk's intermediates.
    """
    N = min_set_real.shape[0]
    fn_chunks, ff_chunks, ph_chunks = [], [], []
    for i in range(0, N, chunk_size):
        fn, ff, ph = _per_chunk_diagnostics(
            min_set_real[i:i + chunk_size], coeffs, psi, metric
        )
        fn_chunks.append(np.asarray(fn))
        ff_chunks.append(np.asarray(ff))
        ph_chunks.append(np.asarray(ph))
    return (
        np.concatenate(fn_chunks),
        np.concatenate(ff_chunks),
        np.concatenate(ph_chunks),
    )

def run_fitness_pipeline(
    points_real: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    k: int = 100000,
    n_refine_steps: int = 100,
    metric: str = 'FS',
    compare_with=None,
    out_dir: Optional[str] = 'plots_slag',
    patch_index: Optional[int] = None,
    chunk_size: int = 10000,
    primary_label: str = 'Potential sLag',
    compare_label: str = 'Random intersection',
    primary_color: str = 'skyblue',
    compare_color: str = 'orange',
    fix_kahler_x_range: Optional[bool] = None,
    extra_comparisons: Optional[list] = None,
    num_devices: int = 1,
    min_set_override: Optional[jnp.ndarray] = None,
    min_set_source: Optional[str] = None,
    ) -> None:
    """Plot Kahler-norm and Omega-phase distributions for `coeffs`, optionally
    overlaid with one or more comparison distributions.

    compare_with:
        None         -- primary only
        "random"     -- generate random coeffs (matching `coeffs.shape`) and overlay
        jnp.ndarray  -- use these coeffs as the comparison set

    extra_comparisons: optional list of dicts {"coeffs", "label", "color"} adding
    further overlays beyond compare_with. Their coeffs are run through
    filter_and_refine + diagnostics separately (any shape compatible with the
    static dispatch in evaluate_equations_single_point is accepted).

    primary_color / compare_color / primary_label / compare_label tune the
    histograms.

    fix_kahler_x_range:
        True  -- pin Kähler-histogram bin range + xlim to [0, 3].
        False -- let matplotlib auto-range.
        None (default) -- True iff compare_with == "random", else False.
                          (The [0, 3] window is calibrated to keep a random
                          overlay's bulk visible alongside a tight primary;
                          for self-only or coeffs-vs-coeffs overlays, auto
                          gives a more informative view.)

    num_devices > 1 shards filter_and_refine across GPUs (per-device mines
    k/num_devices points from its slice of points_real). Diagnostics still
    use the host-side chunked loop on the gathered (k, 10) min_set_real,
    which is memory-safe at d=4 because chunk_size bounds per-call VRAM.

    min_set_override: skip mining and use these (N, 10) real points as the
    primary min-set. Used e.g. for plotting fitness on a single UMAP cluster
    of an existing run. `points_real` is then only consumed by `compare_with`
    / `extra_comparisons` mining (if any).

    min_set_source: optional path to the (N, 5) complex pkl that
    min_set_override was loaded from. When given, save_run_sidecars writes
    out_dir/min_set.pkl as a symlink to this path instead of pickling a
    duplicate copy.
    """
    if fix_kahler_x_range is None:
        fix_kahler_x_range = isinstance(compare_with, str) and compare_with == "random"
    os.makedirs(out_dir, exist_ok=True)

    # --- Primary set ---
    if min_set_override is not None:
        min_set_real = np.asarray(min_set_override)
        # Real per-point Newton-step residuals so the printed Min/Max/Mean
        # actually indicate whether the override points lie on the manifold.
        from find_smooth_submanifold import compute_distances_batched
        distances = np.asarray(compute_distances_batched(
            jnp.asarray(min_set_real), coeffs, psi
        ))
    else:
        min_set_real, distances = _mine_on_one_or_many(
            points_real, coeffs, psi, k, n_refine_steps, num_devices,
        )
    if patch_index is not None:
        patch_indices = np.asarray(determine_patches_batch(
            convert_real_to_complex_batch(jnp.asarray(min_set_real))))
        mask = patch_indices == patch_index
        min_set_real = min_set_real[mask]
        distances = distances[mask]

    frobenius_norms, norms_for_fitness, phases = _chunked_diagnostics(
        jnp.asarray(min_set_real), coeffs, psi, metric, chunk_size
    )
    sorted_nf = np.sort(norms_for_fitness)
    cutoff = int(sorted_nf.shape[0] * 0.99)
    lagrangian_fitness = float(np.exp(-10.0 * np.mean(sorted_nf[:cutoff])))
    # special_fitness uses mod-pi phases (consistent with training);
    # the histograms below use the raw mod-2pi phases.
    phases_mod_pi = phases % np.pi
    special_fitness = float(compute_special_condition_fitness(jnp.asarray(phases_mod_pi), n_bins=100))

    print(f"min_set_distance: Min: {distances.min()}, Max: {distances.max()}, Mean: {distances.mean()}")
    print(f"Lagrangian fitness: {lagrangian_fitness}, special_fitness: {special_fitness}")

    # --- Collect comparison overlays (compare_with first, then extra_comparisons in order) ---
    overlays = []  # list of dicts {fnorms, phases, label, color}
    if compare_with is not None:
        if isinstance(compare_with, str):
            if compare_with != "random":
                raise ValueError(f"compare_with={compare_with!r}; expected None, 'random', or an ndarray.")
            key = jax.random.PRNGKey(1230)
            cmp_coeffs = jax.random.uniform(key, coeffs.shape, minval=-1, maxval=1)
            cmp_coeffs = canonicalize_coeffs(cmp_coeffs)
            cmp_coeffs = normalize_coeffs(cmp_coeffs)
        else:
            cmp_coeffs = compare_with
        min_set_real_cmp, _ = _mine_on_one_or_many(
            points_real, cmp_coeffs, psi, k, n_refine_steps, num_devices,
        )
        fnorms_cmp, _, phases_cmp = _chunked_diagnostics(
            jnp.asarray(min_set_real_cmp), cmp_coeffs, psi, metric, chunk_size
        )
        overlays.append({"fnorms": fnorms_cmp, "phases": phases_cmp,
                         "label": compare_label, "color": compare_color})
    if extra_comparisons:
        for ex in extra_comparisons:
            ex_coeffs = ex["coeffs"]
            min_set_real_ex, _ = _mine_on_one_or_many(
                points_real, ex_coeffs, psi, k, n_refine_steps, num_devices,
            )
            fnorms_ex, _, phases_ex = _chunked_diagnostics(
                jnp.asarray(min_set_real_ex), ex_coeffs, psi, metric, chunk_size
            )
            overlays.append({"fnorms": fnorms_ex, "phases": phases_ex,
                             "label": ex["label"], "color": ex["color"]})

    # Persist sidecars before drawing histograms so the run folder is
    # already valid input to viz.plot_histograms.
    save_run_sidecars(out_dir, coeffs, min_set_real, frobenius_norms, phases,
                      min_set_source=min_set_source)

    # --- Histograms (delegated to the single owner) ---
    runs_for_plot = [{
        "fnorms": np.asarray(frobenius_norms),
        "phases": np.asarray(phases),
        "label": primary_label,
        "color": primary_color,
    }]
    for ov in overlays:
        runs_for_plot.append({
            "fnorms": np.asarray(ov["fnorms"]),
            "phases": np.asarray(ov["phases"]),
            "label": ov["label"],
            "color": ov["color"],
        })
    plot_overlay_histograms(runs_for_plot, out_dir,
                            fix_kahler_x_range=fix_kahler_x_range)
    # Coord-scatter via the sidecar contract: single owner for "render
    # coord-scatter from a folder" lives in plot_coord_scatter. Costs one
    # extra pkl/npy load, which is negligible next to the diagnostics above.
    # When min_set_source is given we didn't write min_set.pkl into out_dir;
    # point the scatter renderer at the source pkl and pass fitness_path
    # explicitly so it picks up the freshly-written frobenius_norms.npy.
    if min_set_source is not None:
        render_from_folder(
            Path(min_set_source), out_dir=Path(out_dir), color="fitness",
            fitness_path=Path(out_dir) / "frobenius_norms.npy",
        )
    else:
        render_from_folder(Path(out_dir) / "min_set.pkl", color="fitness")


def save_run_sidecars(out_dir: str,
                      coeffs: jnp.ndarray,
                      min_set_real: jnp.ndarray,
                      frobenius_norms: np.ndarray,
                      phases: np.ndarray,
                      min_set_source: Optional[str] = None) -> None:
    """Write the canonical run-folder sidecars:
        coeffs.pkl              -- the coeffs that defined the submanifold
        min_set.pkl             -- (N, 5) complex points on the mined min-set
        frobenius_norms.npy     -- per-point ||K_R||_F / ||K_U||_F
        phases.npy              -- per-point Omega phase mod 2*pi

    This is the contract consumed by viz.plot_histograms (overlay) and
    viz.plot_coord_scatter (fitness coloring).

    If min_set_source is given (signals that the caller supplied
    min_set_real explicitly, so coeffs.pkl and min_set.pkl already live
    elsewhere on disk), skip writing both inputs into out_dir -- only the
    fitness sidecars (frobenius_norms.npy, phases.npy) and the plots are
    written. This avoids duplicating large pkls in the --min_set CLI path
    (e.g. cluster-fitness folders).
    """
    os.makedirs(out_dir, exist_ok=True)

    if min_set_source is None:
        with open(os.path.join(out_dir, "coeffs.pkl"), "wb") as f:
            pickle.dump(np.asarray(coeffs), f)
        min_set = np.asarray(min_set_real)[:, :5] + np.asarray(min_set_real)[:, 5:] * 1j
        with open(os.path.join(out_dir, "min_set.pkl"), "wb") as f:
            pickle.dump(min_set, f)

    np.save(os.path.join(out_dir, "frobenius_norms.npy"), np.asarray(frobenius_norms))
    np.save(os.path.join(out_dir, "phases.npy"), np.asarray(phases))


# ---------------------------------------------------------------------------
#  Standalone CLI: fitness plots for a coeffs array. --coeffs is
#  required and accepts a bare (3, w) array or a checkpoint dict with a
#  "coeffs" key.
# ---------------------------------------------------------------------------
def _load_coeffs(path: str) -> jnp.ndarray:
    """Load a coeffs array from a pickle. Accepts a bare (3, w) ndarray or a
    checkpoint dict with a 'coeffs' key (matches gradient_descent checkpoints).
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "coeffs" in obj:
        return jnp.asarray(obj["coeffs"])
    return jnp.asarray(obj)


def main() -> None:
    import argparse
    from helper import assert_metric_psi_compatible, dwork_points_path, load_points

    parser = argparse.ArgumentParser(
        description="Generate fitness plots (Kahler-form histogram, "
                    "Omega-phase polar, fitness-colored coord-scatter) for "
                    "a coeffs array.")
    parser.add_argument("--coeffs", default=None,
                        help="Path to a pickle holding either a (3, w) coeffs "
                             "array (w in {25, 250, 1475, 6375}) or a "
                             "checkpoint dict with a 'coeffs' key. Required "
                             "unless --min_set is given and a coeffs.pkl exists "
                             "in the min_set's parent folder, in which case it "
                             "is auto-discovered.")
    parser.add_argument("--points_file", default=None,
                        help="Override path to point cloud pkl. "
                             "Default: helper.dwork_points_path(psi).")
    parser.add_argument("--min_set", default=None,
                        help="Path to a (N, 5) complex pkl to use as the "
                             "min-set, skipping mining. Used e.g. to plot "
                             "fitness on a single UMAP cluster of an "
                             "existing run.")
    parser.add_argument("--psi", type=complex, default=0+0j,
                        help="Dwork parameter (complex). 0 = Fermat quintic.")
    parser.add_argument("--metric", default="k4_fermat",
                        help="'k4_fermat' (psi=0 only) or 'FS' (any psi).")
    parser.add_argument("--k", type=int, default=80000)
    parser.add_argument("--newton_steps", type=int, default=80,
                        help="Newton refinement steps in filter_and_refine "
                             "(library kwarg: n_refine_steps).")
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument("--out_dir", type=Path, default=None,
                           help="Full output directory. "
                                "Default: parent directory of --coeffs.")
    out_group.add_argument("--out_subdir", type=str, default=None,
                           help="Output subdirectory name appended to "
                                "--coeffs's parent directory.")
    parser.add_argument("--compare_with", default=None,
                        help="'random' to mine a random-coeffs overlay, "
                             "'none' or omitted for self-only. Default: "
                             "'random' when --min_set is NOT given; 'none' "
                             "when --min_set is given (the cluster-fitness "
                             "use case rarely wants a full random overlay).")
    args = parser.parse_args()

    assert_metric_psi_compatible(args.metric, args.psi)
    points_file = (args.points_file if args.points_file is not None
                   else dwork_points_path(args.psi, seed=1024))
    points_real = load_points(points_file)
    print(f"Loaded {points_real.shape[0]} points from {points_file}")

    coeffs_path = args.coeffs
    if coeffs_path is None:
        if args.min_set is None:
            parser.error("--coeffs is required (or pass --min_set whose parent "
                         "folder has a coeffs.pkl sidecar).")
        candidate = Path(args.min_set).parent / "coeffs.pkl"
        if not candidate.exists():
            parser.error(f"--coeffs not given and no coeffs.pkl found at {candidate}")
        coeffs_path = str(candidate)
        print(f"[fitness_pipeline] auto-discovered coeffs at {coeffs_path}")
    coeffs = _load_coeffs(coeffs_path)
    print(f"Coeffs shape: {coeffs.shape}  (from {coeffs_path})")

    # When --min_set is given, the user's mental model is "I'm analyzing THIS
    # min_set" (the coeffs is a sibling lookup), so resolve out_subdir against
    # the min_set's parent. Otherwise resolve against the coeffs' parent.
    default_parent = (Path(args.min_set).parent if args.min_set is not None
                      else Path(coeffs_path).parent)
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.out_subdir is not None:
        out_dir = default_parent / args.out_subdir
    else:
        out_dir = default_parent

    if args.compare_with is None:
        # Default per the help text above: random when mining the primary;
        # self-only when the user has supplied an explicit --min_set.
        compare_with = None if args.min_set is not None else "random"
    elif args.compare_with.lower() == "none":
        compare_with = None
    else:
        compare_with = args.compare_with

    min_set_override = None
    if args.min_set is not None:
        with open(args.min_set, "rb") as f:
            min_set_complex = np.asarray(pickle.load(f))
        # Convert (N, 5) complex -> (N, 10) real (first 5 Re, last 5 Im).
        min_set_override = jnp.concatenate(
            [jnp.asarray(min_set_complex.real),
             jnp.asarray(min_set_complex.imag)], axis=1
        )
        print(f"Loaded min_set override: {min_set_complex.shape[0]} points "
              f"from {args.min_set}")

    # Multi-GPU: shard mining across local devices, same pattern GD uses.
    # Library function defaults num_devices=1 so callers don't accidentally
    # opt into pmap; the CLI explicitly opts in here.
    num_devices = jax.local_device_count()
    if num_devices > 1:
        if args.k % num_devices != 0:
            parser.error(f"--k {args.k} not divisible by num_devices={num_devices}; "
                         f"pass a multiple of {num_devices}.")
        print(f"Detected {num_devices} GPU(s); sharding mining across them.")

    run_fitness_pipeline(
        points_real, coeffs, jnp.asarray(args.psi),
        k=args.k, n_refine_steps=args.newton_steps,
        metric=args.metric, compare_with=compare_with,
        out_dir=str(out_dir),
        min_set_override=min_set_override,
        min_set_source=args.min_set,
        num_devices=num_devices,
    )
    print(f"Plots written to {out_dir}/")


if __name__ == "__main__":
    jax.config.update("jax_default_matmul_precision", "high")
    main()
