import jax
import jax.numpy as jnp
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
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
from helper import canonicalize_coeffs, convert_real_to_complex_batch, determine_patches_batch
from plot_coord_scatter import render_from_folder
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
    """Returns per-point (frobenius_norms_plot, norms_for_fitness, phases_2pi).

    Phases are returned on [0, 2*pi) for plotting (so the user can see the
    raw distribution, not the mod-pi reduction used by training).
    frobenius_norms_plot mirrors compute_combined_fitness debug-mode output:
      ||K_restricted||_F / sqrt(||K_unrestricted||_F)
    norms_for_fitness is what compute_lagrangian_condition_fitness uses:
      ||K_restricted||_F / ||K_unrestricted||_F
    """
    min_set = convert_real_to_complex_batch(min_set_real_chunk)
    patch_indices = determine_patches_batch(min_set)
    jacobians = vmap_compute_affine_jacobian(min_set_real_chunk, patch_indices, coeffs, psi)
    restrictions = vmap_compute_restriction(jacobians)
    kahler_unrestricted = compute_kahler_form_unrestricted(min_set, patch_indices, metric=metric)
    kahler_restricted = compute_kahler_form_restricted(min_set, restrictions, patch_indices, metric=metric)
    norms_unrestricted = jnp.linalg.norm(kahler_unrestricted, axis=(1, 2))
    norms_restricted = jnp.linalg.norm(kahler_restricted, axis=(1, 2))
    frobenius_norms_plot = norms_restricted / jnp.sqrt(norms_unrestricted)
    norms_for_fitness = norms_restricted / norms_unrestricted
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

def make_fitness_plots(
    points_real: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    k: int = 100000,
    n_refine_steps: int = 100,
    metric: str = 'FS',
    compare_with=None,
    parent_folder: Optional[str] = 'plots_slag',
    patch_index: Optional[int] = None,
    chunk_size: int = 10000,
    primary_label: str = 'Potential sLag',
    compare_label: str = 'Random intersection',
    primary_color: str = 'skyblue',
    compare_color: str = 'orange',
    fix_kahler_x_range: bool = True,
    extra_comparisons: Optional[list] = None,
    num_devices: int = 1,
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
    histograms; fix_kahler_x_range=True pins both the bin range and xlim to [0, 3].

    num_devices > 1 shards filter_and_refine across GPUs (per-device mines
    k/num_devices points from its slice of points_real). Diagnostics still
    use the host-side chunked loop on the gathered (k, 10) min_set_real,
    which is memory-safe at d=4 because chunk_size bounds per-call VRAM.
    """
    os.makedirs(parent_folder, exist_ok=True)

    # --- Primary set ---
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

    # --- Kahler-form histogram ---
    plt.figure(figsize=(10, 6))
    hist_kwargs = dict(bins=200, alpha=0.7, density=True)
    if fix_kahler_x_range:
        hist_kwargs['range'] = (0, 3)
    plt.hist(frobenius_norms, label=primary_label, color=primary_color, **hist_kwargs)
    for ov in overlays:
        plt.hist(ov["fnorms"], label=ov["label"], color=ov["color"], **hist_kwargs)
    if fix_kahler_x_range:
        plt.xlim(0, 3)
    plt.xlabel('Frobenius norm')
    plt.ylabel('Probability density')
    plt.title('Distribution of the norm of the Kahler form')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(parent_folder, 'Kahler_form_loss_histogram.png'))
    plt.close()

    # --- Phase histogram (polar, always [0, 2*pi)) ---
    number_of_bins = 1000
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    width = 2 * np.pi / number_of_bins
    counts, bin_edges = np.histogram(phases, bins=number_of_bins, range=(0, 2 * np.pi))
    angles = bin_edges[:-1]
    max_count = int(counts.max())
    overlay_counts = []
    for ov in overlays:
        c, _ = np.histogram(ov["phases"], bins=number_of_bins, range=(0, 2 * np.pi))
        overlay_counts.append(c)
        max_count = max(max_count, int(c.max()))
    baseline_radius = max_count / 2

    ax.bar(angles, counts, width=width, alpha=0.7, color=primary_color,
           label=primary_label, bottom=baseline_radius)
    for ov, c in zip(overlays, overlay_counts):
        ax.bar(angles, c, width=width, alpha=0.7, color=ov["color"],
               label=ov["label"], bottom=baseline_radius)

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2'], fontsize=12)
    if overlays:
        radial_grid_values = [baseline_radius + max_count * 0.25,
                              baseline_radius + max_count * 0.5,
                              baseline_radius + max_count * 0.75]
    else:
        radial_grid_values = [baseline_radius, baseline_radius + max_count * 0.5]
    ax.set_rgrids(radial_grid_values, angle=22.5)
    ax.set_yticklabels([])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_rlim(0, baseline_radius + max_count * 1.05)
    ax.set_title('Distribution of the phases of the holomorphic 3-form', fontsize=16, pad=25)
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    plt.savefig(os.path.join(parent_folder, 'circular_phase_histogram.png'), bbox_inches='tight')
    plt.close()

    save_min_set_and_diagnostics(min_set_real, frobenius_norms, parent_folder)
    # Coord-scatter via the sidecar contract: single owner for "render
    # coord-scatter from a folder" lives in plot_coord_scatter. Costs one
    # extra pkl/npy load, which is negligible next to the diagnostics above.
    render_from_folder(Path(parent_folder) / "min_set.pkl", color="fitness")


def save_min_set_and_diagnostics(min_set_real: jnp.ndarray,
                                 frobenius_norms: np.ndarray,
                                 parent_folder: str) -> None:
    """Write min_set.pkl (legacy (N, 5) complex array) and the sidecar
    frobenius_norms.npy that plot_coord_scatter.py --color fitness consumes.
    """
    min_set = np.asarray(min_set_real)[:, :5] + np.asarray(min_set_real)[:, 5:] * 1j
    with open(os.path.join(parent_folder, "min_set.pkl"), "wb") as f:
        pickle.dump(min_set, f)
    np.save(os.path.join(parent_folder, "frobenius_norms.npy"),
            np.asarray(frobenius_norms))


# ---------------------------------------------------------------------------
#  Standalone CLI: fitness plots for a coeffs array. --coeffs_pkl is
#  required and accepts a bare (3, w) array or a checkpoint dict with a
#  "coeffs" key.
# ---------------------------------------------------------------------------
def _load_coeffs_pkl(path: str) -> jnp.ndarray:
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
    parser.add_argument("--coeffs_pkl", required=True,
                        help="Path to a pickle holding either a (3, w) coeffs "
                             "array (w in {25, 250, 1475, 6375}) or a "
                             "checkpoint dict with a 'coeffs' key.")
    parser.add_argument("--points_file", default=None,
                        help="Override path to point cloud pkl. "
                             "Default: helper.dwork_points_path(psi).")
    parser.add_argument("--psi", type=complex, default=0+0j,
                        help="Dwork parameter (complex). 0 = Fermat quintic.")
    parser.add_argument("--metric", default="k4_fermat",
                        help="'k4_fermat' (psi=0 only) or 'FS' (any psi).")
    parser.add_argument("--k", type=int, default=80000)
    parser.add_argument("--n_refine_steps", type=int, default=80)
    parser.add_argument("--parent_folder", default="plots_fitness_manual",
                        help="Output directory for histograms + sidecars + "
                             "coord-scatter PNGs.")
    parser.add_argument("--compare_with", default="random",
                        help="'None', 'random', or unused (manual ndarray "
                             "not exposed here).")
    args = parser.parse_args()

    assert_metric_psi_compatible(args.metric, args.psi)
    points_file = (args.points_file if args.points_file is not None
                   else dwork_points_path(args.psi, seed=1024))
    points_real = load_points(points_file)
    print(f"Loaded {points_real.shape[0]} points from {points_file}")

    coeffs = _load_coeffs_pkl(args.coeffs_pkl)
    print(f"Coeffs shape: {coeffs.shape}  (from {args.coeffs_pkl})")

    compare_with = None if args.compare_with.lower() == "none" else args.compare_with

    make_fitness_plots(
        points_real, coeffs, jnp.asarray(args.psi),
        k=args.k, n_refine_steps=args.n_refine_steps,
        metric=args.metric, compare_with=compare_with,
        parent_folder=args.parent_folder,
    )
    print(f"Plots written to {args.parent_folder}/")


if __name__ == "__main__":
    jax.config.update("jax_default_matmul_precision", "high")
    main()
