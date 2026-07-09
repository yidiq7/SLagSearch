"""Gradient descent for sLag search with d=1, d=1+2, d=1+2+3, or d=1+2+3+4 ansatz.

Genotype width is set by --max_degree:
  1 -> (3, 25)     (d=1 only)
  2 -> (3, 250)    (d=1 + d=2, default)
  3 -> (3, 1475)   (d=1 + d=2 + d=3)
  4 -> (3, 6375)   (d=1 + d=2 + d=3 + d=4)

The optimization alternates:

1. (Re-)mining: every `mine_interval` steps, `filter_and_refine` produces a
   fresh point cloud on the current submanifold. This is NOT differentiated.
2. Adam steps: with the point cloud frozen as initial conditions, run a short
   Newton refinement (differentiable through `refine_point_iterative`) and
   evaluate Lagrangian / special losses on the refined points.

Init:
- --init scratch     : random Uniform over the current genotype shape
- --init d1_zeropad  : GA.py canonical d=1 baseline, zero-padded (default)
- --init_pkl <path>  : load a (3, w) array; if w < current width, right-pad
                       with zeros. Overrides --init.

Examples:
    # d=1+2 default; plots auto-emit at the end.
    python gradient_descent.py --job_id run1 --steps 2000

    # d=1+2+3, preloading d=1+2 best as init.
    python gradient_descent.py --job_id run1_d3 --max_degree 3 --steps 2000 \
        --init_pkl gd_runs/gd_A_baseline_step10000.pkl

    # d=1+2+3+4, preloading d=1+2+3 best as init.
    python gradient_descent.py --job_id run1_d4 --max_degree 4 --steps 2000 \
        --init_pkl gd_runs/gd_run1_d3_step2000.pkl

    # Resume from a checkpoint and keep training (Adam moments restored).
    python gradient_descent.py --job_id run1_cont \
        --resume gd_runs/gd_run1_step2000.pkl --steps 4000

    # Just plot from an existing checkpoint, no training.
    python gradient_descent.py --job_id run1_plots \
        --resume gd_runs/gd_run1_step2000.pkl --plots_only

    # Smaller/faster plot mining.
    python gradient_descent.py --job_id run1 --steps 2000 \
        --plot_k 20000 --plot_newton_steps 50
"""

import argparse
import os
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from find_smooth_submanifold import (
    filter_and_refine,
    normalize_coeffs,
    refine_point_iterative,
)
from helper import (
    assert_metric_psi_compatible,
    convert_real_to_complex_batch,
    determine_patches_batch,
    dwork_points_path,
    format_array_with_commas,
    load_points as _load_points,
)
from viz.fitness_pipeline import run_fitness_pipeline
from sharding import (
    device_put_sharded, shard_leading_axis, take_replicated, unshard_leading_axis,
)
import cluster_select
import pointcloud_distance as pcd
from scipy.spatial import cKDTree
from slag_condition import (
    compute_holomorphic_form_restricted,
    compute_kahler_form_unrestricted,
    compute_special_condition_fitness,
    compute_special_condition_fitness_smooth,
    lagrangian_per_point_norms,
    top_lag_frac_indices,
    vmap_compute_affine_jacobian,
    vmap_compute_restriction,
)

jax.config.update("jax_enable_x64", True)

# Width per max-degree. Matches the static dispatch in
# helper.evaluate_equations_single_point.
GENOTYPE_WIDTHS = {1: 25, 2: 250, 3: 1475, 4: 6375}
# Default exported for back-compat with other modules (e.g. diagnose_phases).
GENOTYPE_SHAPE = (3, GENOTYPE_WIDTHS[2])


def genotype_shape(max_degree: int) -> tuple[int, int]:
    if max_degree not in GENOTYPE_WIDTHS:
        raise ValueError(f"max_degree must be one of {sorted(GENOTYPE_WIDTHS)}, got {max_degree}")
    return (3, GENOTYPE_WIDTHS[max_degree])


# Canonical d=1 baseline. Loaded from the run-folder pkl written by GA
# (or by tmp_dump_d1_coeffs.py for the historical literal).
D1_BASELINE_COEFFS_PATH = "plots_slag_d1_search/plots_slag_6338568_1_id0/coeffs.pkl"


def _load_d1_baseline_coeffs() -> jnp.ndarray:
    """Lazy load: only the few call sites that need the d=1 baseline pay the
    I/O cost, and a missing file gives an actionable error instead of an
    import-time crash."""
    if not os.path.exists(D1_BASELINE_COEFFS_PATH):
        raise FileNotFoundError(
            f"d=1 baseline coeffs not found at {D1_BASELINE_COEFFS_PATH}. "
            "Run `python tmp_dump_d1_coeffs.py` once to materialize the "
            "historical literal, or point --init_pkl at a GA species "
            "coeffs.pkl instead."
        )
    with open(D1_BASELINE_COEFFS_PATH, "rb") as f:
        return jnp.asarray(pickle.load(f))


def init_coeffs(mode: str, init_pkl, shape: tuple[int, int], key) -> jnp.ndarray:
    """Build initial coefficients of `shape`.

    Precedence: if `init_pkl` is set, load + right-pad (overrides `mode`).
    Otherwise dispatch on `mode`.
    """
    if init_pkl is not None:
        with open(init_pkl, "rb") as f:
            raw = pickle.load(f)
        # Accept either a bare array or a checkpoint dict with a "coeffs" key.
        if isinstance(raw, dict) and "coeffs" in raw:
            arr = jnp.asarray(raw["coeffs"])
        else:
            arr = jnp.asarray(raw)
        if arr.ndim != 2 or arr.shape[0] != shape[0] or arr.shape[1] > shape[1]:
            raise ValueError(
                f"--init_pkl: expected a ({shape[0]}, w) array with w <= {shape[1]}, "
                f"got {arr.shape}"
            )
        coeffs = jnp.zeros(shape).at[:, :arr.shape[1]].set(arr)
        print(f"  [init] loaded {arr.shape} from {init_pkl}, padded to {shape}")
    elif mode == "scratch":
        coeffs = jax.random.uniform(key, shape, minval=-0.1, maxval=0.1)
    elif mode == "d1_zeropad":
        d1 = _load_d1_baseline_coeffs()
        coeffs = jnp.zeros(shape).at[:, :25].set(d1)
    else:
        raise ValueError(f"Unknown init mode {mode}")
    return jnp.asarray(coeffs, dtype=jnp.float64)


def compute_losses_on_fixed_points(
    coeffs: jnp.ndarray,
    min_set_real: jnp.ndarray,
    psi: jnp.ndarray,
    n_refine_steps: int,
    metric: str,
    top_lag_frac: float,
):
    """Refine frozen init points under current coeffs, return (lag_loss, spec_loss).

    Points are ranked by the per-point Lagrangian condition; the best `top_lag_frac`
    fraction is kept and BOTH losses are computed on that subset (so a non-sLag
    disjoint component is excluded from the special loss too).
    """
    refine_fn = partial(
        refine_point_iterative, coeffs=coeffs, psi=psi, n_steps=n_refine_steps
    )
    min_set_real = jax.vmap(refine_fn)(min_set_real)

    min_set = convert_real_to_complex_batch(min_set_real)
    patch_indices = determine_patches_batch(min_set)

    jacobians = vmap_compute_affine_jacobian(min_set_real, patch_indices, coeffs, psi)
    restrictions = vmap_compute_restriction(jacobians)

    kahler_form_unrestricted = compute_kahler_form_unrestricted(
        min_set, patch_indices, metric=metric
    )
    kahler_form_restricted = jnp.einsum(
        "nij,nik,njl->nkl", kahler_form_unrestricted, restrictions, restrictions
    )
    frobenius_norms = jnp.linalg.norm(kahler_form_restricted, axis=(1, 2))
    normalization_factor = jnp.linalg.norm(kahler_form_unrestricted, axis=(1, 2))
    norms_normalized = frobenius_norms / (normalization_factor + 1e-9)

    sel = top_lag_frac_indices(norms_normalized, top_lag_frac)
    lagrangian_loss = jnp.mean(norms_normalized[sel])

    phases = compute_holomorphic_form_restricted(
        min_set, patch_indices, psi, restrictions, phase_only=True
    )
    order_parameter = compute_special_condition_fitness_smooth(phases[sel])
    special_loss = 1.0 - order_parameter

    return lagrangian_loss, special_loss


def compute_ga_fitness(
    min_set_real: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    metric: str,
    top_lag_frac: float,
):
    """GA-comparable (lag_fit, spec_fit) on the given points. No extra Newton.

    Uses the same conventions as compute_combined_fitness in slag_condition.py:
    rank points by the Lagrangian condition, keep the best `top_lag_frac` fraction, and
    on that subset compute
        lagrangian_fitness = exp(-10 * mean of restricted Frobenius norms),
        special_fitness    = histogram Shannon-entropy fitness (n_bins=100).
    """
    min_set = convert_real_to_complex_batch(min_set_real)
    patch_indices = determine_patches_batch(min_set)

    jacobians = vmap_compute_affine_jacobian(min_set_real, patch_indices, coeffs, psi)
    restrictions = vmap_compute_restriction(jacobians)

    kahler_form_unrestricted = compute_kahler_form_unrestricted(
        min_set, patch_indices, metric=metric
    )
    norms_normalized = lagrangian_per_point_norms(kahler_form_unrestricted, restrictions)
    sel = top_lag_frac_indices(norms_normalized, top_lag_frac)
    lag_fit = jnp.exp(-10 * jnp.mean(norms_normalized[sel]))

    phases = compute_holomorphic_form_restricted(
        min_set, patch_indices, psi, restrictions, phase_only=True
    )
    spec_fit = compute_special_condition_fitness(phases[sel], n_bins=100)
    return lag_fit, spec_fit


def make_total_loss(loss_kind: str, lag_weight: float, spec_weight: float, top_lag_frac: float):
    def total_loss(coeffs, min_set_real, psi, n_refine_steps, metric):
        lag, spec = compute_losses_on_fixed_points(
            coeffs, min_set_real, psi, n_refine_steps, metric, top_lag_frac
        )
        if loss_kind == "lag":
            total = lag_weight * lag
        elif loss_kind == "spec":
            total = spec_weight * spec
        elif loss_kind == "both":
            total = lag_weight * lag + spec_weight * spec
        else:
            raise ValueError(f"Unknown loss kind {loss_kind}")
        return total, (lag, spec)

    return total_loss


# ---------------------------------------------------------------------------
# Data-parallel wrappers. Each device computes a local loss/fitness on its
# shard of min_set_real with coeffs/psi replicated; we pmean losses and
# gradients across the device axis. Per-shard reductions (bottom-99% mean of
# Lagrangian norms; Kuramoto |mean exp(2i theta)|) are computed locally then
# averaged. At minset_size//num_devices >= ~1000 the bias vs the true global
# loss is negligible and gradient noise dominates.
# ---------------------------------------------------------------------------


def make_parallel_loss_and_grad(total_loss_fn, num_devices: int):
    """Build the (loss, grad) function. Single-device or pmap'd identically."""
    value_and_grad = jax.value_and_grad(total_loss_fn, argnums=0, has_aux=True)

    if num_devices <= 1:
        return jax.jit(value_and_grad, static_argnames=("n_refine_steps", "metric"))

    def per_device(coeffs, min_set_shard, psi, n_refine_steps, metric):
        (total, (lag, spec)), grads = value_and_grad(
            coeffs, min_set_shard, psi, n_refine_steps, metric
        )
        total = jax.lax.pmean(total, axis_name="x")
        lag = jax.lax.pmean(lag, axis_name="x")
        spec = jax.lax.pmean(spec, axis_name="x")
        grads = jax.lax.pmean(grads, axis_name="x")
        return (total, (lag, spec)), grads

    pmapped = jax.pmap(
        per_device,
        axis_name="x",
        in_axes=(None, 0, None, None, None),
        static_broadcasted_argnums=(3, 4),
    )

    def fn(coeffs, min_set_sharded, psi, n_refine_steps, metric):
        (total, (lag, spec)), grads = pmapped(
            coeffs, min_set_sharded, psi, n_refine_steps, metric
        )
        # pmean'd outputs are replicated along the device axis; collapse it.
        return (
            (take_replicated(total), (take_replicated(lag), take_replicated(spec))),
            take_replicated(grads),
        )

    return fn


def make_parallel_ga_fitness(num_devices: int):
    """Same shape as compute_ga_fitness but sharded over min_set_real.

    metric and top_lag_frac are static call-time args (passed by the caller),
    mirroring how the single-device path already threads metric.
    """
    if num_devices <= 1:
        return jax.jit(compute_ga_fitness, static_argnames=("metric", "top_lag_frac"))

    def per_device(min_set_shard, coeffs, psi, metric, top_lag_frac):
        lag_fit, spec_fit = compute_ga_fitness(min_set_shard, coeffs, psi, metric, top_lag_frac)
        return (
            jax.lax.pmean(lag_fit, axis_name="x"),
            jax.lax.pmean(spec_fit, axis_name="x"),
        )

    pmapped = jax.pmap(
        per_device,
        axis_name="x",
        in_axes=(0, None, None, None, None),
        static_broadcasted_argnums=(3, 4),
    )

    def fn(min_set_sharded, coeffs, psi, metric, top_lag_frac):
        lag_fit, spec_fit = pmapped(min_set_sharded, coeffs, psi, metric, top_lag_frac)
        return take_replicated(lag_fit), take_replicated(spec_fit)

    return fn


def make_parallel_mining(num_devices: int):
    """Sharded filter_and_refine. Each device mines its own slice of points_real
    for k_per_device = k // num_devices points; outputs are concatenated.

    The repulsion step inside filter_and_refine runs intra-shard only (it
    cannot see cross-shard neighbors). This is a small approximation of the
    single-device uniformity heuristic — fine when each shard has >> 100
    points.

    Inputs:
      points_sharded: (D, M, 10) device-sharded array of CY points
      coeffs, psi: replicated (broadcast)
      k: TOTAL desired output size; must be divisible by num_devices
      n_refine_steps: passed through (static)
    Returns:
      min_set_sharded: (D, k/D, 10) sharded
      distances: (D, k/D) sharded
      check: scalar host bool (AND across all devices)
    """
    if num_devices <= 1:
        def fn(points, coeffs, psi, k, n_refine_steps):
            return filter_and_refine(
                points, coeffs, psi, k, n_refine_steps, filter_newton=True,
            )
        return fn

    def per_device(points_shard, coeffs, psi, k_per_dev, n_refine_steps):
        return filter_and_refine(
            points_shard, coeffs, psi, k_per_dev, n_refine_steps, filter_newton=True,
        )

    pmapped = jax.pmap(
        per_device,
        axis_name="x",
        in_axes=(0, None, None, None, None),
        static_broadcasted_argnums=(3, 4),
    )

    def fn(points_sharded, coeffs, psi, k, n_refine_steps):
        if k % num_devices != 0:
            raise ValueError(
                f"minset/plot k={k} not divisible by num_devices={num_devices}"
            )
        k_per_dev = k // num_devices
        min_set_sharded, distances_sharded, check_per_dev = pmapped(
            points_sharded, coeffs, psi, k_per_dev, n_refine_steps,
        )
        # check_per_dev shape (D,); AND across devices.
        return min_set_sharded, distances_sharded, jnp.all(check_per_dev)

    return fn


def mine_one_cluster(mining_fn, points_in, coeffs, psi, args, num_devices, anchor, rng,
                     tube=None):
    """Mine, then extract a fixed-size single-component min-set. Returns
    (min_set_real, distances, new_anchor, info).

    Membership modes:
      - tube set (single-component GD): keep mined points within tube["radius"]
        of the frozen reference cloud in FS-feature space. Membership is
        independent of density/fitness/phase, so the objective covers the whole
        component -- no core-only feedback loop (points the optimizer neglects
        stay in the loss instead of being dropped as HDBSCAN noise).
      - args.target_cluster set: HDBSCAN + anchor tracking (legacy path).
      - neither: passthrough -- mines args.minset_size and returns it unchanged,
        so non-cluster runs are byte-for-byte the old path.
    """
    if args.target_cluster is None and tube is None:
        min_set_real, distances, _ = mining_fn(
            points_in, coeffs, psi, args.minset_size, args.newton_steps,
        )
        return min_set_real, distances, anchor, None

    k_mine = args.mine_oversample * args.cluster_minset_size
    min_set_raw, distances, _ = mining_fn(
        points_in, coeffs, psi, k_mine, args.newton_steps,
    )
    host = (np.asarray(unshard_leading_axis(min_set_raw))
            if num_devices > 1 else np.asarray(min_set_raw))   # (k_mine, 10)
    z = host[:, :5] + 1j * host[:, 5:]                          # (k_mine, 5) complex
    feats = cluster_select.fs_features(z)
    if tube is not None:
        d_ref, _ = tube["tree"].query(feats, k=1)
        member_mask = d_ref < tube["radius"]
        n_mem = int(member_mask.sum())
        if n_mem == 0:
            raise RuntimeError(
                "tube membership kept 0 mined points: the mined zero set no "
                "longer intersects the reference tube (component moved or "
                "coeffs degenerated). Refresh --tube_ref or raise --tube_mult.")
        # W1 drift of the members vs the reference: the "is the torus moving"
        # monitor. Small n_pairs keeps it cheap at every mine.
        drift = pcd.pairwise_distance_drift(
            tube["ref_feats"], feats[member_mask], n_pairs=50_000,
            rng=np.random.default_rng(0))
        print(f"  [tube] kept {n_mem}/{host.shape[0]} mined pts "
              f"(pad -> {args.cluster_minset_size}); W1 vs ref {drift:.2e}")
        if n_mem < args.cluster_minset_size // 5:
            print("  [tube] WARN: <20% unique members -- heavy padding; "
                  "consider raising --mine_oversample")
        new_anchor = anchor
        info = {"n_members": n_mem, "drift_vs_ref": float(drift)}
        member_idx = np.flatnonzero(member_mask)
        fixed_idx = cluster_select.fill_to_size(member_idx, args.cluster_minset_size, rng)
        cluster_real = host[fixed_idx]
        min_set_real = (shard_leading_axis(jnp.asarray(cluster_real), num_devices)
                        if num_devices > 1 else jnp.asarray(cluster_real))
        return min_set_real, distances, new_anchor, info

    member_mask, new_anchor, info = cluster_select.select_cluster(
        feats, anchor, args.target_cluster,
        args.min_cluster_size, args.min_cluster_frac, args.cluster_selection_epsilon,
    )
    if anchor is None:
        print(f"  [cluster] detected {info['n_components']} component(s), "
              f"sizes {info['sizes']}; following component {info['chosen']} "
              f"(of {host.shape[0]} mined pts)")
    else:
        print(f"  [cluster] {info['n_components']} component(s), sizes "
              f"{info['sizes']}; tracked -> component {info['chosen']}")
    member_idx = np.flatnonzero(member_mask)
    fixed_idx = cluster_select.fill_to_size(member_idx, args.cluster_minset_size, rng)
    cluster_real = host[fixed_idx]                             # (cluster_minset_size, 10)
    min_set_real = (shard_leading_axis(jnp.asarray(cluster_real), num_devices)
                    if num_devices > 1 else jnp.asarray(cluster_real))
    return min_set_real, distances, new_anchor, info


def load_points(psi, path=None):
    """Resolve via dwork_points_path if no explicit path, then load.

    Returns (points_real, resolved_path) so the caller can log the source.
    """
    if path is None:
        path = dwork_points_path(psi, seed=1024)
    return _load_points(path), path


def _run_all_plots(points_real, coeffs, psi, args, num_devices: int = 1):
    """Emit one self-contained run folder for the final GD coeffs:

      plots_slag_<job_id>/  GD coeffs vs random  (fixed Kähler x-range)
        coeffs.pkl, min_set.pkl, frobenius_norms.npy, phases.npy
        Kahler_form_loss_histogram.png, circular_phase_histogram.png
        coord_scatter_{re,im,abs}_fitness.png

    Cross-degree / vs-d1 / truncation comparisons are no longer auto-emitted
    here. Run them post-hoc against existing run folders:

        python -m viz.plot_histograms \\
            --runs gd_runs/plots_slag_d2 gd_runs/plots_slag_d3 gd_runs/plots_slag_d4 \\
            --labels d=2 d=3 d=4 --out_dir gd_runs/compare_d2_d3_d4

    num_devices > 1 routes filter_and_refine and the per-point diagnostics
    through a pmap'd path inside run_fitness_pipeline.
    """
    base = os.path.join(args.out_dir, f"plots_slag_{args.job_id}")
    print(f"\n=== Plotting GD coeffs vs random -> {base} ===")
    run_fitness_pipeline(
        points_real, coeffs, psi,
        k=args.plot_k, n_refine_steps=args.plot_newton_steps,
        metric=args.metric, compare_with="random",
        out_dir=base,
        num_devices=num_devices,
        top_lag_frac=args.top_lag_frac,
    )


def make_parallel_lbfgs_step(opt, total_loss_fn, num_devices: int,
                             n_refine_steps: int, metric: str):
    """One full L-BFGS step (value+grad -> opt.update -> apply_updates -> normalize),
    sharded across devices in the same data-parallel-with-pmean pattern Adam uses.

    Replication / sharding:
      coeffs, opt_state : broadcast (in_axes=None) -> identical on every device
      min_set           : sharded on leading axis (in_axes=0)
      psi               : broadcast

    Inside pmap, value_fn = lax.pmean(local_loss(c), 'x'). optax.lbfgs's
    internal zoom line search jit-traces value_fn; the pmean fires on every
    trial coefficient, so every device sees an identical global loss for the
    trial step. The outer value_and_grad's loss/aux/grads are likewise
    pmean'd before opt.update, so every device computes the same updates ->
    coeffs / opt_state remain replicated post-update.

    Bias note: the loss is a per-shard mean (bottom-99% Lagrangian, Kuramoto
    |mean exp(2i theta)|), then pmean'd. This is biased vs the true global
    loss on the union of shards, but it is the *same* biased loss Adam
    minimizes — consistent across iterations, fine for L-BFGS curvature.
    """
    def inner(coeffs, opt_state, min_set, psi):
        def loss_with_aux(c):
            return total_loss_fn(c, min_set, psi, n_refine_steps, metric)

        if num_devices > 1:
            def value_fn(c):
                local_loss, _ = loss_with_aux(c)
                return jax.lax.pmean(local_loss, axis_name="x")
        else:
            def value_fn(c):
                local_loss, _ = loss_with_aux(c)
                return local_loss

        (loss, (lag, spec)), grads = jax.value_and_grad(
            loss_with_aux, has_aux=True
        )(coeffs)

        if num_devices > 1:
            loss = jax.lax.pmean(loss, axis_name="x")
            lag = jax.lax.pmean(lag, axis_name="x")
            spec = jax.lax.pmean(spec, axis_name="x")
            grads = jax.lax.pmean(grads, axis_name="x")

        grads = jnp.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
        updates, new_opt_state = opt.update(
            grads, opt_state, coeffs,
            value=loss, grad=grads, value_fn=value_fn,
        )
        new_coeffs = optax.apply_updates(coeffs, updates)
        new_coeffs = normalize_coeffs(new_coeffs)
        return new_coeffs, new_opt_state, loss, lag, spec, grads

    if num_devices <= 1:
        return jax.jit(inner)

    pmapped = jax.pmap(inner, axis_name="x", in_axes=(None, None, 0, None))

    def fn(coeffs, opt_state, min_set_sharded, psi):
        new_coeffs, new_opt_state, loss, lag, spec, grads = pmapped(
            coeffs, opt_state, min_set_sharded, psi
        )
        # All outputs are replicated along the device axis; collapse it.
        return (
            take_replicated(new_coeffs),
            jax.tree.map(take_replicated, new_opt_state),
            take_replicated(loss),
            take_replicated(lag),
            take_replicated(spec),
            take_replicated(grads),
        )

    return fn


def run_lbfgs_finisher(coeffs, points_in, psi, args, total_loss_fn,
                       mining_fn, ga_fitness_fn, num_devices: int, history):
    """L-BFGS polish on a freshly-mined frozen point set.

    Multi-GPU follows the same pmap+pmean pattern as Adam: replicated
    coeffs/opt_state, sharded min_set. See make_parallel_lbfgs_step for the
    sharding contract. `points_in` is the sharded points array if num_devices
    > 1, else the raw (N, 10) array — same convention as in main().

    Mutates `history` in place. Returns (new_coeffs, opt_state).
    """
    try:
        opt = optax.lbfgs(memory_size=args.lbfgs_memory_size)
    except AttributeError as e:
        raise RuntimeError(
            "optax.lbfgs not available. Requires a recent optax (>=0.2.x with "
            "the L-BFGS optimizer). Upgrade optax."
        ) from e

    print(
        f"\n=== L-BFGS finisher ({num_devices} GPU(s)): "
        f"max {args.lbfgs_steps} steps, tol={args.lbfgs_tol:.2e}, "
        f"memory_size={args.lbfgs_memory_size} ==="
    )

    # Fresh mining (uses the multi-GPU mining path if num_devices > 1).
    min_set_data, distances, _ = mining_fn(
        points_in, coeffs, psi, args.minset_size, args.newton_steps,
    )
    mean_d = float(jnp.mean(distances))
    max_d = float(jnp.max(distances))
    print(f"  [lbfgs mining] mean_dist {mean_d:.2e}  max_dist {max_d:.2e}")
    if mean_d > 1e-4:
        print(f"  [warn] mean Newton distance > 1e-4 -- points may not be on the manifold")

    step_fn = make_parallel_lbfgs_step(
        opt, total_loss_fn, num_devices,
        args.inner_newton_steps, args.metric,
    )
    opt_state = opt.init(coeffs)
    base_step = history[-1]["step"] if history else 0

    for it in range(args.lbfgs_steps):
        t0 = time.time()
        coeffs, opt_state, loss_val, lag_loss, spec_loss, grads = step_fn(
            coeffs, opt_state, min_set_data, psi
        )
        gnorm = float(jnp.linalg.norm(grads))

        lag_fit, spec_fit = ga_fitness_fn(min_set_data, coeffs, psi, args.metric, args.top_lag_frac)
        lag_fit = float(lag_fit)
        spec_fit = float(spec_fit)

        dt = time.time() - t0
        step_num = base_step + it + 1
        print(
            f"lbfgs {it+1:4d} | loss {float(loss_val):.6f} | "
            f"lag_loss {float(lag_loss):.6f} | spec_loss {float(spec_loss):.6f} | "
            f"lag_fit {lag_fit:.4f} | spec_fit {spec_fit:.4f} | "
            f"|grad| {gnorm:.2e} | {dt:.2f}s"
        )
        history.append({
            "step": step_num,
            "phase": "lbfgs",
            "loss": float(loss_val),
            "lag_loss": float(lag_loss),
            "spec_loss": float(spec_loss),
            "lag_fit": lag_fit,
            "spec_fit": spec_fit,
            "gnorm": gnorm,
        })

        if gnorm < args.lbfgs_tol:
            print(f"  [lbfgs] |grad|={gnorm:.2e} < tol={args.lbfgs_tol:.2e}, converged")
            break

    return coeffs, opt_state


def main():
    parser = argparse.ArgumentParser(description="GD for sLag search (d=1+2)")
    parser.add_argument("--psi", type=complex, default=0)
    parser.add_argument("--points_file", type=str, default=None,
                        help="Override path to the point-cloud pkl. "
                             "Default: helper.dwork_points_path(psi).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--mine_interval", type=int, default=10)
    parser.add_argument("--minset_size", type=int, default=10000)
    parser.add_argument("--newton_steps", type=int, default=40,
                        help="Newton steps in (re-)mining (filter_and_refine).")
    parser.add_argument("--inner_newton_steps", type=int, default=10,
                        help="Newton steps inside the differentiated loss.")
    parser.add_argument("--metric", type=str, default="k4_fermat",
                        choices=["FS", "k4_fermat"])
    parser.add_argument("--loss", type=str, default="both",
                        choices=["lag", "spec", "both"])
    parser.add_argument("--lag_weight", type=float, default=1.0,
                        help="Weight on Lagrangian loss (used when --loss is 'lag' or 'both').")
    parser.add_argument("--spec_weight", type=float, default=1.0,
                        help="Weight on special loss (used when --loss is 'spec' or 'both').")
    parser.add_argument("--top_lag_frac", type=float, default=0.99,
                        help="Fraction of mined points (ranked by the Lagrangian "
                             "condition, best first) kept. The special/phase "
                             "condition is evaluated ONLY on these top-Lagrangian "
                             "points (as is the Lagrangian condition). 1.0 = all "
                             "points; 0.99 (default) reproduces the historical "
                             "worst-1%% trim. Lower it (e.g. 0.5) to test whether "
                             "only one disjoint piece of the zero set is sLag. "
                             "Multi-GPU applies it per-shard then averages (biased "
                             "for small top_lag_frac).")
    parser.add_argument("--max_degree", type=int, default=2,
                        choices=sorted(GENOTYPE_WIDTHS),
                        help="Ansatz max degree: 1 -> (3,25), 2 -> (3,250), "
                             "3 -> (3,1475), 4 -> (3,6375).")
    parser.add_argument("--init", type=str, default="d1_zeropad",
                        choices=["scratch", "d1_zeropad"],
                        help="Synthetic init mode. Ignored if --init_pkl is set.")
    parser.add_argument("--init_pkl", type=str, default=None,
                        help="Path to a pkl with a (3, w) array or a checkpoint dict; "
                             "right-padded with zeros to the current genotype width. "
                             "Overrides --init.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--job_id", type=str, default="0")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="./gd_runs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a full checkpoint pkl to resume from. "
                             "Overrides --init and restores coeffs, opt_state, "
                             "step counter, and training history.")
    parser.add_argument("--make_plots", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Call run_fitness_pipeline on the final coeffs "
                             "(same plots as GA.py). Use --no-make_plots to skip.")
    parser.add_argument("--plots_only", action="store_true",
                        help="Skip training. Load --resume <ckpt>, run "
                             "run_fitness_pipeline, exit.")
    parser.add_argument("--plot_k", type=int, default=80000,
                        help="Point cloud size for the final plots.")
    parser.add_argument("--plot_newton_steps", type=int, default=80,
                        help="Newton refinement steps for the final plots.")
    parser.add_argument("--lbfgs_steps", type=int, default=0,
                        help="If > 0, run optax.lbfgs as a finisher after the "
                             "Adam loop on a freshly-mined frozen point set. "
                             "Single-device (multi-GPU pmap is incompatible "
                             "with optax.lbfgs's internal line search).")
    parser.add_argument("--lbfgs_tol", type=float, default=1e-6,
                        help="L-BFGS gradient-norm stopping tolerance.")
    parser.add_argument("--lbfgs_memory_size", type=int, default=10,
                        help="L-BFGS history length (number of (s,y) pairs).")
    parser.add_argument("--target_cluster", type=int, default=None,
                        help="Optimize on ONE geometric component of the mined "
                             "zero set (0-indexed; components ranked by descending "
                             "size at the first mine, then anchor-tracked). HDBSCAN "
                             "on FS features; needs scikit-learn>=1.3. Default None "
                             "= whole manifold (unchanged behavior).")
    parser.add_argument("--min_cluster_size", type=int, default=200,
                        help="HDBSCAN min_cluster_size (only with --target_cluster).")
    parser.add_argument("--cluster_selection_epsilon", type=float, default=0.0,
                        help="HDBSCAN cluster_selection_epsilon (merge components "
                             "closer than this; raise to keep a fat neck merged).")
    parser.add_argument("--min_cluster_frac", type=float, default=0.02,
                        help="Drop components smaller than this fraction of the "
                             "mined points as noise.")
    parser.add_argument("--cluster_minset_size", type=int, default=None,
                        help="Fixed per-cluster min-set size fed to the loss. "
                             "Default: --minset_size.")
    parser.add_argument("--mine_oversample", type=int, default=2,
                        help="Mine this multiple of --cluster_minset_size so the "
                             "target component is well-populated before extraction.")
    parser.add_argument("--tube_ref", type=str, default=None,
                        help="Single-component GD: path to an (N, 5) complex pkl "
                             "sampling the target component (e.g. a cluster_split "
                             "output). Each mine keeps only points within a kNN "
                             "tube of this frozen reference in FS-feature space -- "
                             "membership is independent of density/fitness/phase, "
                             "so the objective covers the WHOLE component instead "
                             "of the HDBSCAN core. Mutually exclusive with "
                             "--target_cluster.")
    parser.add_argument("--tube_mult", type=float, default=2.0,
                        help="Tube radius = tube_mult x the reference cloud's "
                             "NN-distance percentile (--tube_percentile).")
    parser.add_argument("--tube_percentile", type=float, default=95.0,
                        help="Percentile of the reference cloud's internal "
                             "NN distances used as the radius scale.")
    args = parser.parse_args()

    if args.tube_ref is not None and args.target_cluster is not None:
        parser.error("--tube_ref and --target_cluster are mutually exclusive")
    if args.cluster_minset_size is None:
        args.cluster_minset_size = args.minset_size

    os.makedirs(args.out_dir, exist_ok=True)
    shape = genotype_shape(args.max_degree)
    init_desc = f"init_pkl={args.init_pkl}" if args.init_pkl is not None else f"init={args.init}"
    print(f"=== GD for sLag search (max_degree={args.max_degree}, shape={shape}) ===")
    print(f"job_id={args.job_id} {init_desc} loss={args.loss} "
          f"(lag_w={args.lag_weight} spec_w={args.spec_weight} top_lag_frac={args.top_lag_frac}) "
          f"lr={args.lr} steps={args.steps}")
    print(f"mine_interval={args.mine_interval} minset_size={args.minset_size} "
          f"newton_steps={args.newton_steps} inner_newton_steps={args.inner_newton_steps}")

    assert_metric_psi_compatible(args.metric, args.psi)
    points_real, src_path = load_points(args.psi, path=args.points_file)
    print(f"Loaded {len(points_real)} points from {src_path}")
    psi = jnp.asarray(args.psi, dtype=jnp.complex128)

    num_devices = jax.local_device_count()
    print(f"Detected {num_devices} GPU(s).")
    if num_devices > 1:
        # Truncate points_real to a multiple of num_devices once, then shard.
        n_keep = (points_real.shape[0] // num_devices) * num_devices
        if n_keep != points_real.shape[0]:
            print(f"  [shard] truncating {points_real.shape[0]} -> {n_keep} "
                  f"points to be divisible by {num_devices} GPUs")
            points_real = points_real[:n_keep]
        points_sharded = shard_leading_axis(points_real, num_devices)
        if args.minset_size % num_devices != 0:
            raise ValueError(
                f"--minset_size {args.minset_size} not divisible by "
                f"num_devices={num_devices}"
            )
        if args.plot_k % num_devices != 0:
            raise ValueError(
                f"--plot_k {args.plot_k} not divisible by "
                f"num_devices={num_devices}"
            )
        if args.target_cluster is not None or args.tube_ref is not None:
            if args.cluster_minset_size % num_devices != 0:
                raise ValueError(
                    f"--cluster_minset_size {args.cluster_minset_size} not "
                    f"divisible by num_devices={num_devices}")
            if (args.mine_oversample * args.cluster_minset_size) % num_devices != 0:
                raise ValueError(
                    f"mine_oversample*cluster_minset_size="
                    f"{args.mine_oversample * args.cluster_minset_size} not "
                    f"divisible by num_devices={num_devices}")
    else:
        points_sharded = None  # single-device path uses points_real directly

    if args.plots_only:
        if args.resume is None:
            raise ValueError("--plots_only requires --resume <ckpt.pkl>")
        with open(args.resume, "rb") as f:
            ckpt = pickle.load(f)
        coeffs = jnp.asarray(ckpt["coeffs"], dtype=jnp.float64)
        print(f"=== Plots only: coeffs from {args.resume} ===")
        _run_all_plots(points_real, coeffs, psi, args, num_devices=num_devices)
        print("Done.")
        return

    optimizer = optax.adam(learning_rate=args.lr)
    start_step = 0
    if args.resume is not None:
        with open(args.resume, "rb") as f:
            ckpt = pickle.load(f)
        if "opt_state" not in ckpt or "step" not in ckpt:
            raise ValueError(
                f"Checkpoint {args.resume} is missing opt_state/step "
                "(probably a pre-resume checkpoint). Use --init_pkl <path> "
                "to load bare coeffs instead."
            )
        coeffs = jnp.asarray(ckpt["coeffs"], dtype=jnp.float64)
        if coeffs.shape != shape:
            raise ValueError(
                f"Checkpoint coeffs shape {coeffs.shape} does not match "
                f"--max_degree {args.max_degree} (expects {shape}). Resume "
                "uses the checkpoint shape as-is; pass --max_degree to match."
            )
        opt_state = jax.tree.map(jnp.asarray, ckpt["opt_state"])
        history = list(ckpt["history"])
        start_step = int(ckpt["step"])
        cluster_anchor = ckpt.get("anchor")
        print(f"=== Resumed from {args.resume} at step {start_step} ===")
        if start_step > args.steps or (start_step == args.steps and args.lbfgs_steps == 0):
            raise ValueError(
                f"Checkpoint is at step {start_step} but --steps is {args.steps}. "
                "Pass a larger --steps to continue Adam, or --lbfgs_steps > 0 "
                "to skip Adam and run the L-BFGS finisher only."
            )
    else:
        key = jax.random.PRNGKey(args.seed)
        key, sub = jax.random.split(key)
        coeffs = init_coeffs(args.init, args.init_pkl, shape, sub)
        coeffs = normalize_coeffs(coeffs)
        opt_state = optimizer.init(coeffs)
        history = []
        cluster_anchor = None

    total_loss = make_total_loss(args.loss, args.lag_weight, args.spec_weight, args.top_lag_frac)
    loss_value_and_grad = make_parallel_loss_and_grad(total_loss, num_devices)
    ga_fitness_jit = make_parallel_ga_fitness(num_devices)
    mining_fn = make_parallel_mining(num_devices)

    # In multi-GPU mode, min_set_real is a (D, k/D, 10) sharded array that we
    # carry directly between mining and the loss/fitness call — no host
    # round-trip. In single-GPU mode it's the usual (k, 10) array.
    points_in = points_sharded if num_devices > 1 else points_real

    cluster_rng = np.random.default_rng(args.seed)

    # Single-component GD: build the frozen-reference tube once. The reference
    # cloud defines component membership for every mine; it is deliberately
    # NOT updated during training (frozen => membership independent of the
    # optimization state). Refresh manually by re-pointing --tube_ref.
    tube = None
    if args.tube_ref is not None:
        with open(args.tube_ref, "rb") as f:
            ref_z = np.asarray(pickle.load(f))
        ref_feats = cluster_select.fs_features(ref_z)
        ref_tree = cKDTree(ref_feats)
        nn = ref_tree.query(ref_feats, k=2)[0][:, 1]
        radius = float(args.tube_mult * np.percentile(nn, args.tube_percentile))
        tube = {"tree": ref_tree, "radius": radius, "ref_feats": ref_feats}
        print(f"[tube] single-component GD: reference {ref_z.shape} from "
              f"{args.tube_ref}; radius={radius:.4f} "
              f"(={args.tube_mult} x NN p{args.tube_percentile:.0f})")

    # Initial mining + loss eval (also re-runs on resume to repopulate min_set_real).
    min_set_real, distances, cluster_anchor, _ = mine_one_cluster(
        mining_fn, points_in, coeffs, psi, args, num_devices, cluster_anchor, cluster_rng,
        tube=tube,
    )
    mean_d, max_d = float(jnp.mean(distances)), float(jnp.max(distances))
    print(f"  [mining] mean_dist {mean_d:.2e}  max_dist {max_d:.2e}")
    if mean_d > 1e-4:
        print(f"  [warn] mean Newton distance > 1e-4 -- points may not be on the manifold")
    (init_loss, (init_lag, init_spec)), _ = loss_value_and_grad(
        coeffs, min_set_real, psi, args.inner_newton_steps, args.metric
    )
    init_lag_fit, init_spec_fit = ga_fitness_jit(min_set_real, coeffs, psi, args.metric, args.top_lag_frac)
    init_lag_fit = float(init_lag_fit)
    init_spec_fit = float(init_spec_fit)
    label = "resumed   " if args.resume is not None else "initial   "
    print(
        f"{label}  | loss {float(init_loss):.6f} | "
        f"lag_loss {float(init_lag):.6f} | spec_loss {float(init_spec):.6f} | "
        f"lag_fit {init_lag_fit:.4f} | spec_fit {init_spec_fit:.4f}"
    )
    if args.resume is None:
        history.append({
            "step": 0,
            "loss": float(init_loss),
            "lag_loss": float(init_lag),
            "spec_loss": float(init_spec),
            "lag_fit": init_lag_fit,
            "spec_fit": init_spec_fit,
            "gnorm": None,
        })

    for step in range(start_step, args.steps):
        t0 = time.time()
        # Skip step==0: just mined for the initial eval. Mining schedule
        # then fires at step==mine_interval, 2*mine_interval, etc.
        if step > 0 and step % args.mine_interval == 0:
            min_set_real, distances, cluster_anchor, _ = mine_one_cluster(
                mining_fn, points_in, coeffs, psi, args, num_devices,
                cluster_anchor, cluster_rng, tube=tube,
            )
            mean_d, max_d = float(jnp.mean(distances)), float(jnp.max(distances))
            print(f"  [mining @ step {step}] mean_dist {mean_d:.2e}  max_dist {max_d:.2e}")
            if mean_d > 1e-4:
                print(f"  [warn] mean Newton distance > 1e-4 -- points may not be on the manifold")

        (loss_val, (lag_loss, spec_loss)), grads = loss_value_and_grad(
            coeffs, min_set_real, psi, args.inner_newton_steps, args.metric
        )
        grads = jnp.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
        updates, opt_state = optimizer.update(grads, opt_state, coeffs)
        coeffs = optax.apply_updates(coeffs, updates)
        coeffs = normalize_coeffs(coeffs)

        # GA-comparable fitness on the post-update coeffs and (un-inner-Newton'd) min_set.
        lag_fit, spec_fit = ga_fitness_jit(min_set_real, coeffs, psi, args.metric, args.top_lag_frac)
        lag_fit = float(lag_fit)
        spec_fit = float(spec_fit)

        gnorm = float(jnp.linalg.norm(grads))
        dt = time.time() - t0
        print(
            f"step {step+1:5d} | loss {float(loss_val):.6f} | "
            f"lag_loss {float(lag_loss):.6f} | spec_loss {float(spec_loss):.6f} | "
            f"lag_fit {lag_fit:.4f} | spec_fit {spec_fit:.4f} | "
            f"|grad| {gnorm:.2e} | {dt:.2f}s"
        )
        history.append({
            "step": step + 1,
            "loss": float(loss_val),
            "lag_loss": float(lag_loss),
            "spec_loss": float(spec_loss),
            "lag_fit": lag_fit,
            "spec_fit": spec_fit,
            "gnorm": gnorm,
        })

        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            ckpt = os.path.join(args.out_dir, f"gd_{args.job_id}_step{step+1}.pkl")
            payload = {
                "coeffs": np.asarray(coeffs),
                "opt_state": jax.tree.map(np.asarray, opt_state),
                "history": history,
                "step": step + 1,
                "args": vars(args),
                "anchor": None if cluster_anchor is None else np.asarray(cluster_anchor),
                "target_cluster": args.target_cluster,
            }
            tmp = ckpt + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(payload, f)
            os.replace(tmp, ckpt)
            print(f"  [save] wrote {ckpt}")

    if args.lbfgs_steps > 0:
        coeffs, lbfgs_opt_state = run_lbfgs_finisher(
            coeffs, points_in, psi, args, total_loss,
            mining_fn, ga_fitness_jit, num_devices, history,
        )
        lbfgs_ckpt = os.path.join(args.out_dir, f"gd_{args.job_id}_lbfgs.pkl")
        last_step = history[-1]["step"] if history else 0
        payload = {
            "coeffs": np.asarray(coeffs),
            "opt_state": jax.tree.map(np.asarray, lbfgs_opt_state),
            "history": history,
            "step": last_step,
            "args": vars(args),
            "phase": "lbfgs",
            "anchor": None if cluster_anchor is None else np.asarray(cluster_anchor),
            "target_cluster": args.target_cluster,
        }
        tmp = lbfgs_ckpt + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp, lbfgs_ckpt)
        print(f"  [save] wrote {lbfgs_ckpt}")

    print("\nFinal coeffs:")
    print(format_array_with_commas(coeffs))

    if args.make_plots:
        _run_all_plots(points_real, coeffs, psi, args, num_devices=num_devices)


if __name__ == "__main__":
    main()
