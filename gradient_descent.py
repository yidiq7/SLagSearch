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
from plots import make_fitness_plots
from sharding import device_put_sharded, shard_leading_axis, take_replicated
from slag_condition import (
    compute_holomorphic_form_restricted,
    compute_kahler_form_unrestricted,
    compute_lagrangian_condition_fitness,
    compute_special_condition_fitness,
    compute_special_condition_fitness_smooth,
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


# Canonical d=1 baseline. Mirrors GA.py:409 d1_coeffs.
D1_COEFFS = jnp.array([
    [-0.2085878998041153, 0.08078225702047348, 0.12364989519119263, 0.42693421244621277, -0.4276507794857025, 0.05941963940858841, -0.19358153641223907, 0.2884068787097931, 0.2374262660741806, 0.17124612629413605, -0.03099866583943367, 0.07415380328893661, -0.22672683000564575, -0.1914607286453247, 0.09337177127599716, -0.053066715598106384, -0.06608302891254425, -0.3771730363368988, 0.05378381162881851, 0.0064529310911893845, 0.2938925623893738, 0.08852922171354294, 0.020463770255446434, 0.09666207432746887, -0.006990742404013872],
    [-0.1065014973282814, 0.20087268948554993, 0.18935158848762512, -0.17352613806724548, 0.05884088575839996, -0.4646260440349579, -0.10628655552864075, -0.28338274359703064, -0.03379037603735924, 0.007989203557372093, -0.06132059171795845, -0.13810740411281586, 0.04504100978374481, 0.015115765854716301, -0.4030528962612152, -0.025872472673654556, -0.4061300754547119, -0.02022559940814972, -0.13893099129199982, 0.10193423181772232, 0.29334160685539246, 0.22542181611061096, -0.050897762179374695, 0.21366965770721436, -0.04277477413415909],
    [0.054688308387994766, 0.07500440627336502, 0.060474496334791183, -0.3848169445991516, -0.3781052529811859, 0.38639041781425476, 0.021527282893657684, 0.4060642719268799, -0.15761728584766388, -0.1271764189004898, -0.01066557876765728, -0.13985656201839447, 0.1605837494134903, 0.15716029703617096, -0.32516127824783325, 0.016290534287691116, 0.2249980866909027, -0.2878168523311615, -0.12032820284366608, -0.04713383689522743, 0.025025269016623497, 0.08448748290538788, 0.05337755009531975, 0.05431513488292694, -0.03361976519227028]
])


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
        coeffs = jnp.zeros(shape).at[:, :25].set(D1_COEFFS)
    else:
        raise ValueError(f"Unknown init mode {mode}")
    return jnp.asarray(coeffs, dtype=jnp.float64)


def compute_losses_on_fixed_points(
    coeffs: jnp.ndarray,
    min_set_real: jnp.ndarray,
    psi: jnp.ndarray,
    n_refine_steps: int,
    metric: str,
):
    """Refine frozen init points under current coeffs, return (lag_loss, spec_loss)."""
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

    sorted_norms = jnp.sort(norms_normalized)
    cutoff_index = int(sorted_norms.shape[0] * 0.99)
    lagrangian_loss = jnp.mean(sorted_norms[:cutoff_index])

    phases = compute_holomorphic_form_restricted(
        min_set, patch_indices, psi, restrictions, phase_only=True
    )
    order_parameter = compute_special_condition_fitness_smooth(phases)
    special_loss = 1.0 - order_parameter

    return lagrangian_loss, special_loss


def compute_ga_fitness(
    min_set_real: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    metric: str,
):
    """GA-comparable (lag_fit, spec_fit) on the given points. No extra Newton.

    Uses the same conventions as compute_combined_fitness in slag_condition.py:
    lagrangian_fitness = exp(-10 * mean of bottom-99% restricted Frobenius norms),
    special_fitness    = histogram Shannon-entropy fitness (n_bins=100).
    """
    min_set = convert_real_to_complex_batch(min_set_real)
    patch_indices = determine_patches_batch(min_set)

    jacobians = vmap_compute_affine_jacobian(min_set_real, patch_indices, coeffs, psi)
    restrictions = vmap_compute_restriction(jacobians)

    kahler_form_unrestricted = compute_kahler_form_unrestricted(
        min_set, patch_indices, metric=metric
    )
    lag_fit = compute_lagrangian_condition_fitness(
        kahler_form_unrestricted, restrictions, k=10
    )

    phases = compute_holomorphic_form_restricted(
        min_set, patch_indices, psi, restrictions, phase_only=True
    )
    spec_fit = compute_special_condition_fitness(phases, n_bins=100)
    return lag_fit, spec_fit


def make_total_loss(loss_kind: str, lag_weight: float, spec_weight: float):
    def total_loss(coeffs, min_set_real, psi, n_refine_steps, metric):
        lag, spec = compute_losses_on_fixed_points(
            coeffs, min_set_real, psi, n_refine_steps, metric
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
    """Same shape as compute_ga_fitness but sharded over min_set_real."""
    if num_devices <= 1:
        return jax.jit(compute_ga_fitness, static_argnames=("metric",))

    def per_device(min_set_shard, coeffs, psi, metric):
        lag_fit, spec_fit = compute_ga_fitness(min_set_shard, coeffs, psi, metric)
        return (
            jax.lax.pmean(lag_fit, axis_name="x"),
            jax.lax.pmean(spec_fit, axis_name="x"),
        )

    pmapped = jax.pmap(
        per_device,
        axis_name="x",
        in_axes=(0, None, None, None),
        static_broadcasted_argnums=(3,),
    )

    def fn(min_set_sharded, coeffs, psi, metric):
        lag_fit, spec_fit = pmapped(min_set_sharded, coeffs, psi, metric)
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


def load_points(psi, path=None):
    """Resolve via dwork_points_path if no explicit path, then load.

    Returns (points_real, resolved_path) so the caller can log the source.
    """
    if path is None:
        path = dwork_points_path(psi, seed=1024)
    return _load_points(path), path


def _run_all_plots(points_real, coeffs, psi, args, num_devices: int = 1):
    """Driven by make_fitness_plots:
      plots_slag_{job_id}/           GD coeffs vs random       (fixed x-range)
      plots_slag_{job_id}_d1/        d=1 baseline vs random    (fixed x-range)
      plots_slag_{job_id}_vs_d1/     GD vs d=1 canonical       (auto x-range, blue/blue)

    Additionally, when coeffs encodes a d=1+2+3 ansatz (shape (3, 1475)):
      plots_slag_{job_id}_d2_vs_d3/  d=3 (full) vs d=2 truncation (coeffs[:, :250]), steel/sky blue
      plots_slag_{job_id}_d1_d2_d3/  three-way: d=3 (steel) vs d=2 (sky) vs d=1 truncation (light blue)
    The d=2/d=1 truncations slice the GD result and re-normalize each row.

    When coeffs encodes a d=1+2+3+4 ansatz (shape (3, 6375)):
      plots_slag_{job_id}_d3_vs_d4/     d=4 (full) vs d=3 truncation
      plots_slag_{job_id}_d1_d2_d3_d4/  four-way: d=4 (steel) vs d=3 (sky) vs d=2 (light blue) vs d=1 (lightsteelblue)

    num_devices > 1 routes filter_and_refine and the per-point diagnostics
    through a pmap'd path inside make_fitness_plots (see plots.py).
    """
    d1_coeffs_full = jnp.zeros(coeffs.shape).at[:, :25].set(D1_COEFFS)
    d1_coeffs_full = normalize_coeffs(d1_coeffs_full)

    base = os.path.join(args.out_dir, f"plots_slag_{args.job_id}")
    print(f"\n=== Plotting GD coeffs vs random -> {base} ===")
    make_fitness_plots(
        points_real, coeffs, psi,
        k=args.plot_k, n_refine_steps=args.plot_newton_steps,
        metric=args.metric, compare_with="random",
        parent_folder=base,
        num_devices=num_devices,
    )

    d1_folder = base + "_d1"
    print(f"\n=== Plotting d=1 baseline vs random -> {d1_folder} ===")
    make_fitness_plots(
        points_real, d1_coeffs_full, psi,
        k=args.plot_k, n_refine_steps=args.plot_newton_steps,
        metric=args.metric, compare_with="random",
        parent_folder=d1_folder,
        primary_label="d=1 baseline",
        num_devices=num_devices,
    )

    vs_d1_folder = base + "_vs_d1"
    print(f"\n=== Plotting GD vs d=1 baseline -> {vs_d1_folder} ===")
    make_fitness_plots(
        points_real, coeffs, psi,
        k=args.plot_k, n_refine_steps=args.plot_newton_steps,
        metric=args.metric,
        compare_with=d1_coeffs_full,
        primary_label="GD result",
        compare_label="d=1 baseline",
        primary_color="steelblue",
        compare_color="skyblue",
        fix_kahler_x_range=False,
        parent_folder=vs_d1_folder,
        num_devices=num_devices,
    )

    if coeffs.shape[1] == GENOTYPE_WIDTHS[3]:
        d2_truncated = normalize_coeffs(coeffs[:, :GENOTYPE_WIDTHS[2]])
        d1_truncated = normalize_coeffs(coeffs[:, :GENOTYPE_WIDTHS[1]])

        d2_vs_d3_folder = base + "_d2_vs_d3"
        print(f"\n=== Plotting d=3 vs d=2 truncation -> {d2_vs_d3_folder} ===")
        make_fitness_plots(
            points_real, coeffs, psi,
            k=args.plot_k, n_refine_steps=args.plot_newton_steps,
            metric=args.metric,
            compare_with=d2_truncated,
            primary_label="d=3",
            compare_label="d=2",
            primary_color="steelblue",
            compare_color="skyblue",
            fix_kahler_x_range=False,
            parent_folder=d2_vs_d3_folder,
            num_devices=num_devices,
        )

        d1_d2_d3_folder = base + "_d1_d2_d3"
        print(f"\n=== Plotting d=1 vs d=2 vs d=3 -> {d1_d2_d3_folder} ===")
        make_fitness_plots(
            points_real, coeffs, psi,
            k=args.plot_k, n_refine_steps=args.plot_newton_steps,
            metric=args.metric,
            compare_with=d2_truncated,
            primary_label="d=3",
            compare_label="d=2",
            primary_color="steelblue",
            compare_color="skyblue",
            fix_kahler_x_range=False,
            extra_comparisons=[{
                "coeffs": d1_truncated,
                "label": "d=1",
                "color": "lightblue",
            }],
            parent_folder=d1_d2_d3_folder,
            num_devices=num_devices,
        )

    if coeffs.shape[1] == GENOTYPE_WIDTHS[4]:
        d3_truncated = normalize_coeffs(coeffs[:, :GENOTYPE_WIDTHS[3]])
        d2_truncated = normalize_coeffs(coeffs[:, :GENOTYPE_WIDTHS[2]])
        d1_truncated = normalize_coeffs(coeffs[:, :GENOTYPE_WIDTHS[1]])

        d3_vs_d4_folder = base + "_d3_vs_d4"
        print(f"\n=== Plotting d=4 vs d=3 truncation -> {d3_vs_d4_folder} ===")
        make_fitness_plots(
            points_real, coeffs, psi,
            k=args.plot_k, n_refine_steps=args.plot_newton_steps,
            metric=args.metric,
            compare_with=d3_truncated,
            primary_label="d=4",
            compare_label="d=3",
            primary_color="steelblue",
            compare_color="skyblue",
            fix_kahler_x_range=False,
            parent_folder=d3_vs_d4_folder,
            num_devices=num_devices,
        )

        d1_d2_d3_d4_folder = base + "_d1_d2_d3_d4"
        print(f"\n=== Plotting d=1 vs d=2 vs d=3 vs d=4 -> {d1_d2_d3_d4_folder} ===")
        make_fitness_plots(
            points_real, coeffs, psi,
            k=args.plot_k, n_refine_steps=args.plot_newton_steps,
            metric=args.metric,
            compare_with=d3_truncated,
            primary_label="d=4",
            compare_label="d=3",
            primary_color="steelblue",
            compare_color="skyblue",
            fix_kahler_x_range=False,
            extra_comparisons=[
                {"coeffs": d2_truncated, "label": "d=2", "color": "lightblue"},
                {"coeffs": d1_truncated, "label": "d=1", "color": "lightsteelblue"},
            ],
            parent_folder=d1_d2_d3_d4_folder,
            num_devices=num_devices,
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

        lag_fit, spec_fit = ga_fitness_fn(min_set_data, coeffs, psi, args.metric)
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
                        help="Call make_fitness_plots on the final coeffs "
                             "(same plots as GA.py). Use --no-make_plots to skip.")
    parser.add_argument("--plots_only", action="store_true",
                        help="Skip training. Load --resume <ckpt>, run "
                             "make_fitness_plots, exit.")
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
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    shape = genotype_shape(args.max_degree)
    init_desc = f"init_pkl={args.init_pkl}" if args.init_pkl is not None else f"init={args.init}"
    print(f"=== GD for sLag search (max_degree={args.max_degree}, shape={shape}) ===")
    print(f"job_id={args.job_id} {init_desc} loss={args.loss} "
          f"(lag_w={args.lag_weight} spec_w={args.spec_weight}) "
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

    total_loss = make_total_loss(args.loss, args.lag_weight, args.spec_weight)
    loss_value_and_grad = make_parallel_loss_and_grad(total_loss, num_devices)
    ga_fitness_jit = make_parallel_ga_fitness(num_devices)
    mining_fn = make_parallel_mining(num_devices)

    # In multi-GPU mode, min_set_real is a (D, k/D, 10) sharded array that we
    # carry directly between mining and the loss/fitness call — no host
    # round-trip. In single-GPU mode it's the usual (k, 10) array.
    points_in = points_sharded if num_devices > 1 else points_real

    # Initial mining + loss eval (also re-runs on resume to repopulate min_set_real).
    min_set_real, distances, _ = mining_fn(
        points_in, coeffs, psi, args.minset_size, args.newton_steps,
    )
    mean_d, max_d = float(jnp.mean(distances)), float(jnp.max(distances))
    print(f"  [mining] mean_dist {mean_d:.2e}  max_dist {max_d:.2e}")
    if mean_d > 1e-4:
        print(f"  [warn] mean Newton distance > 1e-4 -- points may not be on the manifold")
    (init_loss, (init_lag, init_spec)), _ = loss_value_and_grad(
        coeffs, min_set_real, psi, args.inner_newton_steps, args.metric
    )
    init_lag_fit, init_spec_fit = ga_fitness_jit(min_set_real, coeffs, psi, args.metric)
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
            min_set_real, distances, _ = mining_fn(
                points_in, coeffs, psi, args.minset_size, args.newton_steps,
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
        lag_fit, spec_fit = ga_fitness_jit(min_set_real, coeffs, psi, args.metric)
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
