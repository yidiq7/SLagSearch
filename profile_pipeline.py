"""
Standalone profiler for the SLagSearch fitness pipeline.

Runs each phase of the fitness evaluation (initial distance compute,
Newton refine, repulsion loop, kahler/fitness eval, end-to-end
single-individual, end-to-end mini-batch) under a controlled load and
reports a timing breakdown so you can see where wall time is actually spent.

Examples
--------
# Single-GPU phase profile
python profile_pipeline.py --n_iters 5

# Multi-GPU mini-batch (matches GA.py mini-batch path)
python profile_pipeline.py --multi_gpu --batch_size 100 --n_iters 5

# Compare with vs. without tensor cores
python profile_pipeline.py --precision highest --n_iters 5
python profile_pipeline.py --precision high    --n_iters 5

# Also write a Perfetto-viewable trace
python profile_pipeline.py --trace_dir ./jax_trace --n_iters 3
"""

import argparse
import os
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--points_file', type=str, default=None,
                   help='Path to point cloud pkl. If unset, resolved from --psi '
                        'via helper.dwork_points_path.')
    p.add_argument('--precision', choices=['default', 'high', 'highest'], default='high',
                   help="jax_default_matmul_precision. 'high'=TF32 on Ampere; 'highest'=strict FP32.")
    p.add_argument('--minset_size', type=int, default=10000)
    p.add_argument('--newton_steps', type=int, default=40)
    p.add_argument('--metric', type=str, default='k4_fermat')
    p.add_argument('--psi', type=complex, default=0)
    p.add_argument('--n_repulsion', type=int, default=15)
    p.add_argument('--dist_chunk_size', type=int, default=50000)
    p.add_argument('--batch_size', type=int, default=100,
                   help='Mini-batch size for the batched-fitness measurement (matches GA.py).')
    p.add_argument('--n_iters', type=int, default=5,
                   help='Number of timed iterations per phase. Best (min) is reported.')
    p.add_argument('--multi_gpu', action='store_true',
                   help='Distribute the mini-batch across all local GPUs via pmap, mirroring GA.py.')
    p.add_argument('--trace_dir', type=str, default=None,
                   help='If set, also capture a jax.profiler trace into this directory.')
    return p.parse_args()


def block(x):
    """Force GPU completion for sync timing."""
    jax.tree.map(
        lambda v: v.block_until_ready() if hasattr(v, 'block_until_ready') else v, x
    )


def time_phase(label, fn, n_iters):
    """Compile + warmup + best-of-n timing. Returns (best_seconds, last_output)."""
    # warmup / compile
    out = fn(); block(out)
    out = fn(); block(out)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = fn()
        block(out)
        times.append(time.perf_counter() - t0)
    best = min(times)
    median = sorted(times)[len(times) // 2]
    print(f'  {label:55s} best={best*1000:9.2f} ms   median={median*1000:9.2f} ms')
    return best, out


# d=1 coefficients (copied from GA.py — used as a representative individual)
D1_COEFFS = jnp.array([
    [-0.2085878998041153, 0.08078225702047348, 0.12364989519119263, 0.42693421244621277,
     -0.4276507794857025, 0.05941963940858841, -0.19358153641223907, 0.2884068787097931,
     0.2374262660741806, 0.17124612629413605, -0.03099866583943367, 0.07415380328893661,
     -0.22672683000564575, -0.1914607286453247, 0.09337177127599716, -0.053066715598106384,
     -0.06608302891254425, -0.3771730363368988, 0.05378381162881851, 0.0064529310911893845,
     0.2938925623893738, 0.08852922171354294, 0.020463770255446434, 0.09666207432746887,
     -0.006990742404013872],
    [-0.1065014973282814, 0.20087268948554993, 0.18935158848762512, -0.17352613806724548,
     0.05884088575839996, -0.4646260440349579, -0.10628655552864075, -0.28338274359703064,
     -0.03379037603735924, 0.007989203557372093, -0.06132059171795845, -0.13810740411281586,
     0.04504100978374481, 0.015115765854716301, -0.4030528962612152, -0.025872472673654556,
     -0.4061300754547119, -0.02022559940814972, -0.13893099129199982, 0.10193423181772232,
     0.29334160685539246, 0.22542181611061096, -0.050897762179374695, 0.21366965770721436,
     -0.04277477413415909],
    [0.054688308387994766, 0.07500440627336502, 0.060474496334791183, -0.3848169445991516,
     -0.3781052529811859, 0.38639041781425476, 0.021527282893657684, 0.4060642719268799,
     -0.15761728584766388, -0.1271764189004898, -0.01066557876765728, -0.13985656201839447,
     0.1605837494134903, 0.15716029703617096, -0.32516127824783325, 0.016290534287691116,
     0.2249980866909027, -0.2878168523311615, -0.12032820284366608, -0.04713383689522743,
     0.025025269016623497, 0.08448748290538788, 0.05337755009531975, 0.05431513488292694,
     -0.03361976519227028]
])

GENOTYPE_SHAPE = (3, 250)


def device_put_sharded(shards, devices):
    """Mirror of GA.device_put_sharded so the multi-GPU path matches exactly."""
    mesh = Mesh(np.array(devices), ('x',))
    sharding = NamedSharding(mesh, P('x'))
    return jax.tree.map(
        lambda *xs: jax.device_put(jnp.stack(xs), sharding), *shards
    )


def main():
    args = parse_args()

    # Set precision *before* importing the project modules (some jit caches
    # depend on the global config at trace time).
    jax.config.update('jax_default_matmul_precision', args.precision)

    from find_smooth_submanifold import (
        filter_and_refine,
        refine_point_iterative,
        compute_distances_batched,
        normalize_coeffs,
    )
    from slag_condition import compute_combined_fitness
    from helper import assert_metric_psi_compatible, canonicalize_coeffs, dwork_points_path, load_points

    assert_metric_psi_compatible(args.metric, args.psi)

    print(f'JAX backend:    {jax.default_backend()}')
    print(f'Devices:        {jax.devices()}')
    print(f'Local devices:  {jax.local_device_count()}')
    print(f'Precision:      {args.precision}')
    print(f'Iterations:     {args.n_iters} (reporting best+median)')
    print()

    # ---------------------------------------------------------------- inputs
    if args.points_file is None:
        args.points_file = dwork_points_path(args.psi, seed=1024)
    print(f'Loading points from {args.points_file} ...')
    points_real = load_points(args.points_file)
    print(f'  points shape: {points_real.shape}')

    # representative individual
    base = jnp.zeros(GENOTYPE_SHAPE).at[:, :25].set(D1_COEFFS)
    coeffs = normalize_coeffs(canonicalize_coeffs(base))
    psi = jnp.complex64(args.psi)

    # mini-batch of individuals for the batched/pmapped fitness measurement
    rng = jax.random.PRNGKey(0)
    noise = jax.random.uniform(rng, (args.batch_size, *GENOTYPE_SHAPE),
                               minval=-0.01, maxval=0.01)
    pop_batch = vmap(lambda p: normalize_coeffs(canonicalize_coeffs(p)))(base + noise)
    block(pop_batch)

    # build the same calculate_fitness wrapper GA.py uses
    @partial(jax.jit, static_argnames=('k', 'n_refine_steps', 'metric'))
    def calculate_fitness_for_one_individual(ind_coeffs, points, psi_, k,
                                             n_refine_steps, metric):
        min_set_real, _, ok = filter_and_refine(
            points, ind_coeffs, psi_, k, n_refine_steps, filter_newton=True,
            dist_chunk_size=args.dist_chunk_size,
        )
        return jax.lax.cond(
            ok,
            lambda p: compute_combined_fitness(min_set_real, ind_coeffs, psi_, metric),
            lambda p: jnp.float32(0.0),
            min_set_real,
        )

    vmap_fitness_batch = vmap(
        calculate_fitness_for_one_individual,
        in_axes=(0, None, None, None, None, None), out_axes=0,
    )

    if args.multi_gpu and jax.local_device_count() > 1:
        evaluate_fitness = jax.pmap(
            vmap_fitness_batch,
            in_axes=(0, None, None, None, None, None),
            static_broadcasted_argnums=(3, 4, 5),
        )
    else:
        evaluate_fitness = vmap_fitness_batch

    # ----------------------------------------------------- optional trace
    trace_ctx = None
    if args.trace_dir:
        os.makedirs(args.trace_dir, exist_ok=True)
        print(f'Capturing JAX profiler trace to {args.trace_dir}')
        trace_ctx = jax.profiler.trace(args.trace_dir, create_perfetto_link=False)
        trace_ctx.__enter__()

    # ===================================================== phase timings
    print('\n=== Phase timings (best of N over all warm calls) ===')

    # 1) initial distance over the whole point cloud
    def phase_initial_distance():
        return compute_distances_batched(points_real, coeffs, psi,
                                         chunk_size=args.dist_chunk_size)

    t_initdist, _ = time_phase(
        f'compute_distances_batched (full {points_real.shape[0]} pts)',
        phase_initial_distance, args.n_iters)

    # Pick the same top-2k subset filter_and_refine uses, for the refine timing
    initial_dist = compute_distances_batched(
        points_real, coeffs, psi, chunk_size=args.dist_chunk_size)
    block(initial_dist)
    top2k_idx = jnp.argsort(initial_dist)[: 2 * args.minset_size]
    top2k_pts = points_real[top2k_idx]
    block(top2k_pts)

    # 2) Newton refine on the top-2k points (single vmapped call, no chunking)
    refine_partial = partial(refine_point_iterative,
                             coeffs=coeffs, psi=psi, n_steps=args.newton_steps)
    vmapped_refine = jax.jit(jax.vmap(refine_partial))

    t_refine, refined = time_phase(
        f'Newton refine ({args.newton_steps} steps, '
        f'{2 * args.minset_size} pts)',
        lambda: vmapped_refine(top2k_pts), args.n_iters)

    # 3) filter_and_refine WITHOUT repulsion → "everything except repulsion"
    def phase_far_no_rep():
        return filter_and_refine(
            points_real, coeffs, psi, args.minset_size, args.newton_steps,
            filter_newton=True, n_repulsion_steps=0,
            dist_chunk_size=args.dist_chunk_size,
        )
    t_far_norep, _ = time_phase(
        'filter_and_refine (n_repulsion=0)',
        phase_far_no_rep, args.n_iters)

    # 4) full filter_and_refine
    def phase_far_full():
        return filter_and_refine(
            points_real, coeffs, psi, args.minset_size, args.newton_steps,
            filter_newton=True, n_repulsion_steps=args.n_repulsion,
            dist_chunk_size=args.dist_chunk_size,
        )
    t_far_full, far_out = time_phase(
        f'filter_and_refine (n_repulsion={args.n_repulsion})',
        phase_far_full, args.n_iters)

    rep_total = max(t_far_full - t_far_norep, 0.0)
    rep_per_step = rep_total / max(args.n_repulsion, 1)
    print(f'  {"  -> repulsion total (by ablation)":55s} '
          f'best={rep_total*1000:9.2f} ms   per-step={rep_per_step*1000:9.2f} ms')

    # Convergence-precision report on the full-pipeline output.
    # far_out = (final_points, final_distances, newton_check_pass)
    final_distances = far_out[1]
    newton_ok = bool(far_out[2])
    d_min  = float(jnp.min(final_distances))
    d_max  = float(jnp.max(final_distances))
    d_mean = float(jnp.mean(final_distances))
    d_med  = float(jnp.median(final_distances))
    n_bad_1em6 = int(jnp.sum(final_distances > 1e-6))
    n_bad_1em4 = int(jnp.sum(final_distances > 1e-4))
    print()
    print('=== Convergence precision (filter_and_refine output) ===')
    print(f'  newton_check_pass:    {newton_ok}')
    print(f'  min_set_distance:     '
          f'Min: {d_min:.4e}  Max: {d_max:.4e}  '
          f'Mean: {d_mean:.4e}  Median: {d_med:.4e}')
    print(f'  pts > 1e-6 / 1e-4:    {n_bad_1em6} / {n_bad_1em4}    '
          f'(of {final_distances.shape[0]})')
    print()

    # 5) compute_combined_fitness alone, on the refined min_set
    min_set_real = far_out[0]

    @jax.jit
    def fitness_only(ms, c, p):
        return compute_combined_fitness(ms, c, p, args.metric)

    t_kahler, _ = time_phase(
        'compute_combined_fitness (kahler/holomorphic/etc.)',
        lambda: fitness_only(min_set_real, coeffs, psi), args.n_iters)

    # 6) End-to-end single-individual fitness
    t_one, _ = time_phase(
        'calculate_fitness_for_one_individual (1 ind, end-to-end)',
        lambda: calculate_fitness_for_one_individual(
            coeffs, points_real, psi, args.minset_size,
            args.newton_steps, args.metric),
        args.n_iters)

    # 7) Mini-batch (matches GA.py path)
    if args.multi_gpu and jax.local_device_count() > 1:
        nd = jax.local_device_count()
        per_dev = args.batch_size // nd
        shards = [pop_batch[i * per_dev:(i + 1) * per_dev] for i in range(nd)]
        pop_sharded = device_put_sharded(shards, jax.local_devices())
        block(pop_sharded)

        def phase_batch():
            return evaluate_fitness(
                pop_sharded, points_real, psi, args.minset_size,
                args.newton_steps, args.metric)
    else:
        def phase_batch():
            return evaluate_fitness(
                pop_batch, points_real, psi, args.minset_size,
                args.newton_steps, args.metric)

    t_batch, _ = time_phase(
        f'evaluate_fitness mini-batch ({args.batch_size} inds, '
        f'{"pmap" if args.multi_gpu and jax.local_device_count() > 1 else "vmap"})',
        phase_batch, args.n_iters)

    # ===================================================== summary
    print('\n=== Share of full filter_and_refine ===')
    pct = lambda t: 100 * t / t_far_full if t_far_full > 0 else 0.0
    print(f'  initial distance:                    {pct(t_initdist):5.1f}%')
    print(f'  Newton refine (2k pts, isolated):    {pct(t_refine):5.1f}%')
    print(f'  repulsion (by ablation):             {pct(rep_total):5.1f}%')
    print(f'  full filter_and_refine:              100.0%  ({t_far_full*1000:.1f} ms)')

    print('\n=== Share of single-individual end-to-end ===')
    pct1 = lambda t: 100 * t / t_one if t_one > 0 else 0.0
    print(f'  filter_and_refine:                   {pct1(t_far_full):5.1f}%')
    print(f'  compute_combined_fitness:            {pct1(t_kahler):5.1f}%')
    print(f'  total per individual (measured):     100.0%  ({t_one*1000:.1f} ms)')
    print(f'  sum of parts (sanity check):         {(t_far_full + t_kahler)*1000:.1f} ms')

    print('\n=== Mini-batch throughput ===')
    print(f'  per-individual time at batch={args.batch_size}: '
          f'{t_batch / args.batch_size * 1000:.2f} ms')
    print(f'  effective speedup vs single (vmap/pmap): '
          f'{t_one / (t_batch / args.batch_size):.2f}x')

    if trace_ctx is not None:
        trace_ctx.__exit__(None, None, None)
        print(f'\nTrace written to {args.trace_dir}. '
              'Open with: chrome://tracing or https://ui.perfetto.dev/')


if __name__ == '__main__':
    main()
