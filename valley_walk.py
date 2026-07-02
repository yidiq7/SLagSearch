"""Valley-walk: probe the sLag moduli space by perturb-and-reconverge.

Reuses gradient_descent.py building blocks in-process (JIT warm, multi-GPU
inherited); does not modify the GD training loop.
See docs/superpowers/specs/2026-06-28-valley-walk-design.md.
"""
import os
import argparse
import pickle

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gradient_descent import (
    genotype_shape, init_coeffs, load_points, make_total_loss,
    make_parallel_loss_and_grad, make_parallel_mining, make_parallel_ga_fitness,
    mine_one_cluster,
)
from find_smooth_submanifold import normalize_coeffs
from helper import convert_real_to_complex_batch, assert_metric_psi_compatible
from cluster_select import fs_features
from viz.fitness_pipeline import run_fitness_pipeline
from sharding import shard_leading_axis
import pointcloud_distance as pcd


def build_args(argv=None):
    p = argparse.ArgumentParser(description="Valley-walk over the sLag moduli space.")
    # --- mirror the relevant gradient_descent.py flags (defaults must match) ---
    p.add_argument("--psi", type=complex, default=0)
    p.add_argument("--points_file", type=str, default=None)
    p.add_argument("--metric", type=str, default="k4_fermat")
    p.add_argument("--loss", type=str, default="both")
    p.add_argument("--lag_weight", type=float, default=1.0)
    p.add_argument("--spec_weight", type=float, default=1.0)
    p.add_argument("--top_lag_frac", type=float, default=0.99)
    p.add_argument("--minset_size", type=int, default=10000)
    p.add_argument("--newton_steps", type=int, default=40)
    p.add_argument("--inner_newton_steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_degree", type=int, default=4)  # d=4 candidate
    p.add_argument("--target_cluster", type=int, default=None)
    p.add_argument("--min_cluster_size", type=int, default=200)
    p.add_argument("--cluster_selection_epsilon", type=float, default=0.0)
    p.add_argument("--min_cluster_frac", type=float, default=0.02)
    p.add_argument("--cluster_minset_size", type=int, default=None)
    p.add_argument("--mine_oversample", type=int, default=2)
    p.add_argument("--plot_k", type=int, default=80000)
    p.add_argument("--plot_newton_steps", type=int, default=80)
    # --- walk-specific flags ---
    p.add_argument("--init_pkl", type=str, required=True,
                   help="C*: the candidate to perturb (bare (3,w) array or ckpt dict).")
    p.add_argument("--ref_pkl", type=str, default=None,
                   help="Optional C* reference: coeffs for the drift/fitness baseline "
                        "+ cluster anchor. Defaults to --init_pkl. Set it to the "
                        "original GD checkpoint when --init_pkl is a later walk "
                        "endpoint (e.g. a sigma=0 polish run).")
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--sigma_min", type=float, default=0.0025)
    p.add_argument("--reconverge_steps", type=int, default=300)
    p.add_argument("--mine_interval", type=int, default=50)
    p.add_argument("--max_walk_steps", type=int, default=40)
    p.add_argument("--target_floor_mult", type=float, default=8.0)
    p.add_argument("--n_kicks", type=int, default=1)
    p.add_argument("--fitness_tol", type=float, default=0.02)
    p.add_argument("--n_repeats_floor", type=int, default=3)
    p.add_argument("--n_pairs", type=int, default=200000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--job_id", type=str, default="0")
    p.add_argument("--out_dir", type=str, default="./valley_walk_runs")
    p.add_argument("--calibrate_only", action="store_true",
                   help="Compute and print the noise floor, then exit (no walk).")
    args = p.parse_args(argv)
    # Mirror gradient_descent.py: default cluster_minset_size to minset_size when
    # unset (mine_one_cluster computes mine_oversample * cluster_minset_size, which
    # would crash on None). With --target_cluster it's the per-cluster min-set size.
    if args.cluster_minset_size is None:
        args.cluster_minset_size = args.minset_size
    return args


def load_init_anchor(init_pkl):
    """Pull a persisted cluster anchor (25-D FS-feature centroid) out of a GD
    checkpoint dict, if present. Returns None for a bare-array pkl or a checkpoint
    with no anchor (in which case selection bootstraps by --target_cluster size)."""
    with open(init_pkl, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict) and raw.get("anchor") is not None:
        return np.asarray(raw["anchor"])
    return None


def setup(args):
    """Load points, build the (num_devices-aware) JAX building blocks, load C*."""
    assert_metric_psi_compatible(args.metric, args.psi)
    points_real, src = load_points(args.psi, path=args.points_file)
    psi = jnp.asarray(args.psi, dtype=jnp.complex128)
    num_devices = jax.local_device_count()
    print(f"Loaded {len(points_real)} points from {src}; {num_devices} GPU(s).")

    if num_devices > 1:
        n_keep = (points_real.shape[0] // num_devices) * num_devices
        points_real = points_real[:n_keep]
        points_in = shard_leading_axis(points_real, num_devices)
        if args.minset_size % num_devices != 0:
            raise ValueError(f"--minset_size must be divisible by {num_devices}")
        if args.target_cluster is not None:
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
        points_in = points_real

    total_loss = make_total_loss(args.loss, args.lag_weight, args.spec_weight, args.top_lag_frac)
    fns = {
        "loss_value_and_grad": make_parallel_loss_and_grad(total_loss, num_devices),
        "ga_fitness": make_parallel_ga_fitness(num_devices),
        "mining_fn": make_parallel_mining(num_devices),
    }
    shape = genotype_shape(args.max_degree)
    # ref = drift/fitness baseline + anchor source; init = where the walk starts.
    ref_path = args.ref_pkl if args.ref_pkl is not None else args.init_pkl
    cstar = normalize_coeffs(init_coeffs("scratch", ref_path, shape, jax.random.PRNGKey(0)))
    if args.ref_pkl is not None:
        start = normalize_coeffs(init_coeffs("scratch", args.init_pkl, shape, jax.random.PRNGKey(0)))
        print(f"  [ref] baseline/anchor from {ref_path}; walk starts from {args.init_pkl}")
    else:
        start = cstar
    init_anchor = load_init_anchor(ref_path)
    if args.target_cluster is not None:
        if init_anchor is not None:
            print(f"  [cluster] seeded anchor from {ref_path} "
                  f"(tracks the trained component; --target_cluster size rank ignored)")
        else:
            print(f"  [cluster] WARNING: no 'anchor' in {ref_path}; bootstrapping "
                  f"--target_cluster {args.target_cluster} by size -- unstable for "
                  f"near-equal components. Point --ref_pkl at a GD checkpoint.")
    return points_real, points_in, psi, num_devices, fns, cstar, start, init_anchor


def mine_embed(coeffs, fns, points_in, psi, args, num_devices, anchor, seed):
    """Fresh independent mine -> (unique-point FS embedding (m,25), (lag_fit,
    spec_fit), unique (m,10) host min-set, anchor-unchanged).

    Fitness is on the FULL (padded) cluster min-set, matching GD. The embedding and
    returned points are DEDUPED: fill_to_size pads a small component up to
    cluster_minset_size with duplicate rows, which would otherwise inject an
    RNG-dependent zero-distance spike into the pairwise-distance drift and inflate
    the noise floor. `anchor` is used for selection and returned unchanged (no
    per-mine drift onto the wrong component)."""
    rng = np.random.default_rng(seed)
    min_set_real, _dist, _new_anchor, _ = mine_one_cluster(
        fns["mining_fn"], points_in, coeffs, psi, args, num_devices, anchor, rng)
    lag_fit, spec_fit = fns["ga_fitness"](min_set_real, coeffs, psi, args.metric, args.top_lag_frac)
    msr = np.asarray(min_set_real).reshape(-1, np.asarray(min_set_real).shape[-1])
    msr_u = np.unique(msr, axis=0)                                     # drop fill_to_size padding
    z = np.asarray(convert_real_to_complex_batch(jnp.asarray(msr_u)))  # (m,5) complex
    emb = fs_features(z)                                              # (m,25)
    return emb, (float(lag_fit), float(spec_fit)), msr_u, anchor


def reconverge(coeffs, fns, points_in, psi, args, num_devices, anchor, rng, n_steps):
    """Short Adam reconvergence from a perturbed start (fresh optimizer state).
    Uses a FIXED cluster anchor for every mine (no per-mine drift), so a large kick
    can't walk the selection onto the wrong component mid-reconverge."""
    opt = optax.adam(learning_rate=args.lr)
    opt_state = opt.init(coeffs)
    min_set_real, _d, _a, _ = mine_one_cluster(
        fns["mining_fn"], points_in, coeffs, psi, args, num_devices, anchor, rng)
    for s in range(1, n_steps + 1):
        if s % args.mine_interval == 0:
            min_set_real, _d, _a, _ = mine_one_cluster(
                fns["mining_fn"], points_in, coeffs, psi, args, num_devices, anchor, rng)
        (_loss, (_lag, _spec)), grads = fns["loss_value_and_grad"](
            coeffs, min_set_real, psi, args.inner_newton_steps, args.metric)
        grads = jnp.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
        updates, opt_state = opt.update(grads, opt_state, coeffs)
        coeffs = optax.apply_updates(coeffs, updates)
    return coeffs


def main():
    args = build_args()
    run_dir = os.path.join(args.out_dir, f"valley_walk_{args.job_id}")
    os.makedirs(run_dir, exist_ok=True)
    points_real, points_in, psi, num_devices, fns, cstar, start, init_anchor = setup(args)

    drift_rng = np.random.default_rng(args.seed)
    base_seed = args.seed

    # --- calibrate the noise floor on C* (re-mine R times) ---
    # Seed every re-mine with the (frozen) checkpoint anchor so calibration tracks
    # the SAME component each time; a fresh per-mine bootstrap flips between
    # near-equal components and inflates the floor.
    floor = pcd.calibrate_noise_floor(
        lambda s: mine_embed(cstar, fns, points_in, psi, args, num_devices, init_anchor, s)[0],
        n_repeats=args.n_repeats_floor, n_pairs=args.n_pairs, rng=drift_rng)
    print(f"[floor] wass={floor['wass_floor']:.4g}  chamfer={floor['chamfer_floor']:.4g}  "
          f"target=stop above {args.target_floor_mult * floor['wass_floor']:.4g}")

    embC, (lag0, spec0), ms_cstar, _ = mine_embed(
        cstar, fns, points_in, psi, args, num_devices, init_anchor, base_seed)
    print(f"[C*] lag_fit={lag0:.4f} spec_fit={spec0:.4f}")

    if args.calibrate_only:
        with open(os.path.join(run_dir, "floor.pkl"), "wb") as f:
            pickle.dump({"floor": {k: floor[k] for k in ("wass_floor", "chamfer_floor")},
                         "lag0": lag0, "spec0": spec0}, f)
        print(f"[done] calibrate_only -> {run_dir}/floor.pkl")
        return

    shape = cstar.shape
    sigma = args.sigma
    current = start
    prev_emb = embC
    final_coeffs = cstar
    ms_final = ms_cstar
    seed_ctr = base_seed + 1000
    traj = [{"step": 0, "sigma": sigma, "lag_fit": lag0, "spec_fit": spec0,
             "drift_vs_cstar": 0.0, "drift_vs_prev": 0.0, "chamfer_vs_cstar": 0.0,
             "decision": "start"}]

    # The cluster anchor is FROZEN at C*'s good-component centroid for the whole
    # walk: we explore near C*, so it stays the closest component, and a big kick
    # can't drag selection onto the junk component (which caused false rejects).
    for w in range(1, args.max_walk_steps + 1):
        best = None
        for _k in range(args.n_kicks):
            pert = np.random.default_rng(seed_ctr).standard_normal(shape); seed_ctr += 1
            cand = normalize_coeffs(current + jnp.asarray(pert) * sigma)
            cand = reconverge(cand, fns, points_in, psi, args, num_devices,
                              init_anchor, np.random.default_rng(seed_ctr), args.reconverge_steps)
            seed_ctr += 1
            emb, (lag, spec), msr, _ = mine_embed(
                cand, fns, points_in, psi, args, num_devices, init_anchor, seed_ctr); seed_ctr += 1
            drift = pcd.pairwise_distance_drift(embC, emb, args.n_pairs, drift_rng)
            d = pcd.decide(drift, lag, spec, lag0, spec0, args.fitness_tol,
                           floor["wass_floor"], args.target_floor_mult)
            print(f"  [kick] drift={drift:.4g} lag={lag:.4f} spec={spec:.4f} -> {d}")
            if d != "reject" and (best is None or drift > best["drift"]):
                best = {"cand": cand, "emb": emb, "lag": lag, "spec": spec,
                        "drift": drift, "ms": msr}
        if best is None:
            sigma *= 0.5
            if sigma < args.sigma_min:
                print(f"[walk {w}] all kicks rejected, sigma<{args.sigma_min}; stopping")
                break
            print(f"[walk {w}] all kicks rejected; halving sigma -> {sigma:.4g}")
            continue

        current = best["cand"]; final_coeffs = current; ms_final = best["ms"]
        with open(os.path.join(run_dir, f"coeffs_step{w}.pkl"), "wb") as f:
            pickle.dump(np.asarray(current), f)
        drift_prev = pcd.pairwise_distance_drift(prev_emb, best["emb"], args.n_pairs, drift_rng)
        cham = pcd.fs_chamfer(embC, best["emb"])
        decision = pcd.decide(best["drift"], best["lag"], best["spec"], lag0, spec0,
                              args.fitness_tol, floor["wass_floor"], args.target_floor_mult)
        traj.append({"step": w, "sigma": float(sigma), "lag_fit": best["lag"],
                     "spec_fit": best["spec"], "drift_vs_cstar": best["drift"],
                     "drift_vs_prev": drift_prev, "chamfer_vs_cstar": cham,
                     "decision": decision})
        prev_emb = best["emb"]
        print(f"[walk {w}] drift={best['drift']:.4g} (floor {floor['wass_floor']:.4g}) "
              f"lag={best['lag']:.4f} spec={best['spec']:.4f} -> {decision}")
        if decision == "stop":
            print(f"[walk {w}] cleared target; stopping")
            break

    # --- persist trajectory ---
    with open(os.path.join(run_dir, "trajectory.pkl"), "wb") as f:
        pickle.dump({"trajectory": traj,
                     "floor": {k: floor[k] for k in ("wass_floor", "chamfer_floor")},
                     "lag0": lag0, "spec0": spec0, "args": vars(args)}, f)

    # --- drift-vs-step plot (with the noise-floor band shaded) ---
    steps = [t["step"] for t in traj]
    drifts = [t["drift_vs_cstar"] for t in traj]
    fl = floor["wass_floor"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhspan(0, fl, color="grey", alpha=0.25, label="noise floor (same L)")
    ax.axhline(args.target_floor_mult * fl, ls="--", color="red", label="target")
    ax.plot(steps, drifts, "-o", color="navy")
    ax.set_xlabel("walk step"); ax.set_ylabel("drift vs C* (pairwise-distance W1)")
    ax.set_title("Valley walk: submanifold drift"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "drift_vs_step.png"), dpi=130)
    plt.close(fig)

    # --- fitness-vs-step plot (must stay flat-maximal) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, [t["lag_fit"] for t in traj], "-o", label="lag_fit")
    ax.plot(steps, [t["spec_fit"] for t in traj], "-s", label="spec_fit")
    ax.axhline(lag0, ls=":", color="grey"); ax.axhline(spec0, ls=":", color="grey")
    ax.set_xlabel("walk step"); ax.set_ylabel("GA-comparable fitness")
    ax.set_title("Valley walk: fitness along the path"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "fitness_vs_step.png"), dpi=130)
    plt.close(fig)

    # --- run-folders for C* and the final point (coord_scatter + histograms),
    #     using the walk's own (deduped) cluster points so a cluster-restricted run
    #     stays on that single component instead of re-mining the whole manifold. ---
    for coeffs, sub, ms in ((cstar, "cstar", ms_cstar), (final_coeffs, "final", ms_final)):
        run_fitness_pipeline(
            points_real, coeffs, psi, k=args.plot_k,
            n_refine_steps=args.plot_newton_steps, metric=args.metric,
            top_lag_frac=args.top_lag_frac, out_dir=os.path.join(run_dir, sub),
            num_devices=num_devices, min_set_override=ms)

    print(f"[done] trajectory + plots -> {run_dir}")


if __name__ == "__main__":
    main()
