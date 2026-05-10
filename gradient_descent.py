"""Gradient descent for sLag search with the d=1+2 ansatz (3, 250 coeffs).

Ported from the `gradient-descent` branch and generalized from (3, 25) to
(3, 250). The optimization alternates:

1. (Re-)mining: every `mine_interval` steps, `filter_and_refine` produces a
   fresh point cloud on the current submanifold. This is NOT differentiated.
2. Adam steps: with the point cloud frozen as initial conditions, run a short
   Newton refinement (differentiable through `refine_point_iterative`) and
   evaluate Lagrangian / special losses on the refined points.

Init modes:
- scratch:    random Uniform over (3, 250)
- d1_zeropad: GA.py canonical d=1 baseline, zero-padded to (3, 250)
- pkl:        load a (3, 250) array from a pickle (e.g. GA's best individual)
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
    convert_real_to_complex_batch,
    determine_patches_batch,
    format_array_with_commas,
)
from slag_condition import (
    compute_holomorphic_form_restricted,
    compute_kahler_form_unrestricted,
    compute_special_condition_fitness_smooth,
    vmap_compute_affine_jacobian,
    vmap_compute_restriction,
)

jax.config.update("jax_enable_x64", True)

GENOTYPE_SHAPE = (3, 250)

# Canonical d=1 baseline. Mirrors GA.py:409 d1_coeffs.
D1_COEFFS = jnp.array([
    [-0.2085878998041153, 0.08078225702047348, 0.12364989519119263, 0.42693421244621277, -0.4276507794857025, 0.05941963940858841, -0.19358153641223907, 0.2884068787097931, 0.2374262660741806, 0.17124612629413605, -0.03099866583943367, 0.07415380328893661, -0.22672683000564575, -0.1914607286453247, 0.09337177127599716, -0.053066715598106384, -0.06608302891254425, -0.3771730363368988, 0.05378381162881851, 0.0064529310911893845, 0.2938925623893738, 0.08852922171354294, 0.020463770255446434, 0.09666207432746887, -0.006990742404013872],
    [-0.1065014973282814, 0.20087268948554993, 0.18935158848762512, -0.17352613806724548, 0.05884088575839996, -0.4646260440349579, -0.10628655552864075, -0.28338274359703064, -0.03379037603735924, 0.007989203557372093, -0.06132059171795845, -0.13810740411281586, 0.04504100978374481, 0.015115765854716301, -0.4030528962612152, -0.025872472673654556, -0.4061300754547119, -0.02022559940814972, -0.13893099129199982, 0.10193423181772232, 0.29334160685539246, 0.22542181611061096, -0.050897762179374695, 0.21366965770721436, -0.04277477413415909],
    [0.054688308387994766, 0.07500440627336502, 0.060474496334791183, -0.3848169445991516, -0.3781052529811859, 0.38639041781425476, 0.021527282893657684, 0.4060642719268799, -0.15761728584766388, -0.1271764189004898, -0.01066557876765728, -0.13985656201839447, 0.1605837494134903, 0.15716029703617096, -0.32516127824783325, 0.016290534287691116, 0.2249980866909027, -0.2878168523311615, -0.12032820284366608, -0.04713383689522743, 0.025025269016623497, 0.08448748290538788, 0.05337755009531975, 0.05431513488292694, -0.03361976519227028]
])


def init_coeffs(mode: str, init_pkl, key) -> jnp.ndarray:
    if mode == "scratch":
        coeffs = jax.random.uniform(key, GENOTYPE_SHAPE, minval=-0.1, maxval=0.1)
    elif mode == "d1_zeropad":
        coeffs = jnp.zeros(GENOTYPE_SHAPE)
        coeffs = coeffs.at[:, :25].set(D1_COEFFS)
    elif mode == "pkl":
        if init_pkl is None:
            raise ValueError("--init pkl requires --init_pkl <path>")
        with open(init_pkl, "rb") as f:
            arr = jnp.asarray(pickle.load(f))
        if arr.shape != GENOTYPE_SHAPE:
            raise ValueError(
                f"Expected pickle to contain a {GENOTYPE_SHAPE} array, got {arr.shape}"
            )
        coeffs = arr
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


def make_total_loss(loss_kind: str):
    def total_loss(coeffs, min_set_real, psi, n_refine_steps, metric):
        lag, spec = compute_losses_on_fixed_points(
            coeffs, min_set_real, psi, n_refine_steps, metric
        )
        if loss_kind == "lag":
            total = lag
        elif loss_kind == "spec":
            total = spec
        elif loss_kind == "both":
            total = lag + spec
        else:
            raise ValueError(f"Unknown loss kind {loss_kind}")
        return total, (lag, spec)

    return total_loss


def load_points(psi: int):
    """Try cluster path first, fall back to repo-local pkl."""
    cluster = f"/projects/ruehlehet/yidi/sLag/data_psi/1mil_patch_all_psi{psi}_seed1024.pkl"
    local = "1mil_patch_all_psi0_seed1024.pkl"
    for path in [cluster, local]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                arr = np.asarray(pickle.load(f))
            arr = np.concatenate([np.real(arr), np.imag(arr)], axis=1)
            return jax.device_put(jnp.asarray(arr)), path
    raise FileNotFoundError(f"No CY point cloud at {cluster} or {local}")


def main():
    parser = argparse.ArgumentParser(description="GD for sLag search (d=1+2)")
    parser.add_argument("--psi", type=int, default=0)
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
    parser.add_argument("--init", type=str, default="d1_zeropad",
                        choices=["scratch", "d1_zeropad", "pkl"])
    parser.add_argument("--init_pkl", type=str, default=None,
                        help="Path to a pkl with a (3, 250) array (use with --init pkl).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--job_id", type=str, default="0")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="./gd_runs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("=== GD for sLag search (d=1+2) ===")
    print(f"job_id={args.job_id} init={args.init} loss={args.loss} "
          f"lr={args.lr} steps={args.steps}")
    print(f"mine_interval={args.mine_interval} minset_size={args.minset_size} "
          f"newton_steps={args.newton_steps} inner_newton_steps={args.inner_newton_steps}")

    points_real, src_path = load_points(args.psi)
    print(f"Loaded {len(points_real)} points from {src_path}")

    key = jax.random.PRNGKey(args.seed)
    key, sub = jax.random.split(key)
    coeffs = init_coeffs(args.init, args.init_pkl, sub)
    coeffs = normalize_coeffs(coeffs)

    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(coeffs)

    total_loss = make_total_loss(args.loss)
    loss_value_and_grad = jax.jit(
        jax.value_and_grad(total_loss, argnums=0, has_aux=True),
        static_argnames=("n_refine_steps", "metric"),
    )

    psi = jnp.asarray(args.psi, dtype=jnp.complex128)
    history = []

    # Initial mining + loss eval, before any optimizer step.
    min_set_real, _, newton_check_pass = filter_and_refine(
        points_real, coeffs, psi,
        args.minset_size, args.newton_steps, filter_newton=True,
    )
    if not bool(newton_check_pass):
        print("  [warn] Newton check failed during initial mining")
    (init_loss, (init_lag, init_spec)), _ = loss_value_and_grad(
        coeffs, min_set_real, psi, args.inner_newton_steps, args.metric
    )
    init_lag_fit = float(jnp.exp(-10.0 * init_lag))
    init_spec_fit = 1.0 - float(init_spec)
    print(
        f"initial     | loss {float(init_loss):.6f} | "
        f"lag_loss {float(init_lag):.6f} (fit {init_lag_fit:.4f}) | "
        f"spec_loss {float(init_spec):.6f} (fit {init_spec_fit:.4f})"
    )
    history.append({
        "step": 0,
        "loss": float(init_loss),
        "lag_loss": float(init_lag),
        "spec_loss": float(init_spec),
        "lag_fit": init_lag_fit,
        "spec_fit": init_spec_fit,
        "gnorm": None,
    })

    for step in range(args.steps):
        t0 = time.time()
        # Skip step==0: just mined for the initial eval. Mining schedule
        # then fires at step==mine_interval, 2*mine_interval, etc.
        if step > 0 and step % args.mine_interval == 0:
            min_set_real, _, newton_check_pass = filter_and_refine(
                points_real, coeffs, psi,
                args.minset_size, args.newton_steps, filter_newton=True,
            )
            if not bool(newton_check_pass):
                print(f"  [warn] Newton check failed during mining at step {step}")

        (loss_val, (lag_loss, spec_loss)), grads = loss_value_and_grad(
            coeffs, min_set_real, psi, args.inner_newton_steps, args.metric
        )
        grads = jnp.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
        updates, opt_state = optimizer.update(grads, opt_state, coeffs)
        coeffs = optax.apply_updates(coeffs, updates)
        coeffs = normalize_coeffs(coeffs)

        gnorm = float(jnp.linalg.norm(grads))
        # Match GA's fitness conventions for easy comparison.
        lag_fit = float(jnp.exp(-10.0 * lag_loss))
        spec_fit = 1.0 - float(spec_loss)
        dt = time.time() - t0
        print(
            f"step {step+1:5d} | loss {float(loss_val):.6f} | "
            f"lag_loss {float(lag_loss):.6f} (fit {lag_fit:.4f}) | "
            f"spec_loss {float(spec_loss):.6f} (fit {spec_fit:.4f}) | "
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
            with open(ckpt, "wb") as f:
                pickle.dump(
                    {"coeffs": np.asarray(coeffs), "history": history,
                     "args": vars(args)},
                    f,
                )
            print(f"  [save] wrote {ckpt}")

    print("\nFinal coeffs:")
    print(format_array_with_commas(coeffs))


if __name__ == "__main__":
    main()
