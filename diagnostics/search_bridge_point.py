"""Search for a point P on the candidate submanifold (zeros of the 3 user
equations defined by `coeffs` + the quintic) that lies inside the FS lens
between two cross-cluster anchor points A and B:

    d_FS(P, A) < d_FS(A, B)  AND  d_FS(P, B) < d_FS(A, B).

If such a P exists, the apparent gap between the two clusters can be
bridged by another submanifold point -- evidence that the two clusters
are connected in the continuous submanifold. If after dense sampling we
cannot find one, that is suggestive (not proof) that the gap is real.

Algorithm:
  1. Lift A, B to unit C^5 representatives, phase-align so <a, b> in R_+.
  2. Slerp on the unit sphere (CP^4 FS-geodesic) at n_samples t-values in
     [t_min, t_max]:
         gamma(t) = sin((1-t)*theta)/sin(theta) * a
                  + sin(t*theta)/sin(theta) * b
     Every gamma(t) is on the FS geodesic from [a] to [b], at FS distance
     t*theta from A and (1-t)*theta from B (both < theta).
  3. (Optional) For each gamma(t), draw n_perturb random horizontal
     tangent perturbations at FS distance sigma, via the exponential map:
         gamma_perturbed(t, k) = cos(sigma)*gamma(t) + sin(sigma)*u_{t,k},
     with u_{t,k} a unit vector Hermitian-orthogonal to gamma(t) in C^5
     (so gamma_perturbed is unit-norm, at FS distance sigma from
     gamma(t)). This pushes inits off the geodesic into a tube of FS
     radius sigma, broadening Newton's basin coverage.
  4. Newton-refine each init onto the submanifold via
     refine_point_iterative.
  5. For each refined P, compute the Newton-step residual (convergence
     proxy) and the FS distances d_FS(P, A), d_FS(P, B).
  6. Report all P that converged AND landed in the FS lens, sorted by
     max(d_FS(P, A), d_FS(P, B)). Inits whose refined P drifted back to
     A or B (within FS 1e-3) are flagged and excluded from the "bridge"
     count.

Usage:
    python -m diagnostics.search_bridge_point \\
        --coeffs gd_runs/plots_slag_d4_run/coeffs.pkl \\
        --cluster_a .../cluster_split/cluster_0_points.pkl \\
        --cluster_b .../cluster_split/cluster_1_points.pkl \\
        --a_idx <i_A> --b_idx <i_B> \\
        [--psi 0+0j] [--n_samples 1000]
        [--n_perturb 10 --sigma 0.05 --seed 0]
        [--newton_steps 50] [--t_min 0.01 --t_max 0.99]
        [--converged_tol 1e-4] [--save_pkl bridge_candidates.pkl]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from find_smooth_submanifold import (refine_point_iterative,
                                     approx_distance_newton_step)
from helper import (convert_complex_to_real_single,
                    convert_real_to_complex_single)


def _load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_coeffs(path: Path) -> np.ndarray:
    obj = _load_pkl(path)
    if isinstance(obj, dict) and "coeffs" in obj:
        obj = obj["coeffs"]
    arr = np.asarray(obj)
    if arr.ndim != 2 or arr.shape[0] != 3:
        raise ValueError(f"{path}: expected (3, w) coeffs, got {arr.shape}")
    return arr.astype(np.float32)


def load_points(path: Path) -> np.ndarray:
    z = np.asarray(_load_pkl(path))
    if z.ndim != 2 or z.shape[1] != 5:
        raise ValueError(f"{path}: expected (N, 5) complex, got {z.shape}")
    return z.astype(np.complex128)        # keep precision for geodesic


def fs_distance(z: np.ndarray, w: np.ndarray) -> float:
    """d_FS([z],[w]) = arccos(|<z,w>|/(||z|| ||w||)) on CP^4, float64."""
    z = np.asarray(z, dtype=np.complex128)
    w = np.asarray(w, dtype=np.complex128)
    overlap = np.abs(np.vdot(z, w))
    overlap /= (np.linalg.norm(z) * np.linalg.norm(w))
    overlap = min(overlap, 1.0)
    return float(np.arccos(overlap))


def fs_distance_batch(P: np.ndarray, q: np.ndarray) -> np.ndarray:
    """FS distance from each row of P (N, 5) to q (5,). Float64."""
    P = np.asarray(P, dtype=np.complex128)
    q = np.asarray(q, dtype=np.complex128)
    inner = np.abs(P @ np.conjugate(q))                  # (N,)
    inner /= (np.linalg.norm(P, axis=1) * np.linalg.norm(q))
    inner = np.clip(inner, 0.0, 1.0)
    return np.arccos(inner)


def fs_geodesic_samples(a: np.ndarray, b: np.ndarray,
                        n_samples: int,
                        t_min: float, t_max: float):
    """Slerp on the FS-geodesic from [a] to [b] in CP^4.

    Returns (gammas, theta, t_values):
        gammas: (n_samples, 5) complex128, unit-norm representatives
        theta:  float, d_FS([a], [b])
        t_values: (n_samples,) float64, parameter values used
    """
    a_hat = a / np.linalg.norm(a)
    b_hat = b / np.linalg.norm(b)
    overlap = np.vdot(a_hat, b_hat)
    if np.abs(overlap) < 1e-15:
        phase = 1.0
    else:
        phase = overlap / np.abs(overlap)
    b_hat_aligned = b_hat * np.conjugate(phase)

    real_overlap = float(np.real(np.vdot(a_hat, b_hat_aligned)))
    real_overlap = min(max(real_overlap, -1.0), 1.0)
    theta = float(np.arccos(real_overlap))
    if theta < 1e-12:
        raise ValueError("A and B coincide on CP^4; no geodesic to sample.")

    t_values = np.linspace(t_min, t_max, n_samples, dtype=np.float64)
    sin_th = np.sin(theta)
    w_a = np.sin((1.0 - t_values) * theta) / sin_th
    w_b = np.sin(t_values * theta) / sin_th
    gammas = (w_a[:, None] * a_hat[None, :]
              + w_b[:, None] * b_hat_aligned[None, :])
    gammas /= np.linalg.norm(gammas, axis=1, keepdims=True)
    return gammas, theta, t_values


def tangent_perturbations(gammas: jnp.ndarray, sigma: float,
                          n_perturb: int, key: jax.Array) -> jnp.ndarray:
    """Generate n_perturb random horizontal-tangent perturbations per gamma.

    For each unit vector gamma_i in C^5, draws n_perturb random complex
    normal vectors v, projects them Hermitian-orthogonal to gamma_i (so
    they live in the horizontal tangent space of CP^4 at [gamma_i]),
    normalizes, then exponential-maps:
        gamma_perturbed = cos(sigma)*gamma + sin(sigma)*u_hat.
    The result is unit-norm and at FS distance exactly sigma from gamma.

    Args:
        gammas:    (n_samples, 5) complex64, unit-norm.
        sigma:     scalar in (0, pi/2]; FS perturbation magnitude.
        n_perturb: number of perturbations per gamma.
        key:       JAX PRNG key.

    Returns:
        (n_samples, n_perturb, 5) complex64 perturbed unit vectors.
    """
    n_samples = gammas.shape[0]
    # Random complex normal: (n_samples, n_perturb, 5)
    v_real = jax.random.normal(
        key, (n_samples, n_perturb, 5, 2), dtype=jnp.float32)
    v = (v_real[..., 0] + 1j * v_real[..., 1]).astype(jnp.complex64)
    # Project to Hermitian-orthogonal complement of gamma:
    #   inner_{s, p} = sum_k conj(gamma_{s, k}) * v_{s, p, k}
    inner = jnp.einsum('sk,spk->sp', jnp.conjugate(gammas), v)
    u = v - inner[..., None] * gammas[:, None, :]
    u_norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
    # If a random v happens to be parallel to gamma (vanishing u_norm),
    # the gamma itself substitutes -- harmless (just one wasted init).
    u_hat = u / jnp.maximum(u_norm, 1e-12)
    perturbed = (jnp.cos(sigma) * gammas[:, None, :]
                 + jnp.sin(sigma) * u_hat)
    return perturbed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coeffs", type=Path, required=True,
                        help="Path to (3, w) coeffs pkl (bare array or "
                             "checkpoint dict with 'coeffs').")
    parser.add_argument("--cluster_a", type=Path, required=True,
                        help="Path to first (N, 5) complex pkl.")
    parser.add_argument("--cluster_b", type=Path, required=True,
                        help="Path to second (N, 5) complex pkl.")
    parser.add_argument("--a_idx", type=int, required=True,
                        help="Index of anchor point A inside --cluster_a.")
    parser.add_argument("--b_idx", type=int, required=True,
                        help="Index of anchor point B inside --cluster_b.")
    parser.add_argument("--psi", type=complex, default=complex(0.0, 0.0),
                        help="Complex psi for the quintic (default 0).")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of geodesic t-values (default 1000).")
    parser.add_argument("--n_perturb", type=int, default=0,
                        help="Number of random tangent perturbations per "
                             "geodesic t-value (default 0 = pure "
                             "geodesic). Each t produces (1 + n_perturb) "
                             "Newton inits.")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="FS magnitude of each tangent perturbation "
                             "(default 0.05 rad). Ignored if n_perturb=0.")
    parser.add_argument("--seed", type=int, default=0,
                        help="PRNG seed for tangent perturbations.")
    parser.add_argument("--newton_steps", type=int, default=50,
                        help="Newton iterations per refinement (default 50).")
    parser.add_argument("--t_min", type=float, default=0.01,
                        help="Smallest geodesic parameter (default 0.01; "
                             "skip t=0 which is A itself).")
    parser.add_argument("--t_max", type=float, default=0.99,
                        help="Largest geodesic parameter (default 0.99; "
                             "skip t=1 which is B itself).")
    parser.add_argument("--converged_tol", type=float, default=1e-4,
                        help="Max Newton-step residual ||delta_p|| for a "
                             "point to count as converged (default 1e-4, "
                             "matching find_smooth_submanifold.filter_and_"
                             "refine's threshold).")
    parser.add_argument("--save_pkl", type=Path, default=None,
                        help="If given, save a dict of all refined points "
                             "and their distances/residuals to this pkl.")
    parser.add_argument("--save_scatter", type=Path, default=None,
                        help="Path for the d_FS(P,A) vs d_FS(P,B) scatter "
                             "PNG over all in-lens converged inits. "
                             "Default: <coeffs_dir>/bridge_scatter_"
                             "a<a_idx>_b<b_idx>.png. Pass --no_scatter "
                             "to skip plotting entirely.")
    parser.add_argument("--no_scatter", action="store_true",
                        help="Skip the d_FS scatter PNG entirely.")
    parser.add_argument("--top_k_report", type=int, default=10,
                        help="Print the top-K bridge candidates "
                             "(default 10).")
    args = parser.parse_args()

    coeffs_np = load_coeffs(args.coeffs)
    A_pts = load_points(args.cluster_a)
    B_pts = load_points(args.cluster_b)
    if not (0 <= args.a_idx < A_pts.shape[0]):
        raise ValueError(
            f"--a_idx {args.a_idx} out of range for {args.cluster_a} "
            f"(N={A_pts.shape[0]}).")
    if not (0 <= args.b_idx < B_pts.shape[0]):
        raise ValueError(
            f"--b_idx {args.b_idx} out of range for {args.cluster_b} "
            f"(N={B_pts.shape[0]}).")

    A = A_pts[args.a_idx]
    B = B_pts[args.b_idx]
    theta_ab = fs_distance(A, B)
    print(f"Loaded coeffs {args.coeffs}: shape {coeffs_np.shape}")
    print(f"Loaded cluster A {args.cluster_a}: shape {A_pts.shape}")
    print(f"Loaded cluster B {args.cluster_b}: shape {B_pts.shape}")
    print(f"Anchor A = cluster_a[{args.a_idx}]")
    print(f"Anchor B = cluster_b[{args.b_idx}]")
    print(f"d_FS(A, B) = {theta_ab:.6e}  (FS, CP^4)")

    gammas, theta_check, t_values = fs_geodesic_samples(
        A, B, args.n_samples, args.t_min, args.t_max)
    assert abs(theta_check - theta_ab) < 1e-9

    # Build init set: unperturbed gamma at perturb_idx=0, plus optional
    # tangent-noise perturbations at perturb_idx=1..n_perturb.
    gammas_j = jnp.asarray(gammas, dtype=jnp.complex64)
    if args.n_perturb > 0:
        if not (0.0 < args.sigma <= np.pi / 2):
            raise ValueError(
                f"--sigma must be in (0, pi/2]; got {args.sigma}.")
        key = jax.random.PRNGKey(args.seed)
        perturbed = tangent_perturbations(
            gammas_j, args.sigma, args.n_perturb, key)
        # (n_samples, 1 + n_perturb, 5)
        all_inits_5 = jnp.concatenate(
            [gammas_j[:, None, :], perturbed], axis=1)
    else:
        all_inits_5 = gammas_j[:, None, :]  # (n_samples, 1, 5)

    n_per_t = int(all_inits_5.shape[1])
    total_inits = args.n_samples * n_per_t
    all_inits_flat = all_inits_5.reshape(-1, 5)         # (total, 5)
    t_idx_per_init = np.repeat(np.arange(args.n_samples), n_per_t)
    perturb_idx_per_init = np.tile(np.arange(n_per_t), args.n_samples)

    if args.n_perturb > 0:
        print(f"Slerping n_samples={args.n_samples} t-values in "
              f"[{args.t_min}, {args.t_max}], with {args.n_perturb} "
              f"tangent-noise perturbations per t (sigma={args.sigma}, "
              f"seed={args.seed}). Total inits = {total_inits}.")
        # Verify the perturbation FS distance on one sample.
        sample_orig = np.asarray(gammas_j[0]).astype(np.complex128)
        sample_pert = np.asarray(all_inits_5[0, 1]).astype(np.complex128)
        print(f"  sanity: d_FS(gamma(t={t_values[0]:.3f}), "
              f"perturbation 1) = "
              f"{fs_distance(sample_orig, sample_pert):.4e} "
              f"(should be ~{args.sigma})")
    else:
        print(f"Slerping n_samples={args.n_samples} init points in "
              f"[{args.t_min}, {args.t_max}]  (no tangent noise).")

    g_mid = gammas[args.n_samples // 2]
    print(f"  sanity: gamma(t={t_values[args.n_samples//2]:.3f}) -> "
          f"d_FS(A)={fs_distance(g_mid, A):.4e}, "
          f"d_FS(B)={fs_distance(g_mid, B):.4e}")

    # ----- prepare inputs for Newton ---------------------------------
    coeffs_j = jnp.asarray(coeffs_np)
    psi_j = jnp.asarray(args.psi, dtype=jnp.complex64)

    print(f"\nRunning Newton ({args.newton_steps} steps/init) on "
          f"{total_inits} initializations...")

    def to_real_with_rescale(gamma):
        # Rescale to the natural patch: divide by largest-|z_i| coord, so
        # the first Newton iteration is well-scaled.
        abs_g = jnp.abs(gamma)
        i_max = jnp.argmax(abs_g)
        scale = gamma[i_max]
        return convert_complex_to_real_single(gamma / scale)

    inits_real = jax.vmap(to_real_with_rescale)(all_inits_flat)  # (T, 10)

    refine_fn = lambda p: refine_point_iterative(
        p, coeffs_j, psi_j, args.newton_steps)
    refined_real = jax.vmap(refine_fn)(inits_real)               # (T, 10)

    residuals = jax.vmap(approx_distance_newton_step,
                         in_axes=(0, None, None))(
        refined_real, coeffs_j, psi_j)
    residuals_np = np.asarray(residuals)

    refined_complex_j = jax.vmap(convert_real_to_complex_single)(
        refined_real)
    refined_complex = np.asarray(refined_complex_j).astype(np.complex128)

    d_to_A = fs_distance_batch(refined_complex, A)               # (T,)
    d_to_B = fs_distance_batch(refined_complex, B)               # (T,)

    # ----- classification --------------------------------------------
    converged = residuals_np < args.converged_tol
    in_lens = (d_to_A < theta_ab) & (d_to_B < theta_ab)
    near_a = d_to_A < 1e-3
    near_b = d_to_B < 1e-3
    is_bridge_raw = converged & in_lens
    is_bridge = is_bridge_raw & (~near_a) & (~near_b)

    n_conv = int(converged.sum())
    n_lens = int(in_lens.sum())
    n_bridge_raw = int(is_bridge_raw.sum())
    n_back_to_a = int((converged & near_a).sum())
    n_back_to_b = int((converged & near_b).sum())
    n_bridge = int(is_bridge.sum())

    print()
    print("=" * 72)
    print(f"Bridge search summary  (theta = d_FS(A, B) = {theta_ab:.6e})")
    print("=" * 72)
    print(f"  total inits                            : {total_inits}")
    print(f"  converged   (||delta_p|| < {args.converged_tol:.1e})    : "
          f"{n_conv}")
    print(f"  in FS lens  (d_A, d_B both < theta)    : {n_lens}")
    print(f"  bridge raw  (converged AND in lens)    : {n_bridge_raw}")
    print(f"  ... drifted back on A  (d_FS(P,A)<1e-3): {n_back_to_a}")
    print(f"  ... drifted back on B  (d_FS(P,B)<1e-3): {n_back_to_b}")
    print(f"  ==> BRIDGE distinct from A and B       : {n_bridge}")

    # ----- top-K bridge candidates, two rankings ---------------------
    def _print_topk(order, label, score_fn, score_label):
        print()
        print("-" * 72)
        print(f"Top {len(order)} bridge candidates ({label}):")
        print(f"  {'rank':>4}  {'t_init':>7}  {'pert':>4}  "
              f"{'residual':>11}  {'d_FS(P,A)':>13}  "
              f"{'d_FS(P,B)':>13}  {score_label:>11}")
        for rank, idx in enumerate(order, start=1):
            t_idx = int(t_idx_per_init[idx])
            p_idx = int(perturb_idx_per_init[idx])
            print(f"  {rank:>4d}  {t_values[t_idx]:>7.3f}  "
                  f"{p_idx:>4d}  {residuals_np[idx]:>11.3e}  "
                  f"{d_to_A[idx]:>13.6e}  {d_to_B[idx]:>13.6e}  "
                  f"{score_fn(idx):>11.4e}")

    if n_bridge > 0:
        eligible = is_bridge

        # Ranking A: max(d_A, d_B), smallest first -> "most centered" in
        # the L_inf sense (favors candidates where BOTH distances are
        # small).
        score_max = np.maximum(d_to_A, d_to_B)
        order_max = np.argsort(np.where(eligible, score_max, np.inf))
        order_max = order_max[
            :min(args.top_k_report, int(eligible.sum()))]
        _print_topk(order_max,
                    "ranked by max(d_FS(P, A), d_FS(P, B))",
                    lambda i: float(score_max[i]), "max")

        # Ranking B: |d_A - d_B|, smallest first -> "most symmetric"
        # (favors candidates with d_A ~ d_B, i.e., near the midpoint of
        # the lens regardless of how close to A and B they are).
        score_diff = np.abs(d_to_A - d_to_B)
        order_diff = np.argsort(np.where(eligible, score_diff, np.inf))
        order_diff = order_diff[
            :min(args.top_k_report, int(eligible.sum()))]
        _print_topk(order_diff,
                    "ranked by |d_FS(P, A) - d_FS(P, B)| (symmetric)",
                    lambda i: float(score_diff[i]), "|d_A-d_B|")

        # Detail on rank-1 of the max ranking.
        best = int(order_max[0])
        best_t_idx = int(t_idx_per_init[best])
        best_p_idx = int(perturb_idx_per_init[best])
        print()
        print(f"Best by max(d_A, d_B): t_init={t_values[best_t_idx]:.4f}, "
              f"perturb_idx={best_p_idx}")
        print(f"  d_FS(P, A) = {d_to_A[best]:.6e}")
        print(f"  d_FS(P, B) = {d_to_B[best]:.6e}")
        print(f"  d_FS(A, B) = {theta_ab:.6e}")
        print(f"  residual   = {residuals_np[best]:.3e}")
        print(f"  P:")
        for k, zk in enumerate(refined_complex[best]):
            print(f"    z_{k} = {zk}")
    elif n_bridge_raw > 0:
        print("\n  All in-lens converged inits coincide with A or B "
              "(Newton drifted back along the geodesic).")

    # ----- d_A vs d_B scatter PNG ------------------------------------
    if not args.no_scatter:
        scatter_path = args.save_scatter
        if scatter_path is None:
            scatter_path = (args.coeffs.parent /
                            f"bridge_scatter_a{args.a_idx}_"
                            f"b{args.b_idx}.png")
        mask = converged & in_lens
        n_plot = int(mask.sum())
        if n_plot == 0:
            print(f"\n  No converged in-lens points to plot; skipping "
                  f"{scatter_path}.")
        else:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(d_to_A[mask], d_to_B[mask],
                       s=4, alpha=0.4, edgecolors="none",
                       label=f"in-lens converged P  (n = {n_plot})")
            # FS triangle-inequality lower bound: d_A + d_B >= theta_ab,
            # with equality iff P lies on the FS geodesic from A to B.
            xs = np.linspace(0.0, theta_ab, 64)
            ax.plot(xs, theta_ab - xs, "k--", lw=1,
                    label=r"$d_A + d_B = \theta$  (geodesic)")
            ax.axhline(theta_ab, color="r", lw=1, ls=":",
                       label=r"lens boundary  ($d = \theta$)")
            ax.axvline(theta_ab, color="r", lw=1, ls=":")
            ax.scatter([0.0], [theta_ab], marker="*", s=140,
                       color="C2", edgecolors="black", linewidths=0.5,
                       label="A")
            ax.scatter([theta_ab], [0.0], marker="*", s=140,
                       color="C3", edgecolors="black", linewidths=0.5,
                       label="B")
            ax.set_xlabel(r"$d_{FS}(P,\, A)$", fontsize=12)
            ax.set_ylabel(r"$d_{FS}(P,\, B)$", fontsize=12)
            ax.set_xlim(-0.01, theta_ab * 1.05)
            ax.set_ylim(-0.01, theta_ab * 1.05)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(
                rf"Bridge candidates: $d_{{FS}}(P,A)$ vs $d_{{FS}}(P,B)$"
                f"  ($\\theta$ = {theta_ab:.4f})",
                fontsize=13)
            ax.legend(fontsize=9, loc="lower left")
            ax.grid(True, linestyle="--", alpha=0.6)
            fig.tight_layout()
            fig.savefig(scatter_path, dpi=150)
            plt.close(fig)
            print(f"\nSaved scatter: {scatter_path}  "
                  f"({n_plot} in-lens converged points)")

    if args.save_pkl is not None:
        out = {
            "args": vars(args),
            "theta_ab": theta_ab,
            "A": A,
            "B": B,
            "t_values": t_values,                  # (n_samples,)
            "t_idx_per_init": t_idx_per_init,      # (total,)
            "perturb_idx_per_init": perturb_idx_per_init,  # (total,)
            "refined_complex": refined_complex,    # (total, 5)
            "residuals": residuals_np,             # (total,)
            "d_to_A": d_to_A,                      # (total,)
            "d_to_B": d_to_B,                      # (total,)
            "converged": converged,                # (total,)
            "in_lens": in_lens,                    # (total,)
            "is_bridge_raw": is_bridge_raw,        # (total,)
            "is_bridge": is_bridge,                # (total,)
        }
        with open(args.save_pkl, "wb") as f:
            pickle.dump(out, f)
        print(f"\nSaved {args.save_pkl}")


if __name__ == "__main__":
    main()
