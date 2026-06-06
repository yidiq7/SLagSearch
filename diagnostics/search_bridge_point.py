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
  2. Slerp on the unit sphere (CP^4 FS-geodesic) at n_samples values
     t in (0, 1):    gamma(t) = sin((1-t)theta)/sin(theta) * a
                              + sin(t*theta)/sin(theta) * b
     Every gamma(t) is in the FS lens by construction.
  3. Newton-refine each gamma(t) onto the submanifold via
     refine_point_iterative.
  4. For each refined P, compute the Newton-step residual (a measure of
     "did Newton converge?") and the FS distances d_FS(P, A), d_FS(P, B).
  5. Report all P that converged AND landed in the lens, sorted by
     max(d_FS(P, A), d_FS(P, B)).

Usage:
    python -m diagnostics.search_bridge_point \\
        --coeffs gd_runs/plots_slag_d4_run/coeffs.pkl \\
        --cluster_a .../cluster_split/cluster_0_points.pkl \\
        --cluster_b .../cluster_split/cluster_1_points.pkl \\
        --a_idx <i_A> --b_idx <i_B> \\
        [--psi 0+0j] [--n_samples 1000] [--newton_steps 50]
        [--t_min 0.01 --t_max 0.99]
        [--converged_tol 1e-4]
        [--save_pkl bridge_candidates.pkl]
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
    # Phase-align so <a_hat, b_hat> is real and non-negative.
    overlap = np.vdot(a_hat, b_hat)         # complex scalar
    if np.abs(overlap) < 1e-15:
        # Orthogonal in C^5 -> antipodal on CP^4 (theta = pi/2). Any phase
        # of b_hat works.
        phase = 1.0
    else:
        phase = overlap / np.abs(overlap)   # exp(i arg(overlap))
    b_hat_aligned = b_hat * np.conjugate(phase)   # now <a_hat, b_hat_a> in R_+

    real_overlap = float(np.real(np.vdot(a_hat, b_hat_aligned)))
    real_overlap = min(max(real_overlap, -1.0), 1.0)
    theta = float(np.arccos(real_overlap))
    if theta < 1e-12:
        raise ValueError("A and B coincide on CP^4; no geodesic to sample.")

    t_values = np.linspace(t_min, t_max, n_samples, dtype=np.float64)
    sin_th = np.sin(theta)
    w_a = np.sin((1.0 - t_values) * theta) / sin_th  # (n_samples,)
    w_b = np.sin(t_values * theta) / sin_th
    gammas = (w_a[:, None] * a_hat[None, :]
              + w_b[:, None] * b_hat_aligned[None, :])  # (n_samples, 5)
    # Numerical safety: re-normalize each row (slerp gives unit vectors in
    # exact arithmetic; float64 may drift by a few ULP).
    gammas /= np.linalg.norm(gammas, axis=1, keepdims=True)
    return gammas, theta, t_values


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
                        help="Number of geodesic init points (default 1000).")
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
    print(f"Slerping n_samples={args.n_samples} init points in "
          f"t in [{args.t_min}, {args.t_max}].")

    gammas, theta_check, t_values = fs_geodesic_samples(
        A, B, args.n_samples, args.t_min, args.t_max)
    assert abs(theta_check - theta_ab) < 1e-9
    # Each gamma(t) is at FS distance t*theta from A and (1-t)*theta from B.
    # Verify on one sample for sanity.
    g_mid = gammas[args.n_samples // 2]
    print(f"  sanity: gamma(t={t_values[args.n_samples//2]:.3f}) -> "
          f"d_FS(A) = {fs_distance(g_mid, A):.4e}, "
          f"d_FS(B) = {fs_distance(g_mid, B):.4e}")

    # ----- prepare inputs for Newton ---------------------------------
    # Convert each gamma (5,) complex -> (10,) real, with patch
    # determination (refine_point_iterative does its own patch detection
    # inside the loop, but we pass in a canonical |z_max|=1 representative
    # so the first iteration is well-conditioned).
    coeffs_j = jnp.asarray(coeffs_np)
    psi_j = jnp.asarray(args.psi, dtype=jnp.complex64)

    print(f"\nRunning Newton ({args.newton_steps} steps/point) on "
          f"{args.n_samples} initializations...")

    def to_real_with_rescale(gamma):
        # Rescale to the natural patch: divide by the largest-|z_i| coord.
        # This is what determine_patch_and_rescale_single also does; doing
        # it here keeps the first Newton iteration well-scaled.
        abs_g = jnp.abs(gamma)
        i_max = jnp.argmax(abs_g)
        scale = gamma[i_max]
        return convert_complex_to_real_single(gamma / scale)

    gammas_j = jnp.asarray(gammas, dtype=jnp.complex64)
    inits_real = jax.vmap(to_real_with_rescale)(gammas_j)  # (n_samples, 10)

    refine_fn = lambda p: refine_point_iterative(
        p, coeffs_j, psi_j, args.newton_steps)
    refined_real = jax.vmap(refine_fn)(inits_real)        # (n_samples, 10)

    # Per-point Newton residual.  Same scalar that
    # find_smooth_submanifold.filter_and_refine uses for convergence.
    residuals = jax.vmap(approx_distance_newton_step,
                         in_axes=(0, None, None))(
        refined_real, coeffs_j, psi_j)
    residuals_np = np.asarray(residuals)

    # Back to (5,) complex for FS comparisons.
    refined_complex_j = jax.vmap(convert_real_to_complex_single)(
        refined_real)
    refined_complex = np.asarray(refined_complex_j).astype(
        np.complex128)

    # Per-point FS distances to A and B.
    d_to_A = np.array([fs_distance(p, A) for p in refined_complex])
    d_to_B = np.array([fs_distance(p, B) for p in refined_complex])

    # ----- classification --------------------------------------------
    converged = residuals_np < args.converged_tol
    in_lens = (d_to_A < theta_ab) & (d_to_B < theta_ab)
    is_bridge = converged & in_lens

    n_conv = int(converged.sum())
    n_lens = int(in_lens.sum())
    n_bridge = int(is_bridge.sum())
    # Also flag P that effectively returned to A or B (Newton drifted
    # back along the geodesic). Threshold: within FS scale 1e-3 (way
    # below any meaningful within-cluster spacing).
    near_a = d_to_A < 1e-3
    near_b = d_to_B < 1e-3
    n_back_to_a = int((converged & near_a).sum())
    n_back_to_b = int((converged & near_b).sum())

    print()
    print("=" * 72)
    print(f"Bridge search summary  (theta = d_FS(A, B) = {theta_ab:.6e})")
    print("=" * 72)
    print(f"  total inits                       : {args.n_samples}")
    print(f"  converged   (||delta_p|| < {args.converged_tol:.1e}): "
          f"{n_conv}")
    print(f"  in FS lens  (d_A, d_B both < theta): {n_lens}")
    print(f"  BRIDGE      (converged AND in lens): {n_bridge}")
    print(f"  ... of which P landed back on A "
          f"(d_FS(P, A) < 1e-3)        : {n_back_to_a}")
    print(f"  ... of which P landed back on B "
          f"(d_FS(P, B) < 1e-3)        : {n_back_to_b}")
    if n_bridge - n_back_to_a - n_back_to_b > 0:
        print(f"  ==> {n_bridge - n_back_to_a - n_back_to_b} candidate "
              f"bridge point(s) distinct from A and B.")
    else:
        print(f"  ==> no bridge point distinct from A or B was found.")

    # ----- top-K bridge candidates -----------------------------------
    if n_bridge > 0:
        # Rank by max(d_to_A, d_to_B) (smaller = better "centered" in lens).
        score = np.maximum(d_to_A, d_to_B)
        # Restrict to bridge AND not back-to-A-or-B.
        eligible = is_bridge & (~near_a) & (~near_b)
        if eligible.any():
            order = np.argsort(np.where(eligible, score, np.inf))
            order = order[:min(args.top_k_report, int(eligible.sum()))]
            print()
            print("-" * 72)
            print(f"Top {len(order)} bridge candidates "
                  f"(ranked by max(d_FS(P, A), d_FS(P, B))):")
            print(f"  {'rank':>4}  {'t_init':>7}  {'residual':>11}  "
                  f"{'d_FS(P,A)':>13}  {'d_FS(P,B)':>13}  "
                  f"{'max/theta':>9}")
            for rank, idx in enumerate(order, start=1):
                ratio = float(score[idx] / theta_ab)
                print(f"  {rank:>4d}  {t_values[idx]:>7.3f}  "
                      f"{residuals_np[idx]:>11.3e}  "
                      f"{d_to_A[idx]:>13.6e}  {d_to_B[idx]:>13.6e}  "
                      f"{ratio:>9.3f}")
            # Print coords of the best one.
            best = int(order[0])
            print()
            print(f"Best bridge candidate (rank 1): "
                  f"t_init={t_values[best]:.4f}")
            print(f"  d_FS(P, A) = {d_to_A[best]:.6e}")
            print(f"  d_FS(P, B) = {d_to_B[best]:.6e}")
            print(f"  d_FS(A, B) = {theta_ab:.6e}")
            print(f"  residual   = {residuals_np[best]:.3e}")
            print(f"  P:")
            for k, zk in enumerate(refined_complex[best]):
                print(f"    z_{k} = {zk}")
        else:
            print("\n  All 'bridge' points coincide with A or B "
                  "(Newton drifted back along the geodesic).")

    if args.save_pkl is not None:
        out = {
            "args": vars(args),
            "theta_ab": theta_ab,
            "A": A,
            "B": B,
            "t_values": t_values,
            "refined_complex": refined_complex,    # (n_samples, 5)
            "residuals": residuals_np,             # (n_samples,)
            "d_to_A": d_to_A,                      # (n_samples,)
            "d_to_B": d_to_B,                      # (n_samples,)
            "converged": converged,                # (n_samples,)
            "in_lens": in_lens,                    # (n_samples,)
            "is_bridge": is_bridge,                # (n_samples,)
        }
        with open(args.save_pkl, "wb") as f:
            pickle.dump(out, f)
        print(f"\nSaved {args.save_pkl}")


if __name__ == "__main__":
    main()
