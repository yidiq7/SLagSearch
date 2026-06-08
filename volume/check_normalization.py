"""Cross-check the codebase's k=4 vs FS metric normalizations on the ambient
Fermat-quintic point cloud, and validate the k-NN volume estimator.

Runs three numerical routes to the volume of the ambient CY threefold X (the
Fermat quintic) on the same FS-uniform sample
1mil_patch_all_psi0_seed1024.pkl:

  (1) Vol_FS(X)        via k-NN with the FS metric, d=6.
  (2) <J>              where J(x) = dvol_{k=4} / dvol_FS at x, via importance
                       sampling against the FS-uniform ambient density.
                       Uses the hypersurface formula
                          J = (det g_k4 / det g_FS) * (|dQ|^2_FS / |dQ|^2_k4).
  (3) Vol_{k=4}(X)     via k-NN with the k=4 metric, d=6.

Cross-checks:
  - (1)               vs the topological prediction Vol_FS(X) = (2 pi)^3 * 5/6
                       in the codebase's convention (omega = i ddbar log,
                       no 1/(2 pi) factor).
  - (3)               vs Vol_FS_knn * <J>     (importance-sampling route).
  - (3)               vs (8 pi)^3 * 5/6       (Donaldson-standard k=4 class).
  - <J>               vs 4^3 = 64             (k=4 class is 4 . FS class).

If all four cross-checks pass, the k=4 normalization is the expected Donaldson
one and the k-NN estimator is validated for use on the sLag computation.

Usage:
    python -m volume.check_normalization
    python -m volume.check_normalization --n_subsample 80000 --k_neighbors 15
"""
import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from helper import (
    assert_metric_psi_compatible,
    convert_real_to_complex_batch,
    delete_index,
    determine_patches_batch,
    dwork_points_path,
)
from slag_condition import (
    _assemble_metric_tensor,
    calculate_complex_metric_FS,
    calculate_complex_metric_k4,
)
from volume.compute_volume import _compute_R_k_chunked


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def _load_points_real(path):
    with open(path, "rb") as f:
        arr = np.asarray(pickle.load(f))
    if np.iscomplexobj(arr):
        arr = np.concatenate([np.real(arr), np.imag(arr)], axis=1)
    return jnp.asarray(arr)


# --------------------------------------------------------------------------
# Pointwise J(x) = dvol_{k=4} / dvol_FS on X
#
# For two Kahler metrics g, g' on P^N and the complex hypersurface X = {Q=0},
# the induced (N-1)-dim Hermitian metrics satisfy
#
#     dvol_X_g / dvol_X_g'  =  det(g|_X) / det(g'|_X)             (intrinsic)
#                           =  (det g / det g') * (|dQ|^2_g / |dQ|^2_g')
#
# where |dQ|^2_g = (d_a Q)^* g^{a bbar} (d_b Q). For uniform scaling g = lambda g',
# det ratio = lambda^N and |dQ|^2 ratio = 1/lambda, giving lambda^{N-1}, which
# matches the Kahler scaling (omega|_X)^{N-1} ~ lambda^{N-1}.
#
# (The first version of this script had |dQ|^2_g' / |dQ|^2_g, which gives
# lambda^{N+1} instead of lambda^{N-1} -- that was wrong and produced
# <J> ~ 1177 instead of ~64. Fixed.)
#
# At psi = 0 the gradient is dQ/d zeta_i = 5 zeta_i^4 in inhomogeneous coords.
# --------------------------------------------------------------------------

def _dQ_psi0(zeta):
    """dQ in inhomogeneous coords at psi = 0. zeta has shape (4,) complex."""
    return 5.0 * zeta ** 4


# --------------------------------------------------------------------------
# Per-point diagnostics in the IFT (implicit-function-theorem) basis of T_x X.
#
# At each point on X = {Q = 0}, pick max_idx = argmax |dQ/dzeta_i|. The
# implicit-function theorem says X is locally parametrised by the 3 other
# inhomogeneous coords {zeta_i : i != max_idx}, with tangent basis
#
#     e_i = d/dzeta_i  -  (dQ/dzeta_i / dQ/dzeta_max) * d/dzeta_max
#
# for each kept index i. In this chart:
#
#     Omega = (sign / dQ/dzeta_max) * dzeta_a ^ dzeta_b ^ dzeta_c
#     |Omega|^2 = 1 / |dQ/dzeta_max|^2.
#
# Computing det(g|_X) in this same IFT basis (3x3 Hermitian Gram det) lets
# us compare det(g_k4|_X) directly against |Omega|^2 with no chart factors
# to track. The Calabi-Yau Monge-Ampere identity says
#
#     det(g_CY|_X)  =  c * |Omega|^2     pointwise, c constant
#
# for any Ricci-flat Kahler metric in a fixed Kahler class. The k=4
# balanced metric satisfies this approximately. So fitting c globally and
# then checking the pointwise residual det(g_k4|_X) / (c * |Omega|^2)
# directly measures the k=4 metric's deviation from Ricci-flat.
# --------------------------------------------------------------------------

def _per_point_diagnostics(z_complex, patch_idx):
    """Returns (det_g_FS_X, det_g_k4_X, omega_sq) at one point, all in the
    IFT basis of T_x X."""
    g_FS = calculate_complex_metric_FS(z_complex, patch_idx)
    g_k4 = calculate_complex_metric_k4(z_complex, patch_idx)

    zeta = delete_index(z_complex, patch_idx)                # (4,) complex
    dQ = _dQ_psi0(zeta)                                       # (4,) complex

    max_idx = jnp.argmax(jnp.abs(dQ))
    keep_mask = jnp.arange(4) != max_idx
    keep_idx = jnp.sort(jnp.where(keep_mask, jnp.arange(4), 4))[:3]   # (3,)

    # Build the (4, 3) IFT-basis matrix E: column k = e_{keep_idx[k]}.
    # E[keep_idx[k], k] = 1; E[max_idx, k] = -dQ[keep_idx[k]] / dQ[max_idx].
    E = jnp.zeros((4, 3), dtype=z_complex.dtype)
    E = E.at[keep_idx, jnp.arange(3)].set(1.0 + 0j)
    E = E.at[max_idx, :].set(-dQ[keep_idx] / dQ[max_idx])

    # Hermitian Gram on T_x X. Kahler metric g_{a\bar b} = d_a d_{bar b} K is
    # sesquilinear: for u, v in T^(1,0), <u, v>_g = g_{a\bar b} u^a conj(v^b)
    # = u^T g conj(v). So G[i, j] = e_i^T g conj(e_j) = (E^T g conj(E))[i, j].
    G_FS_X = E.T @ g_FS @ jnp.conj(E)
    G_k4_X = E.T @ g_k4 @ jnp.conj(E)
    # Return the COMPLEX dets so main() can verify imag(det)/real(det) ~ 0
    # (a sanity check that the Gram matrix is actually Hermitian).
    det_FS_X = jnp.linalg.det(G_FS_X)
    det_k4_X = jnp.linalg.det(G_k4_X)

    # |Omega|^2 in the IFT chart: simply 1 / |dQ/dzeta_max|^2.
    q_max = dQ[max_idx]
    omega_sq = 1.0 / jnp.real(q_max * jnp.conj(q_max))

    return det_FS_X, det_k4_X, omega_sq


@jax.jit
def _diag_chunk(z_chunk, patch_chunk):
    return jax.vmap(_per_point_diagnostics)(z_chunk, patch_chunk)


def _compute_diagnostic_arrays(z_complex, patch_indices, chunk_size):
    """Returns (det_FS_X, det_k4_X, omega_sq) as numpy arrays."""
    N = z_complex.shape[0]
    dFS, dk4, oo = [], [], []
    for c0 in range(0, N, chunk_size):
        c1 = min(c0 + chunk_size, N)
        d1, d2, d3 = _diag_chunk(z_complex[c0:c1], patch_indices[c0:c1])
        dFS.append(np.asarray(d1)); dk4.append(np.asarray(d2)); oo.append(np.asarray(d3))
    return np.concatenate(dFS), np.concatenate(dk4), np.concatenate(oo)


# --------------------------------------------------------------------------
# Real 8x8 metric batches for k-NN
# --------------------------------------------------------------------------

def _real_metric_batch_FS(z_complex, patch_indices):
    def per_point(z, p):
        return _assemble_metric_tensor(calculate_complex_metric_FS(z, p))
    return jax.vmap(per_point)(z_complex, patch_indices)


def _real_metric_batch_k4(z_complex, patch_indices):
    def per_point(z, p):
        return _assemble_metric_tensor(calculate_complex_metric_k4(z, p))
    return jax.vmap(per_point)(z_complex, patch_indices)


# --------------------------------------------------------------------------
# 6-dim k-NN volume
# --------------------------------------------------------------------------

V_6 = float(np.pi ** 3 / 6.0)  # unit-ball volume in R^6


def volume_knn_d6(points_real, real_metrics, k_neighbors, chunk_size):
    """k-NN volume estimator for a 6-real-dim manifold (the CY threefold X)."""
    z_complex = convert_real_to_complex_batch(points_real)
    patch_indices = determine_patches_batch(z_complex)

    R_k = _compute_R_k_chunked(
        z_complex, patch_indices, real_metrics, k_neighbors, chunk_size,
    )
    N = int(z_complex.shape[0])
    rho_hat = k_neighbors / (N * V_6 * R_k ** 6)
    vol = float(jnp.mean(1.0 / rho_hat))
    return vol, np.asarray(R_k)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--points_file", type=str, default=None,
                   help="Ambient point cloud (default: dwork_points_path(0)).")
    p.add_argument("--psi", type=complex, default=0.0,
                   help="Quintic deformation (k=4 requires psi=0).")
    p.add_argument("--n_subsample", type=int, default=50000,
                   help="Subsample size for k-NN (default 50000).")
    p.add_argument("--k_neighbors", type=int, default=10)
    p.add_argument("--chunk_size", type=int, default=200,
                   help="Per-row chunk size for the pairwise k-NN pass.")
    p.add_argument("--j_chunk_size", type=int, default=2000,
                   help="Chunk size for the pointwise J(x) pass.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    assert_metric_psi_compatible("k4_fermat", args.psi)
    if complex(args.psi) != 0:
        raise ValueError("This script's dQ formula is hardcoded for psi=0.")

    points_path = args.points_file or dwork_points_path(args.psi, 1024)
    print(f"Loading {points_path}")
    points_real = _load_points_real(points_path)
    N_total = int(points_real.shape[0])
    print(f"Total ambient N = {N_total}")

    rng = np.random.default_rng(args.seed)
    if N_total > args.n_subsample:
        idx = np.asarray(rng.choice(N_total, args.n_subsample, replace=False))
        points_sub = jnp.asarray(np.asarray(points_real)[idx])
        print(f"Subsampled to N = {args.n_subsample}")
    else:
        points_sub = points_real
    N = int(points_sub.shape[0])

    z_complex = convert_real_to_complex_batch(points_sub)
    patch_indices = determine_patches_batch(z_complex)

    # Per-point diagnostics in the IFT basis: det(g_FS|_X), det(g_k4|_X),
    # |Omega|^2. These don't use k-NN at all; just per-point metric + |Omega|^2.
    print("\nComputing per-point diagnostics (det g_FS|_X, det g_k4|_X, |Omega|^2)...")
    det_FS_X_c, det_k4_X_c, omega_sq = _compute_diagnostic_arrays(
        z_complex, patch_indices, args.j_chunk_size,
    )
    # Hermitian Gram should have a real det; check imag/|det| as a sanity test
    # on the pullback formula.
    imag_frac = float(np.max(np.abs(np.imag(det_k4_X_c)) / (np.abs(det_k4_X_c) + 1e-30)))
    print(f"  Hermiticity check: max |imag(det g_k4|_X)| / |det| = {imag_frac:.2e}  (expect ~1e-10)")
    det_FS_X = np.real(det_FS_X_c)
    det_k4_X = np.real(det_k4_X_c)

    # --- Step 1: CY fit and Omega rescaling ---
    # c = <det(g_k4|_X) / |Omega|^2>  (single global constant fitted from data).
    # Rescale Omega -> sqrt(c) Omega so |Omega'|^2 = c |Omega|^2.
    # CY identity is then det(g_k4|_X) ~= |Omega'|^2 pointwise.
    c_pointwise = det_k4_X / omega_sq
    c_mean = float(np.mean(c_pointwise))
    c_std  = float(np.std(c_pointwise))
    omega_sq_rescaled = c_mean * omega_sq
    cy_residual_std = float(np.std(det_k4_X / omega_sq_rescaled))
    print(f"  c = <det(g_k4|_X) / |Omega|^2> = {c_mean:.4e}  (std/<c> = {c_std/c_mean:.4f})")
    print(f"  After rescaling Omega -> sqrt(c) Omega:")
    print(f"    pointwise CY residual std (det(g_k4|_X) / |Omega'|^2) = {cy_residual_std:.4f}")
    print(f"    ^ should be small if the k=4 metric is approximately Ricci-flat on X.")

    # --- Step 2: two mass-function J's ---
    # J_omega = (rescaled |Omega|^2) / det(g_FS|_X)            (Omega-route)
    # J_k4    = det(g_k4|_X)         / det(g_FS|_X)            (direct pullback)
    # Both are chart-invariant ratios of top-form densities on X. They agree
    # pointwise to the CY-residual error.
    J_omega = omega_sq_rescaled / det_FS_X
    J_k4    = det_k4_X          / det_FS_X
    Jw_mean = float(np.mean(J_omega))
    Jk_mean = float(np.mean(J_k4))

    # --- k-NN passes (FS distances and k=4 distances) ---
    print("\nComputing k-NN passes (FS and k=4 distances, d=6)...")
    rm_FS = _real_metric_batch_FS(z_complex, patch_indices)
    Vol_FS_knn, R_k_FS = volume_knn_d6(points_sub, rm_FS, args.k_neighbors, args.chunk_size)
    rm_k4 = _real_metric_batch_k4(z_complex, patch_indices)
    Vol_k4_knn, R_k_k4 = volume_knn_d6(points_sub, rm_k4, args.k_neighbors, args.chunk_size)
    print(f"  Vol_FS(X)_kNN  = {Vol_FS_knn:.4f}   median R_k_FS = {float(np.median(R_k_FS)):.4f}")
    print(f"  Vol_k4(X)_kNN  = {Vol_k4_knn:.4f}   median R_k_k4 = {float(np.median(R_k_k4)):.4f}")

    # --- Step 3: four estimators of Vol_k4 ---
    # Way 1: point-cloud average. Sample is FS-uniform, so
    #   Vol_k4 = integral over X of (rescaled Omega ∧ Omega-bar)
    #          = integral of J * dVol_FS
    # and the FS density in the integrand cancels with the implicit
    # FS-sample weight, leaving
    #   Vol_k4 ~= (Vol_FS_topological) * <J>_sample.
    # We use Vol_FS_top = (2pi)^3 * 5/6 as the analytic anchor (codebase
    # convention: omega = i d-dbar K, no 1/(2pi)).
    #
    # Way 2: same integral, but estimate Vol_FS pointwise via k-NN with FS
    # distances. Per-point sample mass = 1/rho_hat_FS(x_i). Result:
    #   Vol_k4 ~= sum_i (mass function)(x_i) / rho_hat_FS(x_i)
    #          = (V_6 / k) * sum_i (mass function)(x_i) * R_k_FS(x_i)^6.
    # This treats k-NN as the estimator of FS sampling density and runs
    # the same Monte-Carlo integral against that density.
    Vol_FS_top = (2 * np.pi) ** 3 * 5.0 / 6.0   # ≈ 206.71 (codebase convention)

    R_k_FS6_sum_factor = V_6 / args.k_neighbors  # multiplier on Σ ... R_k_FS^6

    # Way 1 (mass function integrated on the point cloud):
    Vol_way1_Omega = Vol_FS_top * Jw_mean
    Vol_way1_k4    = Vol_FS_top * Jk_mean

    # Way 2 (mass function integrated via k-NN FS-density):
    # The "mass function" here is the chart-invariant scalar that, multiplied
    # by dVol_FS, gives dVol_k4. That's J_omega and J_k4 respectively.
    R_k_FS6 = np.asarray(R_k_FS) ** 6
    Vol_way2_Omega = R_k_FS6_sum_factor * float(np.sum(J_omega * R_k_FS6))
    Vol_way2_k4    = R_k_FS6_sum_factor * float(np.sum(J_k4    * R_k_FS6))

    # All four should approximate Vol_k4(X) (= (8pi)^3 * 5/6 if Donaldson).
    # Divide each by (8pi)^3 to compare directly against the canonical 5/6.
    eight_pi_cubed = (8 * np.pi) ** 3
    five_sixths = 5.0 / 6.0

    print("\n" + "=" * 70)
    print("Four Vol_k4(X) estimators, compared to 5/6 (canonical)")
    print("=" * 70)
    print("Each is Vol_k4 in codebase units; the third column divides by (8pi)^3")
    print("to recover the canonical-class number 5/6 (assuming Donaldson")
    print("standard Kahler class for the codebase k=4 metric).")
    print()
    fmt = "  {label:<48s}  {val:>12.4f}   {canon:>10.6f}"
    print(f"  {'estimator':<48s}  {'Vol_k4':>12s}   {'/(8pi)^3':>10s}")
    print(fmt.format(label="Way 1, mass = rescaled Omega ∧ Omega-bar",
                     val=Vol_way1_Omega, canon=Vol_way1_Omega / eight_pi_cubed))
    print(fmt.format(label="Way 1, mass = det(g_k4|_X)",
                     val=Vol_way1_k4,    canon=Vol_way1_k4    / eight_pi_cubed))
    print(fmt.format(label="Way 2, mass = rescaled Omega ∧ Omega-bar",
                     val=Vol_way2_Omega, canon=Vol_way2_Omega / eight_pi_cubed))
    print(fmt.format(label="Way 2, mass = det(g_k4|_X)",
                     val=Vol_way2_k4,    canon=Vol_way2_k4    / eight_pi_cubed))
    print(fmt.format(label="(reference) Vol_k4_kNN, direct k-NN with g_k4",
                     val=Vol_k4_knn,     canon=Vol_k4_knn     / eight_pi_cubed))
    print(fmt.format(label="(reference) Vol_FS_kNN, direct k-NN with g_FS",
                     val=Vol_FS_knn,     canon=Vol_FS_knn     / (2 * np.pi) ** 3))
    print(f"  {'target: 5/6':<48s}  {'':>12s}   {five_sixths:>10.6f}")
    print()
    print("If all four estimators give ~5/6 (canonical), the codebase IS in")
    print("Donaldson-standard normalization AND k-NN is reliable. Disagreements:")
    print("  Way 1 vs Way 2 (same mass) -> consistency of k-NN density estimator.")
    print("  Mass = Omega vs Mass = det g_k4 (same way) -> k=4 CY-residual error.")
    print("  Direct Vol_k4_kNN vs Way 2 -> k-NN with g_k4 vs k-NN with g_FS.")


if __name__ == "__main__":
    main()
