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
# For two Kahler metrics g, g' on P^4 and the complex hypersurface
# X = {Q = 0}, the volume-form ratio on X is
#
#   dvol_X_g / dvol_X_g'  =  (det g / det g') * (|dQ|^2_g' / |dQ|^2_g),
#
# where |dQ|^2_g = (d_a Q)^* g^{a bbar} d_b Q is computed with the inverse
# Kahler metric. The factor (det g / det g') is the volume-form ratio on the
# ambient P^4; the |dQ|^2 ratio accounts for the relative normal-direction
# length, since the volume form on X is dvol_P / |dQ|^2_g (with appropriate
# constants that cancel in the ratio).
#
# At psi = 0 the gradient is dQ/d zeta_i = 5 zeta_i^4 in inhomogeneous coords.
# --------------------------------------------------------------------------

def _dQ_psi0(zeta):
    """dQ in inhomogeneous coords at psi = 0. zeta has shape (4,) complex."""
    return 5.0 * zeta ** 4


def _per_point_J(z_complex, patch_idx):
    """J(x) = (det g_k4 / det g_FS) * (|dQ|^2_FS / |dQ|^2_k4) at one point."""
    g_FS = calculate_complex_metric_FS(z_complex, patch_idx)
    g_k4 = calculate_complex_metric_k4(z_complex, patch_idx)

    det_FS = jnp.real(jnp.linalg.det(g_FS))
    det_k4 = jnp.real(jnp.linalg.det(g_k4))

    zeta = delete_index(z_complex, patch_idx)
    dQ = _dQ_psi0(zeta)

    # |dQ|^2_g = dQ^H g^{-1} dQ.
    sol_FS = jnp.linalg.solve(g_FS, dQ)
    sol_k4 = jnp.linalg.solve(g_k4, dQ)
    normsq_FS = jnp.real(jnp.vdot(dQ, sol_FS))
    normsq_k4 = jnp.real(jnp.vdot(dQ, sol_k4))

    return (det_k4 / det_FS) * (normsq_FS / normsq_k4)


@jax.jit
def _J_chunk(z_chunk, patch_chunk):
    return jax.vmap(_per_point_J)(z_chunk, patch_chunk)


def _compute_J_array(z_complex, patch_indices, chunk_size):
    N = z_complex.shape[0]
    out = []
    for c0 in range(0, N, chunk_size):
        c1 = min(c0 + chunk_size, N)
        out.append(np.asarray(_J_chunk(z_complex[c0:c1], patch_indices[c0:c1])))
    return np.concatenate(out)


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

    # (2) Pointwise J(x)
    print("\nComputing J(x) pointwise...")
    J_arr = _compute_J_array(z_complex, patch_indices, args.j_chunk_size)
    J_mean = float(np.mean(J_arr))
    J_std = float(np.std(J_arr))
    J_med = float(np.median(J_arr))
    print(f"  <J> = {J_mean:.4f}    median(J) = {J_med:.4f}    std(J)/<J> = {J_std / J_mean:.4f}")

    # (1) Vol_FS via k-NN
    print("\nComputing Vol_FS(X) via k-NN, d=6...")
    rm_FS = _real_metric_batch_FS(z_complex, patch_indices)
    Vol_FS_knn, R_k_FS = volume_knn_d6(points_sub, rm_FS, args.k_neighbors, args.chunk_size)
    print(f"  Vol_FS(X)_knn   = {Vol_FS_knn:.4f}    median R_k_FS = {float(np.median(R_k_FS)):.4f}")

    # (3) Vol_{k=4} via k-NN
    print("\nComputing Vol_{k=4}(X) via k-NN, d=6...")
    rm_k4 = _real_metric_batch_k4(z_complex, patch_indices)
    Vol_k4_knn, R_k_k4 = volume_knn_d6(points_sub, rm_k4, args.k_neighbors, args.chunk_size)
    print(f"  Vol_{{k=4}}(X)_knn = {Vol_k4_knn:.4f}    median R_k_k4 = {float(np.median(R_k_k4)):.4f}")

    # Topological predictions in the codebase's convention
    pi3 = np.pi ** 3
    Vol_FS_top = (2 * np.pi) ** 3 * 5.0 / 6.0     # = 8 pi^3 * 5/6
    Vol_k4_donaldson = (8 * np.pi) ** 3 * 5.0 / 6.0  # = 512 pi^3 * 5/6
    Vol_k4_via_J = Vol_FS_knn * J_mean

    print("\n=== Cross-checks ===")
    print(f"Vol_FS(X) topological  (2pi)^3 * 5/6     = {Vol_FS_top:.4f}")
    print(f"Vol_FS(X) k-NN                            = {Vol_FS_knn:.4f}")
    print(f"  ratio (kNN / topological)              = {Vol_FS_knn / Vol_FS_top:.4f}")
    print()
    print(f"Vol_{{k=4}}(X) k-NN                          = {Vol_k4_knn:.4f}")
    print(f"Vol_{{k=4}}(X) via <J> . Vol_FS_knn          = {Vol_k4_via_J:.4f}")
    print(f"  ratio (kNN / via_J)                    = {Vol_k4_knn / Vol_k4_via_J:.4f}")
    print()
    print(f"Vol_{{k=4}}(X) Donaldson prediction (8pi)^3*5/6 = {Vol_k4_donaldson:.4f}")
    print(f"  ratio (kNN / Donaldson)                = {Vol_k4_knn / Vol_k4_donaldson:.4f}")
    print()
    print(f"<J> measured                              = {J_mean:.4f}")
    print(f"<J> Donaldson-standard prediction (4^3)  = 64.0000")
    print(f"  ratio (measured / 64)                  = {J_mean / 64.0:.4f}")
    print()
    print("If all ratios are close to 1, k-NN works and the codebase is in")
    print("Donaldson-standard normalization. Deviations point to either k-NN")
    print("bias (try larger --n_subsample / --k_neighbors) or a non-standard")
    print("overall constant in `unique_coeffs` of calculate_complex_metric_k4.")


if __name__ == "__main__":
    main()
