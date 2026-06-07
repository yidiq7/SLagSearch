"""Numerical volume of a special-Lagrangian point cloud in the k=4 (Donaldson)
metric on the Fermat quintic.

Two estimators on the same min_set, both using the k=4 balanced metric:

  (A) k-NN density.   Treat the post-Newton point cloud as a sample from some
      density rho on L. Estimate rho from the k-th nearest-neighbour distance
      in the k=4 metric and invert:

          rho_hat(x_i) = k / (N * V_3 * R_k(x_i)^3),
          Vol_A(L)    ~= (1 / N) * sum_i 1 / rho_hat(x_i).

  (B) Calibration form.   For a true sLag, dvol_L = Re(e^{-i theta} Omega) at
      every point. Evaluate Omega on the k=4-orthonormal tangent frame of L
      and integrate against the same k-NN density:

          Vol_B(L) ~= (1 / N) * sum_i |Re(e^{-i theta} Omega)|_orth(x_i)
                                       / rho_hat(x_i).

For a true sLag Vol_B -> Vol_A; the ratio Vol_B / Vol_A is <cos delta theta>,
the average calibration deficit (1.0 perfect, smaller = worse).

CLI:
    python -m volume.compute_volume --run gd_runs/plots_slag_<job>
    python -m volume.compute_volume --coeffs <pkl> --min_set <pkl>
    python -m volume.compute_volume --run <dir> --k_neighbors 15 --save

--run resolves coeffs.pkl + min_set.pkl from a run folder (the layout written
by viz/fitness_pipeline.py). --coeffs / --min_set override individual pieces.
"""
import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from get_restriction import compute_affine_jacobian, compute_restriction
from helper import (
    assert_metric_psi_compatible,
    convert_real_to_complex_batch,
    delete_index,
    determine_patches_batch,
)
from slag_condition import (
    _assemble_metric_tensor,
    calculate_complex_metric_k4,
    compute_holomorphic_form,
)


V_3 = (4.0 / 3.0) * jnp.pi  # unit-ball volume in R^3


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def _load_coeffs(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "coeffs" in obj:
        return jnp.asarray(obj["coeffs"])
    return jnp.asarray(obj)


def _load_min_set(path):
    """Return min_set as (N, 10) real, accepting either (N, 5) complex or
    (N, 10) real on disk."""
    with open(path, "rb") as f:
        arr = np.asarray(pickle.load(f))
    if np.iscomplexobj(arr):
        arr = np.concatenate([np.real(arr), np.imag(arr)], axis=1)
    return jnp.asarray(arr)


def _resolve_inputs(run, coeffs_path, min_set_path):
    if run is None and (coeffs_path is None or min_set_path is None):
        raise ValueError("Provide --run, or both --coeffs and --min_set.")
    c_path = coeffs_path or (Path(run) / "coeffs.pkl")
    m_path = min_set_path or (Path(run) / "min_set.pkl")
    return _load_coeffs(c_path), _load_min_set(m_path)


# --------------------------------------------------------------------------
# k=4 metric, assembled at every min_set point in its own patch
# --------------------------------------------------------------------------

def _real_metric_k4_batch(min_set_complex, patch_indices):
    """Returns (N, 8, 8) real metric tensors G(x_i) on the patch of x_i."""
    def per_point(z, p):
        return _assemble_metric_tensor(calculate_complex_metric_k4(z, p))
    return jax.vmap(per_point)(min_set_complex, patch_indices)


# --------------------------------------------------------------------------
# k-NN distance: d(x_i -> x_j)^2 = Delta^T G(x_i) Delta, with Delta = (x_j
# expressed in x_i's affine patch) minus x_i, packed in the 8 real coords
# (Re of 4 inhomogeneous, then Im of the same 4).
# --------------------------------------------------------------------------

def _R_k_single(i, min_set_complex, patch_indices, real_metrics, k_neighbors):
    x_i = min_set_complex[i]
    p_i = patch_indices[i]
    g_i = real_metrics[i]                            # (8, 8)

    # Re-express every x_j in x_i's patch: x_j' = x_j / x_j[p_i]. Replace any
    # exact zero in the denominator with 1.0 (those distances will be huge in
    # the rest of the calculation and won't enter the k-th smallest anyway).
    denom = min_set_complex[:, p_i]
    denom_safe = jnp.where(jnp.abs(denom) < 1e-12, 1.0 + 0j, denom)
    x_in_patch = min_set_complex / denom_safe[:, None]    # (N, 5)

    x_i_in_patch = x_i / x_i[p_i]                         # (5,)
    delta_5 = x_in_patch - x_i_in_patch                   # (N, 5), [*, p_i] ~ 0
    delta_inhom = jax.vmap(lambda d: delete_index(d, p_i))(delta_5)   # (N, 4)
    delta_real = jnp.concatenate(
        [jnp.real(delta_inhom), jnp.imag(delta_inhom)], axis=1
    )                                                     # (N, 8)

    d_sq = jnp.einsum("na,ab,nb->n", delta_real, g_i, delta_real)
    d = jnp.sqrt(jnp.maximum(d_sq, 0.0))

    # sorted[0] = 0 (self), sorted[k] = k-th nearest non-self
    sorted_d = jnp.sort(d)
    return sorted_d[k_neighbors]


def _compute_R_k_chunked(min_set_complex, patch_indices, real_metrics,
                         k_neighbors, chunk_size):
    """R_k(x_i) for every i, processed in chunks of size `chunk_size`."""
    N = min_set_complex.shape[0]

    @jax.jit
    def chunk_fn(i_chunk):
        return jax.vmap(
            _R_k_single,
            in_axes=(0, None, None, None, None),
        )(i_chunk, min_set_complex, patch_indices, real_metrics, k_neighbors)

    out = []
    for c0 in range(0, N, chunk_size):
        c1 = min(c0 + chunk_size, N)
        out.append(np.asarray(chunk_fn(jnp.arange(c0, c1))))
    return jnp.asarray(np.concatenate(out, axis=0))


def _knn_density(R_k, N, k_neighbors):
    """k-NN density estimator on a 3-real-dim manifold."""
    return k_neighbors / (N * V_3 * R_k**3)


# --------------------------------------------------------------------------
# Approach (A): k-NN volume.
# --------------------------------------------------------------------------

def volume_knn_k4(
    min_set_real,
    coeffs=None,           # accepted for API symmetry; unused
    psi=0.0,
    k_neighbors=10,
    chunk_size=200,
    return_intermediates=False,
):
    """Volume(L) via the k-NN density estimator in the k=4 metric.

        Vol(L) ~= (V_3 / k) * sum_i R_k(x_i)^3,

    where R_k(x_i) is the k_neighbors-th nearest-neighbour distance from x_i
    measured in the k=4 metric at x_i.
    """
    del coeffs

    min_set_complex = convert_real_to_complex_batch(min_set_real)
    patch_indices = determine_patches_batch(min_set_complex)
    real_metrics = _real_metric_k4_batch(min_set_complex, patch_indices)

    R_k = _compute_R_k_chunked(
        min_set_complex, patch_indices, real_metrics, k_neighbors, chunk_size,
    )
    N = int(min_set_complex.shape[0])
    rho_hat = _knn_density(R_k, N, k_neighbors)
    vol = jnp.mean(1.0 / rho_hat)

    if return_intermediates:
        return vol, {"R_k": R_k, "rho_hat": rho_hat, "N": N}
    return vol


# --------------------------------------------------------------------------
# Approach (B): calibration-form volume.
#
# At each x_i we have the 8x3 restriction R that parametrises T_{x_i} L, the
# Poincare-residue scalar Omega_residue, and the choice Omega_coord of which
# 3 of the 4 affine complex coordinates form the residue basis. Then
#
#     Omega(R_1, R_2, R_3) = Omega_residue * det(J),
#
# where J = R[Omega_coord] + i R[Omega_coord + 4] is the 3x3 complex matrix
# of derivatives of the 3 residue-basis coordinates w.r.t. the 3 parameters
# of L. (compute_Omega_restriction in get_restriction.py returns
# det(J / max|J|), which preserves the phase but loses the magnitude. We
# multiply by max|J|^3 to recover the true value; equivalently we just call
# det on the unscaled J, which is what we do below.)
#
# Then for the orthonormal frame in g_k4|_L,
#
#     Omega_orth = Omega(R_1, R_2, R_3) / sqrt(det(R^T G(x) R)),
#
# and the calibration density is |Re(e^{-i theta} Omega_orth)|. theta is
# fixed globally to maximise sum_i Re(e^{-i theta} Omega_orth(x_i)) using the
# squared-phase trick (handles the +-1 orientation gauge on L).
# --------------------------------------------------------------------------

vmap_compute_affine_jacobian = jax.vmap(
    compute_affine_jacobian, in_axes=(0, 0, None, None)
)
vmap_compute_restriction = jax.vmap(compute_restriction, in_axes=0)


def _compute_omega_orth_chunked(
    min_set_real, min_set_complex, patch_indices, real_metrics,
    coeffs, psi, chunk_size,
):
    """Per-chunk: jacobian -> restriction -> Omega(R) -> (Omega_orth, vol_R).

    The full vmap'd pipeline blows up GPU memory at d=4: jax.jacobian's
    backward pass for the d=4 basis materialises a 70x70 outer-product
    workspace per point, and vmap across ~5e4 points holds them all
    simultaneously (~20 GB). Chunking caps peak memory at O(chunk_size).
    """
    N = min_set_real.shape[0]

    @jax.jit
    def chunk_fn(real_c, complex_c, patch_c, metric_c):
        jacs = vmap_compute_affine_jacobian(real_c, patch_c, coeffs, psi)
        restrictions = vmap_compute_restriction(jacs)            # (n, 8, 3)
        Omega_residue, _, Omega_coord = compute_holomorphic_form(
            complex_c, patch_c, psi,
        )

        n = restrictions.shape[0]
        row = jnp.arange(n)[:, None]
        Omega_coord_y = Omega_coord + 4
        jac_C = (
            restrictions[row, Omega_coord]
            + 1j * restrictions[row, Omega_coord_y]
        )                                                         # (n, 3, 3)
        Omega_R = Omega_residue * jnp.linalg.det(jac_C)           # (n,)

        g_L = jnp.einsum(
            "nij,nik,njl->nkl", metric_c, restrictions, restrictions,
        )                                                         # (n, 3, 3)
        vol_R_c = jnp.sqrt(jnp.maximum(jnp.linalg.det(g_L), 0.0)) # (n,)

        eps = 1e-30
        Omega_orth_c = Omega_R / (vol_R_c + eps)
        return Omega_orth_c, vol_R_c

    oo_chunks, vr_chunks = [], []
    for c0 in range(0, N, chunk_size):
        c1 = min(c0 + chunk_size, N)
        oo, vr = chunk_fn(
            min_set_real[c0:c1],
            min_set_complex[c0:c1],
            patch_indices[c0:c1],
            real_metrics[c0:c1],
        )
        oo_chunks.append(np.asarray(oo))
        vr_chunks.append(np.asarray(vr))
    return (
        jnp.asarray(np.concatenate(oo_chunks)),
        jnp.asarray(np.concatenate(vr_chunks)),
    )


def volume_calibration_k4(
    min_set_real,
    coeffs,
    psi=0.0,
    k_neighbors=10,
    chunk_size=200,
    return_intermediates=False,
):
    """Volume(L) via the calibration form integral in the k=4 metric.

    Uses the same k-NN density rho_hat(x_i) as volume_knn_k4 to convert the
    sample sum into an integral over L.
    """
    psi_jnp = jnp.asarray(complex(psi))
    coeffs = jnp.asarray(coeffs)

    min_set_complex = convert_real_to_complex_batch(min_set_real)
    patch_indices = determine_patches_batch(min_set_complex)
    real_metrics = _real_metric_k4_batch(min_set_complex, patch_indices)
    N = int(min_set_complex.shape[0])

    # k-NN density (same as (A))
    R_k = _compute_R_k_chunked(
        min_set_complex, patch_indices, real_metrics, k_neighbors, chunk_size,
    )
    rho_hat = _knn_density(R_k, N, k_neighbors)

    # Per-point Omega in the orthonormal g_k4-frame, chunked to keep peak
    # memory bounded for high-degree coeffs.
    Omega_orth, vol_R = _compute_omega_orth_chunked(
        min_set_real, min_set_complex, patch_indices, real_metrics,
        coeffs, psi_jnp, chunk_size,
    )

    # Global phase theta: Kuramoto-style on Omega_orth^2 (mod-pi identifies
    # theta and theta+pi, matching the L-orientation gauge).
    sum_sq = jnp.sum(Omega_orth ** 2)
    theta = jnp.angle(sum_sq) / 2.0

    cal_density = jnp.abs(jnp.real(jnp.exp(-1j * theta) * Omega_orth))
    vol = jnp.mean(cal_density / rho_hat)

    if return_intermediates:
        return vol, {
            "R_k": R_k,
            "rho_hat": rho_hat,
            "Omega_orth": Omega_orth,
            "vol_R": vol_R,
            "cal_density": cal_density,
            "theta": theta,
            "N": N,
        }
    return vol


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=str, default=None,
                   help="Run folder containing coeffs.pkl + min_set.pkl")
    p.add_argument("--coeffs", type=str, default=None,
                   help="Override coeffs path")
    p.add_argument("--min_set", type=str, default=None,
                   help="Override min_set path")
    p.add_argument("--psi", type=complex, default=0.0,
                   help="Quintic deformation parameter (k=4 requires psi=0)")
    p.add_argument("--k_neighbors", type=int, default=10,
                   help="k for the k-NN density estimator (default 10)")
    p.add_argument("--chunk_size", type=int, default=200,
                   help="Per-row chunk size for the pairwise distance pass")
    p.add_argument("--save", action="store_true",
                   help="Write volume_results.json to --run folder")
    args = p.parse_args()

    assert_metric_psi_compatible("k4_fermat", args.psi)

    coeffs, min_set_real = _resolve_inputs(args.run, args.coeffs, args.min_set)
    N = int(min_set_real.shape[0])
    if args.k_neighbors >= N:
        raise ValueError(
            f"k_neighbors={args.k_neighbors} must be < N={N}"
        )
    print(f"min_set N = {N}, coeffs shape = {tuple(coeffs.shape)}")
    print(f"k_neighbors = {args.k_neighbors}, metric = k4_fermat, psi = {args.psi}")
    print()

    vol_A, intA = volume_knn_k4(
        min_set_real, coeffs, psi=args.psi,
        k_neighbors=args.k_neighbors, chunk_size=args.chunk_size,
        return_intermediates=True,
    )
    rho_finite = jnp.isfinite(intA["rho_hat"]) & (intA["rho_hat"] > 0)
    R_k_med = float(jnp.median(intA["R_k"]))
    print(f"(A) k-NN volume     Vol_A = {float(vol_A):.6f}")
    print(f"    median R_k        = {R_k_med:.6f}")
    print(f"    finite rho_hat    = {int(jnp.sum(rho_finite))} / {N}")
    print()

    vol_B, intB = volume_calibration_k4(
        min_set_real, coeffs, psi=args.psi,
        k_neighbors=args.k_neighbors, chunk_size=args.chunk_size,
        return_intermediates=True,
    )
    print(f"(B) Calibration vol Vol_B = {float(vol_B):.6f}")
    print(f"    fitted phase theta = {float(intB['theta']):.6f}")
    print(f"    mean cal_density   = {float(jnp.mean(intB['cal_density'])):.6f}")
    print()

    ratio = float(vol_B / vol_A)
    print(f"Vol_B / Vol_A = {ratio:.6f}    (-> 1 for a perfect sLag; ratio = <cos d_theta>)")

    if args.save:
        if args.run is None:
            print("\n[warn] --save requires --run; skipping JSON write.")
        else:
            out_path = Path(args.run) / "volume_results.json"
            with open(out_path, "w") as f:
                json.dump({
                    "vol_A": float(vol_A),
                    "vol_B": float(vol_B),
                    "ratio_B_over_A": ratio,
                    "theta": float(intB["theta"]),
                    "median_R_k": R_k_med,
                    "k_neighbors": args.k_neighbors,
                    "N": N,
                    "metric": "k4_fermat",
                    "psi": str(args.psi),
                }, f, indent=2)
            print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
