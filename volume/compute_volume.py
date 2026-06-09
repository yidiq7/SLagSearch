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
    """Return min_set as (N, 10) real, normalised to z[argmax|z|] = 1.

    Accepts either (N, 5) complex or (N, 10) real on disk. The metric and
    Omega code in slag_condition.py now has defensive normalisation, but
    we also normalise here so the sanity check in main() reports the
    actual on-disk state (and any future consumer of _load_min_set gets
    a clean cloud regardless of the producer).
    """
    with open(path, "rb") as f:
        arr = np.asarray(pickle.load(f))
    if np.iscomplexobj(arr):
        z = arr
    else:
        z = arr[:, :5] + 1j * arr[:, 5:]

    max_idx = np.argmax(np.abs(z), axis=1)
    denoms = z[np.arange(z.shape[0]), max_idx]
    z = z / denoms[:, None]

    return jnp.asarray(np.concatenate([np.real(z), np.imag(z)], axis=1))


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
# k-scan: R_k at multiple k values in one sort, shared across the scan.
# --------------------------------------------------------------------------

def _R_k_multi_single(i, min_set_complex, patch_indices, real_metrics, k_indices):
    """Like _R_k_single but returns sorted_d at multiple k values in one sort."""
    x_i = min_set_complex[i]
    p_i = patch_indices[i]
    g_i = real_metrics[i]

    denom = min_set_complex[:, p_i]
    denom_safe = jnp.where(jnp.abs(denom) < 1e-12, 1.0 + 0j, denom)
    x_in_patch = min_set_complex / denom_safe[:, None]
    x_i_in_patch = x_i / x_i[p_i]
    delta_5 = x_in_patch - x_i_in_patch
    delta_inhom = jax.vmap(lambda d: delete_index(d, p_i))(delta_5)
    delta_real = jnp.concatenate(
        [jnp.real(delta_inhom), jnp.imag(delta_inhom)], axis=1,
    )
    d_sq = jnp.einsum("na,ab,nb->n", delta_real, g_i, delta_real)
    d = jnp.sqrt(jnp.maximum(d_sq, 0.0))
    sorted_d = jnp.sort(d)
    return sorted_d[k_indices]                            # (K,)


def _compute_R_k_multi_chunked(
    min_set_complex, patch_indices, real_metrics, k_values, chunk_size,
):
    """(N, K) array of k-NN distances, one column per k in k_values."""
    N = min_set_complex.shape[0]
    k_indices = jnp.asarray(k_values)

    @jax.jit
    def chunk_fn(i_chunk):
        return jax.vmap(
            _R_k_multi_single,
            in_axes=(0, None, None, None, None),
        )(i_chunk, min_set_complex, patch_indices, real_metrics, k_indices)

    out = []
    for c0 in range(0, N, chunk_size):
        c1 = min(c0 + chunk_size, N)
        out.append(np.asarray(chunk_fn(jnp.arange(c0, c1))))
    return jnp.asarray(np.concatenate(out, axis=0))       # (N, K)


# --------------------------------------------------------------------------
# Calabi-Yau (Monge-Ampere) constant for Omega normalisation.
#
# The script's Omega is the bare Poincare residue (f = sign / dQ_max) with
# no calibration scaling. For the calibration relation
#
#     dvol_L = Re(e^{-i theta} Omega)|_L
#
# to give the geometric volume of L directly, Omega needs to be normalised
# so |Omega|_orthonormal = 1 on T_x X. The Monge-Ampere identity for a CY
# threefold says |Omega|^2 = c is constant, with
#
#     c = det(g_X|_chart) / |Omega|^2_chart
#
# (chart-invariant on a Ricci-flat metric). We compute c on the same min_set
# via an IFT-basis pullback of g_k4 onto T_x X, exactly the same machinery
# as volume/check_normalization.py::_per_point_diagnostics. Then we
# rescale Omega -> Omega / sqrt(c_median) before the Vol_B integral so that
# Vol_B comes out at the same geometric scale as Vol_A.
#
# Hardcoded for psi = 0 (k=4 metric is only valid there).
# --------------------------------------------------------------------------

def _per_point_cy_const(z_complex, patch_idx):
    """c_pointwise = det(g_k4|_X) / |Omega|^2 at one point in IFT chart."""
    g_k4 = calculate_complex_metric_k4(z_complex, patch_idx)
    # post-normalisation min_set has z[patch] = 1, so delete_index gives the
    # correct inhomogeneous coords.
    zeta = delete_index(z_complex, patch_idx)             # (4,) complex
    dQ = 5.0 * zeta ** 4                                  # psi=0 hardcoded

    max_idx = jnp.argmax(jnp.abs(dQ))
    keep_mask = jnp.arange(4) != max_idx
    keep_idx = jnp.sort(jnp.where(keep_mask, jnp.arange(4), 4))[:3]

    # IFT-basis matrix E: column k = e_{keep_idx[k]}.
    E = jnp.zeros((4, 3), dtype=z_complex.dtype)
    E = E.at[keep_idx, jnp.arange(3)].set(1.0 + 0j)
    E = E.at[max_idx, :].set(-dQ[keep_idx] / dQ[max_idx])

    # Hermitian Gram on T_x X via E^T g conj(E).
    G_k4_X = E.T @ g_k4 @ jnp.conj(E)
    det_k4_X = jnp.real(jnp.linalg.det(G_k4_X))

    # |Omega|^2 in the IFT chart: 1 / |dQ/dzeta_max|^2.
    q_max = dQ[max_idx]
    omega_sq = 1.0 / jnp.real(q_max * jnp.conj(q_max))

    return det_k4_X / omega_sq


def _compute_cy_constant_chunked(min_set_complex, patch_indices, chunk_size):
    """Returns c_pointwise array (N,) of det(g_k4|_X) / |Omega|^2 values."""
    N = min_set_complex.shape[0]

    @jax.jit
    def chunk_fn(z_chunk, p_chunk):
        return jax.vmap(_per_point_cy_const)(z_chunk, p_chunk)

    out = []
    for c0 in range(0, N, chunk_size):
        c1 = min(c0 + chunk_size, N)
        out.append(np.asarray(chunk_fn(
            min_set_complex[c0:c1], patch_indices[c0:c1],
        )))
    return np.concatenate(out)


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
# k-scan: Vol_A and Vol_B at multiple k, sharing the expensive intermediates.
# --------------------------------------------------------------------------

def volumes_kscan(
    min_set_real,
    coeffs,
    psi=0.0,
    k_values=(10, 20, 50),
    chunk_size=200,
    return_intermediates=False,
):
    """Compute Vol_A (k-NN) and Vol_B (calibration) for each k in k_values.

    Shares the metric assembly, the pairwise R_k pass (one sort per query
    yields all k's), and the Omega-orthonormal-frame pass across the scan.
    Only rho_hat and the two sums get re-evaluated per k.

    Vol_B uses Omega rescaled by sqrt(c_median), where
        c = det(g_k4|_X) / |Omega|^2
    is the Monge-Ampere "constant" (constant on a Ricci-flat metric). The
    bare Poincare-residue Omega in compute_holomorphic_form has intrinsic
    norm |Omega|^2 = 1/c (in any chart, |Omega|^2 = |f|^2/det(g_X|_chart)
    = omega_sq/det(g_X) = 1/c). After rescaling by sqrt(c) the calibrated
    Omega satisfies |Omega|_orth = 1, so for a true sLag at the optimal
    phase, |Re(e^{-i theta} Omega_orth)| = 1 pointwise and Vol_B -> Vol(L)
    = Vol_A in the k-NN limit.

    Without this rescaling Vol_B comes out a factor of 1/sqrt(c) ~ 0.17x
    smaller than the geometric volume (so B/A = 0.029 = 1/c for a true
    sLag, instead of B/A = 1).
    """
    psi_jnp = jnp.asarray(complex(psi))
    coeffs = jnp.asarray(coeffs)
    k_values = tuple(int(k) for k in k_values)

    min_set_complex = convert_real_to_complex_batch(min_set_real)
    patch_indices = determine_patches_batch(min_set_complex)
    real_metrics = _real_metric_k4_batch(min_set_complex, patch_indices)
    N = int(min_set_complex.shape[0])

    # All R_k values in one chunked pairwise pass.
    R_k_all = _compute_R_k_multi_chunked(
        min_set_complex, patch_indices, real_metrics, k_values, chunk_size,
    )                                                     # (N, K)

    # Calibration density (independent of k).
    Omega_orth, vol_R = _compute_omega_orth_chunked(
        min_set_real, min_set_complex, patch_indices, real_metrics,
        coeffs, psi_jnp, chunk_size,
    )

    # Monge-Ampere constant c on X (constant on Ricci-flat k=4 metric).
    # Median is robust to IFT-basis outliers (a handful of points where
    # |dQ_max| is small give artificially large c_pointwise; the bulk is
    # tight if k=4 is approximately Ricci-flat).
    c_pointwise = _compute_cy_constant_chunked(
        min_set_complex, patch_indices, chunk_size,
    )
    c_med = float(np.median(c_pointwise))
    c_mean = float(np.mean(c_pointwise))

<<<<<<< Updated upstream
    # Rescale Omega so |Omega|_orth = 1 on T_x X for the calibration to give
    # Vol(L) directly. Equivalent to Omega -> Omega * sqrt(c_med).
=======
    # Rescale Omega so |Omega_orth| = 1 on the orthonormal X-frame, then
    # for a true sLag at optimal phase |Re(e^{-i theta} Omega_orth)| = 1
    # pointwise and Vol_B = mean(1/rho_hat) = Vol_A. Bare Omega has
    # |Omega|^2 = 1/c (intrinsic), so we MULTIPLY by sqrt(c) to get
    # |Omega_new|^2 = c * (1/c) = 1.
>>>>>>> Stashed changes
    Omega_orth_normalized = Omega_orth * jnp.sqrt(c_med)
    sum_sq = jnp.sum(Omega_orth_normalized ** 2)
    theta = jnp.angle(sum_sq) / 2.0
    cal_density = jnp.abs(
        jnp.real(jnp.exp(-1j * theta) * Omega_orth_normalized)
    )

    results = []
    for k_idx, k in enumerate(k_values):
        R_k = R_k_all[:, k_idx]
        rho_hat = k / (N * V_3 * R_k ** 3)
        vol_A = float(jnp.mean(1.0 / rho_hat))
        vol_B = float(jnp.mean(cal_density / rho_hat))
        results.append({
            "k": k,
            "vol_A": vol_A,
            "vol_B": vol_B,
            "ratio_B_over_A": vol_B / vol_A,
            "median_R_k": float(jnp.median(R_k)),
        })

    summary = {
        "theta": float(theta),
        "N": N,
        "k_values": list(k_values),
        "c_med": c_med,
        "c_mean": c_mean,
    }

    if return_intermediates:
        return results, summary, {
            "R_k_all": R_k_all,
            "Omega_orth": Omega_orth,
            "Omega_orth_normalized": Omega_orth_normalized,
            "c_pointwise": c_pointwise,
            "vol_R": vol_R,
            "cal_density": cal_density,
        }
    return results, summary


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
    p.add_argument("--k_neighbors", type=str, default="10,20,50",
                   help="k for k-NN density. Single int '10' or comma-separated "
                        "'10,20,50' for a k-scan (default).")
    p.add_argument("--chunk_size", type=int, default=200,
                   help="Per-row chunk size for the pairwise distance pass")
    p.add_argument("--save", action="store_true",
                   help="Write volume_results.json to --run folder")
    args = p.parse_args()

    assert_metric_psi_compatible("k4_fermat", args.psi)

    # Parse k values.
    k_values = sorted({int(x.strip()) for x in args.k_neighbors.split(",")})

    coeffs, min_set_real = _resolve_inputs(args.run, args.coeffs, args.min_set)
    N = int(min_set_real.shape[0])
    max_k = max(k_values)
    if max_k >= N:
        raise ValueError(f"max k_neighbors={max_k} must be < N={N}")
    print(f"min_set N = {N}, coeffs shape = {tuple(coeffs.shape)}")
    print(f"k_values = {k_values}, metric = k4_fermat, psi = {args.psi}")

    # Sanity check: the metric / Omega code assumes z[patch_idx] = 1.
    # _load_min_set normalises on load, but verify and warn if it didn't.
    z_check = convert_real_to_complex_batch(min_set_real)
    patches_check = determine_patches_batch(z_check)
    z_at_patch = z_check[jnp.arange(z_check.shape[0]), patches_check]
    mag_at_patch = np.asarray(jnp.abs(z_at_patch))
    print(f"\nSanity check: |z[patch_idx]| over loaded min_set "
          f"(expect = 1.0 if normalised):")
    print(f"  min    = {float(np.min(mag_at_patch)):.4e}    "
          f"max    = {float(np.max(mag_at_patch)):.4e}")
    print(f"  median = {float(np.median(mag_at_patch)):.4e}    "
          f"std    = {float(np.std(mag_at_patch)):.4e}")
    if np.max(np.abs(mag_at_patch - 1.0)) > 1e-6:
        print(f"  WARNING: min_set is not normalised post-load -- this is a")
        print(f"  _load_min_set regression. slag_condition.py has defensive")
        print(f"  normalisation so results should still be correct, but flag it.")
    print()

    print("Running k-scan (shared R_k pass and Omega-orth pass)...")
    results, summary = volumes_kscan(
        min_set_real, coeffs, psi=args.psi, k_values=k_values,
        chunk_size=args.chunk_size,
    )

    print()
    print(f"  Monge-Ampere constant on the min_set:")
    print(f"    c_median = {summary['c_med']:.4e}    c_mean = {summary['c_mean']:.4e}")
    print(f"    Omega rescaled by sqrt(c_med) = {np.sqrt(summary['c_med']):.4e}")
<<<<<<< Updated upstream
    print(f"    (calibration: |Omega|_orth = 1 on T_x X, so Vol_B has same")
    print(f"     geometric scale as Vol_A.)")
=======
    print(f"    (calibration: |Omega_orth| = 1 on the orthonormal X-frame,")
    print(f"     so Vol_B has the same geometric scale as Vol_A.)")
>>>>>>> Stashed changes
    print()
    print("=" * 70)
    print("k-scan: Vol_A (k-NN density) and Vol_B (calibration form) on L")
    print("=" * 70)
    print(f"  {'k':>4}  {'Vol_A':>12}  {'Vol_B':>12}  {'B/A':>8}  "
          f"{'med R_k':>10}")
    for r in results:
        print(f"  {r['k']:>4}  {r['vol_A']:>12.4f}  {r['vol_B']:>12.4f}  "
              f"{r['ratio_B_over_A']:>8.4f}  {r['median_R_k']:>10.4f}")
    print()
    print(f"  fitted global phase theta = {summary['theta']:.6f}")
    print()
<<<<<<< Updated upstream
    print("The CY metric needs to be rescaled by (8pi)^3 to match the value 5/6 of the Fermat quintic")
    print("Therefore, the volume A and B of the slag should be normalized by (8pi)^3/2")
=======
    print("Interpretation:")
    print("  Vol_A and Vol_B both scale with k through rho_hat; the RATIO B/A")
    print("  is k-independent (the calibration density factors out).")
    print()
    print("  Vol_A stable across k -> k-NN estimator converged on this sample.")
    print("  Vol_A drifts with k   -> sample-density non-uniformity dominates;")
    print("                           prefer larger-k value as more reliable.")
    print("  B/A close to 1        -> calibration holds, L is approximately sLag.")
    print("  B/A < 1               -> calibration deficit = <|Omega_orth|*|cos d_theta|>;")
    print("                           Vol_B is a lower bound on Vol(L).")
    print()
    print("  Note: Vol(L) is set by the homology class [L] in H_3(X), NOT by")
    print("  the ambient cohomological volume 5/6 (that's Vol_X^canonical for")
    print("  the 6-real-dim ambient X). They're different dimensional volumes")
    print("  ([length]^3 vs [length]^6) and Vol(L) does NOT have to be smaller.")
    print("  The right reliability test is Vol_B/Vol_A -> 1 for a true sLag.")

    # --- Canonical units conversion ---
    # Two conversions take script G-convention codebase-k=4 volume to
    # convention A (factor-of-2 in g_R) with canonical FS normalisation
    # (omega satisfying integral_{CP^1} omega = 1):
    #   (1) g_R^A = 2 g_R^B -> Vol scales by 2^{d/2} on a real d-manifold.
    #   (2) omega_codebase_k4 = 8 pi * omega_FS_canonical
    #       -> g scales by 8 pi -> Vol by (8 pi)^{d/2}.
    # Combined: Vol_canonical = Vol_script * (2 / (8 pi))^{d/2}
    #                         = Vol_script * (1 / (4 pi))^{d/2}.
    # For d = 3 (L is 3-real-dim): factor (1/(4 pi))^{3/2} ~= 1/44.55.
    # Sanity check: same conversion with d = 6 takes Vol_X^script ~= 1645
    # back to Vol_X^canonical = 5/6.
    canonical_factor_L = (1.0 / (4.0 * np.pi)) ** 1.5
    print()
    print("=" * 70)
    print("Canonical-FS units  (Vol_X^canonical = 5/6 in this convention)")
    print("=" * 70)
    print(f"  Conversion factor for 3-real-dim L: "
          f"(1/(4 pi))^(3/2) = {canonical_factor_L:.4e}")
    print()
    print(f"  {'k':>4}  {'Vol_A^can':>14}  {'Vol_B^can':>14}")
    for r in results:
        vA_can = r['vol_A'] * canonical_factor_L
        vB_can = r['vol_B'] * canonical_factor_L
        print(f"  {r['k']:>4}  {vA_can:>14.6f}  {vB_can:>14.6f}")
    print()
    print(f"  For reference: Vol_X^canonical = 5/6 = {5/6:.6f} (6-real-dim).")
    print(f"  Vol(L) and Vol(X) have different dimensions; Vol(L) > 5/6 is fine.")

    if args.save:
        if args.run is None:
            print("\n[warn] --save requires --run; skipping JSON write.")
        else:
            # Augment each row with canonical-FS-units volumes.
            results_with_canonical = [
                {**r,
                 "vol_A_canonical": r["vol_A"] * canonical_factor_L,
                 "vol_B_canonical": r["vol_B"] * canonical_factor_L}
                for r in results
            ]
            out_path = Path(args.run) / "volume_results.json"
            with open(out_path, "w") as f:
                json.dump({
                    "kscan": results_with_canonical,
                    "theta": summary["theta"],
                    "c_med": summary["c_med"],
                    "c_mean": summary["c_mean"],
                    "canonical_factor_L": canonical_factor_L,
                    "N": N,
                    "metric": "k4_fermat",
                    "psi": str(args.psi),
                }, f, indent=2)
            print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
>>>>>>> Stashed changes
