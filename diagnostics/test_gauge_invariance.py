"""Regression tests: fitness-pipeline outputs must be projective-gauge invariant.

The (N, 10) real point clouds consumed by the pipeline are projective
representatives: multiplying a point by a global phase e^{i alpha} (or
storing it normalised on a different coordinate) does not move the point on
the quintic, so geometric outputs of compute_combined_fitness must not
change.

Phase checks (the original bug, fixed in compute_combined_fitness): when the
patch index used downstream is re-derived with a fresh argmax over moduli
instead of being taken together with the stored normalisation, equal-moduli
loci (the CDGP T^3 diagonal ansatz |Z0|=|Z1|=|Z2|=|Z3|) break the argmax tie
with 1-ulp rounding noise; the picked coordinate holds a value
e^{i phi} != 1, compute_affine_jacobian (raw-point frame) pairs
inconsistently with compute_holomorphic_form (normalised frame), and every
Omega|_L phase shifts by 3*phi. Pre-fix these checks read ~1.5 rad; the true
spread at psi=100 is ~5e-9 (the asymptotic ~psi^-5 law).

Kahler-form checks (characterisation, correct before and after the fix): the
FS metric depends on coordinates only through zbar_a z_b, so a global phase
is an exact isometry and the frame mismatch could never corrupt the
Lagrangian defect. The per-point defect ||R^T w R||_F / ||w||_F is still
chart-covariant (a different patch pick changes the graph basis; observed up
to ~2.6x per point on the T^3), but its median is gauge-stable and its zeros
are exact in every chart. The RP^3 real locus -- exactly FS-Lagrangian with
exactly constant phase, as the fixed locus of an anti-holomorphic FS
isometry -- pins the absolute floor: the pipeline returns defect exactly 0.0
and phase spread exactly 0.0 there.

Checks (T^3 and RP^3 at psi = 100, N = 500):
  1. T^3 phase gauge invariance          < 1e-8   [bug regression]
  2. T^3 phase circular std              < 1e-6   [bug regression; true ~5e-9]
  3. T^3 omega median in [7e-4, 2.7e-3]           [psi^-2 law; true ~1.34e-3]
  4. T^3 omega median gauge drift        < 10%    [observed ~2.6%]
  5. RP^3 omega defect max               < 1e-12  [measured exactly 0.0]
  6. RP^3 omega per-point gauge drift    < 1e-12  [measured ~5e-15]
  7. RP^3 phase spread max               < 1e-12  [measured exactly 0.0]
  8. T^3 mod-2pi phase gauge invariance  < 1e-8   [viz.fitness_pipeline path]

Usage:
    python -m diagnostics.test_gauge_invariance
"""

import argparse
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from find_smooth_submanifold import get_basis_labels, normalize_coeffs
from slag_condition import compute_combined_fitness

PSI = 100.0  # tolerances above are calibrated at this psi


def _coeffs(rows) -> jnp.ndarray:
    """(3, 25) coefficient matrix from {label: value} dicts, one per equation."""
    labels = get_basis_labels()
    c = np.zeros((3, 25))
    for r, row in enumerate(rows):
        for lab, val in row.items():
            c[r, labels.index(lab)] = val
    return normalize_coeffs(jnp.asarray(c))


def t3_coefficients() -> jnp.ndarray:
    """CDGP diagonal ansatz f_k = |Z0|^2 - |Zk|^2, k = 1, 2, 3."""
    return _coeffs([{"Re(z0*z0bar)": 1.0, f"Re(z{k}*z{k}bar)": -1.0}
                    for k in (1, 2, 3)])


def rp3_coefficients() -> jnp.ndarray:
    """Involution ansatz Im(z0 zkbar) = 0, k = 1, 2, 3 (real locus component)."""
    return _coeffs([{f"Im(z0*z{k}bar)": 1.0} for k in (1, 2, 3)])


def sample_t3_small_branch(n: int, seed: int) -> np.ndarray:
    """(n, 5) points on the small-|Z4| (SYZ-fiber) branch of the T^3 locus."""
    rng = np.random.default_rng(seed)
    Z = np.ones((n, 5), dtype=np.complex128)
    Z[:, 1:4] = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=(n, 3)))
    for m in range(n):
        roots = np.roots([1.0, 0.0, 0.0, 0.0, PSI * np.prod(Z[m, :4]),
                          np.sum(Z[m, :4] ** 5)])
        Z[m, 4] = roots[np.argmin(np.abs(roots))]
    return Z


def sample_rp3(n: int, seed: int) -> np.ndarray:
    """(n, 5) real points on the quintic (the RP^3 = Fix(z -> zbar), real psi)."""
    rng = np.random.default_rng(seed)
    X = np.ones((n, 5))
    X[:, :4] = rng.uniform(-1.0, 1.0, size=(n, 4))
    for m in range(n):
        roots = np.roots([1.0, 0.0, 0.0, 0.0, PSI * np.prod(X[m, :4]),
                          np.sum(X[m, :4] ** 5)])
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        X[m, 4] = real_roots[np.argmin(np.abs(real_roots))]
    return X.astype(np.complex128)


def run_pipeline(Z: np.ndarray, coeffs: jnp.ndarray):
    """(omega_defect, phases) from the production fitness path."""
    p10 = jnp.concatenate([jnp.asarray(Z.real), jnp.asarray(Z.imag)], axis=1)
    out = compute_combined_fitness(p10, coeffs, jnp.complex128(PSI),
                                   metric="FS", debug_mode=True)
    return np.asarray(jnp.linalg.norm(out[3], axis=(1, 2))), np.asarray(out[5])


def circular_spread(phases: np.ndarray) -> tuple[float, float]:
    """(std, max) deviation on the mod-pi circle."""
    center = 0.5 * np.angle(np.mean(np.exp(2j * phases)))
    dev = 0.5 * np.angle(np.exp(2j * (phases - center)))
    return float(np.std(dev)), float(np.max(np.abs(dev)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="global C* gauge phase applied to the second cloud")
    args = parser.parse_args()
    gauge = np.exp(1j * args.alpha)

    checks = []  # (description, value string, ok)

    # --- T^3: equal-moduli locus, the tie-flip stress test ---
    Z = sample_t3_small_branch(args.n, args.seed)
    om_a, ph_a = run_pipeline(Z, t3_coefficients())
    om_b, ph_b = run_pipeline(Z * gauge, t3_coefficients())

    gauge_diff = np.max(np.abs(0.5 * np.angle(np.exp(2j * (ph_a - ph_b)))))
    ph_std, _ = circular_spread(ph_a)
    om_med = float(np.median(om_a))
    om_drift = abs(om_med - float(np.median(om_b))) / om_med

    checks.append(("T^3 phase gauge invariance (rad)",
                   f"{gauge_diff:.3e} < 1e-8", gauge_diff < 1e-8))
    checks.append(("T^3 phase circular std (rad)",
                   f"{ph_std:.3e} < 1e-6", ph_std < 1e-6))
    checks.append(("T^3 omega median (psi^-2 law)",
                   f"{om_med:.3e} in [7e-4, 2.7e-3]", 7e-4 < om_med < 2.7e-3))
    checks.append(("T^3 omega median gauge drift",
                   f"{om_drift:.2%} < 10%", om_drift < 0.10))

    # --- RP^3: exact sLag control, distinct moduli (no ties) ---
    Zr = sample_rp3(args.n, args.seed)
    om_r, ph_r = run_pipeline(Zr, rp3_coefficients())
    om_r2, _ = run_pipeline(Zr * gauge, rp3_coefficients())

    om_max = float(np.max(om_r))
    om_gauge = float(np.max(np.abs(om_r - om_r2)))
    _, ph_max = circular_spread(ph_r)

    checks.append(("RP^3 omega defect max (exact sLag)",
                   f"{om_max:.3e} < 1e-12", om_max < 1e-12))
    checks.append(("RP^3 omega per-point gauge drift",
                   f"{om_gauge:.3e} < 1e-12", om_gauge < 1e-12))
    checks.append(("RP^3 phase spread max (rad)",
                   f"{ph_max:.3e} < 1e-12", ph_max < 1e-12))

    # --- viz path: the mod-2pi diagnostic phases (histogram plots) go through
    # viz.fitness_pipeline._per_chunk_diagnostics, which pairs the point frame
    # with the patch index the same way -- must be gauge invariant as well.
    from viz.fitness_pipeline import _per_chunk_diagnostics

    def viz_phases(Zc):
        p10 = jnp.concatenate([jnp.asarray(Zc.real), jnp.asarray(Zc.imag)],
                              axis=1)
        _, _, ph = _per_chunk_diagnostics(p10, t3_coefficients(),
                                          jnp.complex128(PSI), "FS")
        return np.asarray(ph)

    viz_diff = float(np.max(np.abs(np.angle(
        np.exp(1j * (viz_phases(Z) - viz_phases(Z * gauge)))))))
    checks.append(("T^3 mod-2pi phase gauge inv. (viz)",
                   f"{viz_diff:.3e} < 1e-8", viz_diff < 1e-8))

    ok_all = True
    for name, val, ok in checks:
        ok_all &= ok
        print(f"{name:38s}: {val:28s} {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
