"""Candelas T^3 at the large-complex-structure limit, through the sLag pipeline.

Evaluates the CDGP diagonal ansatz f_k = |Z^0|^2 - |Z^k|^2 (k = 1, 2, 3) on
the Dwork pencil sum(Z^5) + psi*prod(Z) = 0 over a sweep of psi, using the
production fitness path (FS metric, compute_combined_fitness). Points are
seeded analytically on the locus -- Z^0..Z^3 = random unit phases, Z^4 = a
root of the dehomogenised quintic in Z^4 -- and then (optionally) polished
with the production Newton refiner. This sidesteps two large-psi failure
modes of the generic cloud pipeline: random FS-uniform seeds starve near the
small torus (its measure fraction scales like ~psi^-2), and the refined cloud
mixes components.

At large |psi| the locus {f_1 = f_2 = f_3 = 0} on X_psi is disconnected, and
the two parts are sampled separately via --branch:
  - 'small': |Z^4| ~ 4/|psi| -- the SYZ-fiber (Candelas) torus; expected
    FS-Lagrangian defect ~psi^-2 and Omega|_L phase spread ~psi^-5 around
    pi/2 (for real psi > 0).
  - 'large': |Z^4| ~ |psi|^(1/4) -- the four remaining Z^4-sheets, joined
    into a single torus by root monodromy; always lands in patch 4, which is
    how mixed clouds can be separated.

Interpretation notes. Nonzero defects at finite psi are geometry, not
numerical error: the algebraic locus is only asymptotically Lagrangian, and
only for the Fubini-Study form -- the phase (calibration) statistics are the
metric-independent part. Sampling is uniform in the phase parameters, not in
the induced volume, so quoted statistics are under that measure. The entropy
fitness F_spec bins with a half-bin-shifted anchor (bin centers at
k*pi/n_bins), so tight peaks at the quintic's natural phases -- including
the pi/2 here -- score ~1 instead of capping at 1 - log(2)/log(n_bins) on a
bin edge; the Kuramoto column is the binless concentration measure and is
what resolves the psi^-5 law. float64 is mandatory (enabled below):
in float32 the phase spread floors at ~2e-7 rad and the psi^-5 law is
unmeasurable beyond psi ~ 30.

Usage (from the repo root):
    python -m diagnostics.t3_lcs_experiment
    python -m diagnostics.t3_lcs_experiment --psi 20 50 100 200 500 1000 \
        --branch both --n 2000 --out t3_lcs_results.csv \
        --save_points t3_lcs_points.pkl
"""

import argparse
import csv
import pickle

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from find_smooth_submanifold import (get_basis_labels, normalize_coeffs,
                                     refine_point_iterative)
from helper import determine_patch_and_rescale_single, evaluate_equations_single_point
from slag_condition import (compute_combined_fitness,
                            compute_special_condition_fitness_smooth)


def t3_coefficients() -> jnp.ndarray:
    """(3, 25) coefficients of the CDGP diagonal ansatz f_k = |Z0|^2 - |Zk|^2."""
    labels = get_basis_labels()
    c = np.zeros((3, 25))
    for k in (1, 2, 3):
        c[k - 1, labels.index("Re(z0*z0bar)")] = 1.0
        c[k - 1, labels.index(f"Re(z{k}*z{k}bar)")] = -1.0
    return normalize_coeffs(jnp.asarray(c))


def sample_t3_locus(psi: complex, n: int, branch: str, seed: int) -> np.ndarray:
    """(n, 5) complex points on one branch of the T^3 locus at psi.

    Z^0 = 1 (C* gauge), Z^1..Z^3 = e^{i theta} with theta uniform, and Z^4 a
    root of Z4^5 + (psi * Z0 Z1 Z2 Z3) * Z4 + sum_{i<=3} Zi^5 = 0. 'small'
    takes the smallest-modulus root (unique for |psi| >~ 6); 'large' draws
    uniformly among the other four, sampling all sheets of the cover.
    """
    if branch not in ("small", "large"):
        raise ValueError(f"branch must be 'small' or 'large', got {branch!r}")
    rng = np.random.default_rng(seed)
    Z = np.ones((n, 5), dtype=np.complex128)
    Z[:, 1:4] = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=(n, 3)))
    for m in range(n):
        roots = np.roots([1.0, 0.0, 0.0, 0.0, psi * np.prod(Z[m, :4]),
                          np.sum(Z[m, :4] ** 5)])
        order = np.argsort(np.abs(roots))
        idx = order[0] if branch == "small" else order[rng.integers(1, 5)]
        Z[m, 4] = roots[idx]
    return Z


def circular_stats(phases: np.ndarray) -> tuple[float, float, float]:
    """(center, std, max deviation) of phases on the mod-pi circle."""
    center = 0.5 * np.angle(np.mean(np.exp(2j * phases)))
    dev = 0.5 * np.angle(np.exp(2j * (phases - center)))
    return float(center % np.pi), float(np.std(dev)), float(np.max(np.abs(dev)))


def evaluate_branch(Z: np.ndarray, coeffs: jnp.ndarray, psi: complex,
                    refine_steps: int) -> dict:
    """Run one point cloud through the production fitness path; return stats."""
    psi_j = jnp.complex128(psi)
    Zr = jax.vmap(lambda z: determine_patch_and_rescale_single(z)[0])(
        jnp.asarray(Z))
    p10 = jnp.concatenate([Zr.real, Zr.imag], axis=1)

    if refine_steps > 0:
        p10 = jax.vmap(
            lambda p: refine_point_iterative(p, coeffs, psi_j, refine_steps))(p10)

    resid = jax.vmap(lambda p: jnp.linalg.norm(
        evaluate_equations_single_point(p, coeffs, psi_j)))(p10)

    (_, f_lag, f_spec, kfr_norm, _, phases) = compute_combined_fitness(
        p10, coeffs, psi_j, metric="FS", debug_mode=True)

    omega_defect = np.asarray(jnp.linalg.norm(kfr_norm, axis=(1, 2)))
    phases_np = np.asarray(phases)
    center, ph_std, ph_max = circular_stats(phases_np)
    patch = np.asarray(jnp.argmax(jnp.abs(p10[:, :5] + 1j * p10[:, 5:]), axis=1))

    return {
        "resid_max": float(jnp.max(resid)),
        "omega_median": float(np.median(omega_defect)),
        "omega_p99": float(np.percentile(omega_defect, 99)),
        "F_lag": float(f_lag),
        "phase_center": center,
        "phase_std": ph_std,
        "phase_max": ph_max,
        "kuramoto": float(compute_special_condition_fitness_smooth(phases)),
        "F_spec": float(f_spec),
        "frac_patch4": float(np.mean(patch == 4)),
        "points_10d": np.asarray(p10),
        "phases": phases_np,
        "omega_defect": omega_defect,
    }


def fmt_psi(psi: complex) -> str:
    return f"{psi.real:g}" if psi.imag == 0 else f"{psi.real:g}{psi.imag:+g}j"


COLUMNS = ["branch", "psi", "n", "resid_max", "omega_median", "omega_p99",
           "F_lag", "phase_center", "phase_std", "phase_max", "kuramoto",
           "F_spec", "frac_patch4"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--psi", type=complex, nargs="+",
                        default=[20, 50, 100, 200, 500, 1000],
                        help="Dwork psi values to sweep (smooth: psi != -5*zeta_5).")
    parser.add_argument("--branch", choices=["small", "large", "both"],
                        default="both")
    parser.add_argument("--n", type=int, default=2000, help="points per cloud")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refine_steps", type=int, default=5,
                        help="production Newton polish steps (0 to skip)")
    parser.add_argument("--out", type=str, default=None, help="CSV output path")
    parser.add_argument("--save_points", type=str, default=None,
                        help="pickle per-point data (points/phases/defects) here")
    args = parser.parse_args()

    coeffs = t3_coefficients()
    branches = ["small", "large"] if args.branch == "both" else [args.branch]

    header = (f"{'branch':7s} {'psi':>9s} {'resid_max':>10s} {'omega_med':>10s} "
              f"{'omega_p99':>10s} {'ph_center':>9s} {'ph_std':>9s} "
              f"{'ph_max':>9s} {'kuramoto':>9s} {'F_lag':>7s} {'F_spec':>7s} "
              f"{'patch4':>6s}")
    print(header)

    rows, per_point = [], {}
    for branch in branches:
        for psi in args.psi:
            Z = sample_t3_locus(psi, args.n, branch, args.seed)
            s = evaluate_branch(Z, coeffs, psi, args.refine_steps)
            print(f"{branch:7s} {fmt_psi(psi):>9s} {s['resid_max']:>10.2e} "
                  f"{s['omega_median']:>10.3e} {s['omega_p99']:>10.3e} "
                  f"{s['phase_center']:>9.5f} {s['phase_std']:>9.2e} "
                  f"{s['phase_max']:>9.2e} {s['kuramoto']:>9.6f} "
                  f"{s['F_lag']:>7.4f} {s['F_spec']:>7.4f} "
                  f"{s['frac_patch4']:>6.2f}")
            rows.append([branch, fmt_psi(psi), args.n] +
                        [s[c] for c in COLUMNS[3:]])
            if args.save_points:
                per_point[(branch, fmt_psi(psi))] = {
                    k: s[k] for k in ("points_10d", "phases", "omega_defect")}

        # Log-log slope fits: the asymptotic laws for the small branch are
        # omega ~ psi^-2 and phase spread ~ psi^-5.
        b_rows = [r for r in rows if r[0] == branch]
        psis = np.array([abs(complex(p)) for p in args.psi])
        if len(set(psis)) >= 2 and len(b_rows) == len(psis):
            om = np.array([r[COLUMNS.index("omega_median")] for r in b_rows])
            ps = np.array([r[COLUMNS.index("phase_std")] for r in b_rows])
            slope_om = np.polyfit(np.log10(psis), np.log10(om), 1)[0]
            slope_ps = np.polyfit(np.log10(psis), np.log10(ps), 1)[0]
            print(f"  [{branch}] fitted power laws: omega_med ~ psi^{slope_om:+.2f}, "
                  f"phase_std ~ psi^{slope_ps:+.2f}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            w.writerows(rows)
        print(f"wrote {args.out}")

    if args.save_points:
        with open(args.save_points, "wb") as f:
            pickle.dump(per_point, f)
        print(f"wrote {args.save_points}")


if __name__ == "__main__":
    main()
