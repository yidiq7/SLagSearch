"""Diagnose Omega phase concentration per patch.

Computes phases of Omega restricted to L (chosen ansatz) using the production
code path (with the (-1)^max_idx Poincaré-residue sign in
compute_holomorphic_form -- the (-1)^patch_idx factor is intentionally dropped,
see the comment there -- plus the canonical-basis sign correction in
compute_Omega_restriction) and prints per-patch histograms.

Usage:
    python -m diagnostics.diagnose_phases --ansatz {d1,rp3} --n_bins 30
"""

import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from find_smooth_submanifold import filter_and_refine, normalize_coeffs
from get_restriction import (
    compute_Omega_restriction,
    compute_affine_jacobian,
    compute_restriction,
)
from gradient_descent import GENOTYPE_SHAPE, _load_d1_baseline_coeffs, load_points
from helper import convert_real_to_complex_batch, determine_patches_batch
from slag_condition import compute_holomorphic_form

jax.config.update("jax_enable_x64", True)


def compute_diagnostic_phases(min_set_real, coeffs, psi):
    """Return (phases, patch_indices, max_idx) using the production code path."""
    min_set = convert_real_to_complex_batch(min_set_real)
    patch_indices = determine_patches_batch(min_set)

    jacobians = jax.vmap(compute_affine_jacobian, in_axes=(0, 0, None, None))(
        min_set_real, patch_indices, coeffs, psi
    )
    restrictions = jax.vmap(compute_restriction)(jacobians)

    Omega, max_idx, Omega_coord = compute_holomorphic_form(
        min_set, patch_indices, psi
    )
    Omega_restriction = compute_Omega_restriction(restrictions, Omega_coord)
    phases = jnp.angle(Omega * Omega_restriction) % (2 * jnp.pi)
    return phases, patch_indices, max_idx


def print_per_patch_histogram(phases, patch_indices, label, n_bins=12):
    print(f"\n--- {label} ---")
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    counts_per_patch = []
    for p in range(5):
        mask = patch_indices == p
        ph = phases[mask]
        counts, _ = np.histogram(ph, bins=bin_edges)
        counts_per_patch.append(counts)

    header = f"  {'bin (rad)':<12}"
    for p in range(5):
        header += f"  patch{p:>1}"
    print(header)

    for b in range(n_bins):
        line = f"  {bin_centers[b]:>10.3f}"
        for p in range(5):
            line += f"  {counts_per_patch[p][b]:>6d}"
        print(line)

    print(f"  {'count':<12}", end="")
    for p in range(5):
        print(f"  {int(counts_per_patch[p].sum()):>6d}", end="")
    print()

    print(f"  {'peak (rad)':<12}", end="")
    for p in range(5):
        if counts_per_patch[p].sum() > 0:
            peak_bin = int(np.argmax(counts_per_patch[p]))
            print(f"  {bin_centers[peak_bin]:>6.2f}", end="")
        else:
            print(f"  {'--':>6}", end="")
    print()


def main():
    parser = argparse.ArgumentParser(description="Diagnose Omega phase consistency across patches.")
    parser.add_argument("--psi", type=complex, default=0)
    parser.add_argument("--minset_size", type=int, default=10000)
    parser.add_argument("--newton_steps", type=int, default=40)
    parser.add_argument(
        "--ansatz", type=str, default="d1", choices=["d1", "rp3"],
        help="d1: GA d=1 baseline coeffs. "
             "rp3: 3 polynomials Im(z_0 z̄_1)=Im(z_0 z̄_2)=Im(z_0 z̄_3)=0, "
             "which (with f=0) selects RP^3-like components; expected to give 5 phase peaks "
             "at multiples of 2*pi/5 if Omega's sign convention is correct.",
    )
    parser.add_argument("--n_bins", type=int, default=30, help="Histogram bins on [0, 2*pi).")
    parser.add_argument("--out_pkl", type=str, default="phase_diagnose.pkl")
    args = parser.parse_args()

    print(f"=== Phase consistency diagnostic (ansatz={args.ansatz}) ===")
    points_real, src = load_points(args.psi)
    print(f"Loaded {len(points_real)} points from {src}")

    coeffs = jnp.zeros(GENOTYPE_SHAPE)
    if args.ansatz == "d1":
        coeffs = coeffs.at[:, :25].set(_load_d1_baseline_coeffs())
    elif args.ansatz == "rp3":
        # Im(z_0 z̄_1)=0, Im(z_0 z̄_2)=0, Im(z_0 z̄_3)=0 (basis indices 0, 1, 2).
        coeffs = coeffs.at[0, 0].set(1.0)
        coeffs = coeffs.at[1, 1].set(1.0)
        coeffs = coeffs.at[2, 2].set(1.0)
    coeffs = normalize_coeffs(coeffs).astype(jnp.float64)
    psi = jnp.asarray(args.psi, dtype=jnp.complex128)

    print("Mining (filter_and_refine)...")
    min_set_real, distances, _ = filter_and_refine(
        points_real, coeffs, psi, args.minset_size, args.newton_steps, filter_newton=True,
    )
    mean_d = float(jnp.mean(distances))
    max_d = float(jnp.max(distances))
    print(f"  mean_dist {mean_d:.2e}  max_dist {max_d:.2e}")

    print("Computing phases (production code path)...")
    phases, patch_indices, max_idx = compute_diagnostic_phases(
        min_set_real, coeffs, psi
    )

    phases_np = np.asarray(phases)
    patch_np = np.asarray(patch_indices)
    max_idx_np = np.asarray(max_idx)

    unique, counts = np.unique(patch_np, return_counts=True)
    patch_dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"\nPatch distribution (patch_idx -> count): {patch_dist}")

    print_per_patch_histogram(phases_np, patch_np, "production (current code)", n_bins=args.n_bins)

    if args.ansatz == "rp3":
        print("\nFor RP^3 (with current production sign convention Omega = (-1)^(I+d)/dfdz_d * ...):")
        print("  Expected 5 peaks at theta = (2k+1)*pi/5 for k=0..4")
        print(f"  i.e. odd multiples of pi/5: {np.pi/5:.3f}, {3*np.pi/5:.3f}, {np.pi:.3f}, {7*np.pi/5:.3f}, {9*np.pi/5:.3f} rad")
        print("  If the histogram collapses to these 5 (not 10), the basis-orientation correction works.")

    with open(args.out_pkl, "wb") as f:
        pickle.dump({
            "phases": phases_np,
            "patch_indices": patch_np,
            "max_idx": max_idx_np,
        }, f)
    print(f"\nSaved raw arrays to {args.out_pkl}")


if __name__ == "__main__":
    main()
