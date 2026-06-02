"""Apply S_5 coordinate relabelings to a coeffs pkl, producing the 10 versions
whose holomorphic Z_2 symmetry sits at each transposition (i j) instead of
(2 3). Used to verify that:

  (a) the (2 3) symmetry is a property of the sLag up to S_5 relabeling
      -- every permuted version should give identical fitness plots;
  (b) the d=2 (0, 4)-tile prominence is a labeling artifact -- it should
      move to (tau(0), tau(4)) under the permutation.

The permutation acts on the Hermitian matrix per degree as H -> U_tau H U_tau^T
(holomorphic, z_i -> z_{tau(i)}), where U_tau is the induced permutation matrix
on the d-monomial basis. The result is repackaged back into a (3, w) coeff
row, saved as {"coeffs": <array>} so it works directly with both
plot_hermitian_coeffs.py and gradient_descent.py --plots_only.

Usage:
    python -m diagnostics.permute_coeffs --coeffs gd_runs/<job>.pkl \
        [--out_dir permuted_coeffs]

Then see the printed next-steps block for the three follow-up commands.
"""
import argparse
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np

from hermitian_coeffs import (
    _BLOCK, _SYM_DIM, coeffs_row_to_hermitian, _load_coeffs,
)
from diagnostics.test_permutation_symmetry import monomial_permutation


def hermitian_to_coeffs_row(H: np.ndarray, N: int) -> np.ndarray:
    """Inverse of coeffs_row_to_hermitian. Pack a Hermitian N x N matrix back
    into the real coeff row layout used by helper.generate_basis_*.
    """
    n_imag = N * (N - 1) // 2
    n_real = N * (N + 1) // 2
    iu0_r, iu0_c = np.triu_indices(N, k=0)
    iu1_r, iu1_c = np.triu_indices(N, k=1)
    on_diag = iu0_r == iu0_c

    # Convention 1: H_{AB} = (b_AB - 1j * a_AB) / 2  for A < B, H_{AA} = b_AA.
    imag_coeffs = -2.0 * H[iu1_r, iu1_c].imag
    real_coeffs = np.where(
        on_diag, H[iu0_r, iu0_c].real, 2.0 * H[iu0_r, iu0_c].real,
    )
    out = np.concatenate([imag_coeffs, real_coeffs])
    assert out.shape == (n_imag + n_real,)
    return out


def apply_permutation_to_coeffs(coeffs: np.ndarray,
                                perm: tuple[int, ...]) -> np.ndarray:
    """Apply a coordinate permutation tau to a (3, w) coeff array.

    Each row encodes a polynomial; the new polynomial is f(tau^-1 z), realized
    on Hermitian matrices as H -> U_tau H U_tau^T (per degree).
    """
    new = np.zeros_like(coeffs)
    w = coeffs.shape[1]
    for d, (lo, hi) in _BLOCK.items():
        if w < hi:
            break
        N = _SYM_DIM[d]
        U = monomial_permutation(perm, d)
        for k in range(coeffs.shape[0]):
            row = np.asarray(coeffs[k, lo:hi])
            H = coeffs_row_to_hermitian(row, N)
            H_new = U @ H @ U.T
            new[k, lo:hi] = hermitian_to_coeffs_row(H_new, N)
    return new


def tau_for_target(i: int, j: int) -> tuple[int, ...]:
    """Build a permutation tau such that tau(2) = i, tau(3) = j (and tau is
    the simplest extension to {0,1,2,3,4}). Used so that the post-permutation
    symmetry transposition is exactly (i j).
    """
    perm = [0, 1, 2, 3, 4]
    # Place i in position 2 by swapping wherever i currently is.
    pos_i = perm.index(i)
    perm[2], perm[pos_i] = perm[pos_i], perm[2]
    # Place j in position 3 (j may have moved during the first swap).
    pos_j = perm.index(j)
    perm[3], perm[pos_j] = perm[pos_j], perm[3]
    return tuple(perm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coeffs", required=True, type=Path,
                        help="Source coeffs pkl (bare (3,w) array or dict "
                             "with 'coeffs' key).")
    parser.add_argument("--out_dir", type=Path, default=Path("permuted_coeffs"),
                        help="Directory to write 10 output pkls into.")
    args = parser.parse_args()

    coeffs = _load_coeffs(args.coeffs)
    print(f"Loaded coeffs {coeffs.shape} from {args.coeffs}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.coeffs.stem
    if stem.endswith(".pkl"):
        stem = stem[:-4]

    print(f"\nWriting 10 permuted coeffs into {args.out_dir}/:")
    print(f"  {'target sym':<11} | {'tau':<22} | output")
    print(f"  {'-'*11} | {'-'*22} | {'-'*40}")
    for (i, j) in combinations(range(5), 2):
        perm = tau_for_target(i, j)
        new_coeffs = apply_permutation_to_coeffs(coeffs, perm)
        out_path = args.out_dir / f"{stem}_perm{i}{j}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({"coeffs": np.asarray(new_coeffs)}, f)
        sym = f"({i} {j})"
        print(f"  {sym:<11} | {str(perm):<22} | {out_path.name}")

    print(f"\nNext steps -- run these on the cluster (need points + JAX):")
    print()
    print(f"  # (1) Fitness plots (Lagrangian + special condition). All 10")
    print(f"  #     should be IDENTICAL up to numerical noise if (2 3) is a")
    print(f"  #     true sLag symmetry up to S_5 relabeling.")
    print(f"  for f in {args.out_dir}/*.pkl; do")
    print(f"      job=$(basename $f .pkl)")
    print(f"      python gradient_descent.py --job_id $job --plots_only \\")
    print(f"          --resume $f --max_degree 3")
    print(f"  done")
    print()
    print(f"  # (2) Hermitian heatmaps + spectra. The d=2 (0,4)-tile should")
    print(f"  #     move to (tau(0), tau(4)) in each version.")
    print(f"  for f in {args.out_dir}/*.pkl; do")
    print(f"      python -m viz.plot_hermitian_coeffs --coeffs $f")
    print(f"  done")
    print()
    print(f"  # (3) Symmetry test -- the holomorphic symmetry transposition")
    print(f"  #     should shift to match each filename suffix.")
    print(f"  for f in {args.out_dir}/*.pkl; do")
    print(f"      python -m diagnostics.test_permutation_symmetry --coeffs $f \\")
    print(f"          --group transpositions --mode holo")
    print(f"  done")


if __name__ == "__main__":
    main()
