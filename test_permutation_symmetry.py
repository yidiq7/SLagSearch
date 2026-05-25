"""Test approximate permutation symmetries of the per-degree Hermitian
matrices, allowing an O(3) twist that mixes the three equations.

For a permutation g in S_5 acting on coordinates we induce a permutation U_g on
each d-monomial basis (size N x N with N in {5,15,35,70}). The symmetry test:
the polynomial system {f_0, f_1, f_2} is g-invariant up to an O(3) rotation iff
there exists O in O(3) with
    U_g H^(k) U_g^T  =  sum_l O_{kl} H^(l)   for each k and each degree d.

We find the best O (Procrustes-style, in closed form via SVD of a 3x3 matrix)
and report the relative Frobenius residual

    rel_res(g) = || U_g H U_g^T - O.H ||_F / || H ||_F

both per-degree (lower bound: each degree gets its own best O) and joint
(physical: single O across all degrees). For reference we also print the
"no-twist" residual ||U_g H U_g^T - H||_F / ||H||_F, i.e. the rigid symmetry
test without allowing equation mixing.

Usage:
    python test_permutation_symmetry.py --coeffs gd_runs/<job>.pkl [--group z2xs3]
"""
import argparse
import pickle
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from plot_hermitian_coeffs import (
    _SYM_DIM, extract_hermitians, _load_coeffs,
)


def _load_hermitians(path: Path) -> dict[int, list[np.ndarray]]:
    """Accept either a coeffs pkl (going through extract_hermitians) or the
    hermitian_matrices_<job>.npz file written by plot_hermitian_coeffs.py.
    """
    if path.suffix == ".npz":
        npz = np.load(path)
        by_d: dict[int, list[tuple[int, np.ndarray]]] = {}
        for key in npz.files:
            # Keys look like "d3_eq0"; pull out the two ints.
            d = int(key.split("_")[0][1:])
            eq = int(key.split("_")[1][2:])
            by_d.setdefault(d, []).append((eq, np.asarray(npz[key])))
        return {d: [H for _, H in sorted(eqs)] for d, eqs in by_d.items()}
    coeffs = _load_coeffs(path)
    return extract_hermitians(coeffs)


def monomial_permutation(perm: tuple[int, ...], d: int) -> np.ndarray:
    """Permutation matrix U on the d-monomial basis induced by `perm` acting
    on the 5 coordinates. `perm[i] = g(i)`.

    Returns a real (N, N) permutation matrix P with (P v)_A' = v_{g^{-1}(A')},
    i.e. column k = e_{pi_g(k)} where pi_g sends the kth monomial multi-index
    to its image under index-relabelling.
    """
    monomials = list(combinations_with_replacement(range(5), d))
    N = len(monomials)
    assert N == _SYM_DIM[d]
    index = {m: k for k, m in enumerate(monomials)}
    P = np.zeros((N, N), dtype=np.float64)
    for k, m in enumerate(monomials):
        m_img = tuple(sorted(perm[i] for i in m))
        P[index[m_img], k] = 1.0
    return P


def best_O3_for_block(
    H_triple: list[np.ndarray], tilde_triple: list[np.ndarray]
) -> tuple[np.ndarray, float, float]:
    """Closed-form best O in O(3) (real) minimizing
        sum_k ||tilde_H^(k) - sum_l O_{kl} H^(l)||_F^2
    over a single degree block. Returns (O, abs_res, rel_res).
    """
    V = np.stack([H.ravel() for H in H_triple], axis=1)         # (N^2, 3)
    tV = np.stack([H.ravel() for H in tilde_triple], axis=1)    # (N^2, 3)
    # tilde_h_k = sum_l O_{kl} h_l  <=>  tV = V O^T.
    # min ||tV - V O^T||_F^2 = const - 2 Re tr(O^T V^H tV)  (since O in O(3),
    # ||V O^T||_F = ||V||_F). Procrustes on C = Re(V^H tV).
    C = np.real(V.conj().T @ tV)                                # (3, 3)
    Uc, _, Vct = np.linalg.svd(C)
    O = Uc @ Vct                                                # in O(3)
    diff = tV - V @ O.T
    abs_res = float(np.linalg.norm(diff))
    rel_res = float(abs_res / max(np.linalg.norm(V), 1e-30))
    return O, abs_res, rel_res


def joint_O3_across_degrees(
    H_by_d: dict[int, list[np.ndarray]],
    tilde_by_d: dict[int, list[np.ndarray]],
) -> tuple[np.ndarray, dict[int, float], float]:
    """Single O in O(3) minimizing the total residual across all degrees,
    weighted naturally by Frobenius mass. Returns (O, per_degree_rel_res,
    overall_rel_res).
    """
    # Sum the 3x3 Procrustes matrices across degrees.
    C_total = np.zeros((3, 3))
    V_total_sq = 0.0
    for d, Hs in H_by_d.items():
        V = np.stack([H.ravel() for H in Hs], axis=1)
        tV = np.stack([H.ravel() for H in tilde_by_d[d]], axis=1)
        C_total += np.real(V.conj().T @ tV)
        V_total_sq += float(np.linalg.norm(V) ** 2)

    Uc, _, Vct = np.linalg.svd(C_total)
    O = Uc @ Vct
    per_d = {}
    abs_total_sq = 0.0
    for d, Hs in H_by_d.items():
        V = np.stack([H.ravel() for H in Hs], axis=1)
        tV = np.stack([H.ravel() for H in tilde_by_d[d]], axis=1)
        diff = tV - V @ O.T
        abs_sq = float(np.linalg.norm(diff) ** 2)
        abs_total_sq += abs_sq
        per_d[d] = float(np.sqrt(abs_sq) / max(np.linalg.norm(V), 1e-30))
    overall = float(np.sqrt(abs_total_sq) / max(np.sqrt(V_total_sq), 1e-30))
    return O, per_d, overall


def no_twist_residual(
    H_triple: list[np.ndarray], tilde_triple: list[np.ndarray]
) -> float:
    """||U H U^T - H||_F / ||H||_F, summed over the 3 equations."""
    num_sq = sum(float(np.linalg.norm(t - H) ** 2)
                 for t, H in zip(tilde_triple, H_triple))
    den_sq = sum(float(np.linalg.norm(H) ** 2) for H in H_triple)
    return float(np.sqrt(num_sq) / max(np.sqrt(den_sq), 1e-30))


# Pre-canned group element lists. Each entry is (perm_tuple, label).
_GROUPS: dict[str, list[tuple[tuple[int, ...], str]]] = {
    # Candidate stabilizer of the partition {0,4} | {1,2,3} (order 12).
    "z2xs3": [
        ((0, 1, 2, 3, 4), "identity"),
        ((4, 1, 2, 3, 0), "(0 4)"),
        ((0, 2, 1, 3, 4), "(1 2)"),
        ((0, 1, 3, 2, 4), "(2 3)"),
        ((0, 3, 2, 1, 4), "(1 3)"),
        ((0, 2, 3, 1, 4), "(1 2 3)"),
        ((0, 3, 1, 2, 4), "(1 3 2)"),
        ((4, 2, 1, 3, 0), "(0 4)(1 2)"),
        ((4, 1, 3, 2, 0), "(0 4)(2 3)"),
        ((4, 3, 2, 1, 0), "(0 4)(1 3)"),
        ((4, 2, 3, 1, 0), "(0 4)(1 2 3)"),
        ((4, 3, 1, 2, 0), "(0 4)(1 3 2)"),
    ],
    # All 10 transpositions of S_5 (for diagnostic comparison).
    "transpositions": [
        ((1, 0, 2, 3, 4), "(0 1)"),
        ((2, 1, 0, 3, 4), "(0 2)"),
        ((3, 1, 2, 0, 4), "(0 3)"),
        ((4, 1, 2, 3, 0), "(0 4)"),
        ((0, 2, 1, 3, 4), "(1 2)"),
        ((0, 3, 2, 1, 4), "(1 3)"),
        ((0, 4, 2, 3, 1), "(1 4)"),
        ((0, 1, 3, 2, 4), "(2 3)"),
        ((0, 1, 4, 3, 2), "(2 4)"),
        ((0, 1, 2, 4, 3), "(3 4)"),
    ],
    # A few elements outside Z_2 x S_3, to confirm they are NOT symmetries.
    "outside_z2xs3": [
        ((1, 0, 2, 3, 4), "(0 1)  [moves 0 out of {0,4}]"),
        ((0, 4, 2, 3, 1), "(1 4)  [moves 4 out of {0,4}]"),
        ((1, 2, 3, 4, 0), "(0 1 2 3 4)  [5-cycle]"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coeffs", required=True, type=Path,
                        help="Path to a coeff pkl (bare (3,w) array or dict "
                             "with 'coeffs' key), OR a hermitian_matrices_*.npz "
                             "written by plot_hermitian_coeffs.py.")
    parser.add_argument("--group", default="z2xs3",
                        choices=list(_GROUPS.keys()),
                        help="Which set of elements to test (default: z2xs3).")
    args = parser.parse_args()

    H_by_d = _load_hermitians(args.coeffs)
    degrees = sorted(H_by_d)
    norms = {d: [float(np.linalg.norm(H)) for H in H_by_d[d]] for d in degrees}
    print(f"Loaded Hermitian matrices from {args.coeffs}")
    print(f"Degrees present: {degrees}")
    for d in degrees:
        print(f"  d={d}: ||H||_F per eq = "
              + ", ".join(f"{n:.4g}" for n in norms[d]))
    print()

    elements = _GROUPS[args.group]

    # Precompute the permutation matrices.
    U_cache: dict[tuple[str, int], np.ndarray] = {
        (name, d): monomial_permutation(perm, d)
        for perm, name in elements for d in degrees
    }

    # --- Per-degree best O(3): each degree gets its own optimal rotation.
    print(f"=== Per-degree best O(3) residual  (group: {args.group}) ===")
    header = (f"{'element':<28} | "
              + "  ".join(f"d={d:1d} no-twist | d={d:1d} w/ O(3)"
                         for d in degrees))
    print(header)
    print("-" * len(header))
    for perm, name in elements:
        cells = []
        for d in degrees:
            U = U_cache[(name, d)]
            tilde = [U @ H @ U.T for H in H_by_d[d]]
            no_twist = no_twist_residual(H_by_d[d], tilde)
            _, _, with_twist = best_O3_for_block(H_by_d[d], tilde)
            cells.append(f"{no_twist:.4f}   |  {with_twist:.4f}")
        print(f"{name:<28} | " + "  ".join(cells))
    print()

    # --- Joint O(3) across all degrees (the physical symmetry test).
    print(f"=== Joint O(3) residual (single O across all degrees) ===")
    print(f"{'element':<28} | overall rel_res | "
          + " ".join(f"d={d:1d}" for d in degrees))
    print("-" * 60)
    for perm, name in elements:
        tilde_by_d = {
            d: [U_cache[(name, d)] @ H @ U_cache[(name, d)].T for H in H_by_d[d]]
            for d in degrees
        }
        O, per_d, overall = joint_O3_across_degrees(H_by_d, tilde_by_d)
        cells = " ".join(f"{per_d[d]:.4f}" for d in degrees)
        det = float(np.linalg.det(O))
        print(f"{name:<28} | {overall:.4f} (det O={det:+.2f})  | {cells}")


if __name__ == "__main__":
    main()
