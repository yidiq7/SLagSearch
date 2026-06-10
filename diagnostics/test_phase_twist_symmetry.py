"""Sweep the (Z_5)^4 phase-twist gauge of the Fermat-quintic ambient symmetry
and test whether elements of a candidate symmetry group act as literal
(anti-)holomorphic permutations after the right phase choice.

The Fermat quintic z_0^5 + ... + z_4^5 = 0 is invariant under
    z_i  ->  omega^{a_i} z_i,    omega = e^{2 pi i / 5},   a in (Z_5)^5,
and a sLag candidate found by GD lives at a generic representative of this
(Z_5)^4 = (Z_5)^5 / Z_5 gauge orbit (the diagonal Z_5 acts trivially on H).
A symmetry g that looks broken in test_permutation_symmetry.py may turn out
to be present in a different phase frame:

    in canonical frame  w_i := omega^{-a_i} z_i,
    f_a(w_{g(.)})  ?=  sum_b O_{ab} f_b(w)        (HOLO test in w-coords)
    f_a(conj(w_{g(.)}))  ?=  sum_b O_{ab} f_b(w)  (ANTI test in w-coords)

The Hermitian matrix H_a transforms as H_a -> D_alpha * H_a * D_alpha^*,
where D_alpha[A] = omega^{alpha(A)} for monomial A = (i_1,...,i_d) and
alpha(A) = sum_j a_{i_j} mod 5. Once H is twisted, the symmetry test is
identical to the untwisted one (compare H_twisted to U_g H_twisted U_g^T).

For each g in the chosen group (other than identity, which is trivially 0),
we sweep a in (Z_5)^4 (fixing a_4 = 0) and report:
  * the per-element best a_g and residual (does this g admit ANY frame?);
  * the joint best a* minimizing LS over the whole group (the canonical
    frame in which the entire group simultaneously becomes a literal
    symmetry, if one exists).

Usage:
    python -m diagnostics.test_phase_twist_symmetry \
        --coeffs gd_runs/plots_slag_d4_run/coeffs.pkl \
        [--group z2xs3] [--mode both] [--print_O]
"""
import argparse
from itertools import combinations_with_replacement, product
from pathlib import Path

import numpy as np

from diagnostics.test_permutation_symmetry import (
    _GROUPS,
    _load_hermitians,
    build_tilde,
    joint_O3_across_degrees,
    monomial_permutation,
)


def phase_factors(a: tuple, d: int) -> np.ndarray:
    """Diagonal phase vector D_alpha[A] = omega^{alpha(A) mod 5} for each
    monomial A in Sym^d(C^5), where alpha(A) = sum of a_i over the indices
    in A. Returns a complex (N,) vector with N = 5/15/35/70 for d = 1..4.
    """
    omega = np.exp(2j * np.pi / 5)
    monomials = list(combinations_with_replacement(range(5), d))
    powers = np.array(
        [sum(a[i] for i in m) % 5 for m in monomials], dtype=np.int64
    )
    return omega ** powers


def apply_phase_twist(
    H_by_d: dict[int, list[np.ndarray]], a: tuple
) -> dict[int, list[np.ndarray]]:
    """Apply z_i -> omega^{a_i} z_i to every Hermitian matrix:
        H_{AB}  ->  omega^{alpha(A) - alpha(B)} H_{AB}.
    Returns a new dict; does not mutate input.
    """
    out: dict[int, list[np.ndarray]] = {}
    for d, Hs in H_by_d.items():
        D = phase_factors(a, d)
        twist = D[:, None] * np.conj(D)[None, :]   # outer product D_alpha D_alpha^*
        out[d] = [twist * H for H in Hs]
    return out


def sweep_phase_twists(
    H_by_d: dict[int, list[np.ndarray]],
    elements: list[tuple[tuple[int, ...], str]],
    mode: str,
) -> tuple[dict[str, dict], tuple, dict[str, dict]]:
    """For each g in `elements` (skipping identity), sweep a in (Z_5)^4
    (a_4 = 0 fixed) and find:
      - per-element best a_g (lowest residual any single g admits);
      - joint best a* minimizing sum_g residual_g^2 across the whole group.

    Returns (per_element_best, a_star, per_element_at_a_star). Each
    inner dict has keys {"a", "residual", "O", "per_d"} for the element.
    `per_element_at_a_star` only has "residual"/"O"/"per_d" populated.
    """
    degrees = sorted(H_by_d)
    non_id = [(p, n) for p, n in elements if n != "identity"]

    # Permutation matrices on the monomial bases, cached across the sweep.
    U_cache: dict[tuple[str, int], np.ndarray] = {
        (name, d): monomial_permutation(perm, d)
        for perm, name in non_id for d in degrees
    }

    per_element_best: dict[str, dict] = {
        name: {"a": None, "residual": np.inf, "O": None, "per_d": None}
        for _, name in non_id
    }
    # For the joint search we only need the per-a total LS score; track the
    # full per-element snapshot for the winning a separately at the end.
    a_scores: dict[tuple, float] = {}

    for a_short in product(range(5), repeat=4):
        a = a_short + (0,)
        H_twisted = apply_phase_twist(H_by_d, a)
        total_sq = 0.0
        for perm, name in non_id:
            tilde_by_d = {
                d: build_tilde(H_twisted[d], U_cache[(name, d)], mode)
                for d in degrees
            }
            O, per_d, overall = joint_O3_across_degrees(H_twisted, tilde_by_d)
            total_sq += overall ** 2
            if overall < per_element_best[name]["residual"]:
                per_element_best[name] = {
                    "a": a, "residual": overall, "O": O, "per_d": per_d,
                }
        a_scores[a] = total_sq

    a_star = min(a_scores, key=a_scores.get)

    # Recompute per-element snapshot at a_star (cheap; just one frame).
    H_twisted_star = apply_phase_twist(H_by_d, a_star)
    per_element_at_star: dict[str, dict] = {}
    for perm, name in non_id:
        tilde_by_d = {
            d: build_tilde(H_twisted_star[d], U_cache[(name, d)], mode)
            for d in degrees
        }
        O, per_d, overall = joint_O3_across_degrees(H_twisted_star, tilde_by_d)
        per_element_at_star[name] = {"residual": overall, "O": O, "per_d": per_d}

    return per_element_best, a_star, per_element_at_star


def _format_a(a: tuple) -> str:
    return "(" + ",".join(str(x) for x in a) + ")"


def _format_O(O: np.ndarray) -> str:
    """Pretty-print a 3x3 matrix as a single bracketed string."""
    rows = []
    for r in O:
        rows.append("[" + " ".join(f"{x:+.2f}" for x in r) + "]")
    return " ".join(rows)


def print_per_element_best(
    per_element_best: dict[str, dict],
    degrees: list[int],
    mode_label: str,
    print_O: bool,
) -> None:
    print(f"=== Per-element best phase twist  ({mode_label}, "
          f"sweep over (Z_5)^4 with a_4 = 0) ===")
    d_cols = "  ".join(f"d={d}" for d in degrees)
    header = (f"{'element':<20} | {'best a':<13} | residual | det O | "
              f"per-degree: {d_cols}")
    print(header)
    print("-" * len(header))
    for name, info in per_element_best.items():
        if info["a"] is None:
            continue
        det = float(np.linalg.det(info["O"]))
        per_d = "  ".join(f"{info['per_d'][d]:.3f}" for d in degrees)
        print(f"{name:<20} | {_format_a(info['a']):<13} | "
              f"{info['residual']:.4f}   | {det:+.2f} | {per_d}")
    print()
    if print_O:
        print(f"--- O(3) matrices at each per-element best a  "
              f"({mode_label}) ---")
        for name, info in per_element_best.items():
            if info["O"] is None:
                continue
            print(f"{name:<20}  a={_format_a(info['a'])}   "
                  f"O = {_format_O(info['O'])}")
        print()


def print_joint_best(
    a_star: tuple,
    per_element_at_star: dict[str, dict],
    degrees: list[int],
    mode_label: str,
    print_O: bool,
) -> None:
    residuals = [info["residual"] for info in per_element_at_star.values()]
    rms = float(np.sqrt(np.mean([r ** 2 for r in residuals])))
    worst_name, worst_info = max(
        per_element_at_star.items(), key=lambda kv: kv[1]["residual"]
    )
    print(f"=== Joint best phase twist a*  ({mode_label}) ===")
    print(f"a* = {_format_a(a_star)}   (a_4 fixed = 0)")
    print(f"LS RMS residual across non-identity = {rms:.4f}")
    print(f"worst element: {worst_name}  (residual {worst_info['residual']:.4f})")
    print()
    d_cols = "  ".join(f"d={d}" for d in degrees)
    header = f"{'element':<20} | residual | det O | per-degree: {d_cols}"
    print(header)
    print("-" * len(header))
    for name, info in per_element_at_star.items():
        det = float(np.linalg.det(info["O"]))
        per_d = "  ".join(f"{info['per_d'][d]:.3f}" for d in degrees)
        print(f"{name:<20} | {info['residual']:.4f}   | {det:+.2f} | {per_d}")
    print()
    if print_O:
        print(f"--- O(3) matrices at a* = {_format_a(a_star)}  "
              f"({mode_label}) ---")
        for name, info in per_element_at_star.items():
            print(f"{name:<20}  O = {_format_O(info['O'])}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coeffs", required=True, type=Path,
                        help="Path to a coeff pkl (bare (3,w) array or dict "
                             "with 'coeffs' key), OR a hermitian_matrices_*.npz.")
    parser.add_argument("--group", default="z2xs3",
                        choices=list(_GROUPS.keys()),
                        help="Which set of elements to test (default: z2xs3).")
    parser.add_argument("--mode", default="both",
                        choices=["holo", "anti", "both"],
                        help="holo / anti / both (default: both).")
    parser.add_argument("--print_O", action="store_true",
                        help="Also print the 3x3 O matrices at each best a.")
    args = parser.parse_args()

    H_by_d = _load_hermitians(args.coeffs)
    degrees = sorted(H_by_d)
    print(f"Loaded Hermitian matrices from {args.coeffs}")
    print(f"Degrees present: {degrees}")
    for d in degrees:
        norms = [float(np.linalg.norm(H)) for H in H_by_d[d]]
        print(f"  d={d}: ||H||_F per eq = "
              + ", ".join(f"{n:.4g}" for n in norms))
    print()
    print(f"Sweeping a in (Z_5)^4 (a_4 = 0 fixed); 625 frames per element.")
    print(f"Comparison: H_twisted vs U_g H_twisted U_g^T  "
          f"(canonical-frame symmetry test).")
    print()

    elements = _GROUPS[args.group]
    modes = ["holo", "anti"] if args.mode == "both" else [args.mode]
    mode_label = {"holo": "HOLO (U H U^T)", "anti": "ANTI (U conj(H) U^T)"}

    for mode in modes:
        per_element_best, a_star, per_element_at_star = sweep_phase_twists(
            H_by_d, elements, mode
        )
        print_per_element_best(
            per_element_best, degrees, mode_label[mode], args.print_O
        )
        print_joint_best(
            a_star, per_element_at_star, degrees, mode_label[mode], args.print_O
        )


if __name__ == "__main__":
    main()
