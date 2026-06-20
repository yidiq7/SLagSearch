"""Put a (3, w) candidate-coefficient matrix into a canonical form in which
each of the 3 equations is a 1-dim character of the equation-space rep of
Z_2 x S_3 (or the chosen --group).

Pipeline:
  1. Choose a phase frame a in (Z_5)^4 (a_4 fixed = 0).
       * --a a0,a1,a2,a3,a4 supplied  -> use it directly.
       * --a omitted                   -> sweep via test_phase_twist_symmetry.
  2. Compute the joint-O(3) matrices at a* for each group element (HOLO or
     ANTI, picked by --mode).
  3. Pick two V_4 generators (default (0 4) and (1 2); they commute as
     elements of (Z_2 x S_3) / A_3 = V_4 = Z_2 x Z_2). Joint-diagonalize
     the symmetric parts of their O matrices via the standard "A + eps*B"
     trick (eps = 1/phi to avoid eigenvalue degeneracy) to obtain
     M in O(3).
  4. Apply (phase twist by a*)  o  (equation rotation by M) to the
     original coeffs.
  5. Save coeffs_canonical.pkl + metadata (a*, M, source path, ...).
  6. Recompute O matrices on the canonical coeffs (a = 0 in the new frame)
     and print the diagonal entries -- these are the per-equation character
     assignments. If V_4 is the actual structure, every O should be
     approximately diagonal with +/- 1 entries.

Usage:
    python -m symmetry.canonicalize_coeffs \
        --coeffs gd_runs/plots_slag_d4_run/coeffs.pkl \
        [--a 2,2,4,4,0]                         # default: sweep
        [--group z2xs3] [--mode holo]
        [--gen_a "(0 4)" --gen_b "(1 2)"]
        [--out_dir <dir> | --out_subdir <name>]

Downstream usage of coeffs_canonical.pkl:
    python -m viz.plot_hermitian_coeffs --coeffs <out>/coeffs_canonical.pkl
    python -m symmetry.test_phase_twist_symmetry \
        --coeffs <out>/coeffs_canonical.pkl    # should hit a=(0,0,0,0,0)
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from hermitian_coeffs import (
    _BLOCK, _SYM_DIM, _load_coeffs, extract_hermitians,
)
from symmetry.permute_coeffs import hermitian_to_coeffs_row
from symmetry.test_permutation_symmetry import (
    _GROUPS, build_tilde, joint_O3_across_degrees, monomial_permutation,
)
from symmetry.test_phase_twist_symmetry import (
    apply_phase_twist, sweep_phase_twists,
)


def apply_phase_twist_to_coeffs(
    coeffs: np.ndarray, a: tuple
) -> np.ndarray:
    """Apply z_i -> omega^{a_i} z_i to each equation by going through the
    Hermitian representation (where the twist is a diagonal-unitary
    conjugation). Returns a new (3, w) coeffs array with the same shape.
    """
    H_by_d = extract_hermitians(coeffs)
    H_twisted = apply_phase_twist(H_by_d, a)
    new = np.zeros_like(coeffs)
    w = coeffs.shape[1]
    for d, (lo, hi) in _BLOCK.items():
        if w < hi:
            break
        N = _SYM_DIM[d]
        for k in range(coeffs.shape[0]):
            new[k, lo:hi] = hermitian_to_coeffs_row(H_twisted[d][k], N)
    return new


def compute_O_at_a(
    H_by_d: dict[int, list[np.ndarray]],
    elements: list[tuple[tuple[int, ...], str]],
    mode: str,
    a: tuple,
) -> dict[str, dict]:
    """Joint-O(3) test for each element at a fixed phase frame a.

    Returns {name: {"residual", "O", "per_d"}}.
    """
    H_twisted = apply_phase_twist(H_by_d, a)
    degrees = sorted(H_by_d)
    U_cache = {(name, d): monomial_permutation(perm, d)
               for perm, name in elements for d in degrees}
    out: dict[str, dict] = {}
    for perm, name in elements:
        tilde = {d: build_tilde(H_twisted[d], U_cache[(name, d)], mode)
                 for d in degrees}
        O, per_d, overall = joint_O3_across_degrees(H_twisted, tilde)
        out[name] = {"residual": overall, "O": O, "per_d": per_d}
    return out


def find_equation_rotation(
    O_a: np.ndarray, O_b: np.ndarray, eps: float = 0.3819
) -> np.ndarray:
    """Find M in O(3) that jointly diagonalizes two commuting reflections.

    Symmetrize each O to handle the small numerical noise from the
    Procrustes fit, then take eigenvectors of A + eps*B (eps = 1/phi, an
    irrational-ish value that breaks any accidental eigenvalue degeneracy
    so eigh returns the joint eigenbasis unambiguously). The returned M
    satisfies M @ O @ M.T ~ diag for both O.
    """
    A = (O_a + O_a.T) / 2
    B = (O_b + O_b.T) / 2
    C = A + eps * B
    _, V = np.linalg.eigh(C)              # V columns are joint eigvecs.
    return V.T                            # new_coeffs = M @ coeffs.


def _format_a(a: tuple) -> str:
    return "(" + ",".join(str(x) for x in a) + ")"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coeffs", required=True, type=Path,
                        help="Source coeffs pkl (bare (3,w) array or dict "
                             "with 'coeffs' key).")
    parser.add_argument("--a", type=str, default=None,
                        help="Phase frame as 'a0,a1,a2,a3,a4'. Default: sweep.")
    parser.add_argument("--group", default="z2xs3",
                        choices=list(_GROUPS.keys()))
    parser.add_argument("--mode", default="holo", choices=["holo", "anti"])
    parser.add_argument("--gen_a", default="(0 4)",
                        help="First V_4 generator (must be a name in --group).")
    parser.add_argument("--gen_b", default="(1 2)",
                        help="Second V_4 generator (must commute with --gen_a "
                             "modulo A_3).")
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument("--out_dir", type=Path, default=None,
                           help="Full output dir. Default: <coeffs_dir>/canonical/.")
    out_group.add_argument("--out_subdir", type=str, default="canonical",
                           help="Subdir of <coeffs_dir>. Default: 'canonical'.")
    args = parser.parse_args()

    coeffs = _load_coeffs(args.coeffs)
    print(f"Loaded coeffs {coeffs.shape} from {args.coeffs}")
    H_by_d = extract_hermitians(coeffs)
    degrees = sorted(H_by_d)
    elements = _GROUPS[args.group]

    # ---- 1. Determine a*.
    if args.a is None:
        print(f"\nSweeping (Z_5)^4 phase twists in mode={args.mode} "
              f"(group={args.group}); 625 frames...")
        _, a_star, _ = sweep_phase_twists(H_by_d, elements, args.mode)
        print(f"Joint best a* = {_format_a(a_star)}")
    else:
        a_star = tuple(int(x) for x in args.a.split(","))
        if len(a_star) != 5:
            raise ValueError(f"--a must have 5 ints, got {a_star}")
        print(f"\nUsing supplied a* = {_format_a(a_star)}")

    # ---- 2. Compute O matrices at a*.
    O_at = compute_O_at_a(H_by_d, elements, args.mode, a_star)
    available = [n for _, n in elements]
    if args.gen_a not in O_at:
        raise ValueError(f"--gen_a {args.gen_a!r} not in --group {args.group}. "
                         f"Available: {available}")
    if args.gen_b not in O_at:
        raise ValueError(f"--gen_b {args.gen_b!r} not in --group {args.group}. "
                         f"Available: {available}")

    O_A = O_at[args.gen_a]["O"]
    O_B = O_at[args.gen_b]["O"]
    commutator_norm = float(np.linalg.norm(O_A @ O_B - O_B @ O_A))
    print(f"\nEquation-diag generators: gen_a = {args.gen_a}, gen_b = {args.gen_b}")
    print(f"  ||[O_a, O_b]||_F = {commutator_norm:.4f}  "
          f"(small => approx. commuting; joint diag valid)")

    # ---- 3. Find M.
    M = find_equation_rotation(O_A, O_B)
    print(f"\nM (equation rotation, new_eq_k = sum_l M[k,l] eq_l):")
    for row in M:
        print("  [" + "  ".join(f"{x:+.4f}" for x in row) + "]")
    print(f"  det M = {float(np.linalg.det(M)):+.4f}")

    # ---- 4. Apply twist + rotation to coeffs.
    coeffs_twisted = apply_phase_twist_to_coeffs(coeffs, a_star)
    coeffs_canonical = M @ coeffs_twisted

    # ---- 5. Verify: compute O matrices on canonical coeffs (a = 0 in new frame).
    H_canon = extract_hermitians(coeffs_canonical)
    O_canon = compute_O_at_a(H_canon, elements, args.mode, (0, 0, 0, 0, 0))

    print(f"\n=== Per-element O after canonicalization "
          f"(a = (0,0,0,0,0) in new frame, mode={args.mode}) ===")
    d_cols = "  ".join(f"d={d}" for d in degrees)
    header = (f"{'element':<20} | diag(O)               | off-diag ||.|| | "
              f"residual | per-degree: {d_cols}")
    print(header)
    print("-" * len(header))
    for perm, name in elements:
        info = O_canon[name]
        diag = np.diag(info["O"])
        offdiag = float(np.linalg.norm(info["O"] - np.diag(diag)))
        diag_str = "  ".join(f"{x:+.3f}" for x in diag)
        per_d_str = "  ".join(f"{info['per_d'][d]:.3f}" for d in degrees)
        print(f"{name:<20} | [{diag_str}]  | {offdiag:.3f}          | "
              f"{info['residual']:.4f}   | {per_d_str}")

    # ---- 6. Per-equation character assignment (sign table).
    print(f"\n=== Per-equation character table (sign of diag(O_g)[k] per equation k) ===")
    print(f"{'g':<20} | eq0    | eq1    | eq2")
    print("-" * 50)
    for perm, name in elements:
        if name == "identity":
            continue
        diag = np.diag(O_canon[name]["O"])
        signs = ["+1" if x > 0 else "-1" for x in diag]
        print(f"{name:<20} | {signs[0]:<6} | {signs[1]:<6} | {signs[2]}")

    # ---- 7. Save.
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = args.coeffs.parent / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    coeffs_path = out_dir / "coeffs_canonical.pkl"
    with open(coeffs_path, "wb") as f:
        pickle.dump({"coeffs": np.asarray(coeffs_canonical)}, f)
    meta_path = out_dir / "canonical_form_metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump({
            "a_star": a_star,
            "M": np.asarray(M),
            "generators": (args.gen_a, args.gen_b),
            "group": args.group,
            "mode": args.mode,
            "source_coeffs": str(args.coeffs),
        }, f)
    print(f"\nSaved canonical coeffs to {coeffs_path}")
    print(f"Saved metadata to {meta_path}")
    print(f"\nInspect with:")
    print(f"  python -m viz.plot_hermitian_coeffs --coeffs {coeffs_path}")
    print(f"  python -m symmetry.test_phase_twist_symmetry "
          f"--coeffs {coeffs_path}    # should land at a=(0,0,0,0,0)")


if __name__ == "__main__":
    main()
