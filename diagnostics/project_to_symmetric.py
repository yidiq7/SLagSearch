"""Project canonical-form coeffs onto exact Z_2 x S_3 symmetry by character
averaging.

For each equation k carrying character chi_k of the equation-space rep,
each Hermitian H_k^(d) is replaced by

    H_proj_k = (1/|G|) sum_{g in G} chi_k(g) U_g H_k U_g^T.

After projection, every equation is *exactly* G-equivariant: any future
symmetry test (test_permutation_symmetry, test_phase_twist_symmetry at
a=0) returns residual ~ 0 to machine precision.

Characters per equation are determined from the input by reading diag(O_g)
(snapped to +/- 1) for each group element after the joint-O(3) Procrustes
fit at a = 0. If diag(O_g) deviates from +/- 1 by more than --tol, the
input is not in canonical form and the script aborts (run
diagnostics.canonicalize_coeffs first).

Side effects:
  * The non-G-symmetric residual is discarded. For a relative symmetry
    residual r, the kept fraction is sqrt(1 - r^2) per equation
    (~96-99% for r in [0.15, 0.27]).
  * The projected polynomial may no longer be a sLag. To check, run
    viz.fitness_pipeline on the output -- if the (lag, spec) fitnesses
    are comparable to the input, the discarded part was numerical noise;
    if they drop, the candidate has real non-symmetric structure.

Usage:
    python -m diagnostics.project_to_symmetric \
        --coeffs gd_runs/plots_slag_d4_run/canonical/coeffs_canonical.pkl \
        [--group z2xs3] [--mode holo] [--tol 0.15]
        [--out_dir <dir> | --out_subdir <name>]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from hermitian_coeffs import (
    _BLOCK, _SYM_DIM, _load_coeffs, extract_hermitians,
)
from diagnostics.permute_coeffs import hermitian_to_coeffs_row
from diagnostics.test_permutation_symmetry import (
    _GROUPS, monomial_permutation,
)
from diagnostics.canonicalize_coeffs import compute_O_at_a


def determine_characters(
    H_by_d: dict[int, list[np.ndarray]],
    elements: list[tuple[tuple[int, ...], str]],
    mode: str,
    tol: float,
) -> tuple[dict[str, np.ndarray], float]:
    """Read diag(O_g) per element from the joint-O(3) fit at a = 0 and
    snap to +/- 1. Raises ValueError if any diag(O_g) deviates from +/- 1
    by more than `tol` (i.e., the input isn't canonical).

    Returns ({g_name: ndarray of 3 +/- 1 ints}, max_deviation).
    """
    O_at = compute_O_at_a(H_by_d, elements, mode, (0, 0, 0, 0, 0))
    chars: dict[str, np.ndarray] = {}
    bad: list[tuple[str, float, np.ndarray]] = []
    max_dev = 0.0
    for perm, name in elements:
        diag = np.diag(O_at[name]["O"]).real
        dev = float(np.max(np.abs(np.abs(diag) - 1.0)))
        max_dev = max(max_dev, dev)
        if dev > tol:
            bad.append((name, dev, diag))
        chars[name] = np.sign(diag).astype(int)
    if bad:
        msg = (f"diag(O) deviates from +/- 1 by > {tol} on {len(bad)} "
               f"element(s) -- input isn't in canonical form:\n")
        for nm, dv, dg in bad[:5]:
            dg_str = " ".join(f"{x:+.3f}" for x in dg)
            msg += f"  {nm}: dev={dv:.4f}  diag=[{dg_str}]\n"
        msg += ("Run `python -m diagnostics.canonicalize_coeffs` first to "
                "produce canonical coeffs (twist + equation rotation).")
        raise ValueError(msg)
    return chars, max_dev


def project_onto_characters(
    coeffs: np.ndarray,
    chars: dict[str, np.ndarray],
    elements: list[tuple[tuple[int, ...], str]],
) -> tuple[np.ndarray, dict[int, list[float]]]:
    """H_proj_k = (1/|G|) sum_g chi_k(g) U_g H_k U_g^T per degree, per eq.

    Returns (projected_coeffs, {d: [||H_proj_k||/||H_k|| for k in 0..2]}).
    """
    H_by_d = extract_hermitians(coeffs)
    w = coeffs.shape[1]
    new = np.zeros_like(coeffs)
    ratios: dict[int, list[float]] = {}

    for d, (lo, hi) in _BLOCK.items():
        if w < hi:
            break
        N = _SYM_DIM[d]
        U_cache = {name: monomial_permutation(perm, d)
                   for perm, name in elements}
        ratios[d] = []
        for k in range(3):
            H_k = H_by_d[d][k]
            H_proj = np.zeros_like(H_k)
            for perm, name in elements:
                U = U_cache[name]
                chi = int(chars[name][k])
                H_proj = H_proj + chi * (U @ H_k @ U.T)
            H_proj = H_proj / len(elements)
            new[k, lo:hi] = hermitian_to_coeffs_row(H_proj, N)
            ratios[d].append(
                float(np.linalg.norm(H_proj) /
                      max(np.linalg.norm(H_k), 1e-30))
            )
    return new, ratios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coeffs", required=True, type=Path,
                        help="Path to canonical coeffs pkl.")
    parser.add_argument("--group", default="z2xs3",
                        choices=list(_GROUPS.keys()))
    parser.add_argument("--mode", default="holo", choices=["holo", "anti"])
    parser.add_argument("--tol", type=float, default=0.15,
                        help="Max |diag(O)-1| deviation accepted as "
                             "'in canonical form' (default: 0.15).")
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument("--out_dir", type=Path, default=None,
                           help="Full output directory. Default: "
                                "<coeffs_dir>/projected/.")
    out_group.add_argument("--out_subdir", type=str, default="projected",
                           help="Subdir of --coeffs's parent dir. Default: 'projected'.")
    args = parser.parse_args()

    coeffs = _load_coeffs(args.coeffs)
    print(f"Loaded coeffs {coeffs.shape} from {args.coeffs}")
    H_by_d = extract_hermitians(coeffs)
    degrees = sorted(H_by_d)
    elements = _GROUPS[args.group]

    # ---- 1. Read characters from the input.
    print(f"\nReading characters from diag(O_g) at a=(0,0,0,0,0)  "
          f"(mode={args.mode}, tol={args.tol})...")
    chars, max_dev = determine_characters(H_by_d, elements, args.mode, args.tol)
    print(f"  Max |diag(O)|-1 deviation across group: {max_dev:.4f}  "
          f"(under tol={args.tol})")

    print(f"\nSnapped character table:")
    print(f"{'g':<20} | eq0  | eq1  | eq2")
    print("-" * 40)
    for perm, name in elements:
        c = chars[name]
        print(f"{name:<20} | {c[0]:+d}   | {c[1]:+d}   | {c[2]:+d}")

    # ---- 2. Project.
    print(f"\nProjecting onto character subspaces "
          f"(|G| = {len(elements)})...")
    coeffs_proj, ratios = project_onto_characters(coeffs, chars, elements)

    print(f"\nProjection ratios ||H_proj_k|| / ||H_k|| per degree per equation:")
    print(f"  (1.0 = no mass lost; ~sqrt(1-r^2) for relative symmetry residual r)")
    print(f"{'d':>3} | eq0      | eq1      | eq2")
    print("-" * 40)
    for d in degrees:
        r = ratios[d]
        print(f"{d:>3} | {r[0]:.4f}   | {r[1]:.4f}   | {r[2]:.4f}")

    overall_kept = float(np.linalg.norm(coeffs_proj) /
                         max(np.linalg.norm(coeffs), 1e-30))
    print(f"\nOverall ||coeffs_proj|| / ||coeffs|| = {overall_kept:.4f}  "
          f"(1.0 = lossless).")

    # ---- 3. Verify exact symmetry on the projected coeffs.
    H_proj_by_d = extract_hermitians(coeffs_proj)
    print(f"\nSanity: rerunning joint-O(3) on projected coeffs "
          f"(residuals should be ~ machine eps)...")
    O_proj = compute_O_at_a(H_proj_by_d, elements, args.mode, (0,)*5)
    max_resid = max(info["residual"] for name, info in O_proj.items()
                    if name != "identity")
    print(f"  Max non-identity residual on projected coeffs: {max_resid:.2e}")

    # ---- 4. Save.
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = args.coeffs.parent / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    proj_path = out_dir / "coeffs_projected.pkl"
    with open(proj_path, "wb") as f:
        pickle.dump({"coeffs": np.asarray(coeffs_proj)}, f)
    meta_path = out_dir / "projection_metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump({
            "source_coeffs": str(args.coeffs),
            "group": args.group,
            "mode": args.mode,
            "characters": {name: chars[name].tolist()
                           for _, name in elements},
            "projection_ratios": {d: ratios[d] for d in ratios},
            "overall_kept": overall_kept,
            "max_residual_after": max_resid,
        }, f)
    print(f"\nSaved projected coeffs to {proj_path}")
    print(f"Saved metadata to {meta_path}")
    print(f"\nNext steps:")
    print(f"  # Exactly-symmetric Hermitian heatmaps:")
    print(f"  python -m viz.plot_hermitian_coeffs --coeffs {proj_path}")
    print(f"  # Fitness check -- did projection destroy the sLag?")
    print(f"  python -m viz.fitness_pipeline --coeffs {proj_path}")
    print(f"  # Sanity: symmetry test should now show ~ 0 residual on Z_2 x S_3:")
    print(f"  python -m diagnostics.test_permutation_symmetry "
          f"--coeffs {proj_path} --group z2xs3 --mode holo")


if __name__ == "__main__":
    main()
