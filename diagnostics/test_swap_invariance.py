"""Direct geometric test of a discrete coordinate permutation symmetry on a
candidate sLag. Reuses the already-mined min_set.pkl (produced inside each
plots_slag_<job>/ folder by run_fitness_pipeline) and the evaluation function
in helper.evaluate_equations_single_point. No refinement is done -- if we
Newton-refined the swapped points they could just slide back onto the
candidate locus, defeating the test.

For a swap sigma = (i j), evaluate the original polynomial system on the
mined points (baseline) and on the swapped points sigma.p. If sigma is a
symmetry of L = {f_k = 0}, sigma.p still satisfies f_k ~ 0; if not, the
swapped points fall off the locus and ||f_k(sigma.p)|| grows substantially.

Output goes to --out_dir (full path) or <min_set_dir>/<out_subdir>/ (subdir
name); these flags are mutually exclusive. Default: <min_set_dir>.

Usage:
    python -m diagnostics.test_swap_invariance \
        --min_set <plots_slag_<job>/min_set.pkl> \
        --coeffs gd_runs/gd_<job>_step<N>.pkl \
        [--swap 2 3] [--psi 0] [--out_dir <dir> | --out_subdir <name>]
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from helper import evaluate_equations_single_point
from hermitian_coeffs import _load_coeffs


def load_min_set_real(path: Path) -> np.ndarray:
    """min_set.pkl holds (N, 5) complex points (already Newton-refined on the
    candidate sLag). Return the (N, 10) real form expected by
    evaluate_equations_single_point.
    """
    with open(path, "rb") as f:
        z = pickle.load(f)
    z = np.asarray(z)
    if z.ndim != 2 or z.shape[1] != 5:
        raise ValueError(f"expected (N, 5) complex array, got {z.shape}")
    return np.concatenate([z.real, z.imag], axis=1).astype(np.float64)


def swap_coords_real(points_real: np.ndarray, i: int, j: int) -> np.ndarray:
    """Swap coords i and j in the (N, 10) real form (both Re and Im halves)."""
    out = points_real.copy()
    out[:, [i, j]] = out[:, [j, i]]
    out[:, [5 + i, 5 + j]] = out[:, [5 + j, 5 + i]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min_set", required=True, type=Path,
                        help="Path to a min_set.pkl produced by "
                             "run_fitness_pipeline (inside plots_slag_<job>/).")
    parser.add_argument("--coeffs", required=True, type=Path,
                        help="Path to the corresponding coeffs pkl.")
    parser.add_argument("--swap", nargs=2, type=int, default=[2, 3],
                        metavar=("I", "J"),
                        help="Two indices to swap (default: 2 3).")
    parser.add_argument("--psi", type=float, default=0.0,
                        help="Quintic deformation parameter (default 0 = "
                             "Fermat).")
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument("--out_dir", type=Path, default=None,
                           help="Full output directory. "
                                "Default: parent directory of --min_set.")
    out_group.add_argument("--out_subdir", type=str, default=None,
                           help="Output subdirectory name appended to "
                                "--min_set's parent directory.")
    parser.add_argument("--job_id", type=str, default=None,
                        help="Label for output filenames; defaults to the "
                             "coeffs stem.")
    args = parser.parse_args()

    # Use float64 throughout to match gradient_descent.py precision.
    jax.config.update("jax_enable_x64", True)

    points_real = load_min_set_real(args.min_set)
    print(f"Loaded {points_real.shape[0]} points from {args.min_set}")

    coeffs = jnp.asarray(_load_coeffs(args.coeffs), dtype=jnp.float64)
    print(f"Loaded coeffs {coeffs.shape} from {args.coeffs}")
    psi = jnp.asarray(args.psi, dtype=jnp.complex128)

    i, j = args.swap
    swapped_real = swap_coords_real(points_real, i, j)

    eval_batch = jax.jit(jax.vmap(
        lambda p: evaluate_equations_single_point(p, coeffs, psi)))

    base_all = np.asarray(eval_batch(jnp.asarray(points_real)))    # (N, 5)
    swap_all = np.asarray(eval_batch(jnp.asarray(swapped_real)))   # (N, 5)

    # Layout of the 5 returned values:
    #   col 0, 1 = Re(quintic), Im(quintic)
    #   col 2..4 = f_0, f_1, f_2  (the user equations defining the sLag)
    base_cy = base_all[:, :2]
    base_user = base_all[:, 2:5]
    swap_cy = swap_all[:, :2]
    swap_user = swap_all[:, 2:5]

    norm_base_user = np.linalg.norm(base_user, axis=1)
    norm_swap_user = np.linalg.norm(swap_user, axis=1)
    norm_base_cy = np.linalg.norm(base_cy, axis=1)
    norm_swap_cy = np.linalg.norm(swap_cy, axis=1)

    print()
    print(f"=== quintic eqn (must be ~0 for any point on the CY) ===")
    print(f"  baseline ||cy(p)||   : mean {norm_base_cy.mean():.3e}, "
          f"99% {np.percentile(norm_base_cy, 99):.3e}")
    print(f"  swapped  ||cy(sp)||  : mean {norm_swap_cy.mean():.3e}, "
          f"99% {np.percentile(norm_swap_cy, 99):.3e}")
    print(f"  (should be unchanged: quintic is S_5-invariant)")
    print()
    print(f"=== user equations (the actual symmetry test) ===")
    print(f"  baseline ||f(p)||    : mean {norm_base_user.mean():.3e}, "
          f"median {np.median(norm_base_user):.3e}, "
          f"99% {np.percentile(norm_base_user, 99):.3e}")
    print(f"  swapped  ||f(sp)||   : mean {norm_swap_user.mean():.3e}, "
          f"median {np.median(norm_swap_user):.3e}, "
          f"99% {np.percentile(norm_swap_user, 99):.3e}")

    ratio = norm_swap_user.mean() / max(norm_base_user.mean(), 1e-30)
    print()
    print(f"  Mean ratio swapped / baseline = {ratio:.2f}")
    print(f"    ratio ~ 1     -> ({i} {j}) IS a symmetry of L "
          f"(swapped points still on L)")
    print(f"    ratio >> 1    -> ({i} {j}) is NOT a symmetry "
          f"(swapped points fell off L)")

    # Per-equation breakdown.
    print()
    print(f"=== per-equation breakdown (mean |f_k| only) ===")
    for k in range(3):
        b = float(np.mean(np.abs(base_user[:, k])))
        s = float(np.mean(np.abs(swap_user[:, k])))
        print(f"  f_{k}: baseline {b:.3e}   swapped {s:.3e}   ratio {s/max(b,1e-30):.2f}")

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.out_subdir is not None:
        out_dir = args.min_set.parent / args.out_subdir
    else:
        out_dir = args.min_set.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    job_id = args.job_id or args.coeffs.stem
    out_path = out_dir / f"swap_invariance_{i}{j}_{job_id}.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    lo = max(min(norm_base_user.min(), norm_swap_user.min()), 1e-15)
    hi = max(norm_base_user.max(), norm_swap_user.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 80)
    ax.hist(norm_base_user, bins=bins, alpha=0.6, color="steelblue",
            label=f"baseline (mean {norm_base_user.mean():.2e})")
    ax.hist(norm_swap_user, bins=bins, alpha=0.6, color="indianred",
            label=f"after ({i} {j}) swap (mean {norm_swap_user.mean():.2e})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\|(f_0, f_1, f_2)(p)\|_2$")
    ax.set_ylabel("count")
    ax.set_title(f"({i} {j}) swap invariance: {job_id}  (N={len(norm_base_user)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
