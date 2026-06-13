"""Write per-degree truncations of a coeffs pkl: coeffs[:, :w] for each
max-degree w below the input's own, with rows re-normalized (the same
convention training and the GD truncation plots use).

Each output is saved as {"coeffs": <(3, w) array>} so it works directly with
viz.fitness_pipeline, plot_hermitian_coeffs, gradient_descent --init_pkl, and
every other consumer of coeffs pkls. Typical use: fitness-vs-degree
comparisons of a single result,

    python -m diagnostics.make_coeffs_truncations --coeffs <dir>/coeffs.pkl
    python -m viz.fitness_pipeline --coeffs <dir>/coeffs_d1.pkl --out_subdir fitness_d1
    python -m viz.fitness_pipeline --coeffs <dir>/coeffs_d2.pkl --out_subdir fitness_d2
    ...then overlay the run folders with viz.plot_histograms.

Usage:
    python -m diagnostics.make_coeffs_truncations --coeffs <pkl> \
        [--degrees 1 2 ...] [--no-normalize] \
        [--out_dir <dir> | --out_subdir <name>]

Output: <stem>_d<n>.pkl (meaning "truncated to max_degree n") next to the
input pkl by default; --out_dir (full path) and --out_subdir (relative to the
input's parent) are mutually exclusive.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from hermitian_coeffs import _BLOCK, _load_coeffs

# max_degree -> truncated width, from the basis-block layout.
_WIDTH = {d: hi for d, (lo, hi) in _BLOCK.items()}
_DEGREE_OF_WIDTH = {hi: d for d, hi in _WIDTH.items()}


def truncate_coeffs(coeffs: np.ndarray, degree: int,
                    normalize: bool = True) -> np.ndarray:
    """coeffs[:, :w] for max_degree `degree`, rows re-normalized by default."""
    out = np.asarray(coeffs)[:, :_WIDTH[degree]].copy()
    if normalize:
        out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coeffs", required=True, type=Path,
                        help="Path to a (3, w) coeffs pkl (bare array or "
                             "checkpoint dict with a 'coeffs' key).")
    parser.add_argument("--degrees", type=int, nargs="+", default=None,
                        choices=[1, 2, 3],
                        help="Max-degrees to truncate to. Default: every "
                             "degree below the input's own.")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Keep raw sliced rows instead of re-normalizing "
                             "each row to unit norm.")
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument("--out_dir", type=Path, default=None,
                           help="Full output directory. Default: the input "
                                "pkl's parent dir.")
    out_group.add_argument("--out_subdir", type=str, default=None,
                           help="Subdir of --coeffs's parent dir.")
    args = parser.parse_args()

    coeffs = np.asarray(_load_coeffs(args.coeffs))
    if coeffs.shape[1] not in _DEGREE_OF_WIDTH:
        raise ValueError(f"coeffs width {coeffs.shape[1]} is not one of the "
                         f"ansatz widths {sorted(_DEGREE_OF_WIDTH)}")
    max_d = _DEGREE_OF_WIDTH[coeffs.shape[1]]
    print(f"Loaded coeffs {coeffs.shape} (max_degree {max_d}) "
          f"from {args.coeffs}")

    degrees = args.degrees if args.degrees is not None else list(range(1, max_d))
    bad = [d for d in degrees if d >= max_d]
    if bad:
        raise ValueError(f"degrees {bad} are not below the input's "
                         f"max_degree {max_d}")

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.out_subdir is not None:
        out_dir = args.coeffs.parent / args.out_subdir
    else:
        out_dir = args.coeffs.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    full_norms = np.linalg.norm(coeffs, axis=1)
    for d in sorted(degrees):
        sliced = np.asarray(coeffs)[:, :_WIDTH[d]]
        kept = np.linalg.norm(sliced, axis=1) / full_norms
        out = truncate_coeffs(coeffs, d, normalize=not args.no_normalize)
        path = out_dir / f"{args.coeffs.stem}_d{d}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"coeffs": out}, f)
        kept_str = ", ".join(f"{x:.3f}" for x in kept)
        print(f"  wrote {path}  shape {out.shape}  "
              f"(rows keep [{kept_str}] of full norm"
              f"{'' if args.no_normalize else '; re-normalized'})")

    print("\nNext steps (fitness per degree, then overlay):")
    for d in sorted(degrees):
        print(f"  python -m viz.fitness_pipeline --coeffs "
              f"{out_dir / (args.coeffs.stem + f'_d{d}.pkl')} "
              f"--out_subdir fitness_d{d}")
    print("  python -m viz.plot_histograms --runs <the run folders> "
          "--labels d=1 d=2 ... --out_dir <compare_dir>")


if __name__ == "__main__":
    main()
