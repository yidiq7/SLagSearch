"""Plot per-degree Hermitian heatmaps, eigenvalue spectra, and a text dump
from a GD/GA coeffs pkl. The coefficient row -> Hermitian matrix conversion
lives in hermitian_coeffs.py (pure numpy), so the symmetry tests and
permute_coeffs.py can share it without pulling in matplotlib.

Usage:
    python plot_hermitian_coeffs.py --coeffs gd_runs/gd_<job>_step<N>.pkl \
        [--out_dir hermitian_plots] [--job_id <label>] [--log_scale]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from hermitian_coeffs import (
    _SYM_DIM, _load_coeffs, extract_hermitians, monomial_labels,
)


def _tick_stride(N: int) -> int:
    """How often to draw a tick label so axes stay legible."""
    if N <= 15:
        return 1
    if N <= 35:
        return 2
    return 5


def plot_heatmaps(hermitians: dict[int, list[np.ndarray]],
                  out_path: Path,
                  log_scale: bool = False,
                  title: str = "") -> None:
    """Grid: rows = 3 equations, cols = degrees, cells = |H_d^(k)| heatmap."""
    degrees = sorted(hermitians.keys())
    n_rows = 3
    n_cols = len(degrees)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4.2 * n_rows),
        squeeze=False,
    )

    for j, d in enumerate(degrees):
        Hs = hermitians[d]
        N = _SYM_DIM[d]
        labels = monomial_labels(d)
        stride = _tick_stride(N)
        # Shared color range per column so the 3 equations are comparable.
        vmax = max(np.max(np.abs(H)) for H in Hs)
        if log_scale:
            # vmin floor at vmax * 1e-4 to keep LogNorm well-defined.
            vmin = max(vmax * 1e-4, 1e-30)
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None

        for i, H in enumerate(Hs):
            ax = axes[i, j]
            mag = np.abs(H)
            im = ax.imshow(
                mag,
                cmap="viridis",
                norm=norm,
                vmin=None if log_scale else 0.0,
                vmax=None if log_scale else vmax,
                interpolation="nearest",
                origin="upper",
            )
            ax.set_title(
                f"d={d}, eq {i}  "
                f"||H||_F = {np.linalg.norm(H, 'fro'):.3g}",
                fontsize=10,
            )
            ax.set_xticks(np.arange(0, N, stride))
            ax.set_yticks(np.arange(0, N, stride))
            ax.set_xticklabels([labels[k] for k in range(0, N, stride)],
                               rotation=90, fontsize=7)
            ax.set_yticklabels([labels[k] for k in range(0, N, stride)],
                               fontsize=7)
            ax.set_xlabel("monomial B")
            ax.set_ylabel("monomial A")
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(title or "|H_d^(k)| heatmaps", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spectra(hermitians: dict[int, list[np.ndarray]],
                 out_path: Path,
                 title: str = "") -> None:
    """Grid: rows = 3 equations, cols = degrees, cells = sorted eigenvalues."""
    degrees = sorted(hermitians.keys())
    n_rows = 3
    n_cols = len(degrees)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for j, d in enumerate(degrees):
        Hs = hermitians[d]
        for i, H in enumerate(Hs):
            ax = axes[i, j]
            # eigvalsh returns ascending real eigenvalues for Hermitian H.
            eigs = np.linalg.eigvalsh(H)
            order = np.argsort(-np.abs(eigs))
            eigs_sorted = eigs[order]
            ax.bar(np.arange(len(eigs_sorted)), eigs_sorted,
                   color=["steelblue" if e >= 0 else "indianred"
                          for e in eigs_sorted])
            ax.axhline(0.0, color="k", lw=0.5)
            ax.set_title(
                f"d={d}, eq {i}  "
                f"#nz>1%: {int(np.sum(np.abs(eigs) > 0.01 * np.max(np.abs(eigs))))}"
                f" / {len(eigs)}",
                fontsize=10,
            )
            ax.set_xlabel("eigenvalue index (by |λ| desc)")
            ax.set_ylabel("λ")
            ax.grid(True, alpha=0.3)

    fig.suptitle(title or "eigenvalue spectra", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _format_complex(z: complex, precision: int) -> str:
    """Format a complex number as 'a+bj' / 'a-bj' with fixed precision and width.

    Imag part is suppressed to '' when |Im| < 10**-precision; real-only diagonal
    entries print as plain 'a' with no trailing sign.
    """
    a = z.real
    b = z.imag
    tol = 10.0 ** (-precision)
    width = precision + 7  # sign + 1 digit + '.' + precision + a little slack
    if abs(b) < tol:
        return f"{a:>{width}.{precision}f}"
    sign = "+" if b >= 0 else "-"
    return f"{a:>{width}.{precision}f}{sign}{abs(b):.{precision}f}j"


def write_matrices_text(hermitians: dict[int, list[np.ndarray]],
                        out_path: Path,
                        precision: int = 4,
                        also_print: bool = True) -> None:
    """Write every Hermitian matrix to a single text file with monomial labels.

    Each matrix is preceded by a header and the column-label row. The diagonal
    is real by construction; off-diagonals print as 'a+bj' / 'a-bj'. The lower
    triangle is omitted (it is the conjugate of the upper triangle, and showing
    both is just clutter).
    """
    lines: list[str] = []
    for d in sorted(hermitians):
        labels = monomial_labels(d)
        N = len(labels)
        label_width = max(len(labels[0]), precision + 8)
        col_header_width = max(precision + 8, len(labels[0]) + 2)
        for k, H in enumerate(hermitians[d]):
            lines.append("")
            lines.append(f"=== d={d}, eq {k}  "
                         f"(N={N}, ||H||_F={np.linalg.norm(H, 'fro'):.4g}) ===")
            # Column header.
            header_cells = [f"{lbl:>{col_header_width}}" for lbl in labels]
            lines.append(" " * (label_width + 2) + " ".join(header_cells))
            # Rows: only upper triangle (A <= B); lower is the conjugate.
            for A in range(N):
                cells: list[str] = []
                for B in range(N):
                    if B < A:
                        cells.append(" " * col_header_width)
                    else:
                        cells.append(
                            f"{_format_complex(H[A, B], precision):>{col_header_width}}"
                        )
                lines.append(f"{labels[A]:>{label_width}}  " + " ".join(cells))

    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    if also_print:
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coeffs", required=True, type=Path,
                        help="Path to a coeff pkl (bare (3,w) array or dict "
                             "with 'coeffs' key, e.g. a GD checkpoint).")
    parser.add_argument("--out_dir", type=Path, default=Path("hermitian_plots"),
                        help="Directory to write PNGs into.")
    parser.add_argument("--job_id", type=str, default=None,
                        help="Label for filenames; defaults to coeffs stem.")
    parser.add_argument("--log_scale", action="store_true",
                        help="Use log color scale for heatmaps.")
    parser.add_argument("--normalize", action="store_true",
                        help="Frobenius-normalize each H before plotting "
                             "(makes magnitudes comparable across degrees).")
    parser.add_argument("--text_precision", type=int, default=4,
                        help="Decimal places for the text matrix dump.")
    parser.add_argument("--no_print", action="store_true",
                        help="Save text dump to file but don't echo to stdout.")
    args = parser.parse_args()

    coeffs = _load_coeffs(args.coeffs)
    print(f"Loaded coeffs of shape {coeffs.shape} from {args.coeffs}")

    hermitians = extract_hermitians(coeffs)
    print(f"Extracted Hermitian matrices for degrees: {sorted(hermitians)}")
    for d, Hs in hermitians.items():
        for k, H in enumerate(Hs):
            herm_err = np.max(np.abs(H - H.conj().T))
            print(f"  d={d}, eq {k}: shape={H.shape}, "
                  f"||H||_F={np.linalg.norm(H, 'fro'):.4g}, "
                  f"||H - H^dagger||_inf={herm_err:.2e}")

    if args.normalize:
        hermitians = {
            d: [H / max(np.linalg.norm(H, "fro"), 1e-30) for H in Hs]
            for d, Hs in hermitians.items()
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    job_id = args.job_id or args.coeffs.stem

    heatmap_path = args.out_dir / f"hermitian_heatmaps_{job_id}.png"
    spectra_path = args.out_dir / f"hermitian_spectra_{job_id}.png"
    text_path = args.out_dir / f"hermitian_matrices_{job_id}.txt"
    npz_path = args.out_dir / f"hermitian_matrices_{job_id}.npz"

    write_matrices_text(
        hermitians, text_path,
        precision=args.text_precision,
        also_print=not args.no_print,
    )
    np.savez(
        npz_path,
        **{f"d{d}_eq{k}": H
           for d, Hs in hermitians.items() for k, H in enumerate(Hs)},
    )

    plot_heatmaps(
        hermitians, heatmap_path,
        log_scale=args.log_scale,
        title=f"|H_d^(k)|  ({job_id}"
              + (", normalized" if args.normalize else "")
              + (", log scale" if args.log_scale else "")
              + ")",
    )
    plot_spectra(
        hermitians, spectra_path,
        title=f"eigenvalue spectra of H_d^(k)  ({job_id}"
              + (", normalized" if args.normalize else "")
              + ")",
    )

    print(f"Wrote {heatmap_path}")
    print(f"Wrote {spectra_path}")
    print(f"Wrote {text_path}")
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
