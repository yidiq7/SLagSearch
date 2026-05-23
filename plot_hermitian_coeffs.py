"""Convert GD/GA coefficient rows into per-degree Hermitian matrices on
Sym^d(C^5) and plot heatmaps + eigenvalue spectra for visual symmetry
inspection (option A).

Convention matches helper.reconstruct_hermitian_matrices (generalized to all
degrees): a coeff row of length N^2 packs into an N x N Hermitian H with
    f(z) = sum_{A,B} H_{AB} v_A v_B_bar,
where v is the monomial vector of Sym^d(C^5), N = 5/15/35/70 for d = 1..4.
Off-diagonal H_{AB} = (Re_coeff - 1j * Im_coeff) / 2 for A<B; diagonal entries
take the full real coeff. Monomial ordering = lex with non-decreasing indices.

Usage:
    python plot_hermitian_coeffs.py --coeffs gd_runs/gd_<job>_step<N>.pkl \
        [--out_dir hermitian_plots] [--job_id <label>] [--log_scale]
"""
import argparse
import pickle
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


# Sym^d(C^5) dimension and basis-block layout — must match helper.py.
_SYM_DIM = {1: 5, 2: 15, 3: 35, 4: 70}
_BLOCK = {  # (start, stop) into the flat (3, w) row, per degree
    1: (0, 25),
    2: (25, 250),
    3: (250, 1475),
    4: (1475, 6375),
}


def monomial_labels(d: int) -> list[str]:
    """Length-N strings labelling each monomial in Sym^d(C^5).

    d=1 -> ['0','1','2','3','4'] (z_i)
    d=2 -> ['00','01',...,'44']  (z_i z_j, i<=j), 15 entries
    d=3 -> 35 entries, d=4 -> 70 entries.
    """
    return ["".join(str(i) for i in tup)
            for tup in combinations_with_replacement(range(5), d)]


def coeffs_row_to_hermitian(row: np.ndarray, N: int) -> np.ndarray:
    """Pack a real coeff row of length N^2 into an N x N Hermitian matrix H.

    Generalizes helper.reconstruct_hermitian_matrices (which is d=1 / N=5 only)
    to arbitrary N, and vectorizes it. Convention 1:
        f(z) = sum_{A,B} H_{AB} v_A v_B_bar
    => H_{AA} = b_{AA}, H_{AB} = (b_{AB} - 1j * a_{AB}) / 2 for A<B,
       H_{BA} = conj(H_{AB}). Here v is the monomial vector of size
       N = 5, 15, 35, 70 for d = 1, 2, 3, 4 respectively, and a_{AB}, b_{AB}
       are the imag / real coeffs read off the basis at positions
       triu_indices(N, k=1) / triu_indices(N, k=0).
    """
    n_imag = N * (N - 1) // 2
    n_real = N * (N + 1) // 2
    if row.shape != (n_imag + n_real,):
        raise ValueError(
            f"row shape {row.shape} does not match N={N} "
            f"(expected ({n_imag + n_real},))"
        )

    imag_coeffs = row[:n_imag]
    real_coeffs = row[n_imag:]

    iu0_r, iu0_c = np.triu_indices(N, k=0)   # A <= B (length n_real)
    iu1_r, iu1_c = np.triu_indices(N, k=1)   # A <  B (length n_imag)
    on_diag = iu0_r == iu0_c                  # diagonal entries inside iu0

    H = np.zeros((N, N), dtype=np.complex128)
    # Real part: full coeff on diagonal, /2 on strict upper.
    H[iu0_r, iu0_c] = np.where(on_diag, real_coeffs, real_coeffs * 0.5)
    # Imag part: -1j * a_AB / 2 on strict upper (convention 1).
    H[iu1_r, iu1_c] -= 1j * imag_coeffs * 0.5
    # Hermitize: fill strict lower with conjugate of strict upper.
    H = H + np.conj(np.triu(H, k=1).T)
    return H


def extract_hermitians(coeffs: np.ndarray) -> dict[int, list[np.ndarray]]:
    """coeffs: (3, w) with w in {25, 250, 1475, 6375}.
    Returns {d: [H_d^(0), H_d^(1), H_d^(2)]} for every degree present.
    """
    w = coeffs.shape[1]
    out: dict[int, list[np.ndarray]] = {}
    for d, (lo, hi) in _BLOCK.items():
        if w < hi:
            break
        N = _SYM_DIM[d]
        block = coeffs[:, lo:hi]  # (3, N^2)
        out[d] = [coeffs_row_to_hermitian(np.asarray(block[k]), N)
                  for k in range(coeffs.shape[0])]
    return out


def _load_coeffs(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "coeffs" in obj:
        arr = np.asarray(obj["coeffs"])
    else:
        arr = np.asarray(obj)
    if arr.ndim != 2 or arr.shape[0] != 3:
        raise ValueError(f"expected (3, w) coeffs, got {arr.shape}")
    return arr


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


if __name__ == "__main__":
    main()
