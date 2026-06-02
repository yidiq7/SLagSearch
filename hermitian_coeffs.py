"""Convert GD/GA coefficient rows into per-degree Hermitian matrices on
Sym^d(C^5). Pure numpy, no JAX dependency -- safe to import from analysis
scripts that don't need the geometry pipeline.

Layout:
  Each coeff row of length N^2 packs into an N x N Hermitian H with
      f(z) = sum_{A,B} H_{AB} v_A v_B_bar,
  where v is the monomial vector of Sym^d(C^5), N = 5/15/35/70 for
  d = 1..4. Off-diagonal H_{AB} = (Re_coeff - 1j * Im_coeff) / 2 for A<B;
  diagonal entries take the full real coeff. Monomial ordering = lex with
  non-decreasing indices.

Consumers: plot_hermitian_coeffs.py (CLI for heatmaps/spectra/text dumps),
permute_coeffs.py, test_permutation_symmetry.py, test_swap_invariance.py.
"""
import pickle
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np


# Sym^d(C^5) dimension and basis-block layout -- must match helper.py.
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

    Convention 1:
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
    """Load a (3, w) coeffs array from a pickle: accepts either a bare ndarray
    or a checkpoint dict with a 'coeffs' key (matches gradient_descent dumps).
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "coeffs" in obj:
        arr = np.asarray(obj["coeffs"])
    else:
        arr = np.asarray(obj)
    if arr.ndim != 2 or arr.shape[0] != 3:
        raise ValueError(f"expected (3, w) coeffs, got {arr.shape}")
    return arr
