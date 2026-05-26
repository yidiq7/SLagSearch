"""Persistent homology of points in CP^4 via the (weak) witness complex.

Sister script to persistent_homology_gtda_h0h1.py (Vietoris-Rips). Witness
complexes scale better in N, so this script targets N ~ 50k with a sweep
over the number of landmarks L. The sweep doubles as a stability check:
if the diagram is sample-stable, the persistent H0/H1 features should be
consistent across L.

Pipeline:
  1. Load points; optionally Newton-refine + drop points whose per-point
     Newton-step residual exceeds `--newton_threshold` (delegates to the
     slag pipeline's filter_and_refine with k=N, n_repulsion_steps=0;
     threshold semantics match newton_check at
     find_smooth_submanifold.py:418).
  2. Max-min landmark selection up to L_max in the Fubini-Study metric
     (JAX, runs on GPU). Builds a single (L_max, N) landmark-to-witness
     distance table; the sweep slices it (max-min is prefix-monotone).
  3. For each L in --landmarks: build gudhi.WitnessComplex on the first L
     landmarks, compute persistence, extract H0/H1.
  4. Plot: per-L PH figure (mirrors VR script layout) + a combined
     across-L comparison figure (H0/H1 diagrams + overlaid Betti curves).

Units: gudhi's WitnessComplex consumes a `nearest_landmark_table` whose
entries are *squared* distances, and its filtration is alpha^2. We pass
squared FS distances in, then sqrt the persistence values on the way out
so the diagrams are in raw FS-distance units (comparable to the VR script).

Usage:
    # d=4 coeffs, 50k subsample, default L sweep:
    uv run python persistent_homology/persistent_homology_witness_h0h1.py \
        --filepath plots_slag_d4_run/min_set.pkl \
        --coeffs_pkl gd_runs/gd_d4_run_step3000.pkl --psi 0

    # Custom L sweep and Newton step count:
    uv run python persistent_homology/persistent_homology_witness_h0h1.py \
        --filepath plots_slag_d4_run/min_set.pkl \
        --coeffs_pkl gd_runs/gd_d4_run_step3000.pkl \
        --landmarks 500,1000,2000 --newton_steps 100

    # Skip the Newton filter (PH on raw min_set):
    uv run python persistent_homology/persistent_homology_witness_h0h1.py \
        --filepath plots_slag_d4_run/min_set.pkl --no_newton_filter
"""

import argparse
import os
import pickle
import sys
import time
import warnings

import numpy as np

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster
import matplotlib.pyplot as plt

# Match the slag pipeline's FP64 convention (gradient_descent.py).
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

# Make repo-root modules (find_smooth_submanifold, helper, ...) importable when
# this script is launched from inside persistent_homology/. Same pattern as
# points_gen/points_generation.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from find_smooth_submanifold import filter_and_refine

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------- JAX kernels

def _normalize_rows_complex(Z):
    """Row-normalize complex (N, m) to unit norm. numpy in, numpy out."""
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


@jax.jit
def _fs_distance_from_index(Zn, idx):
    """FS distance from Zn[idx] (one row) to every row of Zn.

    One BLAS matvec on GPU + arccos. Zn assumed already unit-norm.
    """
    inner = Zn @ jnp.conj(Zn[idx])
    return jnp.arccos(jnp.clip(jnp.abs(inner), 0.0, 1.0))


# ------------------------------------------------------------------- data layer

def load_points(filepath, subsamp, seed):
    print("=== LOADING POINTS FROM PICKLE ===")
    with open(filepath, 'rb') as f:
        Z = pickle.load(f)
    Z = np.asarray(Z, dtype=np.complex128)
    n0 = Z.shape[0]
    print(f"Loaded {n0} points")

    max_imag = np.max(np.abs(np.imag(Z)))
    print(f"Maximum imaginary part: {max_imag:.3e}")
    if max_imag <= 1e-10:
        print("Warning: points are real; treating as complex with zero imaginary part.")

    if n0 > subsamp:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n0, subsamp, replace=False)
        Z = Z[idx]
        print(f"Subsampled to {len(Z)} (seed={seed})")
    else:
        print(f"Using all {n0} points (no subsample)")
    return Z


def filter_newton_check(Z_complex, coeffs_pkl, psi, n_steps, threshold,
                        dist_chunk_size=50000):
    """Newton-refine + per-point residual filter via slag's filter_and_refine.

    Thin wrapper: load coeffs, complex->real, call filter_and_refine with
    k=N and n_repulsion_steps=0 (so it's just refine + per-point residual,
    no top-k culling, no repulsion), then threshold by `threshold` per-point.
    Threshold semantics match newton_check (find_smooth_submanifold.py:418).
    """
    print(f"\n=== NEWTON_CHECK FILTER ({n_steps} steps, threshold={threshold:.1e}) ===")
    with open(coeffs_pkl, 'rb') as f:
        coeffs_obj = pickle.load(f)
    if isinstance(coeffs_obj, dict) and 'coeffs' in coeffs_obj:
        coeffs_np = np.asarray(coeffs_obj['coeffs'])
    else:
        coeffs_np = np.asarray(coeffs_obj)
    print(f"  coeffs shape: {coeffs_np.shape}")

    Z_real = np.concatenate([Z_complex.real, Z_complex.imag], axis=1).astype(np.float64)
    N = Z_real.shape[0]

    t0 = time.time()
    final_points, final_distances, _ = filter_and_refine(
        jnp.asarray(Z_real),
        jnp.asarray(coeffs_np),
        jnp.asarray(complex(psi)),
        k=N,
        n_refine_steps=n_steps,
        filter_newton=False,
        n_repulsion_steps=0,
        dist_chunk_size=dist_chunk_size,
    )
    distances_np = np.asarray(final_distances)
    points_real_np = np.asarray(final_points)
    print(f"  filter_and_refine: {time.time() - t0:.1f}s")

    finite = np.isfinite(distances_np)
    mask = (distances_np <= threshold) & finite
    n_drop = int((~mask).sum())
    print(f"  residual stats: min={distances_np[finite].min():.3e}  "
          f"median={np.median(distances_np[finite]):.3e}  "
          f"max={distances_np[finite].max():.3e}")
    print(f"  dropped: {n_drop} / {N}")

    Z_filtered_real = points_real_np[mask]
    Z_filtered_complex = Z_filtered_real[:, :5] + 1j * Z_filtered_real[:, 5:]
    return Z_filtered_complex, mask, distances_np


# ------------------------------------------------------------ max-min landmarks

def maxmin_landmarks(Zn_np, L_max, seed):
    """Farthest-point sampling in FS metric (GPU-accelerated).

    The inner FS-distance call is JIT'd on GPU; the host-side Python loop
    just picks the next landmark via argmax and updates the running min.
    JAX arrays persist on device between iterations, so there's no per-step
    transfer beyond the ~1-int argmax sync.

    Returns:
        lm_idx: (L_max,) int64 numpy array of landmark indices in Zn_np.
        dist_table: (L_max, N) float64 numpy array; dist_table[l, j] = FS
                    distance from landmark l to point j.
    """
    n = Zn_np.shape[0]
    Zn = jnp.asarray(Zn_np)
    rng = np.random.default_rng(seed)
    lm_idx = np.empty(L_max, dtype=np.int64)
    dist_rows = []
    min_to_lm = jnp.full(n, jnp.inf)

    lm_idx[0] = int(rng.integers(0, n))
    for ell in range(L_max):
        if ell > 0:
            lm_idx[ell] = int(jnp.argmax(min_to_lm))
        row = _fs_distance_from_index(Zn, int(lm_idx[ell]))
        dist_rows.append(row)
        min_to_lm = jnp.minimum(min_to_lm, row)

    dist_table = np.asarray(jnp.stack(dist_rows, axis=0))
    return lm_idx, dist_table


# ------------------------------------------------------------- witness complex

def build_witness_diagram(dist_table_slice, max_alpha_square,
                          limit_dimension=2, top_k_landmarks=50):
    """gudhi WitnessComplex on a precomputed landmark-to-witness distance table.

    Args:
        dist_table_slice: (L, N) raw FS distances, landmark -> witness.
        max_alpha_square: filtration cap in squared-distance units.
        limit_dimension: max simplex dim (2 sufficient for H0/H1).
        top_k_landmarks: keep only the top-K nearest landmarks per witness
                         when building the table (saves memory; the witness
                         complex only needs ~limit_dimension+2 nearest).

    Returns:
        (H0, H1): each (n_features, 2) array of (birth, death) in *raw* FS
                  distance (filtration values sqrt'd from gudhi's alpha^2).
    """
    import gudhi

    L, N = dist_table_slice.shape
    K = min(top_k_landmarks, L)
    dt = dist_table_slice.T  # (N, L)

    nearest = []
    for j in range(N):
        if K < L:
            cand = np.argpartition(dt[j], K - 1)[:K]
            order = cand[np.argsort(dt[j, cand])]
        else:
            order = np.argsort(dt[j])
        d_sq = dt[j, order] ** 2
        nearest.append([(int(order[k]), float(d_sq[k])) for k in range(K)])

    wc = gudhi.WitnessComplex(nearest_landmark_table=nearest)
    st = wc.create_simplex_tree(
        max_alpha_square=max_alpha_square,
        limit_dimension=limit_dimension,
    )
    persistence = st.persistence()

    H0, H1 = [], []
    for dim, (b, d) in persistence:
        b_raw = np.sqrt(max(b, 0.0)) if np.isfinite(b) else 0.0
        d_raw = np.inf if np.isinf(d) else np.sqrt(max(d, 0.0))
        if dim == 0:
            H0.append([b_raw, d_raw])
        elif dim == 1:
            H1.append([b_raw, d_raw])
    return (np.array(H0, dtype=np.float64) if H0 else np.empty((0, 2)),
            np.array(H1, dtype=np.float64) if H1 else np.empty((0, 2)))


# ------------------------------------------------------------------- analysis

def betti_at(points, t):
    if len(points) == 0:
        return 0
    return int(np.sum((points[:, 0] <= t) & (points[:, 1] > t)))


def analyze(H0, H1, infinity_val, label=""):
    print(f"\n=== TOPOLOGICAL FEATURES {label} ===")
    print(f"H0 (connected components): {len(H0)}")
    print(f"H1 (loops):                {len(H1)}")

    filt = (np.array([0.2, 0.3, 0.4, 0.5]) * (infinity_val / (np.pi / 2))).tolist()
    print("Betti at sampled filtration values:")
    for t in filt:
        print(f"  r={t:.3f}: beta0={betti_at(H0, t)}, beta1={betti_at(H1, t)}")

    def top_features(points, name):
        if len(points) == 0:
            return
        pers = points[:, 1] - points[:, 0]
        order = np.argsort(pers)[-5:][::-1]
        print(f"{name} top 5 by persistence:")
        for i in order:
            print(f"  birth={points[i, 0]:.4f}  death={points[i, 1]:.4f}  pers={pers[i]:.4f}")

    top_features(H0, "H0")
    top_features(H1, "H1")


# ------------------------------------------------------------------- plotting

def _clip_inf(p, infinity_val):
    if len(p) == 0:
        return p
    out = p.copy()
    out[np.isinf(out[:, 1]), 1] = infinity_val
    return out


def plot_one_L(H0, H1, infinity_val, n_sample, L, output_file):
    """Per-L PH figure mirroring the VR script's 2x4 layout."""
    print(f"  plotting L={L} -> {output_file}")
    h0_color = 'skyblue'
    h1_color = 'orange'

    H0d = _clip_inf(H0, infinity_val)
    H1d = _clip_inf(H1, infinity_val)
    H0f = H0[np.isfinite(H0[:, 1])] if len(H0) else H0
    H1f = H1[np.isfinite(H1[:, 1])] if len(H1) else H1

    fig = plt.figure(figsize=(16, 10))

    def diag_ax(ax, pts, color, name):
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], alpha=0.7, s=30, c=color)
            ax.plot([0, infinity_val], [0, infinity_val], 'k--', alpha=0.3)
        ax.set_xlabel('Birth (FS distance)')
        ax.set_ylabel('Death (FS distance)')
        ax.set_title(f'{name} Persistence Diagram\n({len(pts)} features)')
        ax.set_xlim([0, infinity_val])
        ax.set_ylim([0, infinity_val * 1.05])
        ax.grid(True, linestyle='--', alpha=0.6)

    ax1 = plt.subplot(2, 4, 1)
    diag_ax(ax1, H0d, h0_color, 'H0')
    ax2 = plt.subplot(2, 4, 2)
    diag_ax(ax2, H1d, h1_color, 'H1')

    ax3 = plt.subplot(2, 4, 3)
    if len(H0):
        ax3.scatter(H0d[:, 0], H0d[:, 1], alpha=0.7, s=30, c=h0_color, label='H0')
    if len(H1):
        ax3.scatter(H1d[:, 0], H1d[:, 1], alpha=0.7, s=30, c=h1_color, label='H1')
    ax3.plot([0, infinity_val], [0, infinity_val], 'k--', alpha=0.3)
    ax3.set_xlim([0, infinity_val])
    ax3.set_ylim([0, infinity_val * 1.05])
    ax3.set_xlabel('Birth')
    ax3.set_ylabel('Death')
    ax3.set_title('Combined Persistence Diagram')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    ax4 = plt.subplot(2, 4, 4)
    all_p, labels, cols = [], [], []
    if len(H0f):
        all_p.append(H0f[:, 1] - H0f[:, 0])
        labels.append('H0')
        cols.append(h0_color)
    if len(H1f):
        all_p.append(H1f[:, 1] - H1f[:, 0])
        labels.append('H1')
        cols.append(h1_color)
    if all_p:
        ax4.hist(all_p, bins=30, label=labels, color=cols, alpha=0.7)
    ax4.set_xlabel('Persistence')
    ax4.set_ylabel('Count')
    ax4.set_title('Persistence Distribution')
    if all_p:
        ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.6)

    def plot_barcode(ax, pts, color, title):
        if len(pts) == 0:
            ax.set_title(title)
            return
        order = np.argsort(pts[:, 1] - pts[:, 0])[::-1]
        sorted_pts = pts[order]
        n_bars = min(50, len(sorted_pts))
        for i in range(n_bars):
            b, d = sorted_pts[i]
            if np.isinf(d):
                ax.barh(i, infinity_val - b, left=b, color=color, alpha=0.8,
                        edgecolor='red', linewidth=2)
            else:
                ax.barh(i, d - b, left=b, color=color, alpha=0.7)
        ax.set_xlim([0, infinity_val])
        ax.set_ylim([-1, n_bars])
        ax.set_xlabel('Filtration Value')
        ax.set_ylabel('Feature Index')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6, axis='x')

    ax5 = plt.subplot(2, 4, 5)
    plot_barcode(ax5, H0, h0_color, f'H0 Barcode (top {min(50, len(H0))})')
    ax6 = plt.subplot(2, 4, 6)
    plot_barcode(ax6, H1, h1_color, f'H1 Barcode (top {min(50, len(H1))})')

    ax7 = plt.subplot(2, 4, 7)
    ax8 = plt.subplot(2, 4, 8)
    t_grid = np.linspace(0, infinity_val, 100)
    for ax, dim, color, pts in zip([ax7, ax8], [0, 1],
                                   [h0_color, h1_color], [H0, H1]):
        if len(pts):
            bv = [betti_at(pts, t) for t in t_grid]
            ax.plot(t_grid, bv, color=color, linewidth=2)
            ax.fill_between(t_grid, bv, alpha=0.3, color=color)
        ax.set_xlim([0, infinity_val])
        ax.set_xlabel('Filtration Value')
        ax.set_ylabel(f'beta_{dim}')
        ax.set_title(f'H{dim} Betti Curve')
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(
        f'Witness Complex Persistence (FS metric)  |  '
        f'N={n_sample}, L={L}',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(per_L, infinity_val, n_sample, output_file):
    """Across-L comparison: H0 diagrams (row 1), H1 diagrams (row 2),
    overlaid Betti curves (row 3)."""
    Ls = sorted(per_L.keys())
    n_L = len(Ls)
    fig = plt.figure(figsize=(4 * n_L, 12))
    cmap_L = plt.cm.viridis(np.linspace(0.1, 0.9, n_L))

    # Row 1: H0 diagrams
    for ci, L in enumerate(Ls):
        H0, _ = per_L[L]
        ax = plt.subplot(3, n_L, ci + 1)
        H0d = _clip_inf(H0, infinity_val)
        if len(H0d):
            ax.scatter(H0d[:, 0], H0d[:, 1], s=15, alpha=0.7, c='steelblue')
            ax.plot([0, infinity_val], [0, infinity_val], 'k--', alpha=0.3)
        ax.set_title(f'H0  (L={L}, n_feat={len(H0)})')
        ax.set_xlim([0, infinity_val])
        ax.set_ylim([0, infinity_val * 1.05])
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Row 2: H1 diagrams
    for ci, L in enumerate(Ls):
        _, H1 = per_L[L]
        ax = plt.subplot(3, n_L, n_L + ci + 1)
        H1d = _clip_inf(H1, infinity_val)
        if len(H1d):
            ax.scatter(H1d[:, 0], H1d[:, 1], s=15, alpha=0.7, c='darkorange')
            ax.plot([0, infinity_val], [0, infinity_val], 'k--', alpha=0.3)
        ax.set_title(f'H1  (L={L}, n_feat={len(H1)})')
        ax.set_xlim([0, infinity_val])
        ax.set_ylim([0, infinity_val * 1.05])
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Row 3: overlaid Betti curves, all L on one panel per dim
    ax_b0 = plt.subplot(3, 2, 5)
    ax_b1 = plt.subplot(3, 2, 6)
    t_grid = np.linspace(0, infinity_val, 200)
    for L, color in zip(Ls, cmap_L):
        H0, H1 = per_L[L]
        if len(H0):
            ax_b0.plot(t_grid, [betti_at(H0, t) for t in t_grid],
                       color=color, label=f'L={L}', linewidth=2)
        if len(H1):
            ax_b1.plot(t_grid, [betti_at(H1, t) for t in t_grid],
                       color=color, label=f'L={L}', linewidth=2)
    for ax, dim, name in [(ax_b0, 0, 'H0'), (ax_b1, 1, 'H1')]:
        ax.set_xlim([0, infinity_val])
        ax.set_xlabel('Filtration Value')
        ax.set_ylabel(f'beta_{dim}')
        ax.set_title(f'{name} Betti Curves across L (stability check)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(
        f'Witness Landmark Sweep  |  N={n_sample}, FS metric',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved comparison: {output_file}")


# ----------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--filepath',
                   default='/home/y.qi/projects/SLagSearch/plots_slag/min_set_psi0.pkl')
    p.add_argument('--subsamp', type=int, default=50000)
    p.add_argument('--landmarks', default='300,500,750,1000',
                   help='Comma-separated L values to sweep.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--coeffs_pkl', default=None,
                   help='Coeffs pkl that produced this min_set (3, w) array, '
                        'or a checkpoint dict with a "coeffs" key. Required '
                        'unless --no_newton_filter is set.')
    p.add_argument('--psi', default='0',
                   help='Dwork parameter (complex). Examples: 0, 10, "1+2j".')
    p.add_argument('--newton_steps', type=int, default=70,
                   help='Newton refinement steps before residual check '
                        '(0 to skip refinement; matches filter_and_refine '
                        'n_refine_steps semantics).')
    p.add_argument('--newton_threshold', type=float, default=1e-4,
                   help='Per-point residual cutoff for filtering. Matches '
                        'newton_check (find_smooth_submanifold.py:418).')
    p.add_argument('--dist_chunk_size', type=int, default=50000,
                   help='Passthrough to filter_and_refine dist_chunk_size.')
    p.add_argument('--no_newton_filter', action='store_true',
                   help='Skip the Newton-residual filter entirely.')
    p.add_argument('--top_k_landmarks', type=int, default=50,
                   help='Keep top-K nearest landmarks per witness in the '
                        'gudhi table (>= limit_dimension+2; 50 is generous).')
    p.add_argument('--out_prefix', default='persistent_homology_witness_h0h1')
    p.add_argument('--cache_landmarks', default='witness_landmarks_cache.pkl')
    p.add_argument('--cache_diagrams', default='witness_diagrams_cache.pkl')
    p.add_argument('--no_cache', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    st = time.time()

    L_values = sorted({int(x) for x in args.landmarks.split(',') if x.strip()})
    L_max = max(L_values)
    print(f"Sweep: L values = {L_values}, L_max = {L_max}")

    # ---- load + filter
    Z = load_points(args.filepath, args.subsamp, args.seed)
    if args.no_newton_filter:
        print("\nNewton-residual filter disabled (--no_newton_filter).")
    else:
        if args.coeffs_pkl is None:
            raise SystemExit(
                "ERROR: --coeffs_pkl is required to run the Newton filter. "
                "Pass --no_newton_filter to skip filtering entirely."
            )
        psi_val = complex(args.psi)
        Z, _, _ = filter_newton_check(
            Z,
            coeffs_pkl=args.coeffs_pkl,
            psi=psi_val,
            n_steps=args.newton_steps,
            threshold=args.newton_threshold,
            dist_chunk_size=args.dist_chunk_size,
        )
    Zn = _normalize_rows_complex(Z)
    n_sample = len(Z)
    print(f"\nN after filter: {n_sample}")

    # ---- landmarks (cached on disk)
    filter_sig = {
        'no_newton_filter': args.no_newton_filter,
        'coeffs_pkl': args.coeffs_pkl,
        'psi': args.psi,
        'newton_steps': args.newton_steps,
        'newton_threshold': args.newton_threshold,
    }
    use_lm_cache = False
    if not args.no_cache and os.path.exists(args.cache_landmarks):
        with open(args.cache_landmarks, 'rb') as f:
            lm_data = pickle.load(f)
        if (lm_data.get('filepath') == args.filepath
                and lm_data.get('subsamp') == args.subsamp
                and lm_data.get('seed') == args.seed
                and lm_data.get('filter_sig') == filter_sig
                and lm_data.get('n_sample') == n_sample
                and lm_data.get('L_max', 0) >= L_max):
            print(f"\nLoading cached landmarks/dist_table from "
                  f"'{args.cache_landmarks}'")
            lm_idx = lm_data['lm_idx'][:L_max]
            dist_table = lm_data['dist_table'][:L_max]
            use_lm_cache = True
        else:
            print(f"\nCache '{args.cache_landmarks}' params mismatch; "
                  f"recomputing landmarks.")
    if not use_lm_cache:
        print(f"\n=== MAX-MIN LANDMARK SELECTION (L_max={L_max}) ===")
        t0 = time.time()
        lm_idx, dist_table = maxmin_landmarks(Zn, L_max, args.seed)
        print(f"  done in {time.time() - t0:.1f}s on device {jax.devices()[0]}")
        if not args.no_cache:
            with open(args.cache_landmarks, 'wb') as f:
                pickle.dump({
                    'filepath': args.filepath, 'subsamp': args.subsamp,
                    'seed': args.seed, 'filter_sig': filter_sig,
                    'n_sample': n_sample, 'L_max': L_max,
                    'lm_idx': lm_idx, 'dist_table': dist_table,
                }, f)
            print(f"  cached: {args.cache_landmarks}")

    infinity_val = float(np.max(dist_table))
    max_alpha_sq = infinity_val ** 2
    print(f"\nMax FS distance in landmark/witness pairs: {infinity_val:.4f}")
    print(f"max_alpha_square (filtration cap):         {max_alpha_sq:.4f}")
    print(f"(reference: FS diameter pi/2 = {np.pi / 2:.4f})")

    # ---- witness diagrams per L
    per_L = {}
    for L in L_values:
        print(f"\n=== WITNESS COMPLEX (L={L}) ===")
        t0 = time.time()
        H0, H1 = build_witness_diagram(
            dist_table[:L], max_alpha_sq,
            limit_dimension=2, top_k_landmarks=args.top_k_landmarks,
        )
        print(f"  built in {time.time() - t0:.1f}s; "
              f"H0={len(H0)}, H1={len(H1)}")

        # Match the VR script: ensure the essential H0 is present.
        if len(H0) and not np.any(np.isinf(H0[:, 1])):
            H0 = np.vstack([H0, [0.0, np.inf]])
            print("  injected essential H0 (birth=0, death=inf)")
        per_L[L] = (H0, H1)

        analyze(H0, H1, infinity_val, label=f'(L={L})')
        png = f'{args.out_prefix}_L{L}.png'
        plot_one_L(H0, H1, infinity_val, n_sample, L, png)

    # ---- cache diagrams
    if not args.no_cache:
        with open(args.cache_diagrams, 'wb') as f:
            pickle.dump({
                'per_L': per_L,
                'infinity_val': infinity_val,
                'n_sample': n_sample,
                'L_values': L_values,
                'filepath': args.filepath,
                'subsamp': args.subsamp,
                'seed': args.seed,
                'filter_sig': filter_sig,
            }, f)
        print(f"\nCached diagrams: {args.cache_diagrams}")

    # ---- comparison plot
    plot_comparison(per_L, infinity_val, n_sample,
                    f'{args.out_prefix}_sweep.png')

    print(f"\n=== DONE in {time.time() - st:.1f}s ===")


if __name__ == '__main__':
    main()
