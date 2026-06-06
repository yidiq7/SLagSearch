"""Persistent homology of points in CP^4 via the (weak) witness complex.

Computes H0/H1/H2 over a landmark sweep. The witness complex scales as L
(landmarks), not N (points), so this script targets N ~ 50k with H2
tractable thanks to a filtration cap (--max_alpha) that bounds simplex
counts at higher dimensions. The L sweep doubles as a stability check.

Pipeline:
  1. Load points; optionally Newton-refine + drop points whose per-point
     Newton-step residual exceeds `--newton_threshold` (delegates to the
     slag pipeline's filter_and_refine with k=N, n_repulsion_steps=0;
     threshold semantics match newton_check at
     find_smooth_submanifold.py:418).
  2. Max-min landmark selection up to L_max in the Fubini-Study metric
     (JAX, runs on GPU). Builds a single (L_max, N) landmark-to-witness
     distance table; the sweep slices it (max-min is prefix-monotone).
  3. For each L in --landmarks: build gudhi.WitnessComplex with
     limit_dimension=3 (tetrahedra for H2), compute persistence, extract
     H0/H1/H2.
  4. Plot: per-L PH figure (3x3 grid: rows = H0/H1/H2, cols = diagram /
     barcode / Betti curve) + a combined across-L comparison figure.

Units: gudhi's WitnessComplex consumes a `nearest_landmark_table` whose
entries are *squared* distances, and its filtration is alpha^2. We pass
squared FS distances in, then sqrt the persistence values on the way out
so the diagrams are in raw FS-distance units (comparable to the VR script).

Usage:
    # d=4 coeffs, 50k subsample, default L sweep:
    uv run python persistent_homology/persistent_homology_witness.py \
        --min_set plots_slag_d4_run/min_set.pkl \
        --coeffs gd_runs/gd_d4_run_step3000.pkl --psi 0

    # Custom L sweep and Newton step count:
    uv run python persistent_homology/persistent_homology_witness.py \
        --min_set plots_slag_d4_run/min_set.pkl \
        --coeffs gd_runs/gd_d4_run_step3000.pkl \
        --landmarks 500,1000,2000 --newton_steps 100

    # Skip the Newton filter (PH on raw min_set):
    uv run python persistent_homology/persistent_homology_witness.py \
        --min_set plots_slag_d4_run/min_set.pkl --no_newton_filter
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

def load_points(min_set, subsamp, seed):
    print("=== LOADING POINTS FROM PICKLE ===")
    with open(min_set, 'rb') as f:
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


def filter_newton_check(Z_complex, coeffs, psi, n_steps, threshold,
                        dist_chunk_size=50000):
    """Newton-refine + per-point residual filter via slag's filter_and_refine.

    Thin wrapper: load coeffs, complex->real, call filter_and_refine with
    k=N and n_repulsion_steps=0 (so it's just refine + per-point residual,
    no top-k culling, no repulsion), then threshold by `threshold` per-point.
    Threshold semantics match newton_check (find_smooth_submanifold.py:418).
    """
    print(f"\n=== NEWTON_CHECK FILTER ({n_steps} steps, threshold={threshold:.1e}) ===")
    with open(coeffs, 'rb') as f:
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
                          limit_dimension=3, top_k_landmarks=50,
                          witness_type='weak'):
    """gudhi WitnessComplex on a precomputed landmark-to-witness distance table.

    Args:
        dist_table_slice: (L, N) raw FS distances, landmark -> witness.
        max_alpha_square: filtration cap in squared-distance units.
        limit_dimension: max simplex dim. Need >=3 for H2 (tetrahedra).
        top_k_landmarks: keep only the top-K nearest landmarks per witness
                         when building the table (saves memory; the witness
                         complex only needs ~limit_dimension+2 nearest).
        witness_type: 'weak' (gudhi.WitnessComplex) or 'strong'
                      (gudhi.StrongWitnessComplex). Strong applies a
                      stricter witnessing condition: every face of sigma
                      must be witnessed by the same w. Cuts the late-alpha
                      noise band by ~1-2 orders of magnitude; same input
                      format as weak.

    Returns:
        (H0, H1, H2): each (n_features, 2) array of (birth, death) in *raw* FS
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

    if witness_type == 'strong':
        wc = gudhi.StrongWitnessComplex(nearest_landmark_table=nearest)
    elif witness_type == 'weak':
        wc = gudhi.WitnessComplex(nearest_landmark_table=nearest)
    else:
        raise ValueError(f"witness_type must be 'weak' or 'strong', got {witness_type!r}")
    st = wc.create_simplex_tree(
        max_alpha_square=max_alpha_square,
        limit_dimension=limit_dimension,
    )
    persistence = st.persistence()

    H0, H1, H2 = [], [], []
    for dim, (b, d) in persistence:
        b_raw = np.sqrt(max(b, 0.0)) if np.isfinite(b) else 0.0
        d_raw = np.inf if np.isinf(d) else np.sqrt(max(d, 0.0))
        if dim == 0:
            H0.append([b_raw, d_raw])
        elif dim == 1:
            H1.append([b_raw, d_raw])
        elif dim == 2:
            H2.append([b_raw, d_raw])
    return (np.array(H0, dtype=np.float64) if H0 else np.empty((0, 2)),
            np.array(H1, dtype=np.float64) if H1 else np.empty((0, 2)),
            np.array(H2, dtype=np.float64) if H2 else np.empty((0, 2)))


# ------------------------------------------------------------------- analysis

def betti_at(points, t):
    if len(points) == 0:
        return 0
    return int(np.sum((points[:, 0] <= t) & (points[:, 1] > t)))


def analyze(H0, H1, H2, infinity_val, label=""):
    print(f"\n=== TOPOLOGICAL FEATURES {label} ===")
    print(f"H0 (connected components): {len(H0)}")
    print(f"H1 (loops):                {len(H1)}")
    print(f"H2 (voids):                {len(H2)}")

    filt = (np.linspace(0.2, 0.9, 4) * infinity_val).tolist()
    print("Betti at sampled filtration values:")
    for t in filt:
        print(f"  r={t:.3f}: beta0={betti_at(H0, t)}, beta1={betti_at(H1, t)}, "
              f"beta2={betti_at(H2, t)}")

    def top_features(points, name):
        if len(points) == 0:
            return
        # Bounded persistence so essential classes are ranked by birth.
        death_bounded = np.where(np.isinf(points[:, 1]), infinity_val,
                                 points[:, 1])
        pers_for_sort = death_bounded - points[:, 0]
        order = np.argsort(pers_for_sort)[-5:][::-1]
        print(f"{name} top 5 by persistence:")
        for i in order:
            pers_actual = points[i, 1] - points[i, 0]
            print(f"  birth={points[i, 0]:.4f}  death={points[i, 1]:.4f}  pers={pers_actual:.4f}")

    top_features(H0, "H0")
    top_features(H1, "H1")
    top_features(H2, "H2")


# ------------------------------------------------------------------- plotting

def _clip_inf(p, infinity_val):
    if len(p) == 0:
        return p
    out = p.copy()
    out[np.isinf(out[:, 1]), 1] = infinity_val
    return out


def plot_one_L(H0, H1, H2, infinity_val, n_sample, L, output_file):
    """Per-L PH figure: 3 cols (H0, H1, H2) x 3 rows (diagram, barcode, Betti).
    """
    print(f"  plotting L={L} -> {output_file}")
    dims = [('H0', H0, 'skyblue'),
            ('H1', H1, 'orange'),
            ('H2', H2, 'forestgreen')]

    fig = plt.figure(figsize=(15, 12))

    def plot_diagram(ax, pts, color, name):
        ptsd = _clip_inf(pts, infinity_val)
        if len(ptsd):
            ax.scatter(ptsd[:, 0], ptsd[:, 1], alpha=0.7, s=30, c=color)
            ax.plot([0, infinity_val], [0, infinity_val], 'k--', alpha=0.3)
        ax.set_xlabel('Birth (FS distance)')
        ax.set_ylabel('Death (FS distance)')
        ax.set_title(f'{name} Persistence Diagram ({len(pts)} features)')
        ax.set_xlim([0, infinity_val])
        ax.set_ylim([0, infinity_val * 1.05])
        ax.grid(True, linestyle='--', alpha=0.6)

    def plot_barcode(ax, pts, color, title):
        if len(pts) == 0:
            ax.set_title(title)
            return
        # Sort by *bounded* persistence: treat inf-death as if it died at the
        # filtration cap. Without this, many essential classes all have
        # persistence = inf and the numpy argsort tie-break ranks them by
        # original gudhi index, surfacing arbitrary high-birth bars instead
        # of the visually-meaningful low-birth ones.
        death_for_sort = np.where(np.isinf(pts[:, 1]), infinity_val, pts[:, 1])
        order = np.argsort(death_for_sort - pts[:, 0])[::-1]
        sorted_pts = pts[order]
        n_bars = min(50, len(sorted_pts))
        for i in range(n_bars):
            b, d = sorted_pts[i]
            length = (infinity_val - b) if np.isinf(d) else (d - b)
            ax.barh(i, length, left=b, color=color, alpha=0.7)
        ax.set_xlim([0, infinity_val])
        ax.set_ylim([-1, n_bars])
        ax.set_xlabel('Filtration Value')
        ax.set_ylabel('Feature Index')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6, axis='x')

    def plot_betti(ax, dim, pts, color):
        t_grid = np.linspace(0, infinity_val, 100)
        if len(pts):
            bv = [betti_at(pts, t) for t in t_grid]
            ax.plot(t_grid, bv, color=color, linewidth=2)
            ax.fill_between(t_grid, bv, alpha=0.3, color=color)
        ax.set_xlim([0, infinity_val])
        ax.set_xlabel('Filtration Value')
        ax.set_ylabel(f'beta_{dim}')
        ax.set_title(f'H{dim} Betti Curve')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Columns = homology dim (H0, H1, H2); rows = plot type (diagram, barcode, Betti).
    for col, (name, pts, color) in enumerate(dims):
        ax_diag = plt.subplot(3, 3, col + 1)
        plot_diagram(ax_diag, pts, color, name)
        ax_bar = plt.subplot(3, 3, 3 + col + 1)
        plot_barcode(ax_bar, pts, color, f'{name} Barcode (top {min(50, len(pts))})')
        ax_betti = plt.subplot(3, 3, 6 + col + 1)
        plot_betti(ax_betti, col, pts, color)

    fig.suptitle(
        f'Witness Complex Persistence (FS metric)  |  '
        f'N={n_sample}, L={L}',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(per_L, infinity_val, n_sample, output_file):
    """Across-L comparison: rows = H0/H1/H2 diagrams per L, bottom row =
    overlaid Betti curves (one panel per dim)."""
    Ls = sorted(per_L.keys())
    n_L = len(Ls)
    n_rows = 4  # H0, H1, H2 diagrams + Betti curve overlay
    fig = plt.figure(figsize=(4 * n_L, 4 * n_rows))
    cmap_L = plt.cm.viridis(np.linspace(0.1, 0.9, n_L))

    dim_specs = [(0, 'H0', 'steelblue'),
                 (1, 'H1', 'darkorange'),
                 (2, 'H2', 'forestgreen')]

    # Rows 0..2: per-L diagrams for H0, H1, H2
    for row, (dim, name, color) in enumerate(dim_specs):
        for ci, L in enumerate(Ls):
            pts = per_L[L][dim]
            ax = plt.subplot(n_rows, n_L, row * n_L + ci + 1)
            ptsd = _clip_inf(pts, infinity_val)
            if len(ptsd):
                ax.scatter(ptsd[:, 0], ptsd[:, 1], s=15, alpha=0.7, c=color)
                ax.plot([0, infinity_val], [0, infinity_val], 'k--', alpha=0.3)
            ax.set_title(f'{name}  (L={L}, n_feat={len(pts)})')
            ax.set_xlim([0, infinity_val])
            ax.set_ylim([0, infinity_val * 1.05])
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.grid(True, linestyle='--', alpha=0.6)

    # Row 3: overlaid Betti curves across L, one panel per dim
    t_grid = np.linspace(0, infinity_val, 200)
    for di, (dim, name, _) in enumerate(dim_specs):
        ax = plt.subplot(n_rows, 3, 3 * 3 + di + 1)
        for L, lc in zip(Ls, cmap_L):
            pts = per_L[L][dim]
            if len(pts):
                ax.plot(t_grid, [betti_at(pts, t) for t in t_grid],
                        color=lc, label=f'L={L}', linewidth=2)
        ax.set_xlim([0, infinity_val])
        ax.set_xlabel('Filtration Value')
        ax.set_ylabel(f'beta_{dim}')
        ax.set_title(f'{name} Betti Curves across L')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(
        f'Witness Landmark Sweep  |  N={n_sample}, FS metric',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved comparison: {output_file}")


# ----------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--min_set',
                   default='/home/y.qi/projects/SLagSearch/plots_slag/min_set_psi0.pkl')
    p.add_argument('--subsamp', type=int, default=50000)
    p.add_argument('--landmarks', default='300,500,750,1000',
                   help='Comma-separated L values to sweep.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--coeffs', default=None,
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
    p.add_argument('--max_alpha', type=float, default=None,
                   help='Filtration cap in raw FS distance units. Simplices '
                        'with filtration value above this are not built. '
                        'Default (auto): 4 * covering_radius at the smallest '
                        'L in the sweep, where covering_radius = '
                        'max_p min_l dist(point_p, landmark_l). The 4x is '
                        'a 2x safety buffer over the de Silva-Carlsson bound '
                        '(meaningful PH up to ~2 * covering_radius). For '
                        'the Fermat quintic at L=300 this typically lands '
                        'near the previous fixed default of 0.8. Diagrams '
                        'display on [0, max_alpha].')
    p.add_argument('--witness_type', choices=['weak', 'strong'], default='weak',
                   help='gudhi witness variant. "strong" enforces that every '
                        'face of sigma is witnessed by the same w, which '
                        'prunes the late-alpha noise band by 1-2 orders of '
                        'magnitude. "weak" is the de Silva-Carlsson default.')
    out_group = p.add_mutually_exclusive_group()
    out_group.add_argument('--out_dir', default=None,
                           help='Full output directory for PNGs and cache '
                                'files. Default: parent directory of --min_set '
                                '(e.g. gd_runs/plots_slag_<job>/).')
    out_group.add_argument('--out_subdir', type=str, default=None,
                           help='Output subdirectory name appended to '
                                "--min_set's parent directory.")
    p.add_argument('--out_prefix', default='persistent_homology_witness')
    p.add_argument('--cache_landmarks', default='witness_landmarks_cache.pkl')
    p.add_argument('--cache_diagrams', default='witness_diagrams_cache.pkl')
    p.add_argument('--no_cache', action='store_true')
    p.add_argument('--plots_only', action='store_true',
                   help='Skip Newton filter, landmark selection, and gudhi '
                        'construction. Load diagrams from the cached pkl '
                        'and re-plot (and re-print analysis). Errors out '
                        'if no matching cache is found.')
    return p.parse_args()


def main():
    args = parse_args()
    st = time.time()

    L_values = sorted({int(x) for x in args.landmarks.split(',') if x.strip()})
    L_min = L_values[0]
    L_max = L_values[-1]
    print(f"Sweep: L values = {L_values}, L_min = {L_min}, L_max = {L_max}")

    # Resolve output directory (defaults to the min_set's folder, e.g.
    # gd_runs/plots_slag_<job>/). All PNGs and caches land here.
    min_set_parent = os.path.dirname(os.path.abspath(args.min_set)) or '.'
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.out_subdir is not None:
        out_dir = os.path.join(min_set_parent, args.out_subdir)
    else:
        out_dir = min_set_parent
    os.makedirs(out_dir, exist_ok=True)

    # Auto-discover coeffs.pkl from the min_set's parent if --coeffs wasn't
    # given. Matches the sidecar contract from viz.fitness_pipeline so the
    # PH script Just Works when pointed at a run folder.
    if args.coeffs is None and not args.no_newton_filter:
        candidate = os.path.join(min_set_parent, 'coeffs.pkl')
        if os.path.exists(candidate):
            args.coeffs = candidate
            print(f"[ph_witness] auto-discovered coeffs at {candidate}")
    cache_landmarks_path = os.path.join(out_dir, args.cache_landmarks)
    cache_diagrams_path = os.path.join(out_dir, args.cache_diagrams)
    print(f"Output directory: {out_dir}")

    # ---- filter_sig is used to match both caches; defined early so the
    # diagrams cache check can use it without running the Newton filter.
    filter_sig = {
        'no_newton_filter': args.no_newton_filter,
        'coeffs': args.coeffs,
        'psi': args.psi,
        'newton_steps': args.newton_steps,
        'newton_threshold': args.newton_threshold,
    }

    # ---- try diagrams cache first (fast path: skip filter + landmarks + gudhi)
    requested_sig = {
        'min_set': args.min_set,
        'subsamp': args.subsamp,
        'seed': args.seed,
        'filter_sig': filter_sig,
        'L_values': L_values,
        'max_alpha': args.max_alpha,
        'witness_type': args.witness_type,
        'top_k_landmarks': args.top_k_landmarks,
    }
    per_L = None
    infinity_val = None
    n_sample = None
    if not args.no_cache and os.path.exists(cache_diagrams_path):
        with open(cache_diagrams_path, 'rb') as f:
            d_data = pickle.load(f)
        missing = [k for k in requested_sig if k not in d_data]
        diffs = [
            (k, d_data[k], requested_sig[k])
            for k in requested_sig
            if k in d_data and d_data[k] != requested_sig[k]
        ]
        # Strict match for auto-load; --plots_only loads anyway and warns.
        accept = (not diffs and not missing) or args.plots_only
        if accept:
            print(f"\nLoading cached diagrams from '{cache_diagrams_path}'")
            if missing or diffs:
                print("  WARNING: plotting cached data despite cache "
                      "mismatch (--plots_only):")
                for k, cached_v, current_v in diffs:
                    print(f"    {k}: cached={cached_v!r}  current={current_v!r}")
                if missing:
                    print(f"    missing keys (cache predates them): {missing}")
            per_L = d_data['per_L']
            infinity_val = d_data['infinity_val']
            n_sample = d_data['n_sample']
        else:
            print(f"\nDiagrams cache '{cache_diagrams_path}' mismatch:")
            for k, cached_v, current_v in diffs:
                print(f"  {k}: cached={cached_v!r}  current={current_v!r}")
            if missing:
                print(f"  missing keys (cache predates them): {missing}")

    if per_L is None and args.plots_only:
        raise SystemExit(
            "ERROR: --plots_only set but no matching diagrams cache at "
            f"'{cache_diagrams_path}'. Run once without --plots_only to "
            "populate the cache."
        )

    if per_L is None:
        # ---- load + filter
        Z = load_points(args.min_set, args.subsamp, args.seed)
        if args.no_newton_filter:
            print("\nNewton-residual filter disabled (--no_newton_filter).")
        else:
            if args.coeffs is None:
                raise SystemExit(
                    "ERROR: --coeffs is required to run the Newton filter. "
                    "Pass --no_newton_filter to skip filtering entirely."
                )
            psi_val = complex(args.psi)
            Z, _, _ = filter_newton_check(
                Z,
                coeffs=args.coeffs,
                psi=psi_val,
                n_steps=args.newton_steps,
                threshold=args.newton_threshold,
                dist_chunk_size=args.dist_chunk_size,
            )
        Zn = _normalize_rows_complex(Z)
        n_sample = len(Z)
        print(f"\nN after filter: {n_sample}")

        # ---- landmarks (cached on disk)
        use_lm_cache = False
        if not args.no_cache and os.path.exists(cache_landmarks_path):
            with open(cache_landmarks_path, 'rb') as f:
                lm_data = pickle.load(f)
            if (lm_data.get('min_set') == args.min_set
                    and lm_data.get('subsamp') == args.subsamp
                    and lm_data.get('seed') == args.seed
                    and lm_data.get('filter_sig') == filter_sig
                    and lm_data.get('n_sample') == n_sample
                    and lm_data.get('L_max', 0) >= L_max):
                print(f"\nLoading cached landmarks/dist_table from "
                      f"'{cache_landmarks_path}'")
                lm_idx = lm_data['lm_idx'][:L_max]
                dist_table = lm_data['dist_table'][:L_max]
                use_lm_cache = True
            else:
                print(f"\nCache '{cache_landmarks_path}' params mismatch; "
                      f"recomputing landmarks.")
        if not use_lm_cache:
            print(f"\n=== MAX-MIN LANDMARK SELECTION (L_max={L_max}) ===")
            t0 = time.time()
            lm_idx, dist_table = maxmin_landmarks(Zn, L_max, args.seed)
            print(f"  done in {time.time() - t0:.1f}s on device {jax.devices()[0]}")
            if not args.no_cache:
                with open(cache_landmarks_path, 'wb') as f:
                    pickle.dump({
                        'min_set': args.min_set, 'subsamp': args.subsamp,
                        'seed': args.seed, 'filter_sig': filter_sig,
                        'n_sample': n_sample, 'L_max': L_max,
                        'lm_idx': lm_idx, 'dist_table': dist_table,
                    }, f)
                print(f"  cached: {cache_landmarks_path}")

        data_diameter = float(np.max(dist_table))
        # Covering radius at the loosest landmark count in the sweep:
        # max over points of distance to nearest of the first L_min
        # landmarks. de Silva-Carlsson bound: meaningful PH up to
        # ~2 * covering_radius; the 4x default below carries a 2x buffer.
        covering_radius = float(np.max(np.min(dist_table[:L_min], axis=0)))
        if args.max_alpha is None:
            resolved_max_alpha = 4.0 * covering_radius
            cap_source = f"auto: 4 * covering_radius(L={L_min})"
        else:
            resolved_max_alpha = float(args.max_alpha)
            cap_source = "user override"
        infinity_val = resolved_max_alpha
        max_alpha_sq = resolved_max_alpha ** 2
        print(f"\nData diameter (max FS distance):  {data_diameter:.4f}")
        print(f"Covering radius at L={L_min}:         {covering_radius:.4f}")
        print(f"Filtration cap (--max_alpha):     {infinity_val:.4f}  "
              f"({cap_source})")
        print(f"max_alpha_square:                 {max_alpha_sq:.4f}")
        print(f"(reference: FS diameter pi/2 = {np.pi / 2:.4f})")

        # ---- witness diagrams per L (H0, H1, H2 via limit_dimension=3)
        per_L = {}
        for L in L_values:
            print(f"\n=== WITNESS COMPLEX (L={L}) ===")
            t0 = time.time()
            H0, H1, H2 = build_witness_diagram(
                dist_table[:L], max_alpha_sq,
                limit_dimension=3, top_k_landmarks=args.top_k_landmarks,
                witness_type=args.witness_type,
            )
            print(f"  built in {time.time() - t0:.1f}s; "
                  f"H0={len(H0)}, H1={len(H1)}, H2={len(H2)}")

            # Match the VR script: ensure the essential H0 is present.
            if len(H0) and not np.any(np.isinf(H0[:, 1])):
                H0 = np.vstack([H0, [0.0, np.inf]])
                print("  injected essential H0 (birth=0, death=inf)")
            per_L[L] = (H0, H1, H2)

        # ---- cache diagrams
        if not args.no_cache:
            with open(cache_diagrams_path, 'wb') as f:
                pickle.dump({
                    'per_L': per_L,
                    'infinity_val': infinity_val,
                    'data_diameter': data_diameter,
                    'max_alpha': args.max_alpha,
                    'witness_type': args.witness_type,
                    'top_k_landmarks': args.top_k_landmarks,
                    'n_sample': n_sample,
                    'L_values': L_values,
                    'min_set': args.min_set,
                    'subsamp': args.subsamp,
                    'seed': args.seed,
                    'filter_sig': filter_sig,
                }, f)
            print(f"\nCached diagrams: {cache_diagrams_path}")

    # ---- analyze + per-L plot (runs whether per_L came from cache or fresh)
    for L in L_values:
        H0, H1, H2 = per_L[L]
        analyze(H0, H1, H2, infinity_val, label=f'(L={L})')
        png = os.path.join(out_dir, f'{args.out_prefix}_L{L}.png')
        plot_one_L(H0, H1, H2, infinity_val, n_sample, L, png)

    # ---- comparison plot
    plot_comparison(per_L, infinity_val, n_sample,
                    os.path.join(out_dir, f'{args.out_prefix}_sweep.png'))

    print(f"\n=== DONE in {time.time() - st:.1f}s ===")


if __name__ == '__main__':
    main()
