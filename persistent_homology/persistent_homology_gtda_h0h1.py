"""Vietoris-Rips persistent homology (H0/H1) for points in CP^4.

Loads a min_set pickle (complex (N, 5)), computes the Fubini-Study distance
matrix via JAX, then runs giotto-tda's VR pipeline on the precomputed
distances. Outputs the standard 2x4 PH figure + a diagrams cache.

Sister script: persistent_homology_witness_h0h1.py (witness complex variant,
scales beyond VR's N ~ 10k memory cap).

Usage:
    uv run python persistent_homology/persistent_homology_gtda_h0h1.py \
        --filepath plots_slag_d4_run/min_set.pkl --subsamp 10000

    # With explicit output / cache names:
    uv run python persistent_homology/persistent_homology_gtda_h0h1.py \
        --filepath plots_slag_d4_run/min_set.pkl \
        --out_file ph_d4.png --cache_file ph_d4_cache.pkl
"""

import argparse
import os
import pickle
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

from gtda.homology import VietorisRipsPersistence

warnings.filterwarnings('ignore')


@jax.jit
def fubini_study_distance_matrix_jax(Z):
    """Pairwise FS geodesic distance on CP^(m-1).

    Z: (N, m) complex. Returns (N, N) FP64. Uses one BLAS matmul + arccos,
    runs on the default JAX device (GPU when available).
    """
    Zn = Z / jnp.linalg.norm(Z, axis=1, keepdims=True)
    inner = Zn @ jnp.conj(Zn).T
    return jnp.arccos(jnp.clip(jnp.abs(inner), 0.0, 1.0))


def fubini_study_distance_matrix(Z):
    """Host-side wrapper returning a numpy array (gtda expects numpy)."""
    return np.asarray(fubini_study_distance_matrix_jax(jnp.asarray(Z)))


def compute_betti_at_filtration(points, filtration_value):
    """Count features alive at a specific filtration value (born <= t < death)"""
    return np.sum((points[:, 0] <= filtration_value) & (points[:, 1] > filtration_value))


def compute_persistence(filepath, subsamp=10000, seed=42, cache_file=None):
    """Load points, compute Fubini-Study distance matrix, run VR persistence.

    Returns (H0_points, H1_points, infinity_val, n_sample). If cache_file is
    given, the diagrams are pickled there for fast re-plotting later.
    """
    print("=== LOADING POINTS FROM PICKLE ===")
    with open(filepath, 'rb') as f:
        Z_good = pickle.load(f)
    Z_good = np.asarray(Z_good, dtype=np.complex128)
    n_points = Z_good.shape[0]
    print(f"Loaded {n_points} points")

    max_imag = np.max(np.abs(np.imag(Z_good)))
    is_complex = max_imag > 1e-10
    print(f"Points are {'complex' if is_complex else 'real'}")
    print(f"Maximum imaginary part: {max_imag:.3e}")
    if not is_complex:
        print("\nWarning: Points appear to be real. Fubini-Study metric is designed for complex projective space.")
        print("Proceeding anyway by treating real points as complex with zero imaginary part.")

    # Subsample if needed for computational efficiency
    if n_points > subsamp:
        np.random.seed(seed)
        idx = np.random.choice(n_points, subsamp, replace=False)
        Z_sample = Z_good[idx]
    else:
        Z_sample = Z_good
    print(f"Using {len(Z_sample)} points for persistent homology")

    print("\n=== COMPUTING FUBINI-STUDY DISTANCE MATRIX (JAX) ===")
    print(f"Computing {len(Z_sample)}×{len(Z_sample)} distance matrix...")
    t0 = time.time()
    distance_matrix = fubini_study_distance_matrix(Z_sample)
    print(f"  done in {time.time() - t0:.1f}s on device {jax.devices()[0]}")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    positive_distances = distance_matrix[distance_matrix > 0]
    if positive_distances.size > 0:
        print(f"Distance range: [{np.min(positive_distances):.4f}, {np.max(distance_matrix):.4f}]")
    else:
        print("Distance range: all distances are 0 (duplicate points)")
    print(f"Max possible FS distance: {np.pi/2:.4f}")

    print("\n=== COMPUTING PERSISTENT HOMOLOGY WITH GIOTTO-TDA ===")
    distance_matrices = distance_matrix[np.newaxis, :, :]
    max_dist = np.max(distance_matrix)
    print("Computing persistent homology for H0 and H1...")
    homology = VietorisRipsPersistence(
        metric='precomputed',
        homology_dimensions=[0, 1],
        max_edge_length=max_dist,
        infinity_values=np.inf,
        n_jobs=-1,
    )
    persistence_diagrams = homology.fit_transform(distance_matrices)
    print(f"Persistence diagram shape: {persistence_diagrams.shape}")
    diagram = persistence_diagrams[0]

    H0_points = diagram[diagram[:, 2] == 0][:, :2]
    H1_points = diagram[diagram[:, 2] == 1][:, :2]
    infinity_val = max_dist
    n_sample = len(Z_sample)

    # gtda/Ripser drops the essential H0 class when max_edge_length equals the
    # cloud diameter. Inject it manually: for a connected point cloud, there's
    # exactly one H0 class with (birth=0, death=inf).
    if not np.any(np.isinf(H0_points[:, 1])):
        H0_points = np.vstack([H0_points, [0.0, np.inf]])
        print("Injected essential H0 class (birth=0, death=inf) — gtda dropped it.")

    if cache_file is not None:
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'H0_points': H0_points,
                'H1_points': H1_points,
                'infinity_val': infinity_val,
                'n_sample': n_sample,
                'filepath': filepath,
                'subsamp': subsamp,
                'seed': seed,
            }, f)
        print(f"Saved persistence diagrams to '{cache_file}'")

    return H0_points, H1_points, infinity_val, n_sample


def analyze_persistence(H0_points, H1_points, infinity_val):
    """Print Betti numbers, gap analysis, top persistent features, and stats."""
    H0_finite = H0_points[np.isfinite(H0_points[:, 1])]
    H1_finite = H1_points[np.isfinite(H1_points[:, 1])]

    # Original tuning was for FS diameter pi/2; scale to the actual data range.
    filtration_values = (np.array([0.2, 0.3, 0.4, 0.5]) * (infinity_val / (np.pi / 2))).tolist()

    print(f"\n=== TOPOLOGICAL FEATURES ===")
    print(f"H0 (connected components): {len(H0_points)} total features")
    print(f"H1 (loops): {len(H1_points)} total features")

    print(f"\n=== BETTI NUMBERS AT DIFFERENT FILTRATION VALUES ===")
    for filt_val in filtration_values:
        b0 = compute_betti_at_filtration(H0_points, filt_val)
        b1 = compute_betti_at_filtration(H1_points, filt_val)
        print(f"At r={filt_val:.2f}: β₀={b0}, β₁={b1}")

    print(f"\n=== PERSISTENCE GAP ANALYSIS ===")
    print("(Heuristic for identifying significant topological features)")

    def analyze_persistence_gap(points, dim_name):
        finite_points = points[np.isfinite(points[:, 1])]
        infinite_count = len(points) - len(finite_points)

        if len(finite_points) > 0:
            persistences = finite_points[:, 1] - finite_points[:, 0]
            persistence_sorted = np.sort(persistences)[::-1]

            print(f"\n{dim_name}:")
            print(f"  Infinite persistence features: {infinite_count}")
            print(f"  Finite persistence features: {len(finite_points)}")

            if len(persistence_sorted) > 1:
                gaps = persistence_sorted[:-1] - persistence_sorted[1:]
                max_gap_idx = np.argmax(gaps)
                max_gap = gaps[max_gap_idx]
                relative_gap = max_gap / persistence_sorted[max_gap_idx] if persistence_sorted[max_gap_idx] > 0 else 0

                print(f"  Largest persistence gap: {max_gap:.4f} (between ranks {max_gap_idx+1} and {max_gap_idx+2})")
                print(f"  Relative gap: {relative_gap:.1%}")
                print(f"  Persistence before gap: {persistence_sorted[max_gap_idx]:.4f}")
                print(f"  Persistence after gap: {persistence_sorted[max_gap_idx+1]:.4f}")

                suggested_betti = max_gap_idx + 1 + infinite_count
                print(f"  Gap analysis suggests β_{dim_name[-1]} ≈ {suggested_betti}")

                threshold_gap = 0.1 * persistence_sorted[0] if persistence_sorted[0] > 0 else 0.01
                significant_gaps = np.where(gaps > threshold_gap)[0]
                if len(significant_gaps) > 1:
                    print(f"  Multiple significant gaps found at ranks: {significant_gaps + 1}")
                    print(f"  Consider β_{dim_name[-1]} could be between {significant_gaps[0]+1+infinite_count} and {significant_gaps[-1]+1+infinite_count}")
            else:
                print(f"  Only {len(persistence_sorted)} feature(s), cannot perform gap analysis")

            n_show = min(5, len(persistence_sorted))
            print(f"  Top {n_show} persistence values: {persistence_sorted[:n_show]}")

            return infinite_count + (max_gap_idx + 1 if len(persistence_sorted) > 1 else len(persistence_sorted))
        else:
            print(f"\n{dim_name}: Only infinite persistence features found ({infinite_count})")
            return infinite_count

    gap_betti_0 = analyze_persistence_gap(H0_points, "H0")
    gap_betti_1 = analyze_persistence_gap(H1_points, "H1")

    print("!!!!!!!!!!!! Infinite features", H0_points[np.isinf(H0_points[:, 1])])

    print(f"\n=== BETTI NUMBER ESTIMATES (Gap Analysis) ===")
    print(f"β₀ ≈ {gap_betti_0}, β₁ ≈ {gap_betti_1}")
    print("\nNote: Gap analysis is a heuristic. Compare with:")
    print("  1. Betti curves for stable plateaus")
    print("  2. Domain knowledge about expected topology")
    print("  3. Multiple filtration values above")

    def find_most_persistent_features(points, n_top=5):
        if len(points) == 0:
            return np.array([])
        persistences = points[:, 1] - points[:, 0]
        # np.argsort puts inf at the end of ascending order; reverse to surface it first.
        top_indices = np.argsort(persistences)[-n_top:][::-1]
        return points[top_indices]

    print(f"\n=== MOST PERSISTENT FEATURES (TOP 5) ===")
    for dim, points, name in [(0, H0_points, "H0"), (1, H1_points, "H1")]:
        top_features = find_most_persistent_features(points, 5)
        if len(top_features) > 0:
            print(f"\n{name}:")
            for i, (birth, death) in enumerate(top_features):
                persistence = death - birth
                print(f"  {i+1}. Birth: {birth:.3f}, Death: {death:.3f}, Persistence: {persistence:.3f}")

    def compute_persistence_stats(points, dim_name):
        if len(points) > 0:
            persistences = points[:, 1] - points[:, 0]
            finite_persistences = persistences[np.isfinite(points[:, 1])]
            if len(finite_persistences) > 0:
                print(f"\n{dim_name} persistence statistics:")
                print(f"  Mean persistence: {np.mean(finite_persistences):.4f}")
                print(f"  Max persistence: {np.max(finite_persistences):.4f}")
                print(f"  Std persistence: {np.std(finite_persistences):.4f}")

    compute_persistence_stats(H0_points, "H0")
    compute_persistence_stats(H1_points, "H1")


def make_plots(H0_points, H1_points, infinity_val, n_sample, output_file,
               h0_color='skyblue', h1_color='orange', suggested_filtration=None):
    """Render the 2x4 persistent homology figure."""
    print("\n=== CREATING VISUALIZATION ===")

    if suggested_filtration is None:
        suggested_filtration = 0.5 * (infinity_val / (np.pi / 2))

    H0_finite = H0_points[np.isfinite(H0_points[:, 1])]
    H1_finite = H1_points[np.isfinite(H1_points[:, 1])]

    # Clip infinite deaths to infinity_val for scatter display so the survivor
    # is visible at the top edge of the diagram.
    def _clip_for_display(points):
        if len(points) == 0:
            return points
        out = points.copy()
        out[np.isinf(out[:, 1]), 1] = infinity_val
        return out

    H0_display = _clip_for_display(H0_points)
    H1_display = _clip_for_display(H1_points)

    fig = plt.figure(figsize=(16, 10))

    # 1-2. Persistence Diagrams for each dimension
    ax1 = plt.subplot(2, 4, 1)
    if len(H0_points) > 0:
        ax1.scatter(H0_display[:, 0], H0_display[:, 1], alpha=0.7, s=30, c=h0_color)
        ax1.plot([0, infinity_val], [0, infinity_val], 'k--', alpha=0.3)
    ax1.set_xlabel('Birth')
    ax1.set_ylabel('Death')
    ax1.set_title(f'H0 Persistence Diagram\n({len(H0_points)} features)')
    ax1.set_xlim([0, infinity_val])
    ax1.set_ylim([0, infinity_val * 1.05])
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = plt.subplot(2, 4, 2)
    if len(H1_points) > 0:
        ax2.scatter(H1_display[:, 0], H1_display[:, 1], alpha=0.7, s=30, c=h1_color)
        ax2.plot([0, infinity_val], [0, infinity_val], 'k--', alpha=0.3)
    ax2.set_xlabel('Birth')
    ax2.set_ylabel('Death')
    ax2.set_title(f'H1 Persistence Diagram\n({len(H1_points)} features)')
    ax2.set_xlim([0, infinity_val])
    ax2.set_ylim([0, infinity_val * 1.05])
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. Combined persistence diagram
    ax3 = plt.subplot(2, 4, 3)
    if len(H0_points) > 0:
        ax3.scatter(H0_display[:, 0], H0_display[:, 1], alpha=0.7, s=30, c=h0_color, label='H0')
    if len(H1_points) > 0:
        ax3.scatter(H1_display[:, 0], H1_display[:, 1], alpha=0.7, s=30, c=h1_color, label='H1')
    ax3.plot([0, infinity_val], [0, infinity_val], 'k--', alpha=0.3)
    ax3.set_xlabel('Birth')
    ax3.set_ylabel('Death')
    ax3.set_title('Combined Persistence Diagram')
    ax3.set_xlim([0, infinity_val])
    ax3.set_ylim([0, infinity_val * 1.05])
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    # 4. Persistence histogram
    ax4 = plt.subplot(2, 4, 4)
    all_persistences = []
    labels = []
    colors = []

    if len(H0_finite) > 0:
        pers = H0_finite[:, 1] - H0_finite[:, 0]
        all_persistences.append(pers)
        labels.append('H0')
        colors.append(h0_color)

    if len(H1_finite) > 0:
        pers = H1_finite[:, 1] - H1_finite[:, 0]
        all_persistences.append(pers)
        labels.append('H1')
        colors.append(h1_color)

    if all_persistences:
        ax4.hist(all_persistences, bins=30, label=labels, color=colors, alpha=0.7)
        ax4.set_xlabel('Persistence')
        ax4.set_ylabel('Count')
        ax4.set_title('Persistence Distribution')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.6)

    # 5-6. Persistence Barcodes
    def plot_barcode(ax, points, color, title):
        if len(points) > 0:
            sorted_indices = np.argsort(points[:, 1] - points[:, 0])[::-1]
            sorted_points = points[sorted_indices]

            n_bars = min(50, len(sorted_points))
            for i in range(n_bars):
                birth, death = sorted_points[i]
                if np.isinf(death):
                    ax.barh(i, infinity_val - birth, left=birth, color=color, alpha=0.8, edgecolor='red', linewidth=2)
                else:
                    ax.barh(i, death - birth, left=birth, color=color, alpha=0.7)

            ax.set_xlim([0, infinity_val])
            ax.set_ylim([-1, n_bars])
            ax.set_xlabel('Filtration Value')
            ax.set_ylabel('Feature Index')
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.6, axis='x')

    ax5 = plt.subplot(2, 4, 5)
    plot_barcode(ax5, H0_points, h0_color, f'H0 Barcode (top {min(50, len(H0_points))})')

    ax6 = plt.subplot(2, 4, 6)
    plot_barcode(ax6, H1_points, h1_color, f'H1 Barcode (top {min(50, len(H1_points))})')

    # 7-8. Betti curves
    ax7 = plt.subplot(2, 4, 7)
    ax8 = plt.subplot(2, 4, 8)

    sampling_range = np.linspace(0, infinity_val, 100)

    axes = [ax7, ax8]
    dims = [0, 1]
    curve_colors = [h0_color, h1_color]
    points_list = [H0_points, H1_points]

    for ax, dim, color, points in zip(axes, dims, curve_colors, points_list):
        if len(points) > 0:
            betti_values = [compute_betti_at_filtration(points, t) for t in sampling_range]

            ax.plot(sampling_range, betti_values, color=color, linewidth=2)
            ax.fill_between(sampling_range, betti_values, alpha=0.3, color=color)
            ax.set_xlabel('Filtration Value')
            ax.set_ylabel(f'β_{dim}')
            ax.set_title(f'H{dim} Betti Curve')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim([0, infinity_val])

            suggested_betti = compute_betti_at_filtration(points, suggested_filtration)
            ax.axvline(x=suggested_filtration, color='red', linestyle='--', alpha=0.5, label=f'r={suggested_filtration:.1f}')
            ax.axhline(y=suggested_betti, color='red', linestyle='--', alpha=0.5)
            ax.legend()

    fig.suptitle(f'Persistent Homology Analysis (Fubini-Study Metric)\n'
                 f'Dataset: {n_sample} points in CP⁴',
                 fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as '{output_file}'")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--filepath', default='plots_slag/min_set_psi0.pkl',
                   help='Path to the min_set pickle (complex (N, 5)).')
    p.add_argument('--subsamp', type=int, default=10000,
                   help='Cap on points used for PH (uses all if dataset is smaller).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_file', default='persistent_homology_h0h1.png')
    p.add_argument('--cache_file', default='persistence_diagrams_h0h1.pkl')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    st = time.time()

    use_cache = False
    if os.path.exists(args.cache_file):
        with open(args.cache_file, 'rb') as f:
            data = pickle.load(f)
        if (data.get('filepath') == args.filepath and
                data.get('subsamp') == args.subsamp and
                data.get('seed') == args.seed):
            print(f"Loading cached persistence diagrams from '{args.cache_file}'")
            H0_points = data['H0_points']
            H1_points = data['H1_points']
            infinity_val = data['infinity_val']
            n_sample = data['n_sample']
            use_cache = True
        else:
            print(f"Cache '{args.cache_file}' parameters do not match current config; recomputing.")

    if not use_cache:
        H0_points, H1_points, infinity_val, n_sample = compute_persistence(
            args.filepath, subsamp=args.subsamp, seed=args.seed,
            cache_file=args.cache_file,
        )

    analyze_persistence(H0_points, H1_points, infinity_val)
    make_plots(H0_points, H1_points, infinity_val, n_sample, args.out_file)

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"  Analyzed {n_sample} points (FS metric on CP^4)")
    print(f"  H0 / H1 only")
    print(f"  Plot:  {args.out_file}")
    print(f"  Elapsed: {time.time() - st:.1f}s")
