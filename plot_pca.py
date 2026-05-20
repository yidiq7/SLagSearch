"""3D PCA visualization of min_set.pkl point clouds.

Usage:
    python plot_pca.py <folder> [<folder> ...]
    python plot_pca.py gd_runs/plots_slag_run1
    python plot_pca.py gd_runs/plots_slag_run1 gd_runs/plots_slag_run1_d1

Each folder must contain a min_set.pkl produced by plots.make_fitness_plots.
Writes pca_3d.png (single view) and pca_2d_pairs.png (PC1-2, PC1-3, PC2-3)
next to the input pickle. Points are colored by their affine patch index
(argmax(|z_i|), 0..4).

Pure numpy + matplotlib. PCA is `np.linalg.svd` on the centered (k, 10)
real point cloud; top-3 singular vectors give the projection.
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # registers '3d' projection


def load_min_set(folder):
    """Returns (min_set_real (k, 10), min_set_complex (k, 5))."""
    path = os.path.join(folder, "min_set.pkl")
    with open(path, "rb") as f:
        min_set_complex = np.asarray(pickle.load(f))  # (k, 5) complex
    min_set_real = np.concatenate(
        [min_set_complex.real, min_set_complex.imag], axis=1
    )
    return min_set_real, min_set_complex


def pca_project(X, n_components=3):
    """SVD-based PCA. Returns (projection (k, n_components), all explained-variance ratios)."""
    mean = X.mean(axis=0)
    Xc = X - mean
    # Xc = U @ diag(S) @ Vt; principal axes are rows of Vt.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    proj = Xc @ components.T
    var = S ** 2 / max(Xc.shape[0] - 1, 1)
    ratio = var / var.sum()
    return proj, ratio


def patch_indices_from_complex(min_set_complex):
    """Affine patch index for each point: argmax_i |z_i|."""
    return np.argmax(np.abs(min_set_complex), axis=1)


def plot_pca_3d(proj, patches, folder, title):
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        proj[:, 0], proj[:, 1], proj[:, 2],
        c=patches, cmap="tab10", vmin=-0.5, vmax=4.5,
        s=0.5, alpha=0.5, edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax, label="patch index", shrink=0.6, pad=0.1)
    cbar.set_ticks([0, 1, 2, 3, 4])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    out = os.path.join(folder, "pca_3d.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_pca_2d_pairs(proj, patches, folder, title):
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sc = None
    for ax, (i, j) in zip(axes, pairs):
        sc = ax.scatter(
            proj[:, i], proj[:, j],
            c=patches, cmap="tab10", vmin=-0.5, vmax=4.5,
            s=0.5, alpha=0.5, edgecolors="none",
        )
        ax.set_xlabel(f"PC{i + 1}")
        ax.set_ylabel(f"PC{j + 1}")
        ax.grid(True, linestyle="--", alpha=0.5)
    cbar = fig.colorbar(sc, ax=axes, label="patch index", shrink=0.8)
    cbar.set_ticks([0, 1, 2, 3, 4])
    fig.suptitle(title)
    out = os.path.join(folder, "pca_2d_pairs.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", help="folders containing min_set.pkl")
    parser.add_argument("--max_points", type=int, default=30000,
                        help="Subsample to this many points if min_set has more "
                             "(matplotlib 3D scatter is slow above ~50k). "
                             "Set 0 to disable.")
    parser.add_argument("--seed", type=int, default=0, help="Subsample RNG seed.")
    args = parser.parse_args()

    for folder in args.folders:
        path = os.path.join(folder, "min_set.pkl")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        print(f"Processing {folder}")
        min_set_real, min_set_complex = load_min_set(folder)
        k = min_set_real.shape[0]
        print(f"  Loaded {k} points in {min_set_real.shape[1]}D")

        if args.max_points and k > args.max_points:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(k, args.max_points, replace=False)
            min_set_real = min_set_real[idx]
            min_set_complex = min_set_complex[idx]
            print(f"  Subsampled to {args.max_points} points")

        # PCA
        proj, ratio = pca_project(min_set_real, n_components=3)
        cum = np.cumsum(ratio)
        print(f"  Explained variance ratio (all 10 PCs):")
        for i, (r, c) in enumerate(zip(ratio, cum)):
            print(f"    PC{i+1}: {r:.4f}   (cumulative {c:.4f})")
        print(f"  Top 3 capture {cum[2]:.3f} of total variance")

        patches = patch_indices_from_complex(min_set_complex)
        title = f"{os.path.basename(folder.rstrip('/'))}  (top 3 PCs: {cum[2]:.1%})"
        plot_pca_3d(proj, patches, folder, title)
        plot_pca_2d_pairs(proj, patches, folder, title)


if __name__ == "__main__":
    main()
