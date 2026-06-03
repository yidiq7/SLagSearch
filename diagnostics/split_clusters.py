"""Split a sLag point cloud into separately-connected pieces via UMAP +
KMeans, so each piece can be analyzed in isolation. The output files are in
the same (N, 5) complex format as min_set.pkl, so the persistent_homology/
scripts (or anything else that consumes min_set.pkl) can be pointed at each
cluster directly.

Pipeline:
  1. Load --min_set  (N, 5) complex pickle.
  2. Build the 25-dim FS feature embedding (projective-invariant).
  3. Run UMAP -> 3D embedding.
  4. KMeans with --n_clusters on the chosen --basis ('umap' = on the 3D
     embedding, matches visual split; 'fs' = on the 25D FS features, the
     'principled' option that ignores UMAP's distortions).
  5. Save each cluster as cluster_<k>_points.pkl (complex (n_k, 5)).
  6. Save a 4-view PNG showing the split in UMAP space.

Output goes to --out_dir (full path) or <min_set_dir>/<out_subdir>/ (subdir
name, default 'cluster_split'); these flags are mutually exclusive.

Usage:
    python -m diagnostics.split_clusters --min_set plots_slag_run/min_set.pkl \\
        [--n_clusters 2] [--basis umap]
    # then for each cluster:
    python persistent_homology/persistent_homology_witness.py \\
        --min_set plots_slag_run/cluster_split/cluster_0_points.pkl
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viz.plot_3D import (
    load_min_set_complex, patch_indices_from_complex, subsample,
    to_features, _run_umap, _PATCH_COLORS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min_set", type=Path, required=True,
                        help="Path to an (N, 5) complex pickle (e.g. min_set.pkl).")
    parser.add_argument("--n_clusters", type=int, default=2,
                        help="Number of clusters (default 2).")
    parser.add_argument("--basis", choices=["umap", "fs"], default="umap",
                        help="Where to cluster. 'umap' (default): KMeans "
                             "on the 3D UMAP embedding, so clusters match "
                             "what you see in the UMAP picture. 'fs': "
                             "KMeans on the 25D FS features directly -- "
                             "ignores any UMAP-induced distortion.")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Subsample to this many points first "
                             "(default: use all).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--umap_n_neighbors", type=int, default=100)
    parser.add_argument("--umap_min_dist", type=float, default=0.3)
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument("--out_dir", type=Path, default=None,
                           help="Full output directory. "
                                "Default: <min_set_dir>/cluster_split/.")
    out_group.add_argument("--out_subdir", type=str, default="cluster_split",
                           help="Subdirectory name appended to --min_set's "
                                "parent directory. Default: 'cluster_split'.")
    args = parser.parse_args()

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("scikit-learn missing. `uv sync --extra viz-3d`.")
        return

    z = load_min_set_complex(args.min_set)
    patches = patch_indices_from_complex(z)
    print(f"Loaded {z.shape[0]} points from {args.min_set}")
    print(f"Patch counts: " + ", ".join(
        f"p{i}:{int((patches == i).sum())}" for i in range(5)))

    z_sub, patches_sub = subsample(z, patches, args.max_points,
                                   seed=args.seed)
    X_fs = to_features(z_sub, "fs")
    print(f"\nFS feature embedding: {X_fs.shape}")

    print(f"Running UMAP (n_neighbors={args.umap_n_neighbors}, "
          f"min_dist={args.umap_min_dist})...")
    emb = _run_umap(X_fs, args.umap_n_neighbors, args.umap_min_dist,
                    args.seed)
    print(f"UMAP done; embedding shape = {emb.shape}")

    cluster_basis = emb if args.basis == "umap" else X_fs
    print(f"\nKMeans on {args.basis} basis (k={args.n_clusters})...")
    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed,
                n_init=10)
    labels = km.fit_predict(cluster_basis)
    print("  cluster sizes: " + ", ".join(
        f"c{c}:{int((labels == c).sum())}" for c in range(args.n_clusters)
    ))

    # Cross-tab of cluster vs patch (informative if patches partition by
    # cluster, which would corroborate the {0,4} | {1,2,3} reading).
    print("\n  cluster x patch cross-tab:")
    print("            " + "  ".join(f" p{p}  " for p in range(5)))
    for c in range(args.n_clusters):
        row = [int(((labels == c) & (patches_sub == p)).sum())
               for p in range(5)]
        print(f"   c{c}:    " + "  ".join(f"{v:5d}" for v in row))

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = args.min_set.parent / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # If the input min_set lives next to a coeffs.pkl sidecar (the
    # viz.fitness_pipeline run-folder contract), propagate it so the
    # downstream cluster-fitness / persistent-homology auto-discovery works
    # against the cluster_split/ folder.
    src_coeffs = args.min_set.parent / "coeffs.pkl"
    if src_coeffs.exists():
        import shutil
        dst_coeffs = out_dir / "coeffs.pkl"
        if not dst_coeffs.exists():
            shutil.copy2(src_coeffs, dst_coeffs)
            print(f"  copied coeffs sidecar {src_coeffs} -> {dst_coeffs}")

    # ----- save each cluster's complex points (min_set.pkl format) -----
    for c in range(args.n_clusters):
        mask = labels == c
        points_c = z_sub[mask]
        out_path = out_dir / f"cluster_{c}_points.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(points_c, f)
        print(f"  wrote {out_path}  ({int(mask.sum())} points)")

    # ----- visualization PNG -----
    angles = [(15, az) for az in (0, 90, 180, 270)]
    alpha = float(min(0.5, 12000.0 / max(emb.shape[0], 1)))
    # Use a distinct color palette for clusters (not patch colors).
    cluster_palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a",
                       "#66a61e", "#e6ab02"]
    fig = plt.figure(figsize=(20, 5))
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        for c in range(args.n_clusters):
            mask = labels == c
            ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2],
                       s=0.8, alpha=alpha,
                       color=cluster_palette[c % len(cluster_palette)],
                       label=(f"c{c}: {int(mask.sum())}" if i == 0 else None),
                       edgecolors="none")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"az={azim}", fontsize=9)
        if i == 0:
            ax.legend(fontsize=9, loc="upper right")
    fig.suptitle(
        f"Cluster split:  {args.n_clusters} clusters via KMeans on "
        f"{args.basis} basis,  UMAP view  "
        f"(nn={args.umap_n_neighbors}, md={args.umap_min_dist}, "
        f"N={emb.shape[0]})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png_path = out_dir / "cluster_split.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {png_path}")

    # ----- interactive HTML so you can verify the split looks right -----
    try:
        import plotly.graph_objects as go
        traces = []
        for c in range(args.n_clusters):
            mask = labels == c
            if not mask.any():
                continue
            traces.append(go.Scatter3d(
                x=emb[mask, 0], y=emb[mask, 1], z=emb[mask, 2],
                mode="markers",
                marker=dict(
                    size=1.5,
                    color=cluster_palette[c % len(cluster_palette)],
                    opacity=float(min(0.6, 12000.0 / max(emb.shape[0], 1))),
                ),
                name=f"cluster {c}  ({int(mask.sum())} pts)",
            ))
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"Cluster split  ({args.n_clusters} clusters, "
                  f"basis={args.basis})",
            scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2",
                       zaxis_title="UMAP 3", aspectmode="data"),
            legend=dict(itemsizing="constant"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        html_path = out_dir / "cluster_split.html"
        fig.write_html(str(html_path))
        print(f"  wrote {html_path}")
    except ImportError:
        pass

    # ----- next-steps printout -----
    print(f"\nNext: run persistent homology on each cluster, e.g.")
    for c in range(args.n_clusters):
        print(f"  uv run python persistent_homology/persistent_homology_witness.py \\")
        print(f"      --min_set {out_dir}/cluster_{c}_points.pkl")


if __name__ == "__main__":
    main()
