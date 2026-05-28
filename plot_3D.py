"""3D topology-aware visualizations of min_set.pkl point clouds.

Three methods, all colored by affine patch index to match the 2D plots:

  coord  : coordinate-aligned 3D scatter for a small set of physically
           interesting triples (|z_0|, |z_4|, |z_1|; (Re z_2, Re z_3, Re z_0);
           etc.). Always renders three orbit views per triple.

  umap   : UMAP embedding of the (N, 10) real point cloud into 3D. Preserves
           local neighborhoods + component structure, so a real
           two-piece-with-neck topology should show up as two clusters
           joined by a thin bridge. Requires `umap-learn`.

  mapper : Topological-data-analysis Mapper graph: clusters the point
           cloud along level sets of a filter function and draws the
           resulting simplicial 1-complex. Loops in the graph correspond
           to nontrivial H_1 generators (we expect 5 of them). Requires
           `kmapper` and `networkx`. Writes both an interactive HTML
           (kepler-mapper default) and a static PNG.

Usage:
    python plot_3D.py <folder> [--methods coord umap mapper]
                              [--max_points N] [--filter coord-pc|first-coord|...]
"""
import argparse
import os
import pickle
from itertools import combinations
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # registers 3d projection


# ---------------------------------------------------------------------------
#  Shared utilities
# ---------------------------------------------------------------------------
def load_min_set_complex(folder: Path) -> np.ndarray:
    with open(folder / "min_set.pkl", "rb") as f:
        z = np.asarray(pickle.load(f))  # (N, 5) complex
    if z.ndim != 2 or z.shape[1] != 5:
        raise ValueError(f"expected (N, 5) complex array, got {z.shape}")
    return z


def patch_indices_from_complex(z: np.ndarray) -> np.ndarray:
    """Affine patch = argmax_i |z_i| per point."""
    return np.argmax(np.abs(z), axis=1)


def subsample(z: np.ndarray, patches: np.ndarray, n: int,
              seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if n is None or z.shape[0] <= n:
        return z, patches
    idx = np.random.default_rng(seed).choice(z.shape[0], n, replace=False)
    return z[idx], patches[idx]


def real_form(z: np.ndarray) -> np.ndarray:
    """(N, 5) complex -> (N, 10) real (concat Re and Im)."""
    return np.concatenate([z.real, z.imag], axis=1)


# ---------------------------------------------------------------------------
#  Method 1: coordinate-aligned 3D scatter
# ---------------------------------------------------------------------------
def _coord_value(z: np.ndarray, spec: str) -> tuple[np.ndarray, str]:
    """Resolve a coord spec like 'Re z_0', 'Im z_4', '|z_1|' into a (N,)
    array plus an axis label string.
    """
    spec = spec.strip()
    if spec.startswith("|") and spec.endswith("|"):
        i = int(spec.strip("| z_"))
        return np.abs(z[:, i]), rf"$|z_{i}|$"
    if spec.lower().startswith("re"):
        i = int(spec[2:].strip(" z_"))
        return z[:, i].real, rf"$\mathrm{{Re}}\,z_{i}$"
    if spec.lower().startswith("im"):
        i = int(spec[2:].strip(" z_"))
        return z[:, i].imag, rf"$\mathrm{{Im}}\,z_{i}$"
    raise ValueError(f"unknown coord spec: {spec!r}")


# Curated triples chosen to highlight specific structures.
DEFAULT_TRIPLES: list[tuple[str, str, str, str]] = [
    # (label, x_spec, y_spec, z_spec)
    ("partition_abs",  "|z_0|", "|z_4|", "|z_1|"),
    ("partition_abs2", "|z_0|", "|z_4|", "|z_2|"),
    ("symmetry_23",    "Re z_2", "Re z_3", "Re z_0"),
    ("symmetry_23_im", "Im z_2", "Im z_3", "Im z_0"),
    ("pair_04",        "Re z_0", "Im z_0", "|z_4|"),
    ("pair_04_swap",   "Re z_4", "Im z_4", "|z_0|"),
]


def plot_coord_triples(z: np.ndarray, patches: np.ndarray, out_dir: Path,
                       triples: Sequence[tuple[str, str, str, str]],
                       max_points: int) -> None:
    """For each triple, render one 3D scatter from three viewing angles."""
    z_sub, patches_sub = subsample(z, patches, max_points)
    angles = [(20, 45), (20, 135), (60, 30)]

    for label, sx, sy, sz in triples:
        try:
            x, xl = _coord_value(z_sub, sx)
            y, yl = _coord_value(z_sub, sy)
            zc, zl = _coord_value(z_sub, sz)
        except ValueError as e:
            print(f"  skipping {label}: {e}")
            continue

        fig = plt.figure(figsize=(15, 5))
        for k, (elev, azim) in enumerate(angles):
            ax = fig.add_subplot(1, 3, k + 1, projection="3d")
            ax.scatter(x, y, zc, c=patches_sub,
                       cmap="tab10", vmin=-0.5, vmax=4.5,
                       s=1.0, alpha=0.4, edgecolors="none")
            ax.set_xlabel(xl, fontsize=9)
            ax.set_ylabel(yl, fontsize=9)
            ax.set_zlabel(zl, fontsize=9)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"view {k + 1}  (elev={elev}, az={azim})", fontsize=9)

        fig.suptitle(f"{label}  :  ({sx}, {sy}, {sz})", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_path = out_dir / f"coord3d_{label}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
#  Method 2: UMAP 3D embedding
# ---------------------------------------------------------------------------
def plot_umap_3d(z: np.ndarray, patches: np.ndarray, out_dir: Path,
                 n_neighbors: int, min_dist: float, max_points: int,
                 seed: int) -> None:
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed. `uv pip install umap-learn` "
              "(or `uv add umap-learn`) to enable.")
        return

    z_sub, patches_sub = subsample(z, patches, max_points, seed=seed)
    X = real_form(z_sub)
    print(f"  UMAP on {X.shape[0]} points, n_neighbors={n_neighbors}, "
          f"min_dist={min_dist}")
    reducer = umap.UMAP(
        n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
        random_state=seed, verbose=False,
    )
    emb = reducer.fit_transform(X)

    angles = [(20, 45), (20, 135), (60, 30)]
    fig = plt.figure(figsize=(15, 5))
    for k, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=patches_sub,
                   cmap="tab10", vmin=-0.5, vmax=4.5,
                   s=1.0, alpha=0.4, edgecolors="none")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"view {k + 1}", fontsize=9)
    fig.suptitle(f"UMAP 3D (n_neighbors={n_neighbors}, min_dist={min_dist})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = out_dir / f"umap3d_nn{n_neighbors}_md{min_dist}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
#  Method 3: Mapper graph
# ---------------------------------------------------------------------------
def plot_mapper(z: np.ndarray, patches: np.ndarray, out_dir: Path,
                filter_spec: str, n_cubes: int, perc_overlap: float,
                max_points: int, seed: int) -> None:
    try:
        import kmapper as km
        import networkx as nx
        from sklearn.cluster import DBSCAN
    except ImportError:
        print("  kmapper or networkx not installed. "
              "`uv pip install kmapper networkx scikit-learn` to enable.")
        return

    z_sub, patches_sub = subsample(z, patches, max_points, seed=seed)
    X = real_form(z_sub)

    # Filter (lens) function.
    if filter_spec == "first-pc":
        # Project onto first PC.
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        lens = Xc @ Vt[0]
        lens = lens.reshape(-1, 1)
        lens_label = "first PC"
    elif filter_spec == "abs0_minus_abs4":
        # |z_0| - |z_4| -- designed to separate the two pieces under the
        # hypothesized partition {0,4} | {1,2,3}.
        lens = (np.abs(z_sub[:, 0]) - np.abs(z_sub[:, 4])).reshape(-1, 1)
        lens_label = "|z_0| - |z_4|"
    elif filter_spec == "abs0":
        lens = np.abs(z_sub[:, 0]).reshape(-1, 1)
        lens_label = "|z_0|"
    else:
        raise ValueError(f"unknown --filter {filter_spec!r}")

    mapper = km.KeplerMapper(verbose=1)
    cover = km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap)
    clusterer = DBSCAN(eps=0.15, min_samples=5)
    graph = mapper.map(
        lens, X,
        cover=cover,
        clusterer=clusterer,
    )
    n_nodes = len(graph["nodes"])
    n_edges = sum(len(v) for v in graph["links"].values())
    print(f"  Mapper: {n_nodes} nodes, {n_edges} edges "
          f"(filter={lens_label}, n_cubes={n_cubes}, "
          f"overlap={perc_overlap})")

    # Interactive HTML.
    html_path = out_dir / f"mapper_{filter_spec}.html"
    mapper.visualize(graph, path_html=str(html_path),
                     title=f"Mapper on min_set ({lens_label})",
                     color_values=patches_sub.astype(float),
                     color_function_name="patch index")
    print(f"  wrote {html_path}")

    # Static PNG via networkx.
    G = nx.Graph()
    for node_id, members in graph["nodes"].items():
        # Color each node by majority patch among its members.
        majority = int(np.bincount(patches_sub[members], minlength=5).argmax())
        G.add_node(node_id, size=len(members), patch=majority)
    for src, tgts in graph["links"].items():
        for tgt in tgts:
            G.add_edge(src, tgt)

    cmap = plt.get_cmap("tab10")
    node_colors = [cmap(G.nodes[n]["patch"] / 9.0) for n in G.nodes]
    node_sizes = [max(20, G.nodes[n]["size"]) for n in G.nodes]

    pos = nx.spring_layout(G, seed=seed)
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           alpha=0.85, edgecolors="black", linewidths=0.3)
    ax.set_title(
        f"Mapper graph  (filter={lens_label}, {n_nodes} nodes, "
        f"{n_edges} edges)\n"
        f"node size ~ cluster size, color = majority patch",
        fontsize=11,
    )
    ax.axis("off")
    png_path = out_dir / f"mapper_{filter_spec}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png_path}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path,
                        help="Folder containing min_set.pkl.")
    parser.add_argument("--methods", nargs="+",
                        choices=["coord", "umap", "mapper", "all"],
                        default=["all"],
                        help="Which methods to run (default: all).")
    parser.add_argument("--max_points", type=int, default=20000,
                        help="Subsample to this many points (default 20000).")
    parser.add_argument("--seed", type=int, default=0)

    # UMAP knobs.
    parser.add_argument("--umap_n_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.05)

    # Mapper knobs.
    parser.add_argument("--mapper_filter",
                        choices=["first-pc", "abs0_minus_abs4", "abs0"],
                        default="abs0_minus_abs4",
                        help="Lens function (default: |z_0| - |z_4|, which "
                             "separates the two pieces under the "
                             "{0,4} | {1,2,3} partition).")
    parser.add_argument("--mapper_n_cubes", type=int, default=20)
    parser.add_argument("--mapper_overlap", type=float, default=0.3)
    args = parser.parse_args()

    z = load_min_set_complex(args.folder)
    patches = patch_indices_from_complex(z)
    print(f"Loaded {z.shape[0]} points from {args.folder}/min_set.pkl")
    print(f"Patch counts: " + ", ".join(
        f"p{i}:{int((patches == i).sum())}" for i in range(5)))

    methods = set(args.methods)
    if "all" in methods:
        methods = {"coord", "umap", "mapper"}

    if "coord" in methods:
        print("\n--- coord-aligned 3D scatters ---")
        plot_coord_triples(z, patches, args.folder, DEFAULT_TRIPLES,
                           max_points=args.max_points)

    if "umap" in methods:
        print("\n--- UMAP 3D ---")
        plot_umap_3d(z, patches, args.folder,
                     n_neighbors=args.umap_n_neighbors,
                     min_dist=args.umap_min_dist,
                     max_points=args.max_points,
                     seed=args.seed)

    if "mapper" in methods:
        print("\n--- Mapper ---")
        plot_mapper(z, patches, args.folder,
                    filter_spec=args.mapper_filter,
                    n_cubes=args.mapper_n_cubes,
                    perc_overlap=args.mapper_overlap,
                    max_points=args.max_points,
                    seed=args.seed)


if __name__ == "__main__":
    main()
