"""3D topology-aware visualizations of min_set.pkl point clouds.

Three methods, all colored by affine patch index to match the 2D plots:

  coord  : coordinate-aligned 3D scatter for a small set of physically
           interesting triples (|z_0|, |z_4|, |z_1|; (Re z_2, Re z_3, Re z_0);
           etc.). Always renders three orbit views per triple.

  umap   : UMAP embedding into 3D. Preserves local neighborhoods + component
           structure. Requires `umap-learn`.

  mapper : Topological-data-analysis Mapper graph: clusters the point
           cloud along level sets of a filter function and draws the
           resulting simplicial 1-complex. Loops in the graph correspond
           to nontrivial H_1 generators. Requires `kmapper` and `networkx`.
           Writes both an interactive HTML (kepler-mapper default) and a
           static PNG.

  intrinsic_dim
         : Three intrinsic-dimension estimators in one diagnostic PNG:
           TwoNN (Facco 2017), Levina-Bickel MLE histograms over k in
           {10, 20, 50}, and Schweinhart-style PH-dim (scaling of total
           H_0 persistence = MST edge-weight sum vs sub-sample size N).
           For a sLag in a CY 3-fold the expected answer is d = 3.

For umap and mapper, the distance metric on min_set is selectable via
--metric:
  - 'euclidean': raw 10D real coords. Patch-dependent (a point's 10D
    representation depends on which affine patch normalizes it), so two
    physically-close points in different patches can look far apart.
  - 'fs' (default): map each point z to its rank-1 projector P_z =
    z conj(z)^T / ||z||^2 flattened to 25 real DOF. Euclidean distance on
    these features equals sqrt(2) * sin(d_FS) where d_FS is Fubini-Study
    distance on CP^4 -- monotone with FS, patch-independent, exact (not
    a small-distance approximation).

Usage:
    python plot_3D.py <folder> [--methods coord umap mapper]
                              [--metric fs|euclidean]
                              [--sweep] [--max_points N]
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


def subsample(z: np.ndarray, patches: np.ndarray, n: int | None,
              seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if n is None or z.shape[0] <= n:
        return z, patches
    idx = np.random.default_rng(seed).choice(z.shape[0], n, replace=False)
    return z[idx], patches[idx]


def _auto_alpha(n: int) -> float:
    """Marker alpha that doesn't saturate for large clouds."""
    return float(min(0.4, 8000.0 / max(n, 1)))


def real_form(z: np.ndarray) -> np.ndarray:
    """(N, 5) complex -> (N, 10) real (concat Re and Im).

    Patch-dependent: two points that project to the same point in CP^4 can
    have very different 10-vectors here. Use to_features(..., metric='fs')
    instead when feeding distance-sensitive tools like UMAP / Mapper.
    """
    return np.concatenate([z.real, z.imag], axis=1)


def fs_feature_embedding(z: np.ndarray) -> np.ndarray:
    """(N, 5) complex -> (N, 25) real such that Euclidean distance on the
    features equals the Frobenius distance between rank-1 projector matrices
    P_z = z conj(z)^T / ||z||^2. Concretely

        ||P_z - P_w||_F^2  =  2 - 2 |<z, w>|^2 / (||z||^2 ||w||^2)
                           =  2 sin^2(d_FS)

    so Euclidean distance on these features equals sqrt(2) * sin(d_FS),
    monotone in Fubini-Study distance on CP^4 over [0, pi/2]. The mapping
    is intrinsically projective-invariant -- the projector P_z is unchanged
    by z -> lambda*z for any nonzero lambda, so no per-point phase fix or
    norm fix is needed. Patch labels become irrelevant.

    Layout of the 25 real features:
      [0:5]   diag entries P_ii = |z_i|^2 / ||z||^2 (5 reals)
      [5:15]  sqrt(2) * Re(P_ij) for i < j (10 reals)
      [15:25] sqrt(2) * Im(P_ij) for i < j (10 reals)
    The sqrt(2) weighting on off-diagonals is so that Euclidean distance on
    the 25-vector recovers the full Frobenius norm of the 5x5 Hermitian
    (which counts each off-diagonal twice via Hermiticity).
    """
    norm_sq = (z.conj() * z).sum(axis=1).real  # (N,)
    z_n = z / np.sqrt(norm_sq)[:, None]
    P = z_n[:, :, None] * z_n[:, None, :].conj()  # (N, 5, 5) Hermitian
    iu1 = np.triu_indices(5, k=1)
    diag = np.diagonal(P, axis1=1, axis2=2).real  # (N, 5)
    off_re = P[:, iu1[0], iu1[1]].real * np.sqrt(2.0)  # (N, 10)
    off_im = P[:, iu1[0], iu1[1]].imag * np.sqrt(2.0)  # (N, 10)
    return np.concatenate([diag, off_re, off_im], axis=1).astype(np.float64)


def to_features(z: np.ndarray, metric: str) -> np.ndarray:
    """Dispatch: build the feature matrix used by UMAP / Mapper.

    metric='euclidean': raw 10D real form (patch-dependent).
    metric='fs':        25D projector embedding (projective-invariant,
                        Euclidean = sqrt(2)*sin(d_FS), monotone with FS).
    """
    if metric == "euclidean":
        return real_form(z)
    if metric == "fs":
        return fs_feature_embedding(z)
    raise ValueError(f"unknown metric: {metric!r}")


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
    alpha = _auto_alpha(z_sub.shape[0])

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
                       s=1.0, alpha=alpha, edgecolors="none")
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
def _run_umap(X: np.ndarray, n_neighbors: int, min_dist: float,
              seed: int) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
        random_state=seed, verbose=False,
    )
    return reducer.fit_transform(X)


def plot_umap_3d(z: np.ndarray, patches: np.ndarray, out_dir: Path,
                 n_neighbors: int, min_dist: float, max_points: int,
                 seed: int, metric: str = "fs",
                 sweep: bool = False) -> None:
    try:
        import umap  # noqa: F401
    except ImportError:
        print("  umap-learn not installed. `uv sync --extra viz-3d` to "
              "enable.")
        return

    z_sub, patches_sub = subsample(z, patches, max_points, seed=seed)
    X = to_features(z_sub, metric)
    print(f"  feature dim: {X.shape[1]} ({metric} metric)")
    angles = [(20, 45), (20, 135), (60, 30)]

    if not sweep:
        print(f"  UMAP on {X.shape[0]} points, n_neighbors={n_neighbors}, "
              f"min_dist={min_dist}")
        emb = _run_umap(X, n_neighbors, min_dist, seed)
        fig = plt.figure(figsize=(15, 5))
        for k, (elev, azim) in enumerate(angles):
            ax = fig.add_subplot(1, 3, k + 1, projection="3d")
            ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=patches_sub,
                       cmap="tab10", vmin=-0.5, vmax=4.5,
                       s=1.0, alpha=_auto_alpha(emb.shape[0]),
                       edgecolors="none")
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_zlabel("UMAP 3")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"view {k + 1}", fontsize=9)
        fig.suptitle(f"UMAP 3D (metric={metric}, n_neighbors={n_neighbors}, "
                     f"min_dist={min_dist})", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_path = out_dir / f"umap3d_{metric}_nn{n_neighbors}_md{min_dist}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  wrote {out_path}")
        return

    # Sweep: 3x3 grid over (n_neighbors, min_dist), single viewing angle.
    nn_values = [50, 100, 200]
    md_values = [0.05, 0.3, 0.5]
    fig = plt.figure(figsize=(15, 15))
    for i, nn_v in enumerate(nn_values):
        for j, md_v in enumerate(md_values):
            print(f"  sweep cell ({i},{j}): nn={nn_v}, md={md_v}")
            emb = _run_umap(X, nn_v, md_v, seed)
            ax = fig.add_subplot(3, 3, i * 3 + j + 1, projection="3d")
            ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=patches_sub,
                       cmap="tab10", vmin=-0.5, vmax=4.5,
                       s=0.6, alpha=_auto_alpha(emb.shape[0]),
                       edgecolors="none")
            ax.view_init(elev=20, azim=45)
            ax.set_title(f"nn={nn_v}, md={md_v}", fontsize=10)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"UMAP 3D sweep  (metric={metric}, N={X.shape[0]})",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = out_dir / f"umap3d_{metric}_sweep.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
#  Method 3: Mapper graph
# ---------------------------------------------------------------------------
def _lens_from_spec(z_sub: np.ndarray, X: np.ndarray,
                    filter_spec: str) -> tuple[np.ndarray, str]:
    # ----- 1D filters -----
    if filter_spec == "first-pc":
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        lens = (Xc @ Vt[0]).reshape(-1, 1)
        return lens, "first PC"
    if filter_spec == "abs0_minus_abs4":
        lens = (np.abs(z_sub[:, 0]) - np.abs(z_sub[:, 4])).reshape(-1, 1)
        return lens, "|z_0| - |z_4|"
    if filter_spec == "abs0":
        return np.abs(z_sub[:, 0]).reshape(-1, 1), "|z_0|"
    # ----- 2D filters (n_cubes^2 cover cells -> many more nodes) -----
    if filter_spec == "abs04_abs123":
        # Axis 1: |z_0| - |z_4|  (separates the two suspected pieces)
        # Axis 2: |z_1| + |z_2| + |z_3|  (orthogonal: overall scale of triple)
        a = np.abs(z_sub[:, 0]) - np.abs(z_sub[:, 4])
        b = np.abs(z_sub[:, 1]) + np.abs(z_sub[:, 2]) + np.abs(z_sub[:, 3])
        return np.stack([a, b], axis=1), "|z_0|-|z_4|  vs  Σ|z_{1,2,3}|"
    if filter_spec == "first-two-pc":
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:2].T, "first 2 PCs"
    raise ValueError(f"unknown --filter {filter_spec!r}")


def _autoscale_eps(X: np.ndarray, k: int = 5,
                   percentile: float = 90.0) -> float:
    """Robust DBSCAN eps = percentile of k-th nearest-neighbor distance.

    Defaults: k=5 (matches min_samples=5), percentile=90 (eps large enough
    that 90% of points have >=5 neighbors within eps).
    """
    from sklearn.neighbors import NearestNeighbors
    sub_n = min(X.shape[0], 5000)
    idx = np.random.default_rng(0).choice(X.shape[0], sub_n, replace=False)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X[idx])
    d, _ = nn.kneighbors(X[idx])
    return float(np.percentile(d[:, k], percentile))


def _build_mapper_graph(z_sub, patches_sub, X, filter_spec, n_cubes,
                        perc_overlap, eps_percentile):
    import kmapper as km
    from sklearn.cluster import DBSCAN
    lens, lens_label = _lens_from_spec(z_sub, X, filter_spec)
    eps = _autoscale_eps(X, k=5, percentile=eps_percentile)
    mapper = km.KeplerMapper(verbose=0)
    cover = km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap)
    clusterer = DBSCAN(eps=eps, min_samples=5)
    graph = mapper.map(lens, X, cover=cover, clusterer=clusterer)
    n_nodes = len(graph["nodes"])
    n_edges = sum(len(v) for v in graph["links"].values())
    return mapper, graph, lens_label, eps, n_nodes, n_edges


def _render_mapper_png(graph, patches_sub, ax, title, seed):
    import networkx as nx
    G = nx.Graph()
    for node_id, members in graph["nodes"].items():
        majority = int(np.bincount(patches_sub[members], minlength=5).argmax())
        G.add_node(node_id, size=len(members), patch=majority)
    for src, tgts in graph["links"].items():
        for tgt in tgts:
            G.add_edge(src, tgt)
    cmap = plt.get_cmap("tab10")
    node_colors = [cmap(G.nodes[n]["patch"] / 9.0) for n in G.nodes]
    node_sizes = [max(20, G.nodes[n]["size"]) for n in G.nodes]
    pos = nx.spring_layout(G, seed=seed)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           alpha=0.85, edgecolors="black", linewidths=0.3)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_mapper(z: np.ndarray, patches: np.ndarray, out_dir: Path,
                filter_spec: str, n_cubes: int, perc_overlap: float,
                eps_percentile: float, max_points: int, seed: int,
                metric: str = "fs", sweep: bool = False) -> None:
    try:
        import kmapper as km  # noqa: F401
        import networkx as nx  # noqa: F401
        from sklearn.cluster import DBSCAN  # noqa: F401
    except ImportError:
        print("  kmapper / networkx / sklearn missing. "
              "`uv sync --extra viz-3d` to enable.")
        return

    z_sub, patches_sub = subsample(z, patches, max_points, seed=seed)
    X = to_features(z_sub, metric)
    print(f"  feature dim: {X.shape[1]} ({metric} metric)")

    if not sweep:
        mapper, graph, lens_label, eps, n_nodes, n_edges = _build_mapper_graph(
            z_sub, patches_sub, X, filter_spec, n_cubes, perc_overlap,
            eps_percentile,
        )
        print(f"  Mapper: {n_nodes} nodes, {n_edges} edges  "
              f"(filter={lens_label}, n_cubes={n_cubes}, "
              f"overlap={perc_overlap}, eps={eps:.4f} "
              f"@p{eps_percentile})")

        html_path = out_dir / f"mapper_{metric}_{filter_spec}.html"
        mapper.visualize(graph, path_html=str(html_path),
                         title=f"Mapper ({metric} metric, {lens_label})",
                         color_values=patches_sub.astype(float),
                         color_function_name="patch index")
        print(f"  wrote {html_path}")

        fig, ax = plt.subplots(figsize=(10, 10))
        _render_mapper_png(
            graph, patches_sub, ax,
            f"Mapper  (metric={metric}, filter={lens_label}, "
            f"{n_nodes} nodes, {n_edges} edges)\n"
            f"n_cubes={n_cubes}, overlap={perc_overlap}, eps={eps:.4f}",
            seed,
        )
        png_path = out_dir / f"mapper_{metric}_{filter_spec}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {png_path}")
        return

    # Sweep: 3x3 grid over (n_cubes, overlap). eps stays auto-scaled
    # at the single eps_percentile. n_cubes is the main knob for node
    # count; overlap is the main knob for edge count.
    n_cubes_list = [20, 50, 100]
    overlap_list = [0.3, 0.5, 0.7]
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    for i, nc in enumerate(n_cubes_list):
        for j, ov in enumerate(overlap_list):
            print(f"  sweep ({i},{j}): n_cubes={nc}, overlap={ov}")
            _, graph, lens_label, eps, n_nodes, n_edges = _build_mapper_graph(
                z_sub, patches_sub, X, filter_spec, nc, ov, eps_percentile,
            )
            _render_mapper_png(
                graph, patches_sub, axes[i, j],
                f"n_cubes={nc}, ov={ov}\n"
                f"{n_nodes} nodes, {n_edges} edges",
                seed,
            )
    fig.suptitle(f"Mapper sweep  (metric={metric}, filter={filter_spec}, "
                 f"eps@p{eps_percentile}, N={X.shape[0]})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = out_dir / f"mapper_sweep_{metric}_{filter_spec}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
#  Method 4: intrinsic dimension estimators
# ---------------------------------------------------------------------------
def _twonn_estimate(X: np.ndarray, n_jobs: int = -1
                    ) -> tuple[float, np.ndarray, np.ndarray]:
    """TwoNN (Facco et al. 2017). Returns (d_hat, log_mu, -log(1 - F)).

    For each point, take 1st and 2nd NN distances r_1 < r_2; ratio
    mu = r_2 / r_1 has CDF 1 - mu^{-d} under locally-uniform Poisson
    sampling on a d-manifold. Linear fit slope through origin estimates d.
    Fits only the F < 0.9 region (tail deviates from the model).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=3, n_jobs=n_jobs).fit(X)
    dists, _ = nn.kneighbors(X)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    # Drop degenerate points (duplicates) where r_1 == 0 and mu undefined.
    mask = r1 > 0
    mu = r2[mask] / r1[mask]
    mu = mu[mu > 1.0 + 1e-12]
    mu_sorted = np.sort(mu)
    N = len(mu_sorted)
    # Use (i - 0.5)/N to avoid F=1 at the top (log divergence).
    F = (np.arange(1, N + 1) - 0.5) / N
    log_mu = np.log(mu_sorted)
    nlog_1mF = -np.log(1 - F)
    # Linear fit through origin on the lower 90% of the empirical CDF.
    fit_mask = F < 0.9
    x = log_mu[fit_mask]
    y = nlog_1mF[fit_mask]
    d_hat = float((x * y).sum() / (x * x).sum())
    return d_hat, log_mu, nlog_1mF


def _mle_estimate(X: np.ndarray, k_values: tuple[int, ...] = (10, 20, 50),
                  n_jobs: int = -1) -> dict[int, np.ndarray]:
    """Levina-Bickel MLE with MacKay-Ghahramani (k-2) correction.

    Per-point estimate
       d_hat_k(x) = [ (1/(k-2)) * sum_{j=1..k-1} log(r_k / r_j) ]^{-1}

    Returns dict {k: array of per-point d estimates}.
    """
    from sklearn.neighbors import NearestNeighbors

    k_max = max(k_values)
    nn = NearestNeighbors(n_neighbors=k_max + 1, n_jobs=n_jobs).fit(X)
    dists, _ = nn.kneighbors(X)  # dists[:, 0] = self distance (0)

    results: dict[int, np.ndarray] = {}
    for k in k_values:
        rk = dists[:, k]                # (N,) k-th NN distance
        rj = dists[:, 1:k]              # (N, k-1) distances to 1..k-1 NN
        mask = (rj > 0).all(axis=1) & (rk > 0)
        log_ratio = np.log(rk[mask, None] / rj[mask])  # (N', k-1)
        inv_d = log_ratio.sum(axis=1) / (k - 2)
        d_per_point = 1.0 / inv_d
        finite = np.isfinite(d_per_point) & (d_per_point > 0)
        results[k] = d_per_point[finite]
    return results


def _phdim_estimate(X: np.ndarray, n_samples: int = 6, smallest: int = 500,
                    largest: int | None = None, alpha: float = 1.0,
                    n_trials: int = 3, k_for_graph: int = 30,
                    seed: int = 0
                    ) -> tuple[float, np.ndarray, np.ndarray, float]:
    """PH-dim via Schweinhart-style scaling of H_0 total persistence.

    H_0 persistence diagram bar lengths == minimum-spanning-tree edge
    weights. Total persistence E_alpha(N) = sum |e_MST|^alpha scales as
    N^{(d-alpha)/d} for an underlying d-manifold (Schweinhart 2020). Fit
    slope of log E vs log N to extract d = alpha / (1 - slope).

    Sub-samples are drawn at n_samples geometrically-spaced sizes between
    `smallest` and `largest` (defaults to min(N, 20000)).
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import minimum_spanning_tree

    if largest is None:
        largest = min(X.shape[0], 20000)
    if largest <= smallest:
        smallest = max(largest // 4, 50)
    sample_sizes = np.unique(np.round(
        np.logspace(np.log10(smallest), np.log10(largest), n_samples)
    ).astype(int))

    rng = np.random.default_rng(seed)
    log_N_list: list[float] = []
    log_E_list: list[float] = []
    for N_sub in sample_sizes:
        E_vals = []
        for _ in range(n_trials):
            idx = rng.choice(X.shape[0], int(N_sub), replace=False)
            Y = X[idx]
            k = min(k_for_graph, int(N_sub) - 1)
            G = kneighbors_graph(Y, n_neighbors=k, mode="distance",
                                 include_self=False)
            # Symmetrize (kneighbors_graph is directed).
            G_sym = G.maximum(G.T)
            mst = minimum_spanning_tree(G_sym)
            edges = mst.data
            E_vals.append(float((edges ** alpha).sum()))
        log_N_list.append(np.log(float(N_sub)))
        log_E_list.append(np.log(np.mean(E_vals)))

    log_N = np.array(log_N_list)
    log_E = np.array(log_E_list)
    slope, _intercept = np.polyfit(log_N, log_E, 1)
    slope = float(slope)
    if slope >= 1.0:
        d_hat = float("inf")
    else:
        d_hat = float(alpha / (1.0 - slope))
    return d_hat, log_N, log_E, slope


def plot_intrinsic_dim(z: np.ndarray, patches: np.ndarray, out_dir: Path,
                       metric: str = "fs", max_points: int | None = None,
                       seed: int = 0) -> None:
    """Estimate intrinsic dim three ways and write a 3-panel diagnostic PNG."""
    try:
        from sklearn.neighbors import NearestNeighbors  # noqa: F401
        from scipy.sparse.csgraph import minimum_spanning_tree  # noqa: F401
    except ImportError:
        print("  sklearn / scipy missing. `uv sync --extra viz-3d`.")
        return

    z_sub, _ = subsample(z, patches, max_points, seed=seed)
    X = to_features(z_sub, metric)
    N = X.shape[0]
    print(f"  feature dim: {X.shape[1]} ({metric} metric), points: {N}")

    print("  [1/3] TwoNN ...")
    d_twonn, log_mu, nlog_1mF = _twonn_estimate(X)

    print("  [2/3] MLE  (k = 10, 20, 50) ...")
    mle_results = _mle_estimate(X, k_values=(10, 20, 50))

    print("  [3/3] PH-dim (MST H_0 total persistence vs N) ...")
    d_phdim, ph_logN, ph_logE, ph_slope = _phdim_estimate(X, seed=seed)

    # ----- plot -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # TwoNN
    ax = axes[0]
    plot_idx = np.linspace(0, len(log_mu) - 1, min(1500, len(log_mu))).astype(int)
    ax.scatter(log_mu[plot_idx], nlog_1mF[plot_idx],
               s=4, alpha=0.4, color="steelblue", edgecolors="none")
    x_line = np.linspace(log_mu.min(), max(log_mu.max(), 0.01), 50)
    ax.plot(x_line, d_twonn * x_line, color="crimson", lw=1.5,
            label=f"fit (slope through origin):  d = {d_twonn:.3f}")
    ax.set_xlabel(r"$\log\,\mu$,    $\mu = r_2 / r_1$", fontsize=10)
    ax.set_ylabel(r"$-\log(1 - F(\mu))$", fontsize=10)
    ax.set_title(f"TwoNN:   d = {d_twonn:.3f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # MLE
    ax = axes[1]
    colors = ["steelblue", "darkorange", "seagreen"]
    for (k, d_arr), c in zip(mle_results.items(), colors):
        d_arr_clip = d_arr[(d_arr > 0) & (d_arr < 15)]
        median = float(np.median(d_arr_clip))
        ax.hist(d_arr_clip, bins=80, alpha=0.45, color=c,
                label=f"k = {k}  (median {median:.3f})")
    ax.set_xlabel("local dim (MLE per point)", fontsize=10)
    ax.set_ylabel("count")
    ax.set_title("MLE per-point histograms")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # PH-dim
    ax = axes[2]
    ax.scatter(ph_logN, ph_logE, color="steelblue", s=40,
               edgecolors="black", linewidths=0.5, zorder=3)
    x_line = np.linspace(ph_logN.min(), ph_logN.max(), 50)
    intercept = np.mean(ph_logE - ph_slope * ph_logN)
    y_line = ph_slope * x_line + intercept
    ax.plot(x_line, y_line, color="crimson", lw=1.5,
            label=f"slope = {ph_slope:.3f}    d = {d_phdim:.3f}")
    ax.set_xlabel(r"$\log\,N$", fontsize=10)
    ax.set_ylabel(r"$\log\,\sum_{e\in MST(N)} |e|^\alpha,\ \alpha = 1$",
                  fontsize=10)
    ax.set_title(f"PH-dim:   d = {d_phdim:.3f}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Intrinsic dimension estimates  (metric={metric}, N={N})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / f"intrinsic_dim_{metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")

    # ----- stdout summary -----
    print()
    print("  ===== Intrinsic dimension summary =====")
    print(f"  expected (sLag in CY 3-fold):  d = 3")
    print(f"  TwoNN:                         d = {d_twonn:.3f}")
    for k, d_arr in mle_results.items():
        d_arr_clip = d_arr[(d_arr > 0) & (d_arr < 15)]
        print(f"  MLE  k={k:<3}:                    "
              f"median = {np.median(d_arr_clip):.3f},  "
              f"mean = {d_arr_clip.mean():.3f}")
    print(f"  PH-dim (MST scaling):          "
          f"d = {d_phdim:.3f}  (slope = {ph_slope:.3f})")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path,
                        help="Folder containing min_set.pkl.")
    parser.add_argument("--methods", nargs="+",
                        choices=["coord", "umap", "mapper", "intrinsic_dim",
                                 "all"],
                        default=["all"],
                        help="Which methods to run (default: all -- includes "
                             "intrinsic_dim).")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Subsample to this many points "
                             "(default: use all points). Pass an integer "
                             "if UMAP / Mapper runtime needs to be capped.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_subdir", type=str, default=None,
                        help="If set, write all outputs to "
                             "<folder>/<out_subdir>/ instead of <folder>/. "
                             "E.g., --out_subdir topology.")
    parser.add_argument("--sweep", action="store_true",
                        help="For UMAP / Mapper: run a 3x3 hyperparameter "
                             "sweep instead of a single setting. coord "
                             "method is unaffected.")
    parser.add_argument("--metric", choices=["euclidean", "fs"], default="fs",
                        help="Distance metric for UMAP / Mapper. "
                             "'euclidean' uses raw 10D real coords "
                             "(patch-dependent). 'fs' (default) maps each "
                             "point to its rank-1 projector z conj(z)^T / "
                             "||z||^2 flattened to 25 real DOF, giving "
                             "Euclidean distance = sqrt(2) * sin(d_FS), "
                             "monotone with Fubini-Study distance on CP^4. "
                             "Projective-invariant.")

    # UMAP knobs (used only if --sweep is not set).
    parser.add_argument("--umap_n_neighbors", type=int, default=100)
    parser.add_argument("--umap_min_dist", type=float, default=0.3)

    # Mapper knobs.
    parser.add_argument("--mapper_filter",
                        choices=["first-pc", "abs0_minus_abs4", "abs0",
                                 "abs04_abs123", "first-two-pc"],
                        default="abs04_abs123",
                        help="Lens function. 1D options: first-pc, abs0, "
                             "abs0_minus_abs4. 2D options (n_cubes^2 cover "
                             "cells, many more nodes): abs04_abs123 "
                             "(|z_0|-|z_4|, |z_1|+|z_2|+|z_3|; default), "
                             "first-two-pc.")
    parser.add_argument("--mapper_n_cubes", type=int, default=50,
                        help="Cover intervals per filter axis. For 2D "
                             "filters this gives n_cubes^2 cells. Default "
                             "50 (gives ~2500 cells for 2D filters, vs "
                             "~50 for 1D).")
    parser.add_argument("--mapper_overlap", type=float, default=0.5)
    parser.add_argument("--mapper_eps_percentile", type=float, default=75.0,
                        help="Percentile of k-NN distances used to set "
                             "DBSCAN eps (auto-scaled to local density). "
                             "Higher = more permissive clustering = more "
                             "edges. Default 75.")
    args = parser.parse_args()

    z = load_min_set_complex(args.folder)
    patches = patch_indices_from_complex(z)
    print(f"Loaded {z.shape[0]} points from {args.folder}/min_set.pkl")
    print(f"Patch counts: " + ", ".join(
        f"p{i}:{int((patches == i).sum())}" for i in range(5)))

    out_dir = args.folder if args.out_subdir is None else (
        args.folder / args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_subdir is not None:
        print(f"Writing outputs to {out_dir}/")

    methods = set(args.methods)
    if "all" in methods:
        methods = {"coord", "umap", "mapper", "intrinsic_dim"}

    if "coord" in methods:
        print("\n--- coord-aligned 3D scatters ---")
        plot_coord_triples(z, patches, out_dir, DEFAULT_TRIPLES,
                           max_points=args.max_points)

    if "umap" in methods:
        print("\n--- UMAP 3D ---")
        plot_umap_3d(z, patches, out_dir,
                     n_neighbors=args.umap_n_neighbors,
                     min_dist=args.umap_min_dist,
                     max_points=args.max_points,
                     seed=args.seed,
                     metric=args.metric,
                     sweep=args.sweep)

    if "mapper" in methods:
        print("\n--- Mapper ---")
        plot_mapper(z, patches, out_dir,
                    filter_spec=args.mapper_filter,
                    n_cubes=args.mapper_n_cubes,
                    perc_overlap=args.mapper_overlap,
                    eps_percentile=args.mapper_eps_percentile,
                    max_points=args.max_points,
                    seed=args.seed,
                    metric=args.metric,
                    sweep=args.sweep)

    if "intrinsic_dim" in methods:
        print("\n--- Intrinsic dimension (TwoNN + MLE + PH-dim) ---")
        plot_intrinsic_dim(z, patches, out_dir,
                           metric=args.metric,
                           max_points=args.max_points,
                           seed=args.seed)


if __name__ == "__main__":
    main()
