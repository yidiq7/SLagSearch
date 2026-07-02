"""Host-side (pure numpy + HDBSCAN) component selection for single-cluster GD.

Clusters the post-Newton mined points by density (HDBSCAN) on the 25-D
Fubini-Study projector embedding, picks one component (anchor-tracked across
re-mines), and returns a fixed-size index set into the mined points. No JAX, no
matplotlib, so it is cheap to import in the GD training loop.
"""
import warnings

import numpy as np


def fs_features(z: np.ndarray) -> np.ndarray:
    """(N,5) complex -> (N,25) float64 FS-projector embedding.

    Euclidean distance on the output equals sqrt(2)*sin(d_FS): projective-
    invariant and patch-independent (the projector P_z = z conj(z)^T / ||z||^2
    is unchanged by z -> lambda z). Same embedding as
    viz.plot_3D.fs_feature_embedding; duplicated here (8 lines, mathematically
    fixed) to keep this module matplotlib-free for the GD hot path.
    """
    z = np.asarray(z)
    norm_sq = (z.conj() * z).sum(axis=1).real
    z_n = z / np.sqrt(norm_sq)[:, None]
    P = z_n[:, :, None] * z_n[:, None, :].conj()        # (N,5,5) Hermitian
    iu = np.triu_indices(5, k=1)
    diag = np.diagonal(P, axis1=1, axis2=2).real         # (N,5)
    off_re = P[:, iu[0], iu[1]].real * np.sqrt(2.0)      # (N,10)
    off_im = P[:, iu[0], iu[1]].imag * np.sqrt(2.0)      # (N,10)
    return np.concatenate([diag, off_re, off_im], axis=1).astype(np.float64)


def _get_hdbscan():
    """Lazy HDBSCAN lookup: sklearn>=1.3 first, then the standalone package."""
    try:
        from sklearn.cluster import HDBSCAN
        return HDBSCAN
    except ImportError:
        pass
    try:
        from hdbscan import HDBSCAN
        return HDBSCAN
    except ImportError as e:
        raise ImportError(
            "--target_cluster needs scikit-learn>=1.3 (sklearn.cluster.HDBSCAN) "
            "or the standalone `hdbscan` package. `uv sync --extra viz-3d`."
        ) from e


def cluster_labels(features, min_cluster_size, cluster_selection_epsilon=0.0):
    """Integer HDBSCAN labels (noise = -1). Deterministic given the params."""
    HDBSCAN = _get_hdbscan()
    clusterer = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
    )  # allow_single_cluster stays False: True biases EOM to the root and
       # merges real multi-component data into one (see detect_components fallback
       # for the genuine single-component case).
    with warnings.catch_warnings():
        # sklearn's HDBSCAN warns that the `copy` default flips in 1.10; either
        # value is fine here (features is a fresh throwaway array), so silence
        # the per-fit spam. Message filter is a no-op for the standalone package.
        warnings.filterwarnings("ignore", message=r"The default value of `copy`",
                                category=FutureWarning)
        return clusterer.fit_predict(np.asarray(features))


def detect_components(features, min_cluster_size, min_cluster_frac,
                      cluster_selection_epsilon=0.0):
    """Cluster, drop components smaller than min_cluster_frac*N as noise, relabel
    survivors 0..n-1 by DESCENDING size. Returns (labels, n, sizes)."""
    raw = cluster_labels(features, min_cluster_size, cluster_selection_epsilon)
    n_total = np.asarray(features).shape[0]
    floor = max(1, int(min_cluster_frac * n_total))
    keep = [(int(lbl), int((raw == lbl).sum()))
            for lbl in np.unique(raw) if lbl != -1]
    keep = [(lbl, sz) for lbl, sz in keep if sz >= floor]
    keep.sort(key=lambda t: -t[1])                      # descending size
    if not keep:
        # HDBSCAN found no multi-cluster structure (with allow_single_cluster
        # at its default False, a lone blob is labeled all-noise). Treat the
        # whole set as one component so a single-component manifold reduces to
        # whole-manifold GD instead of erroring.
        return np.zeros(n_total, dtype=int), 1, [n_total]
    labels = np.full(n_total, -1, dtype=int)
    sizes = []
    for new_lbl, (old_lbl, sz) in enumerate(keep):
        labels[raw == old_lbl] = new_lbl
        sizes.append(sz)
    return labels, len(keep), sizes


def component_centroid(features, labels, label):
    return np.asarray(features)[labels == label].mean(axis=0)


def select_cluster(features, anchor, target_k, min_cluster_size, min_cluster_frac,
                   cluster_selection_epsilon=0.0):
    """Pick one component. Bootstrap (anchor is None): component `target_k` by
    descending-size label. Tracking: component whose centroid is nearest
    `anchor`. Returns (member_mask, new_anchor, info)."""
    features = np.asarray(features)
    labels, n, sizes = detect_components(
        features, min_cluster_size, min_cluster_frac, cluster_selection_epsilon
    )
    info = {"n_components": n, "sizes": sizes}
    if n == 0:
        raise ValueError(
            "HDBSCAN found no component above min_cluster_frac; lower "
            "--min_cluster_size or --min_cluster_frac."
        )
    if anchor is None:
        if target_k >= n:
            raise ValueError(
                f"--target_cluster {target_k} but only {n} component(s) detected."
            )
        chosen = int(target_k)
    else:
        centroids = np.stack([component_centroid(features, labels, c)
                              for c in range(n)])
        chosen = int(np.argmin(np.linalg.norm(centroids - np.asarray(anchor)[None, :],
                                              axis=1)))
    member_mask = labels == chosen
    new_anchor = features[member_mask].mean(axis=0)
    info["chosen"] = chosen
    return member_mask, new_anchor, info


def fill_to_size(member_idx, size, rng):
    """Exactly `size` indices: every member once + uniform resample padding.
    Subsample (no replacement) only when there are more than `size` members."""
    member_idx = np.asarray(member_idx)
    m = member_idx.shape[0]
    if m == 0:
        raise ValueError("empty component: cannot fill")
    if m >= size:
        return rng.choice(member_idx, size=size, replace=False)
    pad = rng.choice(member_idx, size=size - m, replace=True)
    return np.concatenate([member_idx, pad])


def stability_sweep(features, sizes_to_try, min_cluster_frac,
                    cluster_selection_epsilon=0.0):
    """#components for each min_cluster_size in sizes_to_try. A plateau is the
    robust component count (cross-check against the PH b0)."""
    out = {}
    for s in sizes_to_try:
        _, n, _ = detect_components(features, s, min_cluster_frac,
                                    cluster_selection_epsilon)
        out[int(s)] = n
    return out
