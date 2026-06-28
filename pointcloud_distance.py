"""Resampling-stable distances between point clouds sampled from submanifolds,
for the valley-walk experiment. All metrics operate on the (N, 25) FS-projector
embedding (cluster_select.fs_features), where Euclidean distance = sqrt(2)*sin(d_FS):
projective-invariant and patch-independent. Pure numpy + scipy.

See docs/superpowers/specs/2026-06-28-valley-walk-design.md.
"""
import numpy as np
from scipy.spatial import cKDTree


def pairwise_fs_distances(emb, n_pairs=200_000, rng=None):
    """(n_pairs,) Euclidean distances of random index pairs in (N, d) `emb`.
    O(n_pairs); self-pairs (i == j) are resampled out so the zero-distance spike
    does not bias the low end of the distribution."""
    rng = np.random.default_rng() if rng is None else rng
    emb = np.asarray(emb, dtype=np.float64)
    n = emb.shape[0]
    if n < 2:
        raise ValueError(f"need >= 2 points, got {n}")
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    same = i == j
    while same.any():
        j[same] = rng.integers(0, n, size=int(same.sum()))
        same = i == j
    return np.linalg.norm(emb[i] - emb[j], axis=1)


def _wasserstein1_sorted(a, b):
    """1-D Wasserstein-1. Equal-length samples: mean|sort(a)-sort(b)|. Unequal:
    compare on a shared quantile grid."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.shape[0] == b.shape[0]:
        return float(np.mean(np.abs(a - b)))
    q = np.linspace(0.0, 1.0, max(a.shape[0], b.shape[0]))
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def pairwise_distance_drift(embA, embB, n_pairs=200_000, rng=None):
    """PRIMARY drift scalar: 1-D Wasserstein-1 between the pairwise-distance
    distributions of A and B. Resampling-stable (population statistic) and
    isometry-invariant (so an S5/phase relabeling of the same L reads ~0)."""
    rng = np.random.default_rng() if rng is None else rng
    dA = pairwise_fs_distances(embA, n_pairs, rng)
    dB = pairwise_fs_distances(embB, n_pairs, rng)
    return _wasserstein1_sorted(dA, dB)


def fs_chamfer(embA, embB):
    """SECONDARY local-shape check: symmetric mean nearest-neighbor distance in
    the FS embedding (cKDTree). Floor ~ N^(-1/3); NOT relabeling-invariant, so use
    only between clouds not expected to differ by a discrete symmetry."""
    embA = np.asarray(embA, dtype=np.float64)
    embB = np.asarray(embB, dtype=np.float64)
    dA, _ = cKDTree(embB).query(embA, k=1)
    dB, _ = cKDTree(embA).query(embB, k=1)
    return float(dA.mean() + dB.mean())


def calibrate_noise_floor(mine_and_embed, n_repeats=3, n_pairs=200_000, rng=None):
    """Re-mine the SAME coeffs `n_repeats` times (different seeds) and return the
    among-sample drift as the 'same sLag, different sample' floor.

    mine_and_embed(seed:int) -> (N, d) embedding of a fresh independent mining.
    Returns {'wass_floor', 'chamfer_floor', 'samples'} (floors = median over pairs)."""
    rng = np.random.default_rng() if rng is None else rng
    if n_repeats < 2:
        raise ValueError("n_repeats must be >= 2 to form a floor")
    embs = [np.asarray(mine_and_embed(int(rng.integers(0, 2**31 - 1))))
            for _ in range(n_repeats)]
    wass, cham = [], []
    for a in range(len(embs)):
        for b in range(a + 1, len(embs)):
            wass.append(pairwise_distance_drift(embs[a], embs[b], n_pairs, rng))
            cham.append(fs_chamfer(embs[a], embs[b]))
    return {"wass_floor": float(np.median(wass)),
            "chamfer_floor": float(np.median(cham)),
            "samples": embs}


def decide(drift, lag, spec, lag0, spec0, tol, wass_floor, target_mult):
    """Walk-control decision (pure, no JAX).
    'reject' if fitness fell below tol of the C* baseline (fell off the floor);
    'stop'   if fitness is OK and drift exceeded target_mult * wass_floor;
    'accept' otherwise."""
    if lag < lag0 - tol or spec < spec0 - tol:
        return "reject"
    if drift > target_mult * wass_floor:
        return "stop"
    return "accept"
