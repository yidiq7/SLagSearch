"""Find the closest pairs of points across two sLag clusters, measured in
Fubini-Study distance on CP^4, with an optional within-cluster nearest-
neighbor baseline so you can tell a real gap apart from sampling sparsity.

Loads two (N, 5) complex pickles (typically produced by
diagnostics/split_clusters.py, but any (N, 5) complex pkl works), then runs
a brute-force pairwise search on GPU with chunking across all local
devices. Reports:

  1. Top-K closest cross-cluster pairs (default K=10).
  2. Cross-cluster nearest-neighbor stats: for each point in the bigger
     cluster, FS distance to its nearest point in the smaller cluster ->
     min / mean / max.
  3. Within-cluster nearest-neighbor stats (each cluster, separately).
     For each point in X, FS distance to its nearest OTHER point in X.
     If the cross-cluster closest pair is much larger than the within-
     cluster NN max, there is a real gap between the clusters; if they
     are comparable, the apparent gap is probably just sparse sampling.
     Disable with --no-within_cluster (the within pass roughly doubles
     the runtime).

Search is done in complex64 for speed; the surviving top-K cross-cluster
pairs are re-ranked and reported in complex128 to avoid float32 tie-break
artifacts.

Usage:
    python -m diagnostics.closest_pair_fs \\
        --cluster_a plots_slag_run/cluster_split/cluster_0_points.pkl \\
        --cluster_b plots_slag_run/cluster_split/cluster_1_points.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

# Repo's drop-in for the deprecated jax.device_put_sharded; pmap-compatible.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharding import device_put_sharded


def load_points(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        z = pickle.load(f)
    z = np.asarray(z)
    if z.ndim != 2 or z.shape[1] != 5:
        raise ValueError(
            f"{path}: expected (N, 5) complex, got {z.shape}")
    return z.astype(np.complex64)


def fs_distance_exact(z: np.ndarray, w: np.ndarray) -> float:
    """Reference FS distance d_FS([z],[w]) = arccos(|<z,w>|/(||z|| ||w||))
    in complex128, for the final reported pairs."""
    z = np.asarray(z, dtype=np.complex128)
    w = np.asarray(w, dtype=np.complex128)
    overlap = np.abs(np.vdot(z, w))
    overlap /= (np.linalg.norm(z) * np.linalg.norm(w))
    overlap = min(overlap, 1.0)
    return float(np.arccos(overlap))


def make_per_chunk(top_k: int):
    """Returns a pmapped function that gives, for each row of A_chunk, the
    top-`top_k` smallest FS distances and their B indices.

    Correctness for global top-K: if entry (i, j) is in the global top-K
    smallest of the (Na, Nb) distance matrix, then it must also be in row
    i's top-K smallest (since fewer than K row-i entries can be strictly
    smaller). So global top-K ⊆ union of per-row top-K -- we can collect
    per-row top-K and globally re-rank without missing anything.
    """
    @jax.pmap
    def _per_chunk(A_chunk: jnp.ndarray, B_norm: jnp.ndarray):
        # A_chunk: (k, 5) unit-normalized; B_norm: (Nb, 5) unit-normalized.
        # |<a, b>| is invariant under which side gets conjugated, so the
        # matmul orientation does not affect the FS distance.
        M = A_chunk @ jnp.conjugate(B_norm).T        # (k, Nb)
        abs_M = jnp.clip(jnp.abs(M), 0.0, 1.0)
        D = jnp.arccos(abs_M)                        # (k, Nb)
        # top_k returns LARGEST; negate to get smallest.
        neg_topk_vals, topk_j = jax.lax.top_k(-D, top_k)  # (k, K), (k, K)
        topk_vals = -neg_topk_vals
        return topk_vals, topk_j
    return _per_chunk


def per_row_topk(A: np.ndarray, B: np.ndarray, chunk_size: int,
                 top_k: int, label: str = ""):
    """Per-row top-K smallest FS distances from A to B.

    Args:
        A: (Na, 5) complex.  Chunked across local devices.
        B: (Nb, 5) complex.  Replicated on each device.
        chunk_size: rows of A processed per device per step.
            Per-device peak memory ~ chunk_size * Nb * 8B (complex64).
        top_k: K. Per-row output shape is (Na, K).
        label: prefix for progress prints (e.g. "A->B", "A->A").

    Returns:
        (topk_vals, topk_idx): float32 (Na, K), int64 (Na, K).
    """
    devices = jax.local_devices()
    D = len(devices)
    A_norm = (A / np.linalg.norm(A, axis=1, keepdims=True)).astype(
        np.complex64)
    B_norm = (B / np.linalg.norm(B, axis=1, keepdims=True)).astype(
        np.complex64)

    # B is broadcast to every device. Replication via D identical shards
    # along the leading device axis (drop-in for the deprecated
    # jax.device_put_replicated).
    B_repl = device_put_sharded([B_norm] * D, devices)

    Na = A_norm.shape[0]
    K = top_k
    step = D * chunk_size                            # rows per outer iter
    per_chunk = make_per_chunk(K)

    topk_vals = np.empty((Na, K), dtype=np.float32)
    topk_idx = np.empty((Na, K), dtype=np.int64)

    for s in range(0, Na, step):
        e = min(s + step, Na)
        block = A_norm[s:e]
        pad = step - block.shape[0]
        if pad > 0:
            block = np.concatenate(
                [block, np.zeros((pad, 5), dtype=block.dtype)], axis=0)
        block_sharded = device_put_sharded(
            list(block.reshape(D, chunk_size, 5)), devices)

        vals, idx = per_chunk(block_sharded, B_repl)
        vals_np = np.asarray(vals)                   # (D, chunk_size, K)
        idx_np = np.asarray(idx)                     # (D, chunk_size, K)

        flat_vals = vals_np.reshape(-1, K)[: e - s]
        flat_idx = idx_np.reshape(-1, K)[: e - s]
        topk_vals[s:e] = flat_vals
        topk_idx[s:e] = flat_idx

        running_min = float(topk_vals[:e, 0].min())
        print(f"  [{label}] scanned {e}/{Na}  "
              f"(running min FS = {running_min:.6e})")

    return topk_vals, topk_idx


def closest_pairs_fs(A: np.ndarray, B: np.ndarray, chunk_size: int,
                     top_k: int):
    """Top-K cross-cluster closest pairs + per-A nearest-B distribution.

    Returns dict:
      'topk_pairs'      list of (fs_search, i_in_A, j_in_B), len top_k
      'mean','max'      stats over per-A nearest-B FS
      'all_min_per_A'   (Na,) float32: per-A nearest-B FS
      'all_argmin_per_A'(Na,) int64:   per-A nearest-B index
    """
    devices = jax.local_devices()
    print(f"Using {len(devices)} local device(s): "
          f"{[f'{d.platform}:{d.id}' for d in devices]}")

    topk_vals, topk_idx = per_row_topk(
        A, B, chunk_size, top_k, label="cross")
    Na, K = topk_vals.shape

    # ----- global top-K via flattened argpartition ------------------
    flat_vals = topk_vals.reshape(-1)
    flat_i = np.repeat(np.arange(Na, dtype=np.int64), K)
    flat_j = topk_idx.reshape(-1)
    k_select = min(K, flat_vals.shape[0])
    top_idx = np.argpartition(flat_vals, k_select - 1)[:k_select]
    order = np.argsort(flat_vals[top_idx])
    top_idx = top_idx[order]
    topk_search = [
        (float(flat_vals[t]), int(flat_i[t]), int(flat_j[t]))
        for t in top_idx
    ]

    nearest_per_A = topk_vals[:, 0]
    return {
        "topk_pairs": topk_search,
        "mean": float(nearest_per_A.mean()),
        "max": float(nearest_per_A.max()),
        "all_min_per_A": nearest_per_A,
        "all_argmin_per_A": topk_idx[:, 0],
    }


def within_cluster_nn_stats(X: np.ndarray, chunk_size: int,
                            label: str = "X"):
    """For each x in X, FS distance to the nearest OTHER point in X.

    Uses the same kernel with K=2: the top-1 entry per row is self (with
    distance ~0), the top-2 is the actual nearest neighbor. We pick the
    second whenever the first's index equals the row's own index.

    Returns dict with min/mean/median/max/p10/p90 plus the per-row arrays.
    """
    topk_vals, topk_idx = per_row_topk(
        X, X, chunk_size, top_k=2, label=f"{label}->{label}")
    Nx = topk_vals.shape[0]
    abs_idx = np.arange(Nx, dtype=np.int64)
    # The kernel's top-1 (smallest) for self-vs-self is the point itself
    # (distance 0). If that's not true for some row (e.g. an exact
    # duplicate landed first), we still pick the non-self candidate; if
    # both candidates are non-self (no duplicate at all), we pick top-1.
    is_self_0 = topk_idx[:, 0] == abs_idx
    nn_dist = np.where(is_self_0, topk_vals[:, 1], topk_vals[:, 0])
    nn_j = np.where(is_self_0, topk_idx[:, 1], topk_idx[:, 0])
    pct = np.percentile(nn_dist.astype(np.float64), [10, 50, 90])
    return {
        "nn_dist": nn_dist,
        "nn_j": nn_j,
        "min": float(nn_dist.min()),
        "mean": float(nn_dist.mean()),
        "max": float(nn_dist.max()),
        "p10": float(pct[0]),
        "p50": float(pct[1]),
        "p90": float(pct[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster_a", type=Path, required=True,
                        help="Path to first (N, 5) complex pkl.")
    parser.add_argument("--cluster_b", type=Path, required=True,
                        help="Path to second (N, 5) complex pkl.")
    parser.add_argument("--chunk_size", type=int, default=4096,
                        help="Rows of A processed per device per step "
                             "(default 4096). Per-device memory ~ "
                             "chunk_size * |B| * 8B (complex64).")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of closest pairs to report "
                             "(default 10).")
    parser.add_argument("--within_cluster", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Also compute within-cluster nearest-"
                             "neighbor stats for each cluster, as a "
                             "baseline to interpret the cross-cluster "
                             "closest pair (default: enabled). Roughly "
                             "doubles runtime. Use --no-within_cluster "
                             "to skip.")
    args = parser.parse_args()

    A_orig = load_points(args.cluster_a)
    B_orig = load_points(args.cluster_b)
    print(f"Loaded {args.cluster_a}: shape {A_orig.shape}")
    print(f"Loaded {args.cluster_b}: shape {B_orig.shape}")

    # Put the larger set as the "A" that gets chunked across devices; the
    # smaller set is replicated so per-device memory stays small.
    if A_orig.shape[0] >= B_orig.shape[0]:
        big, small = A_orig, B_orig
        swapped = False
    else:
        big, small = B_orig, A_orig
        swapped = True
    print(f"Sharding the larger set ({'B' if swapped else 'A'}, "
          f"N={big.shape[0]}) across devices; replicating the smaller "
          f"({'A' if swapped else 'B'}, N={small.shape[0]}).")

    result = closest_pairs_fs(big, small, args.chunk_size, args.top_k)
    big_label, small_label = ("B", "A") if swapped else ("A", "B")

    # Re-rank surviving top-K in complex128.
    refined = []  # list of (fs_exact, i_in_A, j_in_B, fs_search)
    for fs_search, i_big, j_small in result["topk_pairs"]:
        if swapped:
            i_a, j_b = j_small, i_big
        else:
            i_a, j_b = i_big, j_small
        fs_exact = fs_distance_exact(A_orig[i_a], B_orig[j_b])
        refined.append((fs_exact, i_a, j_b, fs_search))
    refined.sort(key=lambda r: r[0])

    print()
    print("=" * 72)
    print(f"Top {len(refined)} closest pairs across clusters  "
          f"(Fubini-Study distance on CP^4)")
    print("=" * 72)
    print(f"  cluster A file:  {args.cluster_a}  (N = {A_orig.shape[0]})")
    print(f"  cluster B file:  {args.cluster_b}  (N = {B_orig.shape[0]})")
    print()
    print(f"  {'rank':>4}  {'A_idx':>8}  {'B_idx':>8}  "
          f"{'FS (complex128)':>20}  {'FS (complex64)':>18}")
    for rank, (fs_exact, i_a, j_b, fs_search) in enumerate(refined,
                                                            start=1):
        print(f"  {rank:>4}  {i_a:>8d}  {j_b:>8d}  "
              f"{fs_exact:>20.12e}  {fs_search:>18.6e}")

    # Detailed listing for rank 1.
    fs_exact_top, i_a_top, j_b_top, fs_search_top = refined[0]
    z_a = A_orig[i_a_top]
    z_b = B_orig[j_b_top]
    overlap = float(np.cos(fs_exact_top))
    print()
    print("-" * 72)
    print(f"Closest pair (rank 1) coordinates:")
    print(f"  A index: {i_a_top}    B index: {j_b_top}")
    print(f"  |<a,b>| / (||a|| ||b||) :  {overlap:.12f}")
    print(f"  A point:")
    for k, zk in enumerate(z_a):
        print(f"    z_{k} = {zk}")
    print(f"  B point:")
    for k, zk in enumerate(z_b):
        print(f"    z_{k} = {zk}")

    # Cross-cluster nearest-neighbor distribution.
    print()
    print("-" * 72)
    print(f"Cross-cluster nearest-neighbor stats "
          f"(for each point in {big_label}, FS distance to its nearest "
          f"point in {small_label};  N = {result['all_min_per_A'].shape[0]}):")
    print(f"  min  FS: {fs_search_top:.6e}    (= rank-1 closest pair)")
    print(f"  mean FS: {result['mean']:.6e}")
    print(f"  max  FS: {result['max']:.6e}")

    # Within-cluster NN baselines (the diagnostic for "real gap vs sparse
    # sampling": if the cross-cluster min FS dwarfs the within-cluster
    # max NN, there is a real gap; if they are similar, the gap is
    # consistent with sampling sparsity).
    if args.within_cluster:
        print()
        print("=" * 72)
        print("Within-cluster nearest-neighbor stats  (baseline)")
        print("=" * 72)
        a_nn = within_cluster_nn_stats(A_orig, args.chunk_size,
                                       label="A")
        b_nn = within_cluster_nn_stats(B_orig, args.chunk_size,
                                       label="B")

        def _print_within(label, stats, N):
            print(f"  cluster {label}  (N = {N}):")
            print(f"    min  FS: {stats['min']:.6e}")
            print(f"    p10  FS: {stats['p10']:.6e}")
            print(f"    p50  FS: {stats['p50']:.6e}  (median)")
            print(f"    mean FS: {stats['mean']:.6e}")
            print(f"    p90  FS: {stats['p90']:.6e}")
            print(f"    max  FS: {stats['max']:.6e}")

        _print_within("A", a_nn, A_orig.shape[0])
        print()
        _print_within("B", b_nn, B_orig.shape[0])

        print()
        print("-" * 72)
        print("Gap interpretation:")
        cross_min = fs_search_top
        within_max = max(a_nn["max"], b_nn["max"])
        within_p90 = max(a_nn["p90"], b_nn["p90"])
        print(f"  cross-cluster closest pair (FS):  {cross_min:.6e}")
        print(f"  within-cluster p90 NN  (max over A, B):  "
              f"{within_p90:.6e}")
        print(f"  within-cluster max NN  (max over A, B):  "
              f"{within_max:.6e}")
        print(f"  ratio  cross_min / within_p90  =  "
              f"{cross_min / within_p90:.3f}")
        print(f"  ratio  cross_min / within_max  =  "
              f"{cross_min / within_max:.3f}")
        print("  (large ratio -> real gap;  ~1 -> consistent with "
              "sampling sparsity.)")


if __name__ == "__main__":
    main()
