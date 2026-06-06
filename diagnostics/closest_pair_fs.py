"""Find the closest pairs of points across two sLag clusters, measured in
Fubini-Study distance on CP^4.

Loads two (N, 5) complex pickles (typically produced by
diagnostics/split_clusters.py, but any (N, 5) complex pkl works), then runs
a brute-force pairwise search on GPU with chunking across all local
devices. Reports the global top-K closest pairs plus mean / max
nearest-neighbor FS distances.

Search is done in complex64 for speed; the surviving top-K candidate pairs
are re-ranked and reported in complex128 to avoid float32 tie-break
artifacts.

Usage:
    python -m diagnostics.closest_pair_fs \\
        --cluster_a plots_slag_run/cluster_split/cluster_0_points.pkl \\
        --cluster_b plots_slag_run/cluster_split/cluster_1_points.pkl
"""
import argparse
import pickle
import sys
from functools import partial
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


def closest_pairs_fs(A: np.ndarray, B: np.ndarray, chunk_size: int,
                     top_k: int):
    """Brute-force top-K closest pairs (FS distance) across A and B.

    Args:
        A: (Na, 5) complex.  Chunked across local devices.
        B: (Nb, 5) complex.  Replicated on each device.
        chunk_size: rows of A processed per device per step.
            Per-device peak memory ~ chunk_size * Nb * 8B (complex64).
        top_k: per-row top-K, also the global top-K reported. Per-row
            top-K is the smallest K that the algorithm needs (any global
            top-K entry is in its row's top-K), so we set both to the same
            value.

    Returns:
        dict with keys
          'topk_pairs'      -- list of (fs_search, i_in_A, j_in_B) tuples,
                               sorted ascending (length top_k)
          'mean', 'max'     -- aggregated over (Na,) per-A nearest-B FS
          'all_min_per_A'   -- (Na,) array of nearest-B FS per A point
          'all_argmin_per_A'-- (Na,) array of nearest-B index per A point
    """
    devices = jax.local_devices()
    D = len(devices)
    print(f"Using {D} local device(s): "
          f"{[f'{d.platform}:{d.id}' for d in devices]}")

    A_norm = (A / np.linalg.norm(A, axis=1, keepdims=True)).astype(
        np.complex64)
    B_norm = (B / np.linalg.norm(B, axis=1, keepdims=True)).astype(
        np.complex64)

    # B is broadcast to every device (so the matmul on each device sees the
    # full B). Replication via D identical shards along the leading device
    # axis (drop-in for the deprecated jax.device_put_replicated).
    B_repl = device_put_sharded([B_norm] * D, devices)

    Na = A_norm.shape[0]
    K = top_k
    step = D * chunk_size                            # rows per outer iter
    per_chunk = make_per_chunk(K)

    # For each a in A, store its K smallest FS distances to B and their
    # B indices.  Used both for global top-K re-ranking and mean / max
    # nearest-neighbor stats (mean / max take the [:, 0] column).
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

        # Flatten device + intra-chunk axes, drop padded rows, store.
        flat_vals = vals_np.reshape(-1, K)[: e - s]
        flat_idx = idx_np.reshape(-1, K)[: e - s]
        topk_vals[s:e] = flat_vals
        topk_idx[s:e] = flat_idx

        # Running global top-1 just for log feedback.
        running_min_local = float(flat_vals[:, 0].min())
        running_min_global = float(topk_vals[:e, 0].min())
        print(f"  scanned {e}/{Na} of A  "
              f"(chunk min FS = {running_min_local:.6e}, "
              f"running global min = {running_min_global:.6e})")

    # ------- global top-K via flattened argpartition -----------------
    flat_vals = topk_vals.reshape(-1)
    flat_i = np.repeat(np.arange(Na, dtype=np.int64), K)   # row index
    flat_j = topk_idx.reshape(-1)                          # B index

    n_total = flat_vals.shape[0]
    k_select = min(K, n_total)
    top_idx = np.argpartition(flat_vals, k_select - 1)[:k_select]
    # Sort the K selected entries ascending by value.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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

    # Re-rank surviving top-K in complex128 (cheap, fixes any float32
    # tie-break artifacts).
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

    # Detailed listing for the closest pair (rank 1).
    fs_exact, i_a, j_b, fs_search = refined[0]
    z_a = A_orig[i_a]
    z_b = B_orig[j_b]
    overlap = float(np.cos(fs_exact))
    print()
    print("-" * 72)
    print(f"Closest pair (rank 1) coordinates:")
    print(f"  A index: {i_a}    B index: {j_b}")
    print(f"  |<a,b>| / (||a|| ||b||) :  {overlap:.12f}")
    print(f"  A point:")
    for k, zk in enumerate(z_a):
        print(f"    z_{k} = {zk}")
    print(f"  B point:")
    for k, zk in enumerate(z_b):
        print(f"    z_{k} = {zk}")

    # Cross-cluster nearest-neighbor distribution stats.
    print()
    print("-" * 72)
    print(f"Cross-cluster nearest-neighbor statistics "
          f"(for each point in {big_label}, FS distance to its nearest "
          f"point in {small_label}; N = {result['all_min_per_A'].shape[0]}):")
    print(f"  min  FS distance: {refined[0][3]:.12e}    "
          f"(complex64; matches rank-1 search above)")
    print(f"  mean FS distance: {result['mean']:.12e}")
    print(f"  max  FS distance: {result['max']:.12e}")


if __name__ == "__main__":
    main()
