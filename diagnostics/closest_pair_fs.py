"""Find the closest pair of points across two sLag clusters, measured in
Fubini-Study distance on CP^4.

Loads two (N, 5) complex pickles (typically produced by
diagnostics/split_clusters.py, but any (N, 5) complex pkl works), then runs
a brute-force pairwise search on GPU with chunking across all local
devices. Prints the closest pair's indices, coordinates, and the FS
distance.

Search is done in complex64 for speed; the final reported distance is
recomputed in complex128 on the chosen pair for precision.

Usage:
    python -m diagnostics.closest_pair_fs \\
        --cluster_a plots_slag_run/cluster_split/cluster_0_points.pkl \\
        --cluster_b plots_slag_run/cluster_split/cluster_1_points.pkl
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

# Repo's drop-in for the deprecated jax.device_put_sharded; pmap-compatible.
import sys
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
    in complex128, for the final reported pair."""
    z = np.asarray(z, dtype=np.complex128)
    w = np.asarray(w, dtype=np.complex128)
    overlap = np.abs(np.vdot(z, w))
    overlap /= (np.linalg.norm(z) * np.linalg.norm(w))
    overlap = min(overlap, 1.0)
    return float(np.arccos(overlap))


@jax.pmap
def _per_chunk(A_chunk: jnp.ndarray, B_norm: jnp.ndarray):
    """A_chunk: (k, 5) unit-normalized; B_norm: (Nb, 5) unit-normalized.

    Computes |<a_i, b_j>| via A_chunk @ conj(B_norm).T (absolute value is
    the same up to conjugation, so the matmul orientation does not matter
    for FS distance). Returns per-row (min_fs, argmin_j) over B_norm.
    """
    M = A_chunk @ jnp.conjugate(B_norm).T            # (k, Nb)
    abs_M = jnp.clip(jnp.abs(M), 0.0, 1.0)
    D = jnp.arccos(abs_M)
    j = jnp.argmin(D, axis=1)                        # (k,)
    mins = jnp.take_along_axis(D, j[:, None], axis=1).squeeze(-1)
    return mins, j


def closest_pair_fs(A: np.ndarray, B: np.ndarray, chunk_size: int):
    """Brute-force closest pair (FS distance) across A and B.

    Args:
        A: (Na, 5) complex.  Chunked across local devices.
        B: (Nb, 5) complex.  Replicated on each device.
        chunk_size: rows of A processed per device per step.
            Per-device peak memory ~ chunk_size * Nb * 8B (complex64).

    Returns:
        dict with keys
          'min', 'i_min_in_A', 'j_min_in_B'   -- the closest pair
          'mean', 'max'                       -- aggregated over the per-A-point
                                                 distance-to-nearest-B
          'all_min_per_A'                     -- (Na,) array of nearest-B
                                                 FS distance for each A point
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
    # full B). For chunk_size = 4096 and |B| ~ 1e5 this is ~3 GB per device
    # -- fine on A100 (40/80 GB). Replication via D identical shards along
    # the leading device axis (drop-in for the deprecated
    # jax.device_put_replicated).
    B_repl = device_put_sharded([B_norm] * D, devices)

    Na = A_norm.shape[0]
    step = D * chunk_size                            # rows per outer iter

    all_min = np.empty(Na, dtype=np.float32)         # nearest-B FS per a
    all_j = np.empty(Na, dtype=np.int64)             # nearest-B index per a
    best_dist = float("inf")
    best_i = -1
    best_j = -1

    for s in range(0, Na, step):
        e = min(s + step, Na)
        block = A_norm[s:e]
        pad = step - block.shape[0]
        if pad > 0:
            block = np.concatenate(
                [block, np.zeros((pad, 5), dtype=block.dtype)], axis=0)
        block_sharded = device_put_sharded(
            list(block.reshape(D, chunk_size, 5)), devices)

        mins, j = _per_chunk(block_sharded, B_repl)
        mins_np = np.asarray(mins)                   # (D, chunk_size)
        j_np = np.asarray(j)                         # (D, chunk_size)

        # Flatten back, drop padded rows, and store.
        flat_mins = mins_np.reshape(-1)[: e - s]
        flat_j = j_np.reshape(-1)[: e - s]
        all_min[s:e] = flat_mins
        all_j[s:e] = flat_j

        local_k = int(np.argmin(flat_mins))
        if flat_mins[local_k] < best_dist:
            best_dist = float(flat_mins[local_k])
            best_i = s + local_k
            best_j = int(flat_j[local_k])

        print(f"  scanned {e}/{Na} of A  "
              f"(running min FS = {best_dist:.6e} "
              f"at A[{best_i}] vs B[{best_j}])")

    return {
        "min": best_dist,
        "i_min_in_A": best_i,
        "j_min_in_B": best_j,
        "mean": float(all_min.mean()),
        "max": float(all_min.max()),
        "all_min_per_A": all_min,
        "all_j_per_A": all_j,
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

    result = closest_pair_fs(big, small, args.chunk_size)
    fs_search = result["min"]
    i_big = result["i_min_in_A"]
    j_small = result["j_min_in_B"]
    fs_mean = result["mean"]
    fs_max = result["max"]

    if swapped:
        i_a, j_b = j_small, i_big
        big_label, small_label = "B", "A"
    else:
        i_a, j_b = i_big, j_small
        big_label, small_label = "A", "B"

    z_a = A_orig[i_a]
    z_b = B_orig[j_b]
    fs_exact = fs_distance_exact(z_a, z_b)
    overlap = float(np.cos(fs_exact))

    print()
    print("=" * 64)
    print("Closest pair across clusters  (Fubini-Study distance on CP^4)")
    print("=" * 64)
    print(f"  cluster A file:   {args.cluster_a}")
    print(f"  cluster B file:   {args.cluster_b}")
    print(f"  cluster A index:  {i_a}   (of {A_orig.shape[0]})")
    print(f"  cluster B index:  {j_b}   (of {B_orig.shape[0]})")
    print()
    print("  A point:")
    for k, zk in enumerate(z_a):
        print(f"    z_{k} = {zk}")
    print()
    print("  B point:")
    for k, zk in enumerate(z_b):
        print(f"    z_{k} = {zk}")
    print()
    print(f"  |<a,b>| / (||a|| ||b||) :  {overlap:.12f}")
    print(f"  FS distance (complex64 search)    : {fs_search:.12e}")
    print(f"  FS distance (complex128 recompute): {fs_exact:.12e}")
    print()
    print("-" * 64)
    print(f"Cross-cluster nearest-neighbor statistics "
          f"(for each point in {big_label}, FS distance to its nearest "
          f"point in {small_label}, N={result['all_min_per_A'].shape[0]} "
          f"values):")
    print(f"  min  FS distance: {fs_search:.12e}    (= closest pair above)")
    print(f"  mean FS distance: {fs_mean:.12e}")
    print(f"  max  FS distance: {fs_max:.12e}")


if __name__ == "__main__":
    main()
