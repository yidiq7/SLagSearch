"""Host-side (pure numpy + HDBSCAN) component selection for single-cluster GD.

Clusters the post-Newton mined points by density (HDBSCAN) on the 25-D
Fubini-Study projector embedding, picks one component (anchor-tracked across
re-mines), and returns a fixed-size index set into the mined points. No JAX, no
matplotlib, so it is cheap to import in the GD training loop.
"""
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
