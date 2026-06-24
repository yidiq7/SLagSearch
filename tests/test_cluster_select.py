import numpy as np
import cluster_select


def test_fs_features_shape():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(7, 5)) + 1j * rng.normal(size=(7, 5))
    f = cluster_select.fs_features(z)
    assert f.shape == (7, 25)
    assert f.dtype == np.float64


def test_fs_features_projective_invariance():
    # P_z = z z*^T / ||z||^2 is unchanged by z -> lambda z (per-row nonzero scale).
    rng = np.random.default_rng(1)
    z = rng.normal(size=(7, 5)) + 1j * rng.normal(size=(7, 5))
    lam = rng.normal(size=(7, 1)) + 1j * rng.normal(size=(7, 1))
    np.testing.assert_allclose(
        cluster_select.fs_features(z), cluster_select.fs_features(z * lam), atol=1e-10
    )


def _two_blobs(rng, n=200, sep=10.0, std=0.4):
    a = rng.normal(loc=[0.0, 0.0], scale=std, size=(n, 2))
    b = rng.normal(loc=[sep, 0.0], scale=std, size=(n, 2))
    return np.vstack([a, b])


def test_detect_two_blobs():
    rng = np.random.default_rng(2)
    X = _two_blobs(rng)
    labels, n, sizes = cluster_select.detect_components(
        X, min_cluster_size=30, min_cluster_frac=0.05
    )
    assert n == 2
    assert sizes == sorted(sizes, reverse=True)   # descending
    assert sum(sizes) <= X.shape[0]
    assert set(np.unique(labels)) <= {-1, 0, 1}


def test_detect_single_blob():
    rng = np.random.default_rng(3)
    X = rng.normal(scale=0.4, size=(300, 2))
    _, n, _ = cluster_select.detect_components(
        X, min_cluster_size=30, min_cluster_frac=0.05
    )
    assert n == 1


def test_detect_thin_neck_splits():
    # Two dense blobs joined by a SPARSE neck. Density-based HDBSCAN must keep
    # them as 2 components (a k-NN connected-components / b0 method would merge).
    rng = np.random.default_rng(4)
    X = _two_blobs(rng, n=200, sep=10.0, std=0.4)
    neck = np.stack([np.linspace(1.0, 9.0, 6), np.zeros(6)], axis=1)  # 6 points
    _, n, _ = cluster_select.detect_components(
        np.vstack([X, neck]), min_cluster_size=30, min_cluster_frac=0.05
    )
    assert n == 2
