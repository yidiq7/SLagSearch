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


def test_select_bootstrap_picks_target_k():
    rng = np.random.default_rng(5)
    X = _two_blobs(rng)
    m0, _, i0 = cluster_select.select_cluster(X, None, 0, 30, 0.05)
    m1, _, i1 = cluster_select.select_cluster(X, None, 1, 30, 0.05)
    assert i0["chosen"] == 0 and i1["chosen"] == 1
    assert m0.sum() >= m1.sum()           # label 0 = largest
    assert not np.any(m0 & m1)            # disjoint


def test_select_tracks_anchor():
    rng = np.random.default_rng(6)
    X = _two_blobs(rng, sep=10.0)
    anchor = np.array([10.0, 0.0])        # near the [10,0] blob
    _, new_anchor, _ = cluster_select.select_cluster(X, anchor, 0, 30, 0.05)
    assert np.linalg.norm(new_anchor - np.array([10.0, 0.0])) < 2.0


def test_select_raises_when_target_k_out_of_range():
    rng = np.random.default_rng(7)
    X = _two_blobs(rng)
    import pytest
    with pytest.raises(ValueError):
        cluster_select.select_cluster(X, None, 5, 30, 0.05)


def test_fill_to_size_pads_short():
    rng = np.random.default_rng(8)
    idx = np.arange(5)
    out = cluster_select.fill_to_size(idx, 12, rng)
    assert out.shape == (12,)
    assert set(idx.tolist()).issubset(set(out.tolist()))   # every member present


def test_fill_to_size_subsamples_large():
    rng = np.random.default_rng(9)
    out = cluster_select.fill_to_size(np.arange(100), 20, rng)
    assert out.shape == (20,)
    assert len(np.unique(out)) == 20                       # no replacement


def test_fill_to_size_exact():
    rng = np.random.default_rng(10)
    out = cluster_select.fill_to_size(np.arange(8), 8, rng)
    assert sorted(out.tolist()) == list(range(8))


def test_stability_sweep_plateau():
    rng = np.random.default_rng(11)
    X = _two_blobs(rng)
    sweep = cluster_select.stability_sweep(X, [20, 30, 40, 50], min_cluster_frac=0.05)
    assert set(sweep) == {20, 30, 40, 50}
    assert all(v == 2 for v in sweep.values())   # clean blobs -> stable n=2
