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
