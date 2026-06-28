import numpy as np
import pytest
import pointcloud_distance as pcd


def _blob(n, d=25, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)) * scale


# --- pairwise-distance drift (primary scalar) ---

def test_pairwise_fs_distances_shape_and_no_self_pairs():
    emb = _blob(500, seed=1)
    rng = np.random.default_rng(0)
    d = pcd.pairwise_fs_distances(emb, n_pairs=10_000, rng=rng)
    assert d.shape == (10_000,)
    assert np.all(d > 0)  # self-pairs (distance 0) are excluded


def test_drift_identical_cloud_near_zero():
    emb = _blob(2000, seed=2)
    rng = np.random.default_rng(0)
    # Same cloud, independent pair draws -> only sampling fluctuation.
    drift = pcd.pairwise_distance_drift(emb, emb, n_pairs=200_000, rng=rng)
    assert drift < 0.02


def test_drift_isometry_invariant():
    # A rigid rotation+translation does NOT change internal pairwise distances.
    emb = _blob(2000, seed=3)
    rng_rot = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng_rot.standard_normal((25, 25)))
    moved = emb @ Q + 5.0
    rng = np.random.default_rng(0)
    drift = pcd.pairwise_distance_drift(emb, moved, n_pairs=200_000, rng=rng)
    assert drift < 0.02


def test_drift_grows_with_scaling():
    # Scaling changes the distance distribution -> drift should increase.
    emb = _blob(2000, seed=4)
    rng = np.random.default_rng(0)
    d_small = pcd.pairwise_distance_drift(emb, emb * 1.05, n_pairs=200_000, rng=rng)
    d_big = pcd.pairwise_distance_drift(emb, emb * 1.30, n_pairs=200_000, rng=rng)
    assert d_small > 0.02
    assert d_big > d_small


# --- FS Chamfer (secondary local check) ---

def test_chamfer_identical_is_zero():
    emb = _blob(1000, seed=10)
    assert pcd.fs_chamfer(emb, emb) == pytest.approx(0.0, abs=1e-12)


def test_chamfer_symmetric():
    a = _blob(800, seed=11)
    b = _blob(800, seed=12) + 0.3
    assert pcd.fs_chamfer(a, b) == pytest.approx(pcd.fs_chamfer(b, a), rel=1e-9)


def test_chamfer_grows_with_shift():
    a = _blob(800, seed=13)
    near = pcd.fs_chamfer(a, a + 0.1)
    far = pcd.fs_chamfer(a, a + 0.5)
    assert 0.0 < near < far


# --- noise-floor calibration + walk decision ---

def test_calibrate_noise_floor_small_for_resamples():
    # mine_and_embed returns fresh independent samples of the SAME blob.
    def mine_and_embed(seed):
        return _blob(2000, scale=1.0, seed=seed + 100)
    floor = pcd.calibrate_noise_floor(mine_and_embed, n_repeats=3,
                                      n_pairs=100_000, rng=np.random.default_rng(0))
    assert set(floor) == {"wass_floor", "chamfer_floor", "samples"}
    assert len(floor["samples"]) == 3
    assert floor["wass_floor"] < 0.05  # same distribution -> tiny floor
    # A genuinely different (scaled) cloud must exceed the floor.
    other = _blob(2000, scale=1.4, seed=999)
    drift = pcd.pairwise_distance_drift(floor["samples"][0], other,
                                        n_pairs=100_000, rng=np.random.default_rng(1))
    assert drift > floor["wass_floor"]


def test_decide_cases():
    fl = 0.01
    # fitness dropped -> reject (even if far)
    assert pcd.decide(drift=1.0, lag=0.5, spec=0.9, lag0=0.9, spec0=0.9,
                      tol=0.02, wass_floor=fl, target_mult=8.0) == "reject"
    # fitness ok and drift beyond target -> stop
    assert pcd.decide(drift=0.2, lag=0.89, spec=0.89, lag0=0.9, spec0=0.9,
                      tol=0.02, wass_floor=fl, target_mult=8.0) == "stop"
    # fitness ok, drift below target -> accept
    assert pcd.decide(drift=0.03, lag=0.9, spec=0.9, lag0=0.9, spec0=0.9,
                      tol=0.02, wass_floor=fl, target_mult=8.0) == "accept"
