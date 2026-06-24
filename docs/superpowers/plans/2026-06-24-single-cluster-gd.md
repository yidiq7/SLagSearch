# Single-Cluster Gradient Descent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `gradient_descent.py` optimize the sLag conditions on *one geometric component* of the mined zero set at a time, selected by density clustering (HDBSCAN) on Fubini–Study features of the post-Newton points.

**Architecture:** A new pure-numpy module `cluster_select.py` does all clustering host-side (FS-projector features → HDBSCAN → component selection with anchor tracking → fixed-size index fill). `gradient_descent.py` calls it from a thin `mine_one_cluster` wrapper right after each (re-)mine; the chosen component's points are handed to the **unchanged** loss/fitness functions. When `--target_cluster` is unset, behavior is byte-for-byte the old path.

**Tech Stack:** Python 3.12, NumPy, JAX (x64), `sklearn.cluster.HDBSCAN` (already in the `viz-3d` extra), pytest.

## Global Constraints

- **Testing is on the cluster, by the user.** No local test execution. Every "Run" step below is a command the *user* runs on the cluster after the relevant commits are pushed; the implementer writes code + tests but does not run them locally. (repo CLAUDE.md)
- **Branch:** `experiment/single-cluster-gd` (already created, off `experiment/top-lag-frac`).
- **Commits:** no Co-Authored-By trailer; commit with explicit pathspecs (`git commit -- <paths>`) — concurrent sessions may stage other files. (memory)
- **JAX precision:** `jax_enable_x64` is set at `gradient_descent.py:85`; keep FP64 throughout.
- **`top_lag_frac` is NOT hardcoded.** The experiment passes `--top_lag_frac 1.0`; the flag already exists (`gradient_descent.py:598`). Do not change its default.
- **HDBSCAN import is lazy** (inside a function), so importing `cluster_select` / `gradient_descent` does not require the `viz-3d` extra.
- **FS features:** Euclidean distance on the 25-D projector embedding = `√2·sin(d_FS)`; projective-invariant and patch-independent. This is the only distance used for clustering.
- **Point convention:** `(N,10)` real = `[Re(z_0..4), Im(z_0..4)]`; complex `z = real[:, :5] + 1j*real[:, 5:]`.

---

# Milestone 1 — `cluster_select.py` + unit tests

Self-contained, pure-numpy. Validation gate at the end: `uv run pytest tests/test_cluster_select.py -v` on the cluster, all green, before starting Milestone 2.

### Task 1: FS-feature embedding

**Files:**
- Create: `cluster_select.py`
- Test: `tests/test_cluster_select.py`

**Interfaces:**
- Produces: `fs_features(z: np.ndarray) -> np.ndarray` — `(N,5)` complex → `(N,25)` float64.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cluster_select.py
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
```

- [ ] **Step 2: Run test to verify it fails** (on the cluster)

Run: `uv run pytest tests/test_cluster_select.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cluster_select'`.

- [ ] **Step 3: Write minimal implementation**

```python
# cluster_select.py
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
```

- [ ] **Step 4: Run test to verify it passes** (cluster)

Run: `uv run pytest tests/test_cluster_select.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_cluster_select.py cluster_select.py
git commit -- tests/test_cluster_select.py cluster_select.py -m "feat(cluster_select): FS-projector feature embedding"
```

### Task 2: HDBSCAN component detection

**Files:**
- Modify: `cluster_select.py` (add `_get_hdbscan`, `cluster_labels`, `detect_components`)
- Modify: `pyproject.toml` (add a `dev` extra with pytest)
- Test: `tests/test_cluster_select.py` (add)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `cluster_labels(features, min_cluster_size, cluster_selection_epsilon=0.0) -> np.ndarray` (int labels, noise = -1).
  - `detect_components(features, min_cluster_size, min_cluster_frac, cluster_selection_epsilon=0.0) -> tuple[np.ndarray, int, list[int]]` — `(labels, n, sizes)` where survivors (size ≥ `min_cluster_frac·N`) are relabeled `0..n-1` by **descending size**, everything else `-1`.

- [ ] **Step 1: Add the dev extra to `pyproject.toml`**

After the `viz-3d` block (line 50), before `[tool.uv]`:

```toml
# Test runner. Install with:   uv sync --extra dev
dev = [
    "pytest>=8.0",
]
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_cluster_select.py  (append)
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
```

- [ ] **Step 3: Run to verify they fail** (cluster)

Run: `uv run pytest tests/test_cluster_select.py -k detect -v`
Expected: FAIL — `AttributeError: module 'cluster_select' has no attribute 'detect_components'`.

- [ ] **Step 4: Implement**

```python
# cluster_select.py  (append)
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
    )
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
    labels = np.full(n_total, -1, dtype=int)
    sizes = []
    for new_lbl, (old_lbl, sz) in enumerate(keep):
        labels[raw == old_lbl] = new_lbl
        sizes.append(sz)
    return labels, len(keep), sizes
```

- [ ] **Step 5: Run to verify they pass** (cluster)

Run: `uv run pytest tests/test_cluster_select.py -k detect -v`
Expected: PASS (3 passed).
*If `test_detect_thin_neck_splits` fails:* HDBSCAN's neck behavior depends on params — try `min_cluster_size=20` or a sparser neck (`np.linspace(2,8,4)`). This is the one test whose outcome must be confirmed on real cluster HDBSCAN.

- [ ] **Step 6: Commit**

```bash
git add tests/test_cluster_select.py cluster_select.py pyproject.toml
git commit -- tests/test_cluster_select.py cluster_select.py pyproject.toml -m "feat(cluster_select): HDBSCAN component detection (descending-size labels, noise floor)"
```

### Task 3: Component selection with anchor tracking

**Files:**
- Modify: `cluster_select.py` (add `component_centroid`, `select_cluster`)
- Test: `tests/test_cluster_select.py` (add)

**Interfaces:**
- Consumes: `detect_components`.
- Produces:
  - `component_centroid(features, labels, label) -> np.ndarray` (mean feature vector of that component).
  - `select_cluster(features, anchor, target_k, min_cluster_size, min_cluster_frac, cluster_selection_epsilon=0.0) -> tuple[np.ndarray, np.ndarray, dict]` — returns `(member_mask, new_anchor, info)`. `anchor is None` ⇒ bootstrap (pick component `target_k`); else pick the component whose centroid is nearest `anchor`. `info` has keys `n_components`, `sizes`, `chosen`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cluster_select.py  (append)
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
```

- [ ] **Step 2: Run to verify they fail** (cluster)

Run: `uv run pytest tests/test_cluster_select.py -k select -v`
Expected: FAIL — `AttributeError: ... 'select_cluster'`.

- [ ] **Step 3: Implement**

```python
# cluster_select.py  (append)
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
```

- [ ] **Step 4: Run to verify they pass** (cluster)

Run: `uv run pytest tests/test_cluster_select.py -k select -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_cluster_select.py cluster_select.py
git commit -- tests/test_cluster_select.py cluster_select.py -m "feat(cluster_select): anchor-tracked component selection"
```

### Task 4: Fixed-size index fill

**Files:**
- Modify: `cluster_select.py` (add `fill_to_size`)
- Test: `tests/test_cluster_select.py` (add)

**Interfaces:**
- Produces: `fill_to_size(member_idx, size, rng) -> np.ndarray` — exactly `size` indices: every member once, remaining slots padded by uniform resample-with-replacement; subsample without replacement only if there are more than `size` members.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cluster_select.py  (append)
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
```

- [ ] **Step 2: Run to verify they fail** (cluster)

Run: `uv run pytest tests/test_cluster_select.py -k fill -v`
Expected: FAIL — `AttributeError: ... 'fill_to_size'`.

- [ ] **Step 3: Implement**

```python
# cluster_select.py  (append)
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
```

- [ ] **Step 4: Run to verify they pass** (cluster)

Run: `uv run pytest tests/test_cluster_select.py -k fill -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_cluster_select.py cluster_select.py
git commit -- tests/test_cluster_select.py cluster_select.py -m "feat(cluster_select): fixed-size index fill (all members + resample pad)"
```

### Task 5: Stability sweep + Milestone-1 gate

**Files:**
- Modify: `cluster_select.py` (add `stability_sweep`)
- Test: `tests/test_cluster_select.py` (add)

**Interfaces:**
- Produces: `stability_sweep(features, sizes_to_try, min_cluster_frac, cluster_selection_epsilon=0.0) -> dict[int,int]` — `{min_cluster_size: n_components}`. A plateau = the robust component count (≈ PH b₀).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cluster_select.py  (append)
def test_stability_sweep_plateau():
    rng = np.random.default_rng(11)
    X = _two_blobs(rng)
    sweep = cluster_select.stability_sweep(X, [20, 30, 40, 50], min_cluster_frac=0.05)
    assert set(sweep) == {20, 30, 40, 50}
    assert all(v == 2 for v in sweep.values())   # clean blobs -> stable n=2
```

- [ ] **Step 2: Run to verify it fails** (cluster)

Run: `uv run pytest tests/test_cluster_select.py -k stability -v`
Expected: FAIL — `AttributeError: ... 'stability_sweep'`.

- [ ] **Step 3: Implement**

```python
# cluster_select.py  (append)
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
```

- [ ] **Step 4: Run the FULL module test (Milestone-1 gate)** (cluster)

Run: `uv run pytest tests/test_cluster_select.py -v`
Expected: PASS (all 12 tests). **Do not start Milestone 2 until this is green.**

- [ ] **Step 5: Commit + push**

```bash
git add tests/test_cluster_select.py cluster_select.py
git commit -- tests/test_cluster_select.py cluster_select.py -m "feat(cluster_select): stability sweep diagnostic"
git push
```

---

# Milestone 2 — `gradient_descent.py` integration

Validated by cluster smoke-runs (no local pytest possible — needs JAX/GPU/point cloud). Each task: confirm a non-cluster run is unchanged AND the cluster path does what the step claims.

### Task 6: CLI flags + imports (no behavior change)

**Files:**
- Modify: `gradient_descent.py` (imports near line 73; argparse near line 620; post-parse default + divisibility checks near lines 648 and 674)

**Interfaces:**
- Produces: `args.target_cluster` (int|None, default None), `args.min_cluster_size`, `args.cluster_selection_epsilon`, `args.min_cluster_frac`, `args.cluster_minset_size` (defaults to `args.minset_size`), `args.mine_oversample`. Module import `cluster_select`; `unshard_leading_axis` from `sharding`.

- [ ] **Step 1: Add imports**

Change `gradient_descent.py:73` from:

```python
from sharding import device_put_sharded, shard_leading_axis, take_replicated
```

to:

```python
from sharding import (
    device_put_sharded, shard_leading_axis, take_replicated, unshard_leading_axis,
)
import cluster_select
```

- [ ] **Step 2: Add argparse flags**

After `gradient_descent.py:646` (the `--lbfgs_memory_size` argument, just before `args = parser.parse_args()`):

```python
    parser.add_argument("--target_cluster", type=int, default=None,
                        help="Optimize on ONE geometric component of the mined "
                             "zero set (0-indexed; components ranked by descending "
                             "size at the first mine, then anchor-tracked). HDBSCAN "
                             "on FS features; needs scikit-learn>=1.3. Default None "
                             "= whole manifold (unchanged behavior).")
    parser.add_argument("--min_cluster_size", type=int, default=200,
                        help="HDBSCAN min_cluster_size (only with --target_cluster).")
    parser.add_argument("--cluster_selection_epsilon", type=float, default=0.0,
                        help="HDBSCAN cluster_selection_epsilon (merge components "
                             "closer than this; raise to keep a fat neck merged).")
    parser.add_argument("--min_cluster_frac", type=float, default=0.02,
                        help="Drop components smaller than this fraction of the "
                             "mined points as noise.")
    parser.add_argument("--cluster_minset_size", type=int, default=None,
                        help="Fixed per-cluster min-set size fed to the loss. "
                             "Default: --minset_size.")
    parser.add_argument("--mine_oversample", type=int, default=2,
                        help="Mine this multiple of --cluster_minset_size so the "
                             "target component is well-populated before extraction.")
```

- [ ] **Step 3: Default `cluster_minset_size`**

Immediately after `gradient_descent.py:647` (`args = parser.parse_args()`):

```python
    if args.cluster_minset_size is None:
        args.cluster_minset_size = args.minset_size
```

- [ ] **Step 4: Divisibility checks for the cluster path**

Inside the `if num_devices > 1:` block, after the `--plot_k` check (after `gradient_descent.py:683`):

```python
        if args.target_cluster is not None:
            if args.cluster_minset_size % num_devices != 0:
                raise ValueError(
                    f"--cluster_minset_size {args.cluster_minset_size} not "
                    f"divisible by num_devices={num_devices}")
            if (args.mine_oversample * args.cluster_minset_size) % num_devices != 0:
                raise ValueError(
                    f"mine_oversample*cluster_minset_size="
                    f"{args.mine_oversample * args.cluster_minset_size} not "
                    f"divisible by num_devices={num_devices}")
```

- [ ] **Step 5: Verify (cluster) — flags present, old path unchanged**

Run: `uv run python gradient_descent.py --help`
Expected: `--target_cluster`, `--min_cluster_size`, `--cluster_minset_size`, etc. listed.

Run a short normal run (no `--target_cluster`), confirm it behaves exactly as before:
`uv run python gradient_descent.py --job_id smoke_baseline --steps 5 --mine_interval 5 --minset_size 2000 --no-make_plots`
Expected: runs to step 5, prints loss/fit lines as before, no cluster output.

- [ ] **Step 6: Commit**

```bash
git add gradient_descent.py
git commit -- gradient_descent.py -m "feat(gd): add --target_cluster CLI flags + cluster_select import (no behavior change)"
```

### Task 7: `mine_one_cluster` wrapper, wired into the mine calls

**Files:**
- Modify: `gradient_descent.py` (add `mine_one_cluster` after `make_parallel_mining`, ~line 379; init anchor/rng before the initial mine ~line 743; replace the two mine calls at ~745 and ~780)

**Interfaces:**
- Consumes: `cluster_select.{fs_features, select_cluster, fill_to_size}`, `unshard_leading_axis`, `shard_leading_axis`.
- Produces: `mine_one_cluster(mining_fn, points_in, coeffs, psi, args, num_devices, anchor, rng) -> (min_set_real, distances, new_anchor, info)`. Passthrough (target_cluster None) returns `mining_fn(... minset_size ...)` + `(anchor, None)`.

- [ ] **Step 1: Implement the wrapper**

After `make_parallel_mining`'s `return fn` (after `gradient_descent.py:378`), add:

```python
def mine_one_cluster(mining_fn, points_in, coeffs, psi, args, num_devices, anchor, rng):
    """Mine, then (if --target_cluster set) extract a fixed-size pure-cluster
    min-set via HDBSCAN on FS features. Returns
    (min_set_real, distances, new_anchor, info).

    Passthrough when args.target_cluster is None: mines args.minset_size and
    returns it unchanged, so non-cluster runs are byte-for-byte the old path.
    """
    if args.target_cluster is None:
        min_set_real, distances, _ = mining_fn(
            points_in, coeffs, psi, args.minset_size, args.newton_steps,
        )
        return min_set_real, distances, anchor, None

    k_mine = args.mine_oversample * args.cluster_minset_size
    min_set_raw, distances, _ = mining_fn(
        points_in, coeffs, psi, k_mine, args.newton_steps,
    )
    host = (np.asarray(unshard_leading_axis(min_set_raw))
            if num_devices > 1 else np.asarray(min_set_raw))   # (k_mine, 10)
    z = host[:, :5] + 1j * host[:, 5:]                          # (k_mine, 5) complex
    feats = cluster_select.fs_features(z)
    member_mask, new_anchor, info = cluster_select.select_cluster(
        feats, anchor, args.target_cluster,
        args.min_cluster_size, args.min_cluster_frac, args.cluster_selection_epsilon,
    )
    if anchor is None:
        print(f"  [cluster] detected {info['n_components']} component(s), "
              f"sizes {info['sizes']}; following component {info['chosen']} "
              f"(of {host.shape[0]} mined pts)")
    else:
        print(f"  [cluster] {info['n_components']} component(s), sizes "
              f"{info['sizes']}; tracked -> component {info['chosen']}")
    member_idx = np.flatnonzero(member_mask)
    fixed_idx = cluster_select.fill_to_size(member_idx, args.cluster_minset_size, rng)
    cluster_real = host[fixed_idx]                             # (cluster_minset_size, 10)
    min_set_real = (shard_leading_axis(jnp.asarray(cluster_real), num_devices)
                    if num_devices > 1 else jnp.asarray(cluster_real))
    return min_set_real, distances, new_anchor, info
```

- [ ] **Step 2: Initialize anchor + rng before the initial mine**

`cluster_anchor` must default to None and be restored on resume. In the resume branch (after `gradient_descent.py:718`, `start_step = int(ckpt["step"])`), add:

```python
        cluster_anchor = ckpt.get("anchor")
```

In the fresh-start `else` branch (after `gradient_descent.py:732`, `history = []`), add:

```python
        cluster_anchor = None
```

Then immediately before the initial mine (before `gradient_descent.py:744`'s comment), add:

```python
    cluster_rng = np.random.default_rng(args.seed)
```

- [ ] **Step 3: Replace the initial mine call**

Change `gradient_descent.py:745-747`:

```python
    min_set_real, distances, _ = mining_fn(
        points_in, coeffs, psi, args.minset_size, args.newton_steps,
    )
```

to:

```python
    min_set_real, distances, cluster_anchor, _ = mine_one_cluster(
        mining_fn, points_in, coeffs, psi, args, num_devices, cluster_anchor, cluster_rng,
    )
```

- [ ] **Step 4: Replace the re-mine call**

Change the in-loop re-mine (`gradient_descent.py:780-782`):

```python
            min_set_real, distances, _ = mining_fn(
                points_in, coeffs, psi, args.minset_size, args.newton_steps,
            )
```

to:

```python
            min_set_real, distances, cluster_anchor, _ = mine_one_cluster(
                mining_fn, points_in, coeffs, psi, args, num_devices,
                cluster_anchor, cluster_rng,
            )
```

- [ ] **Step 5: Verify (cluster)**

Single-GPU short cluster run (substitute your trained d=4 coeffs pkl):
`uv run python gradient_descent.py --job_id smoke_c0 --max_degree 4 --init_pkl <your d4 coeffs>.pkl --steps 20 --mine_interval 10 --minset_size 4000 --top_lag_frac 1.0 --target_cluster 0 --no-make_plots`
Expected: a `[cluster] detected N component(s), sizes [...]; following component 0` line at the first mine and a `tracked -> component 0` line at step 10; loss/fit lines print normally; min-set size is `cluster_minset_size` (=4000).

Also re-confirm the baseline (no `--target_cluster`) is unchanged.

- [ ] **Step 6: Commit + push**

```bash
git add gradient_descent.py
git commit -- gradient_descent.py -m "feat(gd): mine_one_cluster — single-cluster min-set extraction wired into the mine schedule"
git push
```

### Task 8: Persist + restore the anchor in checkpoints

**Files:**
- Modify: `gradient_descent.py` (both checkpoint payloads, ~lines 821 and 841)

**Interfaces:**
- Produces: checkpoint dicts gain `"anchor"` (np array | None) and `"target_cluster"` (int | None). Restored at Task 7 Step 2.

- [ ] **Step 1: Add to the Adam-loop checkpoint payload**

In the payload dict at `gradient_descent.py:821-827`, after `"args": vars(args),`:

```python
                "anchor": None if cluster_anchor is None else np.asarray(cluster_anchor),
                "target_cluster": args.target_cluster,
```

- [ ] **Step 2: Add to the L-BFGS checkpoint payload**

In the payload dict at `gradient_descent.py:841-848`, after `"phase": "lbfgs",`:

```python
            "anchor": None if cluster_anchor is None else np.asarray(cluster_anchor),
            "target_cluster": args.target_cluster,
```

- [ ] **Step 3: Verify (cluster) — resume keeps the same component**

Run a cluster run for a few steps with `--save_every 10`, then resume:
`uv run python gradient_descent.py --job_id smoke_resume --max_degree 4 --init_pkl <d4>.pkl --steps 10 --save_every 10 --minset_size 4000 --top_lag_frac 1.0 --target_cluster 0 --no-make_plots`
`uv run python gradient_descent.py --job_id smoke_resume --resume gd_runs/gd_smoke_resume_step10.pkl --steps 20 --minset_size 4000 --top_lag_frac 1.0 --target_cluster 0 --no-make_plots`
Expected: the resumed run prints `tracked -> component 0` (not a fresh bootstrap), i.e. it loaded the anchor and kept following the same component.

- [ ] **Step 4: Commit + push (Milestone-2 core gate)**

```bash
git add gradient_descent.py
git commit -- gradient_descent.py -m "feat(gd): persist/restore cluster anchor + target_cluster in checkpoints"
git push
```

**Milestone-2 core validation (cluster):** run the real experiment — your trained d=4 coeffs, both clusters, full steps:
```bash
uv run python gradient_descent.py --job_id d4_cluster0 --max_degree 4 --init_pkl <d4>.pkl --steps 2000 --top_lag_frac 1.0 --target_cluster 0
uv run python gradient_descent.py --job_id d4_cluster1 --max_degree 4 --init_pkl <d4>.pkl --steps 2000 --top_lag_frac 1.0 --target_cluster 1
```
Compare `spec_fit` curves (`python -m viz.plot_gd_history --filepath gd_runs/gd_d4_cluster{0,1}_step2000.pkl`) — the real sLag is the component whose `spec_fit` climbs.

---

# Phase 2 (deferred — separable, after the core is validated)

### Task 9: `--target_cluster all` in-process loop

**Files:**
- Modify: `gradient_descent.py` (extract the per-run body — mine→train→save→plots — into a `run_one(args, coeffs0, opt_state0, ...)` function; loop over `k` when `--target_cluster all`).

**Interfaces:**
- Produces: `--target_cluster` also accepts the string `all`. Bootstrap-detects N once, then runs `run_one` for `k=0..N-1`, each with `job_id=f"{args.job_id}_c{k}"` and its own run folder/checkpoints.

- [ ] **Step 1:** change `--target_cluster` to `type=str` and parse: `None` | `"all"` | int-string. Add `args.target_cluster_int` resolution.
- [ ] **Step 2:** refactor the body from the initial mine (line ~744) through the final plots (line ~859) into `def run_one(...)`, parameterized by the target component index and `job_id`.
- [ ] **Step 3:** for `all`, do one bootstrap mine at the init coeffs, `cluster_select.detect_components` → `N`; loop `k in range(N)` calling `run_one(k)` with coeffs re-initialized from `init_coeffs`/resume each time.
- [ ] **Step 4 (verify, cluster):** `--target_cluster all` on the d=4 init runs two sequential GDs, writing `plots_slag_<job>_c0` and `_c1`.
- [ ] **Step 5:** commit + push.

*(Full code deferred to implementation time — the refactor boundary is the per-run body; it depends on the exact final line numbers after Tasks 6–8, so it is specified structurally here and fleshed out when reached.)*

### Task 10: Bootstrap stability-sweep diagnostic mode

**Files:**
- Modify: `gradient_descent.py` (add `--cluster_diagnostic` flag: mine once, print `cluster_select.stability_sweep` + component sizes, exit before training).

- [ ] **Step 1:** add `--cluster_diagnostic` (store_true) and `--cluster_sweep_sizes` (comma-separated ints, default `"50,100,200,400,800"`).
- [ ] **Step 2:** after the initial mine, if set: gather to host, `fs_features`, `stability_sweep`, print `{min_cluster_size: n_components}` and `detect_components` sizes at `--min_cluster_size`, then `return`.
- [ ] **Step 3 (verify, cluster):** `--target_cluster 0 --cluster_diagnostic` prints the sweep table (look for the plateau = b₀) and exits.
- [ ] **Step 4:** commit + push.

---

## Self-Review

**Spec coverage:** HDBSCAN on FS features at mine-time (Tasks 2, 7) ✓; FS distance/patch-independence (Task 1) ✓; `top_lag_frac=1.0` via existing flag, unchanged loss/fitness (Task 7 feeds pure-cluster points; no loss edits) ✓; two clusters sequential / full opt (Milestone-2 validation; Task 9 for the `all` convenience) ✓; mask-only, B ignored (extraction drops non-members) ✓; anchor tracking + checkpoint persistence (Tasks 3, 7, 8) ✓; auto-detect N + stability sweep (Tasks 2, 5, 10) ✓; neck-splitting (Task 2 `test_detect_thin_neck_splits`) ✓; fixed-size static-shape for jit/pmap (Task 4 + Task 7 reshard) ✓; multi-GPU gather/reshard (Task 7) ✓; "all points used" via include-once+pad (Task 4) ✓; HDBSCAN noise-drop replaces the worst-1% guard (Task 2 `min_cluster_frac`) ✓; dependency already in `viz-3d`, lazy import (Task 2) ✓.

**Placeholder scan:** Tasks 1–8 contain complete code. Tasks 9–10 are explicitly deferred/structural (their exact code depends on post-Task-8 line numbers); `<your d4 coeffs>.pkl` is a user-supplied runtime data path, not a code placeholder.

**Type consistency:** `select_cluster` returns `(member_mask, new_anchor, info)` and `mine_one_cluster` consumes exactly that; `info` keys (`n_components`, `sizes`, `chosen`) match between Task 3 and Task 7's prints; `fill_to_size(member_idx, size, rng)` signature matches Task 7's call; `detect_components` returns `(labels, n, sizes)` used consistently in Tasks 3 and 5; checkpoint `"anchor"`/`"target_cluster"` written (Task 8) match the keys read (Task 7 Step 2).
