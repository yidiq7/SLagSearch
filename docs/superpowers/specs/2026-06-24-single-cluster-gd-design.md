# Single-cluster gradient descent — design

**Date:** 2026-06-24
**Branch:** `experiment/single-cluster-gd` (off `experiment/top-lag-frac`)
**Status:** design, pending review

## Summary

Add a mode to `gradient_descent.py` that optimizes the sLag conditions on **one
geometric cluster** of the mined zero set at a time, instead of on the whole
point cloud. The cluster is identified at mine-time by density clustering
(HDBSCAN) on Fubini–Study features of the post-Newton points. The chosen
cluster's points are handed to the **unchanged** loss/fitness functions
(`top_lag_frac=1.0`). "Two clusters" = two independent sequential runs.

## Goal & context

The well-trained d=4 candidate's mined submanifold has (at least) two
components / clusters. We want to run GD that drives **one** cluster toward a
special-Lagrangian, ignoring the other.

Why this is needed: both clusters have **comparable Lagrangian quality**, so the
existing `top_lag_frac` knob (a *global* rank by Lagrangian norm) cannot isolate
one — at `0.5` it just takes the best half of *each* cluster. Isolating a
cluster requires a **geometric** selection, not a quality rank. The **special
(phase) condition** is expected to be what ultimately separates a real sLag
component from a junk one, so running each cluster on its own and comparing how
`spec_fit` evolves is the point of the experiment.

## Non-goals

- **Not** removing the other cluster B from the zero set. The coeffs define the
  whole zero set; this is mask-only — B is excluded from the loss, but still
  exists and will still be mined. (Confirmed: "ignore B".)
- **Not** a multi-branch engine. Two clusters → two sequential full GD runs.
- **Not** changing GA. GA keeps `top_lag_frac=0.99`; this is GD-only.
- **No** triage / cheap-then-refine for now (full optimization on each cluster).

## Workflow (how this runs)

Editing/commits happen here via Claude Code; **execution is on the cluster**.
The user pulls `experiment/single-cluster-gd` on the cluster and runs it there.
There is **no local test step** — all validation is run by the user on the
cluster (per repo CLAUDE.md). Implementation must therefore be correct by
construction; the branch is pushed so the cluster can pull it.

## Key decisions (recap)

| Decision | Choice | Why |
| --- | --- | --- |
| Clustering method | **HDBSCAN** on FS-projector features | Density-based ⇒ splits a *thin neck* (k-NN connected-components / b₀ would merge it); **auto-detects N**; deterministic labels; drops outliers as noise |
| Feature space / distance | 25-D FS projector `P_z = z z̄†/‖z‖²` (reuse `fs_feature_embedding`) | Euclidean on features = `√2·sin(d_FS)` — true Fubini–Study, projective-invariant, **patch-independent** by construction |
| When to cluster | At **mine-time**, on the **post-Newton** mined points | Structure only appears after Newton lands points on the CY; min-set is frozen between mines, so clustering every step is wasted work |
| Point selection | `top_lag_frac = 1.0` (all points of the chosen cluster) | "sLag is smooth"; the worst-1% blow-ups the trim guarded against are exactly what HDBSCAN drops as noise |
| Runs | Two clusters, sequential, full optimization | Simplest; compare `spec_fit` curves afterward |

## Architecture & data flow

**Bootstrap (once, at the init d=4 coeffs):**
1. Mine at the init coeffs (`filter_and_refine`) → post-Newton `min_set_real`.
2. `fs_feature_embedding` → 25-D features.
3. HDBSCAN → labels. Run a small **stability sweep** of `#components` vs
   `min_cluster_size` and pick N from the plateau; cross-check against the PH b₀.
4. Sort surviving components (size ≥ `min_cluster_frac · N_total`) by
   **descending size** → stable labels `0 … N−1`.

**Branch *k* (one full GD run, `--target_cluster k`):**
- Record component *k*'s FS-centroid as the **anchor** (persisted in checkpoint).
- Each (re-)mine: mine (with oversampling) → FS features → HDBSCAN → match the
  target component by **nearest FS-centroid to the anchor** → refresh anchor →
  collect that component's points → build a **fixed-size** `cluster_minset_size`
  array: every component point included **once**, remaining slots padded by
  uniform resample-with-replacement (unbiased for a mean / Kuramoto). Default
  `cluster_minset_size = minset_size` (≥ any single component), so padding — not
  subsampling — is the normal path and **every component point is used**.
- Hand the pure-cluster min-set to the **unchanged**
  `compute_losses_on_fixed_points` / `compute_ga_fitness` with `top_lag_frac=1.0`.
- Adam, checkpoints, plots: unchanged — one standard run folder per branch.

`--target_cluster all` loops `k = 0 … N−1` sequentially in one process.

## Components & boundaries

1. **`cluster_select.py` (new, host-side, numpy + sklearn/hdbscan; no JAX).**
   - `fs_features(z)` — reuse `viz.plot_3D.fs_feature_embedding`.
   - `cluster(features, params) -> labels`.
   - `detect_n(features, params) -> (n, diagnostics)` — stability sweep.
   - `select_target(features, labels, anchor) -> (member_idx, new_anchor)`.
   Runs only at mine-time; non-differentiable selection is fine (same status as
   the existing `argsort`).
2. **Mining-wrapper hook** in `gradient_descent.py`: after `filter_and_refine`,
   call `cluster_select` to extract a fixed-size pure-cluster min-set. This is
   the *only* place new logic touches the training loop; the loss/fitness kernels
   are untouched.
3. **CLI flags:** `--target_cluster {all,<k>}`, `--min_cluster_size`,
   `--cluster_selection_epsilon`, `--min_cluster_frac`, `--cluster_minset_size`,
   `--mine_oversample`. (`--top_lag_frac` already exists; the experiment passes
   `1.0`.)
4. **Checkpoint additions:** `anchor` (25-vector) and `target_cluster` (int),
   so `--resume` keeps following the same component.

## Integration: shapes & multi-GPU

- The pure-cluster min-set is delivered at a **fixed** size
  (`cluster_minset_size`, divisible by `num_devices`), so the existing
  `jit`/`pmap` path is unchanged — no recompiles, no sharding changes.
- Clustering is host-side at mine-time (cadence `mine_interval`), so its cost and
  non-differentiability are non-issues.
- Per-shard reductions remain the existing documented approximation.

## Edge cases & risks

- **`top_lag_frac=1.0` removes `main`'s worst-1% Lagrangian guard.** On `main`
  the GD Lagrangian loss is a hardcoded bottom-99% mean
  (`gradient_descent.py:179–181`); at `1.0` we keep those blow-ups. **HDBSCAN's
  noise-drop is now the replacement guard** — verify on real mined data that it
  actually removes mis-converged points before trusting the run.
- **Sizing:** padding-to-fill is the normal path (see data flow). The only real
  edge is a component *larger* than `cluster_minset_size` (would subsample, i.e.
  not all points) — avoided by defaulting `cluster_minset_size = minset_size`.
  Mining `mine_oversample × minset_size` up front keeps each component
  well-populated.
- **N can drift** (components merge/split as coeffs move): the anchor follows the
  nearest piece; log when the component count changes mid-run.
- **Thin vs fat neck:** whether a neck is cut is governed by `min_cluster_size` /
  `cluster_selection_epsilon` — tune on real mined data. A dense (fat) bridge is
  kept merged (defensibly one piece).
- **Dependency:** HDBSCAN via the standalone `hdbscan` package or
  `sklearn.cluster.HDBSCAN` — believed added in scikit-learn 1.3 (~2023), *verify
  against the cluster env*. Repo already uses sklearn (KMeans) + umap.

## Validation (on the cluster, by the user)

- Bootstrap diagnostic: `#components` vs `min_cluster_size` plateau; confirm it
  matches the PH b₀ reading.
- Run both clusters; compare `lag_fit` / `spec_fit` curves — the real sLag is the
  cluster whose `spec_fit` climbs.
- Sanity: with `N = 1` (no split detected) the run should reduce to a normal
  full-manifold GD at `top_lag_frac=1.0`.

## Open parameters (tunable, not blockers)

`min_cluster_size`, `cluster_selection_epsilon`, `min_cluster_frac`,
`mine_oversample`, `cluster_minset_size`.
