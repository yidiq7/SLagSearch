# SLagSearch

Searches for special-Lagrangian (sLag) submanifolds inside the Fermat quintic
Calabi–Yau threefold by evolving coefficient matrices that define candidate
submanifolds. See `CLAUDE.md` for architecture notes; this README is a
worked example of the end-to-end workflow.

## Pipeline at a glance

```
GA (d=1 broad search)
    └── plots_slag_<job>/<species_folder>/   ← run folder = one species
              coeffs.pkl, min_set.pkl,
              frobenius_norms.npy, phases.npy,
              two histograms, three scatters

GD at d=2 → d=3 → d=4 (local refinement)
    └── gd_runs/gd_<job>_step<N>.pkl         ← checkpoint (opt_state + history)
    └── gd_runs/plots_slag_<job>/            ← run folder (same shape as GA)

Cross-run analyses (post-hoc, against existing run folders)
    ├── plot_histograms   (any subset, overlaid distributions)
    ├── plot_coord_scatter  (already auto-emitted in the run folder)
    ├── plot_hermitian_coeffs  (Hermitian heatmap from coeffs.pkl)
    ├── plot_3D           (UMAP / PCA / Mapper / intrinsic-dim)
    └── split_clusters    (UMAP+KMeans → per-cluster sub-folders)
              └── persistent_homology_witness  (H0/H1/H2)
              └── fitness_pipeline --min_set   (fitness on one cluster)
```

The `coeffs.pkl` / `min_set.pkl` / `frobenius_norms.npy` / `phases.npy`
sidecars defined by `viz.fitness_pipeline.save_run_sidecars` are the
contract between the producer (GA, GD, manual `python -m viz.fitness_pipeline`)
and all the consumers (`plot_histograms`, `plot_coord_scatter`, `plot_3D`,
`plot_hermitian_coeffs`, `split_clusters`, `persistent_homology_witness`).
Any folder that has these sidecars is a valid input to any consumer.

## Worked example

### Step 0 — point cloud

Already in the repo as `1mil_patch_all_psi0_seed1024.pkl` (Fermat quintic, ψ=0).
For other ψ:

```bash
python points_gen/points_generation.py --psi 10 --seed 1024 --out_dir ./
```

### Step 1 — GA at d=1 (broad search)

```bash
python GA.py --job_id d1_search
```

Edit `PSI`, `SEED`, `MINSET_SIZE`, `NEWTON_STEPS`, etc. as module-level
constants at the top of `GA.py` (no CLI flags for these — by design).

Output:

```
plots_slag_d1_search/
  top_coeffs/
    coeffs_rank1_id7.pkl                ← bare (3, 25) coeffs, lowest rank = best
    coeffs_rank2_id12.pkl
    ...
  plots_slag_<gen>_<rank>_id<sid>/      ← one run folder per top species
    coeffs.pkl
    min_set.pkl
    frobenius_norms.npy
    phases.npy
    Kahler_form_loss_histogram.png      ← species vs random overlay
    circular_phase_histogram.png
    coord_scatter_{re,im,abs}_fitness.png
  ...
```

Pick the best species's `coeffs.pkl` as the seed for GD.

### Step 2 — GD ladder (d=2 → d=3 → d=4)

Each call writes a checkpoint into `gd_runs/` and a run folder
`gd_runs/plots_slag_<job>/`:

```bash
# d=2 seeded by the GA d=1 result
python gradient_descent.py --job_id d2 --max_degree 2 --steps 2000 \
    --init_pkl plots_slag_d1_search/plots_slag_6338568_1_id0/coeffs.pkl

# d=3 seeded by the d=2 result (right-zero-padded to width 1475)
python gradient_descent.py --job_id d3 --max_degree 3 --steps 2000 \
    --init_pkl gd_runs/gd_d2_step2000.pkl

# d=4 seeded by d=3 (right-zero-padded to width 6375)
python gradient_descent.py --job_id d4 --max_degree 4 --steps 2000 \
    --init_pkl gd_runs/gd_d3_step2000.pkl
```

`--init_pkl` accepts either a checkpoint dict (`gd_runs/gd_*.pkl` with
`opt_state` + `history`) or a bare coeffs pkl (`coeffs.pkl` from any run
folder). The two source types serve different purposes:

| File | Purpose | When to use |
|---|---|---|
| `gd_runs/gd_<job>_step<N>.pkl` | Full checkpoint (coeffs + opt_state + history + step) | `--resume` to continue Adam from where it left off |
| `<run_folder>/coeffs.pkl` | Just the coeffs | `--init_pkl` to seed a fresh run (Adam moments reset) |

Output after the ladder:

```
gd_runs/
  gd_d2_step2000.pkl, gd_d3_step2000.pkl, gd_d4_step2000.pkl
  plots_slag_d2/   ← one run folder per GD job
  plots_slag_d3/
  plots_slag_d4/
```

### Step 3 — convergence story (cross-run histogram overlay)

```bash
python -m viz.plot_histograms \
    --runs gd_runs/plots_slag_d2 gd_runs/plots_slag_d3 gd_runs/plots_slag_d4 \
    --labels d=2 d=3 d=4 \
    --out_dir gd_runs/compare_d2_d3_d4
```

Writes only the two overlay histograms (no mining, no scatters):

```
gd_runs/compare_d2_d3_d4/
  Kahler_form_loss_histogram.png
  circular_phase_histogram.png
```

Add `--vs random` to also append a random-coeffs curve; it auto-mines
into `gd_runs/_cache/random_w<width>_seed<seed>/` if absent, then reuses
the cache on subsequent calls.

### Step 4 — drill down into the d=4 result

These analyses live inside the d=4 run folder by default (every CLI
defaults `out_subdir` to a subfolder of the input file's parent):

```bash
# 4a. Hermitian heatmaps of the coeffs themselves
python -m viz.plot_hermitian_coeffs \
    --coeffs gd_runs/plots_slag_d4/coeffs.pkl \
    --out_subdir hermitian
# → gd_runs/plots_slag_d4/hermitian/

# 4b. 2D coord-scatter grids (auto-emitted by GD; rerun by hand for variants)
python -m viz.plot_coord_scatter \
    --min_set gd_runs/plots_slag_d4/min_set.pkl \
    --color fitness --part all \
    --out_subdir scatter_all
# → gd_runs/plots_slag_d4/scatter_all/

# 4c. 3D topology-aware viz (UMAP / PCA / Mapper / intrinsic-dim)
python -m viz.plot_3D \
    --min_set gd_runs/plots_slag_d4/min_set.pkl \
    --methods pca umap intrinsic_dim \
    --out_subdir topology
# → gd_runs/plots_slag_d4/topology/
```

### Step 5 — split into UMAP clusters and analyze each

If the d=4 min-set looks like two visually-separated pieces in UMAP:

```bash
python -m diagnostics.split_clusters \
    --min_set gd_runs/plots_slag_d4/min_set.pkl \
    --n_clusters 2 --basis umap
# → gd_runs/plots_slag_d4/cluster_split/cluster_{0,1}_points.pkl + split PNG
```

`split_clusters` propagates `coeffs.pkl` into the `cluster_split/` folder
(if the input min_set had one beside it), so both downstream tools auto-
discover the coeffs from the `--min_set`'s parent — no need to pass
`--coeffs` explicitly:

```bash
# 5a. Self-only fitness histograms for cluster 0
python -m viz.fitness_pipeline \
    --min_set gd_runs/plots_slag_d4/cluster_split/cluster_0_points.pkl \
    --out_subdir cluster_0_fitness
# → gd_runs/plots_slag_d4/cluster_split/cluster_0_fitness/
#     (coeffs.pkl, min_set.pkl, frobenius_norms.npy, phases.npy, two histograms, scatters)

# 5b. Persistent homology (H0/H1/H2) of cluster 0
python persistent_homology/persistent_homology_witness.py \
    --min_set gd_runs/plots_slag_d4/cluster_split/cluster_0_points.pkl \
    --psi 0 --out_subdir ph
# → gd_runs/plots_slag_d4/cluster_split/ph/
```

Same two commands for `cluster_1_points.pkl`. Pass `--coeffs <path>`
explicitly to override the auto-discovered one.

### Step 6 — compare clusters' fitness distributions

Now that each cluster has its own run folder under `cluster_split/`,
overlay them:

```bash
python -m viz.plot_histograms \
    --runs gd_runs/plots_slag_d4/cluster_split/cluster_0_fitness \
           gd_runs/plots_slag_d4/cluster_split/cluster_1_fitness \
    --labels cluster_0 cluster_1 \
    --out_dir gd_runs/plots_slag_d4/cluster_split/compare
```

## End-state directory tree

```
SLagSearch/
  1mil_patch_all_psi0_seed1024.pkl
  checkpoints/                          ← GA checkpoints (gitignored)
  plots_slag_d1_search/                 ← GA output
    top_coeffs/coeffs_rank<r>_id<sid>.pkl
    plots_slag_6338568_1_id0/           ← per-species run folder
      coeffs.pkl, min_set.pkl, frobenius_norms.npy, phases.npy
      Kahler_form_loss_histogram.png, circular_phase_histogram.png
      coord_scatter_{re,im,abs}_fitness.png
    ...
  gd_runs/                              ← GD output
    gd_d2_step2000.pkl, gd_d3_step2000.pkl, gd_d4_step2000.pkl
    plots_slag_d2/  plots_slag_d3/      ← d=2, d=3 run folders (same shape)
    plots_slag_d4/                      ← d=4 run folder
      coeffs.pkl, min_set.pkl, ...                          ← sidecars
      Kahler_form_loss_histogram.png, ...                   ← self-only histograms
      coord_scatter_{re,im,abs}_fitness.png                 ← scatters (auto)
      hermitian/                                            ← step 4a
      scatter_all/                                          ← step 4b
      topology/                                             ← step 4c
      cluster_split/                                        ← step 5
        cluster_0_points.pkl, cluster_1_points.pkl
        coeffs.pkl                                          ← propagated sidecar
        cluster_split.png, cluster_split.html
        cluster_0_fitness/                                  ← step 5a
          coeffs.pkl, min_set.pkl, frobenius_norms.npy, phases.npy
          Kahler_form_loss_histogram.png, ...
        cluster_1_fitness/                                  ← step 5a (mirror)
        ph/                                                 ← step 5b (cluster 0 PH)
        compare/                                            ← step 6
    compare_d2_d3_d4/                   ← step 3
    _cache/random_w6375_seed1230/       ← --vs random auto-cached
```

## Quick reference

| Tool | Input | Output | Role |
|---|---|---|---|
| `GA.py` | (module constants) | `plots_slag_<job>/<species>/` run folders | Search at d=1 |
| `gradient_descent.py` | `--init_pkl` or `--resume` | `gd_runs/gd_<job>_step<N>.pkl` + `gd_runs/plots_slag_<job>/` | Refine at d=2/3/4 |
| `viz.fitness_pipeline` | `--coeffs`, opt `--min_set` | run folder (full sidecars + plots) | Producer (also internal lib for GA/GD) |
| `viz.plot_histograms` | `--runs <dir>...` | two overlay PNGs | Cross-run histogram comparison |
| `viz.plot_coord_scatter` | `--min_set` | scatter PNGs | 2D coord-pair grids |
| `viz.plot_hermitian_coeffs` | `--coeffs` | heatmaps + spectra | Coeffs structure |
| `viz.plot_3D` | `--min_set` | 3D PNGs/HTMLs | UMAP/PCA/Mapper/intrinsic-dim |
| `viz.plot_gd_history` | `--filepath` (log or ckpt) | loss/fitness curves | GD training curves |
| `diagnostics.split_clusters` | `--min_set` | per-cluster `(N,5)` pkls + split PNG | UMAP+KMeans split |
| `diagnostics.diagnose_phases` | `--ansatz {d1,rp3}` | stdout per-patch Ω-phase histograms | Sign-convention sanity check |
| `diagnostics.permute_coeffs` | `--coeffs` | 10 permuted coeffs pkls | S₅ symmetry sweep |
| `diagnostics.test_permutation_symmetry` | `--coeffs` | stdout residuals per permutation | Algebraic symmetry test |
| `diagnostics.test_swap_invariance` | `--min_set`, `--coeffs` | residual histogram PNG | Geometric symmetry test |
| `persistent_homology_witness.py` | `--min_set`, `--coeffs` | PH diagrams/barcodes/Betti curves | H₀/H₁/H₂ via witness complex |

All CLIs follow the same convention: `--out_dir <full>` (full path) or
`--out_subdir <name>` (relative to the input file's parent dir); mutually
exclusive. Default = input file's parent dir (or a sensible subdir of it).
