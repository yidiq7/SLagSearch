# CLAUDE.md

Never open responses with filler phrases like "Great question!", "Of course!", "Certainly!", or similar warmups. Start every response with the actual answer. No preamble, no acknowledgment of the question.
Match response length to task complexity. Simple questions get direct, short answers. Complex tasks get full, detailed responses. Never pad responses with restatements of the question or closing sentences that repeat what you just said.
Before any significant task, show me 2-3 ways you could approach this work. Wait for me to choose before proceeding.
If you are uncertain about any fact, statistic, date, or piece of technical information: say so explicitly before including it. Never fill gaps in your knowledge with plausible-sounding information. When in doubt, say so.


This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Searches for special Lagrangian (sLag) submanifolds inside the Fermat quintic Calabi–Yau threefold by evolving coefficient matrices that define candidate submanifolds. An individual / coeff matrix encodes 3 polynomial equations in a basis of real bilinears built from `Im/Re` parts of `z_i z̄_j`, `z_i z_j z̄_m z̄_n`, etc. The coefficient width selects the ansatz max-degree:

| max_degree | shape | basis sizes (cumulative) |
| --- | --- | --- |
| 1 | `(3, 25)` | 25 |
| 2 | `(3, 250)` | 25 + 225 |
| 3 | `(3, 1475)` | 25 + 225 + 1225 |
| 4 | `(3, 6375)` | 25 + 225 + 1225 + 4900 |

`helper.evaluate_equations_single_point` dispatches on `coeffs.shape[1]` statically; the cutoffs live in `_D1_END`/`_D2_END`/`_D3_END`/`_D4_END`. Fitness measures (a) Lagrangian condition (Kähler form vanishes when restricted to the candidate submanifold) × (b) special condition (phases of the holomorphic 3-form concentrate). Both must be high to count as a sLag.

There are two optimizers in the repo: a speciation **GA** (`GA.py`) and a **gradient-descent** loop (`gradient_descent.py`). They share the fitness pipeline (`find_smooth_submanifold.py` + `slag_condition.py`).

## Recommended workflow

GA is the search tool (broad exploration of the d=1 landscape); GD is the refinement tool (local optimization with higher-order ansätze). The intended loop:

1. **Search at d=1** (GA). Coeffs constant `GENOTYPE_SHAPE = (3, 25)` at the top of `GA.py`:
   ```bash
   python GA.py --job_id d1_search
   ```
   At the end of the run, GA writes one pkl per surviving top species into `plots_slag_d1_search/top_coeffs/coeffs_rank<r>_id<sid>.pkl`. Lower-rank = higher fitness.

2. **Refine with gradient descent.** Pick one of the top-K coeffs and warm-start GD at the desired max-degree. The `--init_pkl` path right-zero-pads the d=1 coeffs to the current genotype width:
   ```bash
   # Skip-the-middle: warm-start d=4 directly from d=1 (empirically works).
   python gradient_descent.py --job_id d4_from_d1 --max_degree 4 --steps 2000 \
       --init_pkl plots_slag_d1_search/top_coeffs/coeffs_rank1_id7.pkl

   # Or step through: d=1+2 → d=1+2+3 → d=1+2+3+4, using each result as the
   # init for the next. Slower but lets you inspect intermediate truncation plots.
   python gradient_descent.py --job_id d2 --max_degree 2 --steps 2000 \
       --init_pkl plots_slag_d1_search/top_coeffs/coeffs_rank1_id7.pkl
   python gradient_descent.py --job_id d3 --max_degree 3 --steps 2000 \
       --init_pkl gd_runs/gd_d2_step2000.pkl
   python gradient_descent.py --job_id d4 --max_degree 4 --steps 2000 \
       --init_pkl gd_runs/gd_d3_step2000.pkl
   ```

3. **Inspect.** GD auto-emits histograms at the end (`plots_slag_<job_id>/`, plus per-degree truncation overlays for `max_degree >= 3`). 3D viz of `min_set.pkl`: `python plot_3D.py gd_runs/plots_slag_<job_id>` (PCA / UMAP / Mapper / intrinsic_dim).

### Extracting candidates from a mid-run checkpoint

The `top_coeffs/` dump happens only in GA's end-of-run "final analysis" block. To pull candidates from an in-progress checkpoint:

```python
from helper import load_ga_checkpoint, top_species
import numpy as np, pickle

ckpt = load_ga_checkpoint('checkpoints/checkpoint_gen_300.pkl')
for rank, s in enumerate(top_species(ckpt['species_list'], top_k=5), start=1):
    pickle.dump(np.asarray(s.representative),
                open(f'ga_top/coeffs_rank{rank}_id{s.id}.pkl', 'wb'))
```

Note: `load_ga_checkpoint` uses each species's stored `representative` (current best), while GA's end-of-run dump re-evaluates fitness and uses the actual best `member`. The two usually agree, but for high-fidelity candidate selection, prefer letting GA finish.

## Point clouds: location & naming

Path resolution has two layers:

1. **Per-script `POINTS_FILE` variable** — every module-level script (`GA.py`, `sLagSearch2.py`) sets `POINTS_FILE` once at the top, default `dwork_points_path(PSI, SEED)`. Edit that single line to point at any pickle. CLI scripts (`gradient_descent.py`, `profile_pipeline.py`, `fitness_plots.py`) accept `--points_file <path>` instead.
2. **Repo-wide `helper.POINTS_DIR`** — default `"."` (cwd). The directory that `dwork_points_path` joins with `dwork_filename`. Edit this if all your data lives in one shared spot and you want every Dwork-family script to find it without per-script edits.

The Dwork-family helpers (`dwork_points_path`, `dwork_filename`) are scoped to the one-parameter Dwork pencil `Σ z_i^5 + ψ · Π z_i = 0`. The filename convention is `1mil_patch_all_psi{psi_str}_seed{seed}.pkl`; integer-real psi keeps the legacy form (`psi0`, `psi10`), fractional real becomes `psi0.5`, complex becomes `psi1+2j` / `psi1-2j`.

For CICY / other CY families that don't have a psi parameter, just bypass the helper:

```python
POINTS_FILE = "data/my_cicy.pkl"
```

There is nothing magic about going through `dwork_points_path` — it's a convenience for the Dwork pencil. CLI scripts accept `--points_file` for the same purpose.

### Generating point clouds

```bash
# Install the optional dependency group (sympy + MLGeometry-JAX).
pip install -e ".[points-gen]"      # or:  uv sync --extra points-gen

# Fermat quintic, psi=0
python points_gen/points_generation.py --psi 0  --seed 1024 --out_dir ./

# Deformed quintic, psi=10  (writes 1mil_patch_all_psi10_seed1024.pkl)
python points_gen/points_generation.py --psi 10 --seed 1024 --out_dir ./

# Complex psi (writes 1mil_patch_all_psi1+2j_seed1024.pkl)
python points_gen/points_generation.py --psi 1+2j --seed 1024 --out_dir ./
```

`--n_pairs` (default 200000) is per-patch; the final cloud has ~5 × n_pairs points. The script writes via `MLGeometry.hypersurface.Hypersurface` (JAX backend); no TensorFlow needed. Default install (`pip install -e .` / `uv sync`) skips these deps — they're only required to regenerate inputs, not to run GA/GD/plots.

### Metric / psi compatibility

`metric='k4_fermat'` is precomputed for the Fermat quintic (psi=0) only. `assert_metric_psi_compatible(metric, psi)` is called at the entry of every script that uses both and raises `ValueError` for any nonzero psi with `k4_fermat`. Use `metric='FS'` (Fubini–Study) for non-zero psi.

## Running the GA

```bash
# Fresh run
python GA.py --job_id <label>

# Resume from latest checkpoint in ./checkpoints/
python GA.py --job_id <label> --load_checkpoint        # 'latest' is the default const

# Resume specific checkpoint
python GA.py --job_id <label> --load_checkpoint checkpoint_gen_300.pkl

# Initialize 25%/75% near a known d=1 baseline + wide d=2 perturbation
python GA.py --job_id <label> --preload_d1
```

GA hyperparameters live as module-level constants at the top of `GA.py` (population, generations, mutation/crossover rates, speciation thresholds, sigma adaptation, `MINSET_SIZE`, `NEWTON_STEPS`, `DIST_CHUNK_SIZE`). Edit them in-source — there are no CLI flags for them. The expensive ones to tune for memory/perf are `MINSET_SIZE`, `NEWTON_STEPS`, `FITNESS_MINI_BATCH_SIZE`, and `DIST_CHUNK_SIZE`.

`PSI` and `SEED` at the top of `GA.py` (and `psi`/`metric` at the top of `sLagSearch2.py`) select the point cloud — resolution goes through `helper.points_path(PSI, SEED)`, so edit `helper.POINTS_DIR` once to point all of them at your data location.

### Cluster (Slurm)

```bash
sbatch GA.script              # one 8h H200 job, resumes from latest checkpoint
./submit_chain.sh             # chain of 10 sbatch jobs with afterany dependency
```

`GA.script` activates the `sLagSearch` conda env at `/home/y.qi/miniconda3`. Adapt for your cluster.

## Running gradient descent

```bash
# d=1+2 default; plots auto-emit at the end.
python gradient_descent.py --job_id run1 --steps 2000

# d=1+2+3, preloading a d=1+2 result as init (right-zero-padded).
python gradient_descent.py --job_id run1_d3 --max_degree 3 --steps 2000 \
    --init_pkl gd_runs/gd_A_baseline_step10000.pkl

# d=1+2+3+4, preloading d=1+2+3 best as init.
python gradient_descent.py --job_id run1_d4 --max_degree 4 --steps 2000 \
    --init_pkl gd_runs/gd_run1_d3_step2000.pkl

# Resume from a full checkpoint (Adam moments + step counter restored).
python gradient_descent.py --job_id run1_cont \
    --resume gd_runs/gd_run1_step2000.pkl --steps 4000

# Just plot from an existing checkpoint, no training.
python gradient_descent.py --job_id run1_plots \
    --resume gd_runs/gd_run1_step2000.pkl --plots_only

# Smaller/faster plot mining.
python gradient_descent.py --job_id run1 --steps 2000 \
    --plot_k 20000 --plot_newton_steps 50
```

Init precedence: `--init_pkl <path>` (accepts either a bare `(3, w)` array or a checkpoint dict with a `"coeffs"` key; right-pads with zeros to the current genotype width) overrides `--init {scratch,d1_zeropad}`. The `d1_zeropad` default is the canonical GA d=1 baseline (`D1_COEFFS` constant in `gradient_descent.py`) zero-padded to the current width.

The training loop alternates **(re-)mining** every `--mine_interval` steps (`filter_and_refine`, not differentiated) with **Adam** steps that differentiate through a short inner Newton refinement (`refine_point_iterative`) on the frozen point cloud. Losses:

- `--loss lag` — Lagrangian only (bottom-99% mean of restricted Frobenius norms).
- `--loss spec` — special only (Kuramoto order parameter on `exp(2iθ)`, mod-π identification).
- `--loss both` (default) — weighted sum with `--lag_weight`/`--spec_weight`.

The Kuramoto path uses `compute_special_condition_fitness_smooth` in `slag_condition.py` — it is the differentiable replacement for the histogram-entropy `compute_special_condition_fitness` used by the GA. GA-comparable `(lag_fit, spec_fit)` is logged each step on the post-update coeffs for direct comparison.

Checkpoints land in `--out_dir` (default `./gd_runs/`) as `gd_<job_id>_step<N>.pkl` and contain `coeffs`, `opt_state`, `history`, `step`, and a copy of `args`. The repo's `gd_runs/gd_A_baseline_step*.pkl` files are the d=1+2 reference run.

### Output plots

`make_fitness_plots` is called automatically unless `--no-make_plots`. It writes several sibling folders under `--out_dir`:

- `plots_slag_<job_id>/` — GD coeffs vs random (fixed `[0, 3]` Kähler x-range).
- `plots_slag_<job_id>_d1/` — d=1 baseline vs random.
- `plots_slag_<job_id>_vs_d1/` — GD vs d=1 (auto x-range, steelblue/skyblue).
- For `max_degree=3`: `plots_slag_<job_id>_d2_vs_d3/` and `plots_slag_<job_id>_d1_d2_d3/` (truncations of the GD result, each row re-normalized).
- For `max_degree=4`: `plots_slag_<job_id>_d3_vs_d4/` and `plots_slag_<job_id>_d1_d2_d3_d4/`.

Each folder contains the two fitness histograms (`Kahler_form_loss_histogram.png`, `circular_phase_histogram.png`), the 5×5 coord-pair scatter grids colored by Lagrangian fitness (`coord_scatter_{re,im,abs}_fitness.png`, via `plot_coord_scatter.plot_pairs`), and the mined point cloud as `min_set.pkl` + sidecar `frobenius_norms.npy` (re-use the sidecar with `python plot_coord_scatter.py <folder> --color fitness` or by hand).

## Profiling

```bash
python profile_pipeline.py --n_iters 5                     # single GPU phase breakdown
python profile_pipeline.py --multi_gpu --batch_size 100    # mirrors GA.py mini-batch path
python profile_pipeline.py --precision highest             # FP32 vs TF32 comparison
python profile_pipeline.py --trace_dir ./jax_trace         # also writes Perfetto trace
```

Reports per-phase timing for `compute_distances_batched`, Newton refine, repulsion (by ablation), `compute_combined_fitness`, and end-to-end single + mini-batch fitness. Useful when tuning `DIST_CHUNK_SIZE` / `MINSET_SIZE` / `NEWTON_STEPS`.

## Auxiliary diagnostic scripts

- `plot_gd_history.py <path>` — plots loss + fitness curves from either a GD stdout log or a `gd_runs/*.pkl` checkpoint's `history` field. Writes `<prefix>_loss.png` and `<prefix>_fitness.png`.
- `fitness_plots.py [--coeffs_pkl <path>] [--points_file <path>] [--psi <c>] [--metric <m>] [--parent_folder <dir>]` — library (`make_fitness_plots`, imported by GA/GD) + CLI. CLI runs the full fitness-diagnostics pipeline on any `(3, w)` coeffs (loads from `--coeffs_pkl`; defaults to a d=1 baseline hardcoded in the file). Outputs the two histograms + sidecars + fitness-colored coord-scatter PNGs.
- `plot_3D.py <folder> [--methods coord pca umap mapper intrinsic_dim]` — 3D topology-aware viz of `min_set.pkl`: coord-aligned scatters, linear PCA (explained-variance readout + 2D-pairs), UMAP, Mapper (β₁ readout), and intrinsic-dim estimators (TwoNN / MLE / PH-dim). Colored by affine-patch index; `--metric fs|euclidean` controls feature representation for PCA/UMAP/Mapper.
- `plot_coord_scatter.py --filepath <pkl> [--out_dir <dir>] [--fitness_path <npy>] [--part re|im|abs|all] [--color patch|fitness]` — 5×5 grid of coord-pair scatters from any `(N, 5)` complex pickle (diagonal panels show Re z_i vs Im z_i; off-diagonals draw y=x for swap-symmetry checks). `--out_dir` defaults to `--filepath`'s parent. `--color patch` (default) is pure-numpy; `--color fitness` reads a `frobenius_norms.npy` sidecar (from `--fitness_path`, or `<filepath_dir>/frobenius_norms.npy`) and colors by `exp(-10·||K_R||_F/√||K_U||_F)`. GA/GD auto-emit the fitness-colored variant at run-end.
- `diagnose_phases.py --ansatz {d1,rp3}` — prints per-patch Ω-phase histograms via the production code path (`compute_holomorphic_form` + `compute_Omega_restriction`, including the `(-1)^(patch_idx+max_idx)` and basis-orientation signs). The `rp3` ansatz (`Im(z_0 z̄_1)=Im(z_0 z̄_2)=Im(z_0 z̄_3)=0`) is a known check: expects 5 peaks at odd multiples of π/5.

## Code architecture

The fitness pipeline is the load-bearing structure — everything else feeds it.

```
GA.py / gradient_descent.py
        │
        ├── filter_and_refine          (find_smooth_submanifold.py)
        │     ├── compute_distances_batched   (Newton-step norm)
        │     ├── refine_point_iterative      (Newton with auto-patch + damping)
        │     └── repulsion fori_loop          (uniform sampling)
        │
        └── compute_combined_fitness   (slag_condition.py)   [GA]
              ├── compute_kahler_form_unrestricted   (FS or k4_fermat)
              ├── compute_lagrangian_condition_fitness
              └── compute_holomorphic_form_restricted
                    → compute_special_condition_fitness        (histogram-entropy, non-diff)
                    → compute_special_condition_fitness_smooth (Kuramoto, diff)   [GD]
```

- `find_smooth_submanifold.py` — Newton-method refinement onto the candidate submanifold (zeros of the 3 user equations + the quintic). Handles the 5 affine patches automatically: `PATCH_ACTIVE_INDICES` defines which 8 of the 10 real coords are free per patch, and `determine_patch_and_rescale_single` rescales so `|z_max|=1` between iterations. The damped step uses `K_DAMP = 8` parallel alpha tiers (vmapped, `0.5^i`) instead of sequential backtracking — important when modifying the Newton inner loop. `filter_and_refine(..., filter_newton=True)` also returns a convergence flag.
- `slag_condition.py` — Geometry. `calculate_complex_metric_FS` is auto-diff'd from the FS Kähler potential; `calculate_complex_metric_k4` uses Donaldson's `k=4` algebraic metric for the Fermat quintic with hardcoded balanced coefficients (see the canonical-form table near `_QUINTIC_EXPONENTS`). Switching `METRIC` between `'FS'` and `'k4_fermat'` selects between them. Differentiable `compute_special_condition_fitness_smooth` is the GD loss; histogram-entropy `compute_special_condition_fitness` is the GA fitness.
- `get_restriction.py` — affine Jacobian, tangent-space restriction matrices, and `compute_Omega_restriction` used by both Newton and the Kähler-form pullback.
- `helper.py` — basis generation up to degree 4 (`generate_basis_*_order_single_point` for d=1/2/3/4 with sizes 25/225/1225/4900), real⇄complex (10D⇄5D) conversions, patch detection, distance matrices. `canonicalize_coeffs` is currently a no-op; the RREF code is kept as a docstring for future use.
- `sharding.py` — tiny helpers (`device_put_sharded`, `shard_leading_axis`, `unshard_leading_axis`, `take_replicated`) for the leading-device-axis sharding pattern used by GD multi-GPU and by `fitness_plots.make_fitness_plots`. The pattern: an outer "device" axis of size `D = jax.local_device_count()` is consumed by `pmap` with `axis_name="x"`.
- `fitness_plots.py` — fitness-diagnostics pipeline (mining → Kähler/phase diagnostics → histograms + sidecar + coord-scatter). Shared by both optimizers via `make_fitness_plots`, plus a standalone CLI (see auxiliary scripts above). `make_fitness_plots` accepts:
  - `compare_with`: `None` | `"random"` | `jnp.ndarray` for the comparison overlay.
  - `extra_comparisons`: list of `{"coeffs", "label", "color"}` dicts for additional overlays (used by the d=1+2+3 / d=1+2+3+4 multi-truncation plots).
  - `primary_color/label`, `compare_color/label`, `fix_kahler_x_range` to style the histograms.
  - `num_devices > 1` shards `filter_and_refine` across GPUs (host-side chunked diagnostics still bound peak VRAM via `chunk_size`).

### GA structure (GA.py)

Speciation-based GA with adaptive niche sigma. A `Species` is a Python object holding members + sigma + stagnation state — Python-level loops mutate `species_list` while JAX handles the heavy numerics. Key flow per generation:

1. Mini-batched fitness over the population (`FITNESS_MINI_BATCH_SIZE=100` per call), pmap-sharded across GPUs if `num_devices > 1` via `device_put_sharded` (a `NamedSharding` drop-in for the deprecated `jax.device_put_sharded`).
2. Speciate: merge close species (lower-fitness merged into higher), assign each individual to nearest representative, spawn a new species when min-distance ≥ `current_speciation_threshold`. The threshold is dynamically adjusted after `WARMUP_GENERATIONS` to keep species count in `[TARGET_SPECIES_COUNT_MIN, TARGET_SPECIES_COUNT_MAX]`.
3. Fitness sharing: divide by niche crowding count → adjusted fitness → proportional offspring allocation per species.
4. Reproduce: tournament selection + polynomial mutation (crossover defined but `CROSSOVER_RATE` path is not used — `generate_padded_offspring_batch` is mutation-only). Per-species adaptive sigma follows a 1/5th-success rule with cooldown.
5. Stagnation pruning: drop species idle for `STAGNATION_THRESHOLD` gens unless their best fitness is above `survival_threshold`.

The GA exits to a "final analysis" block that re-evaluates fitness on the final population, sorts species, and calls `make_fitness_plots` for each species above `0.5 * top_fitness_score`.

### GD structure (gradient_descent.py)

Adam optimizer over `coeffs` with the alternating (re-)mining schedule described above. Multi-GPU is fully supported when `jax.local_device_count() > 1`:

- `make_parallel_mining` shards `filter_and_refine` across devices (each mines `k/num_devices` points from its slice of `points_real`; intra-shard repulsion only — a small approximation when each shard has ≫ 100 points).
- `make_parallel_loss_and_grad` pmean's both loss/aux and gradients across the device axis. Per-shard reductions (bottom-99% mean of Lagrangian norms, Kuramoto `|mean exp(2iθ)|`) are computed locally then averaged — biased vs the true global loss, but negligible at `minset_size // num_devices ≳ 1000`.
- `make_parallel_ga_fitness` mirrors the same sharding for the GA-comparable `(lag_fit, spec_fit)` logged each step.

Both `--minset_size` and `--plot_k` must be divisible by `num_devices`; the loader truncates `points_real` to a multiple of `num_devices` before sharding.

### Conventions worth knowing

- Points are stored as `(N, 10)` real arrays: first 5 = `Re(z)`, last 5 = `Im(z)`. Convert with helpers in `helper.py`. The 5D complex form is used inside metric/Kähler computations.
- An individual's coefficient matrix has shape `(3, w)` with `w ∈ {25, 250, 1475, 6375}` (see the table at the top). Slicing `coeffs[:, :w']` to a smaller `w'` selects a lower-degree truncation; `normalize_coeffs` re-normalizes each row independently (used by both training and the d=1/d=2/d=3 truncation plots).
- `psi` is the complex deformation parameter of the quintic; `PSI=0` gives the Fermat quintic. Several `.pkl` point clouds for different psi values live under `data_psi/` on the cluster.
- JAX precision: `GA.py` and `profile_pipeline.py` set `'jax_default_matmul_precision'='high'` (TF32 on Ampere+); `gradient_descent.py` and `diagnose_phases.py` set `'jax_enable_x64'=True` (strict FP64 throughout). `sLagSearch2.py` uses `'highest'` (strict FP32). Match the precision when reproducing results.
- GA checkpoints (`checkpoints/checkpoint_gen_*.pkl`) save population, key, and a slim version of `species_list` (members pruned, representative + sigma + stagnation kept). `Species._id_counter` is restored on load to avoid ID collisions.
- GD checkpoints (`gd_runs/gd_<job_id>_step<N>.pkl`) save `coeffs`, `opt_state`, `history`, `step`, and the parsed `args` dict.
- `.pkl` files and `checkpoints/` are git-ignored; the 80MB `1mil_patch_all_psi0_seed1024.pkl` in the repo root is the local fallback point cloud (the GD `load_points` tries the cluster path first, then falls back to this local file).
- The `.nb`, `.ipynb` files are exploratory Mathematica/Jupyter notebooks (RP3, T3, restriction, PCA checks); they're not part of the runtime pipeline.
