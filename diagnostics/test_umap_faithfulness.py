"""Persistent homology of the 3D UMAP embedding of a min_set.

Runs plot_3D's UMAP setup (25D FS-projector features -> 3D Euclidean) on a
min_set, then computes H0/H1/H2 on the 3D output via gudhi's witness
complex over a max-min landmark sweep. Same plot/analysis style as
persistent_homology/persistent_homology_witness.py, but Euclidean metric
on the embedding instead of FS on CP^4.

This is NOT a faithfulness comparison against the original cloud; it just
shows what topology UMAP's 3D embedding itself exhibits, complementing the
existing plot_3D Mapper / intrinsic_dim readouts on the same input.

The gudhi witness builder and analyze helper are copied from the FS witness
script rather than imported to keep startup light (the FS script imports
JAX + the slag pipeline for its Newton-filter front-end, none of which is
needed here).

Usage:
    # Defaults: n_neighbors=200, min_dist=0.3, metric='fs',
    # landmarks=300,500,1000, max_alpha auto = 0.5 * UMAP-output diameter.
    python -m diagnostics.test_umap_faithfulness \
        --min_set plots_slag_d4_run/min_set.pkl

    # Cap UMAP at 30k points + custom landmark sweep.
    python -m diagnostics.test_umap_faithfulness \
        --min_set plots_slag_d4_run/min_set.pkl \
        --max_points 30000 --landmarks 500,1000,2000

    # Override the auto max_alpha (raw Euclidean units on UMAP output).
    python -m diagnostics.test_umap_faithfulness \
        --min_set plots_slag_d4_run/min_set.pkl --max_alpha 5.0
"""
import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viz.plot_3D import to_features, load_min_set_complex, _run_umap

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------- helpers

def betti_at(points, t):
    if len(points) == 0:
        return 0
    return int(np.sum((points[:, 0] <= t) & (points[:, 1] > t)))


def _clip_inf(p, infinity_val):
    if len(p) == 0:
        return p
    out = p.copy()
    out[np.isinf(out[:, 1]), 1] = infinity_val
    return out


# -------------------------------------------------- max-min landmarks (Euclidean)

def maxmin_landmarks_euclidean(X, L_max, seed):
    """Farthest-point sampling in Euclidean metric on (N, d) real array.

    Pure numpy (UMAP output is low-d so the BLAS cost per step is small).
    Mirrors the FS version in persistent_homology_witness.maxmin_landmarks.

    Returns:
        lm_idx:     (L_max,) int64 landmark indices into X.
        dist_table: (L_max, N) float64 Euclidean distances landmark->point.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    lm_idx = np.empty(L_max, dtype=np.int64)
    dist_rows = []
    min_to_lm = np.full(n, np.inf)

    lm_idx[0] = int(rng.integers(0, n))
    for ell in range(L_max):
        if ell > 0:
            lm_idx[ell] = int(np.argmax(min_to_lm))
        row = np.linalg.norm(X - X[int(lm_idx[ell])], axis=1)
        dist_rows.append(row)
        min_to_lm = np.minimum(min_to_lm, row)

    return lm_idx, np.stack(dist_rows, axis=0)


# ------------------------------------------------------------- witness complex

def build_witness_diagram(dist_table_slice, max_alpha_square,
                          limit_dimension=3, top_k_landmarks=50,
                          witness_type="weak"):
    """gudhi WitnessComplex on a precomputed landmark-to-witness distance table.

    Args:
        dist_table_slice: (L, N) raw distances, landmark -> witness.
        max_alpha_square: filtration cap in squared-distance units.
        limit_dimension: max simplex dim. limit_dimension=3 yields tetrahedra
            (needed for H2).
        top_k_landmarks: keep top-K nearest landmarks per witness in the
            gudhi table. limit_dimension=3 only strictly needs K >= 5;
            50 is the same generous default as the FS witness script.
        witness_type: 'weak' (gudhi.WitnessComplex) or 'strong'
            (gudhi.StrongWitnessComplex). See the FS witness script for the
            tradeoff (strong prunes the late-alpha noise band).

    Returns:
        (H0, H1, H2): each (n_features, 2) array of (birth, death) in raw
        distance units (filtration values sqrt'd from gudhi's alpha^2).
    """
    import gudhi

    L, N = dist_table_slice.shape
    K = min(top_k_landmarks, L)
    dt = dist_table_slice.T  # (N, L)

    nearest = []
    for j in range(N):
        if K < L:
            cand = np.argpartition(dt[j], K - 1)[:K]
            order = cand[np.argsort(dt[j, cand])]
        else:
            order = np.argsort(dt[j])
        d_sq = dt[j, order] ** 2
        nearest.append([(int(order[k]), float(d_sq[k])) for k in range(K)])

    if witness_type == "strong":
        wc = gudhi.StrongWitnessComplex(nearest_landmark_table=nearest)
    elif witness_type == "weak":
        wc = gudhi.WitnessComplex(nearest_landmark_table=nearest)
    else:
        raise ValueError(
            f"witness_type must be 'weak' or 'strong', got {witness_type!r}"
        )
    st = wc.create_simplex_tree(
        max_alpha_square=max_alpha_square,
        limit_dimension=limit_dimension,
    )
    persistence = st.persistence()

    H0, H1, H2 = [], [], []
    for dim, (b, d) in persistence:
        b_raw = np.sqrt(max(b, 0.0)) if np.isfinite(b) else 0.0
        d_raw = np.inf if np.isinf(d) else np.sqrt(max(d, 0.0))
        if dim == 0:
            H0.append([b_raw, d_raw])
        elif dim == 1:
            H1.append([b_raw, d_raw])
        elif dim == 2:
            H2.append([b_raw, d_raw])
    return (np.array(H0, dtype=np.float64) if H0 else np.empty((0, 2)),
            np.array(H1, dtype=np.float64) if H1 else np.empty((0, 2)),
            np.array(H2, dtype=np.float64) if H2 else np.empty((0, 2)))


# ----------------------------------------------------------------- analysis

def analyze(H0, H1, H2, infinity_val, label=""):
    print(f"\n=== TOPOLOGICAL FEATURES {label} ===")
    print(f"H0 (connected components): {len(H0)}")
    print(f"H1 (loops):                {len(H1)}")
    print(f"H2 (voids):                {len(H2)}")

    filt = (np.linspace(0.2, 0.9, 4) * infinity_val).tolist()
    print("Betti at sampled filtration values:")
    for t in filt:
        print(f"  r={t:.3f}: beta0={betti_at(H0, t)}, "
              f"beta1={betti_at(H1, t)}, beta2={betti_at(H2, t)}")

    def top_features(points, name):
        if len(points) == 0:
            return
        death_bounded = np.where(np.isinf(points[:, 1]), infinity_val,
                                 points[:, 1])
        pers_for_sort = death_bounded - points[:, 0]
        order = np.argsort(pers_for_sort)[-5:][::-1]
        print(f"{name} top 5 by persistence:")
        for i in order:
            pers_actual = points[i, 1] - points[i, 0]
            print(f"  birth={points[i, 0]:.4f}  death={points[i, 1]:.4f}  "
                  f"pers={pers_actual:.4f}")

    top_features(H0, "H0")
    top_features(H1, "H1")
    top_features(H2, "H2")


# ----------------------------------------------------------------- plotting

_DISTANCE_LABEL = "Euclidean distance (UMAP 3D)"


def plot_one_L(H0, H1, H2, infinity_val, n_sample, L, output_file,
               umap_nn, umap_md):
    """Per-L PH figure: 3 cols (H0, H1, H2) x 3 rows (diagram, barcode, Betti).

    Adapted from persistent_homology_witness.plot_one_L: Euclidean axis
    labels + UMAP context in the suptitle.
    """
    print(f"  plotting L={L} -> {output_file}")
    dims = [("H0", H0, "skyblue"),
            ("H1", H1, "orange"),
            ("H2", H2, "forestgreen")]

    fig = plt.figure(figsize=(15, 12))

    def plot_diagram(ax, pts, color, name):
        ptsd = _clip_inf(pts, infinity_val)
        if len(ptsd):
            ax.scatter(ptsd[:, 0], ptsd[:, 1], alpha=0.7, s=30, c=color)
            ax.plot([0, infinity_val], [0, infinity_val], "k--", alpha=0.3)
        ax.set_xlabel(f"Birth ({_DISTANCE_LABEL})")
        ax.set_ylabel(f"Death ({_DISTANCE_LABEL})")
        ax.set_title(f"{name} Persistence Diagram ({len(pts)} features)")
        ax.set_xlim([0, infinity_val])
        ax.set_ylim([0, infinity_val * 1.05])
        ax.grid(True, linestyle="--", alpha=0.6)

    def plot_barcode(ax, pts, color, title):
        if len(pts) == 0:
            ax.set_title(title)
            return
        death_for_sort = np.where(np.isinf(pts[:, 1]), infinity_val,
                                  pts[:, 1])
        order = np.argsort(death_for_sort - pts[:, 0])[::-1]
        sorted_pts = pts[order]
        n_bars = min(50, len(sorted_pts))
        for i in range(n_bars):
            b, d = sorted_pts[i]
            if np.isinf(d):
                ax.barh(i, infinity_val - b, left=b, color=color, alpha=0.8,
                        edgecolor="red", linewidth=2)
            else:
                ax.barh(i, d - b, left=b, color=color, alpha=0.7)
        ax.set_xlim([0, infinity_val])
        ax.set_ylim([-1, n_bars])
        ax.set_xlabel("Filtration Value")
        ax.set_ylabel("Feature Index")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.6, axis="x")

    def plot_betti(ax, dim, pts, color):
        t_grid = np.linspace(0, infinity_val, 100)
        if len(pts):
            bv = [betti_at(pts, t) for t in t_grid]
            ax.plot(t_grid, bv, color=color, linewidth=2)
            ax.fill_between(t_grid, bv, alpha=0.3, color=color)
        ax.set_xlim([0, infinity_val])
        ax.set_xlabel("Filtration Value")
        ax.set_ylabel(f"beta_{dim}")
        ax.set_title(f"H{dim} Betti Curve")
        ax.grid(True, linestyle="--", alpha=0.6)

    for col, (name, pts, color) in enumerate(dims):
        ax_diag = plt.subplot(3, 3, col + 1)
        plot_diagram(ax_diag, pts, color, name)
        ax_bar = plt.subplot(3, 3, 3 + col + 1)
        plot_barcode(ax_bar, pts, color,
                     f"{name} Barcode (top {min(50, len(pts))})")
        ax_betti = plt.subplot(3, 3, 6 + col + 1)
        plot_betti(ax_betti, col, pts, color)

    fig.suptitle(
        f"Witness Complex Persistence on UMAP 3D embedding  |  "
        f"N={n_sample}, L={L}, nn={umap_nn}, md={umap_md}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(per_L, infinity_val, n_sample, output_file,
                    umap_nn, umap_md):
    """Across-L: rows = H0/H1/H2 diagrams per L, bottom row = overlaid Betti.

    Adapted from persistent_homology_witness.plot_comparison.
    """
    Ls = sorted(per_L.keys())
    n_L = len(Ls)
    n_rows = 4
    fig = plt.figure(figsize=(4 * n_L, 4 * n_rows))
    cmap_L = plt.cm.viridis(np.linspace(0.1, 0.9, n_L))

    dim_specs = [(0, "H0", "steelblue"),
                 (1, "H1", "darkorange"),
                 (2, "H2", "forestgreen")]

    for row, (dim, name, color) in enumerate(dim_specs):
        for ci, L in enumerate(Ls):
            pts = per_L[L][dim]
            ax = plt.subplot(n_rows, n_L, row * n_L + ci + 1)
            ptsd = _clip_inf(pts, infinity_val)
            if len(ptsd):
                ax.scatter(ptsd[:, 0], ptsd[:, 1], s=15, alpha=0.7, c=color)
                ax.plot([0, infinity_val], [0, infinity_val], "k--", alpha=0.3)
            ax.set_title(f"{name}  (L={L}, n_feat={len(pts)})")
            ax.set_xlim([0, infinity_val])
            ax.set_ylim([0, infinity_val * 1.05])
            ax.set_xlabel("Birth")
            ax.set_ylabel("Death")
            ax.grid(True, linestyle="--", alpha=0.6)

    t_grid = np.linspace(0, infinity_val, 200)
    for di, (dim, name, _) in enumerate(dim_specs):
        ax = plt.subplot(n_rows, 3, 3 * 3 + di + 1)
        for L, lc in zip(Ls, cmap_L):
            pts = per_L[L][dim]
            if len(pts):
                ax.plot(t_grid, [betti_at(pts, t) for t in t_grid],
                        color=lc, label=f"L={L}", linewidth=2)
        ax.set_xlim([0, infinity_val])
        ax.set_xlabel("Filtration Value")
        ax.set_ylabel(f"beta_{dim}")
        ax.set_title(f"{name} Betti Curves across L")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(
        f"Witness Landmark Sweep on UMAP 3D embedding  |  "
        f"N={n_sample}, nn={umap_nn}, md={umap_md}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved comparison: {output_file}")


# ----------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min_set", type=Path, required=True,
                   help="Path to (N, 5) complex pickle (e.g. min_set.pkl).")
    p.add_argument("--max_points", type=int, default=None,
                   help="Subsample to this many points before UMAP. "
                        "Default: use all points (matches plot_3D).")
    p.add_argument("--umap_n_neighbors", type=int, default=200)
    p.add_argument("--umap_min_dist", type=float, default=0.3)
    p.add_argument("--metric", choices=["fs", "euclidean"], default="fs",
                   help="UMAP feature representation (input to UMAP, not "
                        "the metric on UMAP output -- that's always "
                        "Euclidean). 'fs' (default) = 25D rank-1 projector, "
                        "matches plot_3D. 'euclidean' = raw 10D real "
                        "(patch-dependent).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--landmarks", default="300,500,1000",
                   help="Comma-separated L values to sweep.")
    p.add_argument("--top_k_landmarks", type=int, default=50,
                   help="Keep top-K nearest landmarks per witness in the "
                        "gudhi table. K >= 5 suffices for limit_dimension=3; "
                        "50 is the generous default the FS witness uses.")
    p.add_argument("--max_alpha", type=float, default=None,
                   help="Filtration cap in raw Euclidean distance on the "
                        "UMAP output. Default (auto): 4 * covering_radius "
                        "at the smallest L in the sweep, where "
                        "covering_radius = max_p min_l dist(point_p, "
                        "landmark_l). The 4x is a 2x safety buffer over "
                        "the de Silva-Carlsson bound (meaningful PH up "
                        "to ~2 * covering_radius).")
    p.add_argument("--witness_type", choices=["weak", "strong"],
                   default="weak",
                   help="gudhi witness variant (see FS witness script).")
    out_group = p.add_mutually_exclusive_group()
    out_group.add_argument("--out_dir", type=Path, default=None,
                           help="Full output directory. "
                                "Default: parent dir of --min_set.")
    out_group.add_argument("--out_subdir", type=str, default=None,
                           help="Output subdirectory name under --min_set's "
                                "parent dir.")
    p.add_argument("--out_prefix", default="test_umap_faithfulness")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    L_values = sorted({int(x) for x in args.landmarks.split(",") if x.strip()})
    L_min = L_values[0]
    L_max = L_values[-1]
    print(f"Sweep: L values = {L_values}, L_min = {L_min}, L_max = {L_max}")

    # Out dir resolution (matches other diagnostics).
    min_set_parent = args.min_set.parent
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.out_subdir is not None:
        out_dir = min_set_parent / args.out_subdir
    else:
        out_dir = min_set_parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ---- load
    print("\n=== LOADING POINTS ===")
    z = load_min_set_complex(args.min_set)
    print(f"Loaded {z.shape[0]} points from {args.min_set}")

    # ---- subsample (optional)
    if args.max_points is not None and z.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(z.shape[0], args.max_points, replace=False)
        z = z[idx]
        print(f"Subsampled to {z.shape[0]} (seed={args.seed})")

    # ---- features
    print(f"\n=== BUILDING {args.metric.upper()} FEATURES ===")
    X = to_features(z, args.metric)
    print(f"Feature dim: {X.shape[1]} ({args.metric} metric)")

    # ---- UMAP (same call plot_3D makes)
    print(f"\n=== UMAP (n_neighbors={args.umap_n_neighbors}, "
          f"min_dist={args.umap_min_dist}, seed={args.seed}) ===")
    t0 = time.time()
    emb = _run_umap(X, args.umap_n_neighbors, args.umap_min_dist, args.seed)
    print(f"UMAP: {time.time() - t0:.1f}s; output shape: {emb.shape}")
    print(f"UMAP output bounds per axis: "
          f"x in [{emb[:, 0].min():.3f}, {emb[:, 0].max():.3f}], "
          f"y in [{emb[:, 1].min():.3f}, {emb[:, 1].max():.3f}], "
          f"z in [{emb[:, 2].min():.3f}, {emb[:, 2].max():.3f}]")

    # ---- landmarks (Euclidean on UMAP output)
    print(f"\n=== MAX-MIN LANDMARK SELECTION (L_max={L_max}, Euclidean) ===")
    t0 = time.time()
    lm_idx, dist_table = maxmin_landmarks_euclidean(emb, L_max, args.seed)
    print(f"Landmarks: {time.time() - t0:.1f}s")
    data_diameter = float(np.max(dist_table))

    # ---- max_alpha auto-compute
    # Covering radius at the loosest L in the sweep: max over points of
    # distance to nearest of the first L_min landmarks. de Silva-Carlsson
    # bound: meaningful PH up to ~2 * covering_radius; 4x is a 2x buffer.
    covering_radius = float(np.max(np.min(dist_table[:L_min], axis=0)))
    if args.max_alpha is None:
        max_alpha = 4.0 * covering_radius
        print(f"max_alpha (auto): 4 * covering_radius(L={L_min}) "
              f"= {max_alpha:.4f}")
    else:
        max_alpha = float(args.max_alpha)
        print(f"max_alpha (user override): {max_alpha:.4f}")
    print(f"Data diameter (Euclidean on UMAP 3D): {data_diameter:.4f}")
    print(f"Covering radius at L={L_min}: {covering_radius:.4f}")
    max_alpha_sq = max_alpha ** 2
    infinity_val = max_alpha

    # ---- witness diagrams per L
    per_L = {}
    for L in L_values:
        print(f"\n=== WITNESS COMPLEX (L={L}, {args.witness_type}) ===")
        t0 = time.time()
        H0, H1, H2 = build_witness_diagram(
            dist_table[:L], max_alpha_sq,
            limit_dimension=3,
            top_k_landmarks=args.top_k_landmarks,
            witness_type=args.witness_type,
        )
        print(f"  built in {time.time() - t0:.1f}s; "
              f"H0={len(H0)}, H1={len(H1)}, H2={len(H2)}")

        # Match the FS witness script: ensure the essential H0 bar is present.
        if len(H0) and not np.any(np.isinf(H0[:, 1])):
            H0 = np.vstack([H0, [0.0, np.inf]])
            print("  injected essential H0 (birth=0, death=inf)")
        per_L[L] = (H0, H1, H2)

    # ---- analyze + per-L plot
    n_sample = z.shape[0]
    for L in L_values:
        H0, H1, H2 = per_L[L]
        analyze(H0, H1, H2, infinity_val, label=f"(L={L})")
        png = str(out_dir / f"{args.out_prefix}_L{L}.png")
        plot_one_L(H0, H1, H2, infinity_val, n_sample, L, png,
                   args.umap_n_neighbors, args.umap_min_dist)

    # ---- comparison plot
    sweep_png = str(out_dir / f"{args.out_prefix}_sweep.png")
    plot_comparison(per_L, infinity_val, n_sample, sweep_png,
                    args.umap_n_neighbors, args.umap_min_dist)

    print(f"\n=== DONE in {time.time() - t_start:.1f}s ===")


if __name__ == "__main__":
    main()
