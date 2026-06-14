"""Cup-product rank of an sLag point cloud via ripser -- the T^3 vs #(S^1xS^2) test.

Persistent homology recovers Betti numbers but NOT the cohomology ring, and
T^3 and #^3(S^1 x S^2) share b = (1,3,3,1). They are separated by the cup map

        mu : Lambda^2 H^1 --> H^2,   (alpha, beta) |-> alpha cup beta

whose rank over F_2 is the discriminant (per connected component):

        rank mu == C(b_1, 2)  (mu injective on Lambda^2 H^1)  -> torus T^3
        rank mu == 0          (all degree-1 products vanish) -> #^k(S^1 x S^2)

For a closed orientable 3-manifold with b_1 >= 1, Poincare duality already
forces cup-length >= 2 (some H^1 cup H^2 is nonzero); the informative jump is
the 2 -> 3 transition, i.e. whether three 1-classes multiply to the volume
form. rank mu probes exactly the degree-1 x degree-1 part of that.

Pipeline (jax-free; consumes an already-refined min_set, so no Newton re-filter):
  1. Load min_set (N, 5) complex; row-normalize (CP^4 / Fubini-Study).
  2. Farthest-point (max-min) landmark selection in the FS metric (numpy).
     Prefix-monotone, so one L_max pass serves the whole --landmarks sweep.
  3. ripser on the (L, L) FS distance matrix: maxdim=2, do_cocycles=True,
     coeff=2 -> H^1/H^2 diagrams + H^1 cocycle representatives.
  4. Pick a scale eps on the H^1 plateau (auto from the persistent bars, or
     --epsilon); report b_1(eps), b_2(eps) for sanity.
  5. Build the 2-skeleton at eps, convert the persistent H^1 cocycles to F_2
     edge sets, and rank mu via cup_product.cup_map_rank.
  6. Report cup rank PER CONNECTED COMPONENT per L (cup products across
     components vanish, so mu is block-diagonal); agreement across the L-sweep
     is the stability check -- cup rank is a topological invariant, not a fit.

Components: the cloud may be disconnected (b_0 > 1, e.g. several sLag fibers).
Components at the chosen scale are found by union-find and each is tested on its
own, so a disjoint union of c tori reports c x (b1=3, rank=3) rather than a
single misleading total. (diagnostics/split_clusters.py can still split upstream.)

Usage:
    python persistent_homology/persistent_homology_cup_length.py \
        --min_set min_set.pkl --landmarks 1000,2000,3000

    # If you know the expected b_1 (e.g. 6 for two T^3 components), pin it:
    python persistent_homology/persistent_homology_cup_length.py \
        --min_set min_set.pkl --landmarks 1000,2000,3000 --n_h1 6

    # Validate the ripser -> cup-rank wiring on synthetic manifolds (cluster):
    python persistent_homology/persistent_homology_cup_length.py --selftest
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

# cup_product.py lives alongside this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cup_product import cup_map_rank


# ------------------------------------------------------------ FS metric layer

def _normalize_rows_complex(Z):
    """Row-normalize complex (N, m) to unit norm."""
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def _fs_row(Zn, k):
    """FS distance arccos|<z_k, z_j>| from row k to every row (Zn unit-norm)."""
    inner = np.abs(Zn @ np.conj(Zn[k]))
    return np.arccos(np.clip(inner, 0.0, 1.0))


def fs_distance_matrix(Zn):
    """Dense (L, L) Fubini-Study distance matrix for unit-norm rows."""
    gram = np.abs(Zn @ np.conj(Zn).T)
    return np.arccos(np.clip(gram, 0.0, 1.0))


def fps_landmarks(Zn, L_max, seed):
    """Farthest-point (max-min) sampling in the FS metric. numpy, O(L_max * N).

    Keeps only a running (N,) nearest-landmark distance, so memory is O(N), not
    O(L_max * N) like the witness script's full table. Prefix-monotone: the
    first L of the returned indices are themselves a valid L-landmark set.
    """
    n = Zn.shape[0]
    rng = np.random.default_rng(seed)
    lm = np.empty(L_max, dtype=np.int64)
    lm[0] = int(rng.integers(0, n))
    min_to_lm = _fs_row(Zn, int(lm[0]))
    for ell in range(1, L_max):
        lm[ell] = int(np.argmax(min_to_lm))
        min_to_lm = np.minimum(min_to_lm, _fs_row(Zn, int(lm[ell])))
    return lm


# --------------------------------------------------- ripser -> cup-rank bridge

def _betti_at(dgm, t):
    """Number of bars in a (n,2) birth/death diagram alive at filtration t."""
    if len(dgm) == 0:
        return 0
    return int(np.sum((dgm[:, 0] <= t) & (dgm[:, 1] > t)))


def _auto_n_h1(lifetimes):
    """Number of 'long' H^1 bars via the largest multiplicative gap.

    Sort lifetimes descending; the biggest ratio drop between consecutive
    values separates signal (persistent generators) from noise.
    """
    s = np.sort(lifetimes)[::-1]
    if len(s) <= 1:
        return len(s)
    ratios = s[:-1] / np.maximum(s[1:], 1e-12)
    return int(np.argmax(ratios)) + 1


def _choose_epsilon(H1, chosen):
    """A scale on the common H^1 plateau of the chosen bars.

    The bars are simultaneously alive on [max births, min deaths]; pick its
    midpoint. If they have no common overlap, fall back to the midpoint of the
    single most persistent bar (and the caller warns via b_1(eps) reporting).
    """
    b = float(H1[chosen, 0].max())
    d = float(H1[chosen, 1].min())
    if b < d:
        return 0.5 * (b + d)
    top = chosen[int(np.argmax(H1[chosen, 1] - H1[chosen, 0]))]
    return 0.5 * (H1[top, 0] + H1[top, 1])


def _two_skeleton(D, eps):
    """Vertices/edges/triangles of the Vietoris-Rips complex at scale eps.

    Returns (n_vertices, edges, triangles); triangles are sorted (i<j<k) tuples
    -- the format cup_product.cup_map_rank expects.
    """
    L = D.shape[0]
    adj = D <= eps
    np.fill_diagonal(adj, False)
    neighbors = [set(np.nonzero(adj[i])[0]) for i in range(L)]
    edges = [(i, j) for i in range(L) for j in neighbors[i] if j > i]
    triangles = []
    for (i, j) in edges:
        for k in neighbors[i] & neighbors[j]:
            if k > j:
                triangles.append((i, j, k))
    return L, edges, triangles


def _connected_components(n_vert, edges):
    """Union-find: component id per vertex over the given edges.

    The cup map is block-diagonal across components (products of classes from
    different components vanish -- disjoint simplicial supports), so cup rank is
    computed per component and summed. Without this, a disconnected cloud (e.g.
    several sLag fibers) reports a misleading total against C(b_1, 2).
    """
    parent = list(range(n_vert))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return [find(x) for x in range(n_vert)]


def _cocycle_to_edges(coc, D, eps):
    """ripser dim-1 cocycle (rows [i, j, coeff]) -> set of F_2 edges at eps.

    Keep edges with odd coefficient (mod 2) that are present at scale eps.
    Vertices are ripser's original point indices.
    """
    out = set()
    for row in np.asarray(coc):
        i, j, v = int(row[0]), int(row[1]), int(row[2])
        if v % 2 == 0:
            continue
        a, b = (i, j) if i < j else (j, i)
        if D[a, b] <= eps:
            out.add((a, b))
    return out


def cup_rank_from_distances(D, n_h1=None, epsilon=None, thresh=None,
                            verbose=True):
    """Run ripser on a distance matrix and return the rank of mu : Lambda^2 H^1 -> H^2.

    Metric-agnostic: ``D`` is any (L, L) distance matrix (FS for the sLag data,
    Euclidean for the synthetic self-test). Returns a dict with the rank and the
    diagnostics needed to interpret / reproduce it.
    """
    from ripser import ripser

    res = ripser(D, distance_matrix=True, maxdim=2, do_cocycles=True, coeff=2,
                 thresh=(np.inf if thresh is None else float(thresh)))
    dgms = res["dgms"]
    H1 = dgms[1]
    H2 = dgms[2] if len(dgms) > 2 else np.empty((0, 2))
    coc1 = res["cocycles"][1]

    lifetimes = (H1[:, 1] - H1[:, 0]) if len(H1) else np.array([])
    if n_h1 is None:
        n_h1 = _auto_n_h1(lifetimes) if len(lifetimes) else 0
    n_h1 = min(n_h1, len(H1))

    if n_h1 < 2:
        # Fewer than two H^1 generators: Lambda^2 H^1 = 0, rank mu = 0 trivially.
        if verbose:
            print(f"    b_1 used = {n_h1} (< 2): no degree-1 pairs to multiply.")
        return {"rank": 0, "b1": n_h1, "epsilon": epsilon,
                "b1_at_eps": None, "b2_at_eps": None,
                "n_vertices": 0, "n_edges": 0, "n_triangles": 0,
                "n_components": 0, "components": [], "dgms": dgms,
                "lifetimes_sorted": np.sort(lifetimes)[::-1]}

    order = np.argsort(lifetimes)[::-1]
    chosen = order[:n_h1]
    eps = float(epsilon) if epsilon is not None else _choose_epsilon(H1, chosen)

    n_vert, edges, triangles = _two_skeleton(D, eps)
    comp_of = _connected_components(n_vert, edges)
    # b_0 self-consistency: the components we split over (at scale eps) should
    # match ripser's H_0 at eps. A mismatch means eps merged or fragmented
    # pieces -- the per-component split (hence rank) would be untrustworthy.
    b0_uf = len(set(comp_of))
    b0_ripser = _betti_at(dgms[0], eps)
    cocycles = [_cocycle_to_edges(coc1[c], D, eps) for c in chosen]
    cocycles = [cc for cc in cocycles if cc]  # drop any empty at this eps

    # Bucket triangles + cocycles by component; cup rank is per-component summed.
    tri_by_comp = {}
    for tri in triangles:
        tri_by_comp.setdefault(comp_of[tri[0]], []).append(tri)
    coc_by_comp = {}
    for cc in cocycles:
        coc_by_comp.setdefault(comp_of[next(iter(cc))[0]], []).append(cc)

    components, total_rank = [], 0
    for cid, ccs in sorted(coc_by_comp.items()):
        r = cup_map_rank(tri_by_comp.get(cid, []), ccs)
        total_rank += r
        components.append({"b1": len(ccs), "rank": r})

    info = {"rank": total_rank, "b1": len(cocycles), "epsilon": eps,
            "b1_at_eps": _betti_at(H1, eps), "b2_at_eps": _betti_at(H2, eps),
            "b0_at_eps": b0_ripser, "n_components_total": b0_uf,
            "n_vertices": n_vert, "n_edges": len(edges),
            "n_triangles": len(triangles), "n_components": len(components),
            "components": components, "dgms": dgms,
            "lifetimes_sorted": np.sort(lifetimes)[::-1]}
    if verbose:
        print(f"    H^1 lifetimes (top): "
              f"{np.round(info['lifetimes_sorted'][:max(n_h1 + 2, 4)], 4)}")
        print(f"    b_1 used = {len(cocycles)}   eps = {eps:.4f}   "
              f"b_1(eps) = {info['b1_at_eps']}   b_2(eps) = {info['b2_at_eps']}")
        print(f"    2-skeleton @ eps: V={n_vert}  E={len(edges)}  "
              f"T={len(triangles)}")
        print(f"    components @ eps: {b0_uf} total "
              f"(ripser b_0(eps) = {b0_ripser}), {len(components)} with H^1")
        if b0_uf != b0_ripser:
            print("    WARNING: component count != ripser b_0(eps) -- eps may be "
                  "off (pieces merging or fragmenting); set --epsilon explicitly.")
        for i, c in enumerate(components):
            tag = ("T^3-like" if (c["b1"] == 3 and c["rank"] == 3)
                   else "conn-sum-like" if c["rank"] == 0
                   else f"other(b1={c['b1']},rank={c['rank']})")
            print(f"      component {i}: b_1 = {c['b1']}, rank mu = {c['rank']}"
                  f"  ->  {tag}")
        if info["b2_at_eps"] == 0:
            print("    WARNING: b_2(eps) = 0 -- no H^2 at this scale, so every "
                  "cup product is forced to 0. Pick eps inside the (b_1, b_2) "
                  "plateau via --epsilon.")
    return info


def _verdict(info):
    """One-line per-component interpretation of the cup map mu : Lambda^2 H^1 -> H^2.

    For a closed orientable 3-manifold component the discriminant is sharp:
        b_1 = 3 and rank mu = 3 (= C(3,2))  -> T^3 (three 1-classes multiply up)
        rank mu = 0                          -> #^k(S^1 x S^2)
    """
    from collections import Counter
    comps = info["components"]
    if not comps:
        return "no persistent H^1 -> not torus-like (cannot form 1-class products)."
    tags = []
    for c in comps:
        if c["b1"] == 3 and c["rank"] == 3:
            tags.append("T^3")
        elif c["rank"] == 0:
            tags.append("#(S^1xS^2)")
        else:
            tags.append(f"other(b1={c['b1']},rank={c['rank']})")
    cnt = Counter(tags)
    return (f"{len(comps)} component(s) with H^1: "
            + ", ".join(f"{v}x {k}" for k, v in cnt.items())
            + f"   (total rank mu = {info['rank']})")


# --------------------------------------------------------------------- data IO

def load_points(min_set, subsamp, seed):
    print("=== LOADING POINTS ===")
    with open(min_set, "rb") as f:
        Z = np.asarray(pickle.load(f), dtype=np.complex128)
    print(f"Loaded {Z.shape[0]} points, shape {Z.shape}")
    if Z.shape[0] > subsamp:
        rng = np.random.default_rng(seed)
        Z = Z[rng.choice(Z.shape[0], subsamp, replace=False)]
        print(f"Subsampled to {len(Z)} (seed={seed})")
    return Z


# ------------------------------------------------------------------- self-test

def _sample_torus_n(n_circles, n_pts, seed):
    """Uniform-ish sample of T^{n_circles} embedded as a product of circles in
    R^{2*n_circles} (radii spread so the circles don't collide in VR)."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, size=(n_pts, n_circles))
    radii = 1.0 + 0.6 * np.arange(n_circles)
    cols = []
    for c in range(n_circles):
        cols += [radii[c] * np.cos(ang[:, c]), radii[c] * np.sin(ang[:, c])]
    return np.stack(cols, axis=1)


def _sample_sphere(n_pts, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_pts, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _two_disjoint_tori(n_pts, seed):
    """Two T^2 separated far in R^4 -> two components, total cup rank 1+1 = 2.
    Exercises the per-component (block-diagonal) cup-rank path."""
    a = _sample_torus_n(2, n_pts, seed)
    b = _sample_torus_n(2, n_pts, seed + 99) + 100.0
    return np.vstack([a, b])


def _euclidean_dist(X):
    g = X @ X.T
    sq = np.diag(g)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * g, 0.0)
    return np.sqrt(d2)


def _selftest():
    """End-to-end ripser -> cup-rank validation on KNOWN spaces.

    Expected rank mu = C(b_1, 2):  T^2 -> 1,  T^3 -> 3,  S^2 -> 0 (no H^1).
    This is the integration test deferred to the cluster (it needs ripser).
    """
    print("=== SELF-TEST (ripser -> cup-rank on known manifolds) ===")
    cases = [
        ("S^2  (b_1=0, expect rank 0)", _sample_sphere(500, 0), 0),
        ("T^2  (b_1=2, expect rank 1)", _sample_torus_n(2, 700, 1), 1),
        ("T^3  (b_1=3, expect rank 3)", _sample_torus_n(3, 1500, 2), 3),
        ("T^2 + T^2 disjoint (expect total rank 2, 2 components)",
         _two_disjoint_tori(700, 4), 2),
    ]
    ok = True
    for label, X, expected in cases:
        print(f"\n--- {label} ---")
        t0 = time.time()
        info = cup_rank_from_distances(_euclidean_dist(X), verbose=True)
        got = info["rank"]
        flag = "PASS" if got == expected else "FAIL"
        ok = ok and (got == expected)
        print(f"    {flag}: total rank mu = {got} (expected {expected}), "
              f"components-with-H^1 = {info['n_components']}  "
              f"[{time.time() - t0:.1f}s]")
    print(f"\n{'ALL SELF-TESTS PASSED' if ok else 'SELF-TEST FAILURES (see above)'}")
    return 0 if ok else 1


# ----------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min_set", default="min_set.pkl",
                   help="(N, 5) complex point cloud pickle (already refined).")
    p.add_argument("--landmarks", default="1000,2000,3000",
                   help="Comma-separated L values to sweep (stability check).")
    p.add_argument("--subsamp", type=int, default=200000,
                   help="Cap on points loaded before landmark selection.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_h1", type=int, default=None,
                   help="Expected b_1 (number of persistent H^1 generators). "
                        "Default: auto-detect via the largest lifetime gap.")
    p.add_argument("--epsilon", type=float, default=None,
                   help="Scale at which to build the 2-skeleton / read cocycles. "
                        "Default: midpoint of the common H^1 plateau.")
    p.add_argument("--thresh", type=float, default=None,
                   help="ripser filtration cap (FS-distance units). Default: "
                        "ripser's enclosing radius.")
    p.add_argument("--cache_landmarks", default=None,
                   help="Optional pkl to cache/reuse the L_max landmark indices.")
    p.add_argument("--selftest", action="store_true",
                   help="Validate the ripser -> cup-rank wiring on synthetic "
                        "T^2/T^3/S^2 instead of loading data. (Needs ripser.)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        return _selftest()

    t_start = time.time()
    L_values = sorted({int(x) for x in args.landmarks.split(",") if x.strip()})
    L_max = L_values[-1]
    print(f"L sweep: {L_values}")

    Z = load_points(args.min_set, args.subsamp, args.seed)
    Zn = _normalize_rows_complex(Z)

    lm = None
    if args.cache_landmarks and os.path.exists(args.cache_landmarks):
        with open(args.cache_landmarks, "rb") as f:
            c = pickle.load(f)
        if c.get("min_set") == args.min_set and c.get("seed") == args.seed \
                and c.get("subsamp") == args.subsamp and len(c["lm"]) >= L_max:
            lm = c["lm"][:L_max]
            print(f"Loaded {len(lm)} cached landmarks from {args.cache_landmarks}")
    if lm is None:
        print(f"=== FARTHEST-POINT LANDMARKS (L_max={L_max}) ===")
        t0 = time.time()
        lm = fps_landmarks(Zn, L_max, args.seed)
        print(f"  done in {time.time() - t0:.1f}s")
        if args.cache_landmarks:
            with open(args.cache_landmarks, "wb") as f:
                pickle.dump({"min_set": args.min_set, "seed": args.seed,
                             "subsamp": args.subsamp, "lm": lm}, f)

    results = {}
    for L in L_values:
        print(f"\n=== L = {L} ===")
        t0 = time.time()
        D = fs_distance_matrix(Zn[lm[:L]])
        info = cup_rank_from_distances(D, n_h1=args.n_h1, epsilon=args.epsilon,
                                       thresh=args.thresh, verbose=True)
        results[L] = info
        print(f"    {_verdict(info)}")
        print(f"    [{time.time() - t0:.1f}s]")

    print("\n=== SUMMARY (cup-product rank across the L-sweep) ===")
    print(f"{'L':>6}  {'comps':>5}  {'b_1':>4}  {'rank mu':>8}  "
          f"per-component (b1:rank)")
    for L in L_values:
        r = results[L]
        per = " ".join(f"{c['b1']}:{c['rank']}" for c in r["components"]) or "-"
        print(f"{L:>6}  {r['n_components']:>5}  {r['b1']:>4}  {r['rank']:>8}  {per}")
    print("\nStable across L -> trustworthy (cup rank is a topological invariant, "
          "not a fit).")
    print("Per closed-3-manifold component:  (b1:rank) = (3:3) -> T^3;  "
          "rank 0 -> #(S^1 x S^2).")
    print(f"\n=== DONE in {time.time() - t_start:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
