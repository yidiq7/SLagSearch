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
  3. ripser on the (L, L) FS distance matrix: maxdim=1, do_cocycles=True,
     coeff=2 -> H^1 diagram + H^1 cocycle representatives. (No H^2/tetrahedra:
     the cup rank is computed on our own 2-skeleton, not ripser's complex.)
  4. SCAN scales across the chosen bars' H^1 plateau; at each, build the
     2-skeleton, convert the H^1 cocycles ALIVE there to F_2 edge sets, and
     rank mu via cup_product.cup_map_rank. Report the MAX over the scan (a
     poor-man's persistent cup rank), since a nonzero product needs eps where
     the H^2 void exists -- a larger scale than where H^1 loops are born.
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
    """Heuristic count of persistent H^1 bars: the largest ADDITIVE gap among
    the top bars. UNRELIABLE on noise and on spaces with no real H^1 -- a
    convenience only; pin --n_h1 from known Betti numbers for real runs.

    Caller passes lifetimes with essential (death=inf) bars capped to thresh, so
    they read as the longest finite lifetimes. The earlier multiplicative-gap
    version blew up: inf lifetimes gave inf/inf = nan, and on a smooth noise
    tail the largest ratio sits between tiny values -> absurd counts (e.g. 137).
    """
    s = np.sort(np.asarray(lifetimes, dtype=float))[::-1]
    s = s[np.isfinite(s)]                     # defensive (caller already caps)
    if len(s) <= 1:
        return int(len(s))
    k = min(len(s) - 1, 15)                    # only the top bars matter
    gaps = s[:k] - s[1:k + 1]
    return int(np.argmax(gaps)) + 1


def _cup_rank_at_eps(D, eps, coc1, chosen_info):
    """Per-component rank of mu : Lambda^2 H^1 -> H^2 at a single scale eps.

    ``chosen_info`` is a list of (cocycle_index, birth, death). Only bars ALIVE
    at eps (birth <= eps < death) are used -- a dead bar's cocycle is no longer
    closed, which would break cup_map_rank's "products are cocycles" invariant
    and yield garbage. Cup products across connected components vanish, so the
    rank is summed per component. Needs only the 2-skeleton (rank modulo
    im delta^1 already equals the rank in H^2), so no H^2 from ripser.
    """
    alive = [c for (c, b, d) in chosen_info if b <= eps < d]
    base = {"eps": float(eps), "n_alive": len(alive), "components": [],
            "n_vertices": 0, "n_edges": 0, "n_triangles": 0,
            "n_components_total": 0, "rank": 0}
    if len(alive) < 2:
        return base
    n_vert, edges, triangles = _two_skeleton(D, eps)
    comp_of = _connected_components(n_vert, edges)
    cocycles = [_cocycle_to_edges(coc1[c], D, eps) for c in alive]
    cocycles = [cc for cc in cocycles if cc]
    tri_by_comp, coc_by_comp = {}, {}
    for tri in triangles:
        tri_by_comp.setdefault(comp_of[tri[0]], []).append(tri)
    for cc in cocycles:
        coc_by_comp.setdefault(comp_of[next(iter(cc))[0]], []).append(cc)
    components, total = [], 0
    for cid, ccs in sorted(coc_by_comp.items()):
        r = cup_map_rank(tri_by_comp.get(cid, []), ccs)
        total += r
        components.append({"b1": len(ccs), "rank": r})
    base.update(rank=total, components=components, n_vertices=n_vert,
                n_edges=len(edges), n_triangles=len(triangles),
                n_components_total=len(set(comp_of)))
    return base


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
                            thresh_factor=4.0, n_eps=8, verbose=True):
    """ripser on a distance matrix -> max over scale of rank(mu: Lambda^2 H^1 -> H^2).

    Metric-agnostic: ``D`` is any (L, L) distance matrix (FS for the sLag data,
    Euclidean for the self-test). Returns a dict with the (persistent) cup rank
    and diagnostics.

    Memory: ripser runs at maxdim=1 (H^1 cocycles only). The cup rank is built
    from this tool's OWN 2-skeleton, so ripser's H^2 -- and its tetrahedra, the
    OOM source -- are never needed. ``thresh`` still caps the filtration; with
    thresh = inf even the triangles explode (C(L,3)). Default thresh =
    ``thresh_factor`` x covering radius; pass --thresh (e.g. a scale from your
    witness run) if features live at a larger scale.

    Scale: a nonzero cup product needs eps where H^2 exists (the void scale,
    larger than where H^1 loops are born). Rather than guess one eps we SCAN
    ``n_eps`` scales across the chosen bars' H^1 plateau and report the MAX cup
    rank (a poor-man's persistent cup rank). Pass ``epsilon`` for a single scale.
    """
    from ripser import ripser

    if thresh is None:
        Dd = D.copy()
        np.fill_diagonal(Dd, np.inf)
        cov = float(Dd.min(axis=1).max())  # covering radius among landmarks
        thresh = thresh_factor * cov
        if verbose:
            print(f"    covering radius = {cov:.4f}  ->  thresh = "
                  f"{thresh_factor:g} x cov = {thresh:.4f}  (caps Rips complex)")
    thresh = float(thresh)
    res = ripser(D, distance_matrix=True, maxdim=1, do_cocycles=True,
                 coeff=2, thresh=thresh)
    H1 = res["dgms"][1]
    coc1 = res["cocycles"][1]

    # Cap essential (death=inf) bars at thresh so lifetimes are finite and the
    # essential generators read as the longest.
    if len(H1):
        deaths = np.where(np.isinf(H1[:, 1]), thresh, H1[:, 1])
        lifetimes = deaths - H1[:, 0]
    else:
        deaths, lifetimes = np.array([]), np.array([])
    sorted_life = np.sort(lifetimes)[::-1] if len(lifetimes) else np.array([])

    if n_h1 is None:
        n_h1 = _auto_n_h1(lifetimes) if len(lifetimes) else 0
        if verbose:
            print(f"    [auto] n_h1 = {n_h1}  (heuristic, unreliable on noise -- "
                  f"pin --n_h1 from your known Betti numbers)")
    n_h1 = min(n_h1, len(H1))

    if n_h1 < 2:
        if verbose:
            print(f"    b_1 used = {n_h1} (< 2): no degree-1 pairs to multiply.")
        return {"rank": 0, "b1": n_h1, "epsilon": epsilon, "components": [],
                "n_components": 0, "n_components_total": None, "scan": [],
                "thresh": thresh, "lifetimes_sorted": sorted_life}

    chosen = np.argsort(lifetimes)[::-1][:n_h1]
    chosen_info = [(int(c), float(H1[c, 0]), float(H1[c, 1])) for c in chosen]

    # Scan scales across the chosen bars' common H^1 plateau (deaths capped).
    lo = float(H1[chosen, 0].max())
    hi = float(deaths[chosen].min())
    if not (lo < hi):                       # chosen bars never co-exist
        lo, hi = float(H1[chosen, 0].min()), float(deaths[chosen].max())
    if epsilon is not None:
        eps_grid = [float(epsilon)]
    else:
        eps_grid = [e for e in np.linspace(lo, hi, n_eps + 2)[1:-1]
                    if 0 < e < thresh] or [0.5 * (lo + min(hi, thresh))]

    if verbose:
        print(f"    H^1 lifetimes (top): "
              f"{np.round(sorted_life[:max(n_h1 + 2, 4)], 4)}")
        print(f"    b_1 used = {n_h1}   plateau [{lo:.4f}, {hi:.4f}]   "
              f"thresh = {thresh:.4f}   scanning {len(eps_grid)} scale(s)")

    # rank mu can never exceed C(n_h1, 2) (that many products exist), so once we
    # hit it the scan can stop -- a big saving since later (larger-eps) steps are
    # by far the most expensive. (Disjoint clouds whose true max < C(n_h1,2),
    # from cross-component vanishing, simply scan the full grid.)
    max_possible = n_h1 * (n_h1 - 1) // 2
    best, scan = None, []
    for eps in eps_grid:
        r = _cup_rank_at_eps(D, eps, coc1, chosen_info)
        scan.append((float(eps), r["rank"]))
        if verbose:
            per = " ".join(f"{c['b1']}:{c['rank']}" for c in r["components"]) or "-"
            print(f"      eps={eps:.4f}  V={r['n_vertices']} E={r['n_edges']} "
                  f"T={r['n_triangles']}  comps={r['n_components_total']}  "
                  f"rank={r['rank']}  ({per})")
        if best is None or r["rank"] > best["rank"]:
            best = r
        if best["rank"] >= max_possible:
            if verbose:
                print(f"      (reached max possible rank C({n_h1},2) = "
                      f"{max_possible}; stopping scan early)")
            break

    info = {"rank": best["rank"], "b1": n_h1, "epsilon": best["eps"],
            "components": best["components"], "n_components": len(best["components"]),
            "n_components_total": best["n_components_total"],
            "n_vertices": best["n_vertices"], "n_edges": best["n_edges"],
            "n_triangles": best["n_triangles"], "thresh": thresh,
            "scan": scan, "lifetimes_sorted": sorted_life}
    if verbose:
        print(f"    => max cup rank over scale = {best['rank']} "
              f"at eps = {best['eps']:.4f}")
        for i, c in enumerate(best["components"]):
            tag = ("T^3-like" if (c["b1"] == 3 and c["rank"] == 3)
                   else "conn-sum-like" if c["rank"] == 0
                   else f"other(b1={c['b1']},rank={c['rank']})")
            print(f"      component {i}: b_1 = {c['b1']}, rank mu = {c['rank']}"
                  f"  ->  {tag}")
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
    """Uniform sample of a flat T^{n_circles} as a product of UNIT circles in
    R^{2*n_circles} (equal radii -> symmetric, cleanly resolved in VR)."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, size=(n_pts, n_circles))
    cols = []
    for c in range(n_circles):
        cols += [np.cos(ang[:, c]), np.sin(ang[:, c])]
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
    """End-to-end ripser -> cup-rank validation on KNOWN spaces (needs ripser).

    n_h1 is PINNED to the known Betti number -- this validates the ripser->AW->
    rank wiring given the right generators, not the heuristic auto-detector.
    thresh is set generously so the H^2 void forms within the scanned range.
    Expected max cup rank:  S^2 -> 0,  T^2 -> 1,  T^3 -> 3,  T^2 u T^2 -> 2.
    """
    print("=== SELF-TEST (ripser -> cup-rank on known manifolds) ===")
    # (label, points, n_h1, thresh, expected rank)
    cases = [
        ("S^2  (no H^1, expect rank 0)", _sample_sphere(400, 0), 0, 1.6, 0),
        ("T^2  (expect rank 1)", _sample_torus_n(2, 400, 1), 2, 1.6, 1),
        ("T^3  (expect rank 3)", _sample_torus_n(3, 1500, 2), 3, 1.6, 3),
        ("T^2 + T^2 disjoint (expect total rank 2)",
         _two_disjoint_tori(400, 4), 4, 1.6, 2),
    ]
    ok = True
    for label, X, n_h1, thr, expected in cases:
        print(f"\n--- {label} ---")
        t0 = time.time()
        info = cup_rank_from_distances(_euclidean_dist(X), n_h1=n_h1,
                                       thresh=thr, verbose=True)
        got = info["rank"]
        flag = "PASS" if got == expected else "FAIL"
        ok = ok and (got == expected)
        print(f"    {flag}: max cup rank = {got} (expected {expected}), "
              f"components = {info['n_components']}  [{time.time() - t0:.1f}s]")
    print(f"\n{'ALL SELF-TESTS PASSED' if ok else 'SELF-TEST FAILURES (see above)'}")
    print("(T^2 and the disjoint case are the core wiring validators; a T^3 "
          "miss with rank < 3 is usually VR under-sampling the voids, not a bug.)")
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
                   help="Fix a single scale instead of scanning the H^1 plateau "
                        "(e.g. once you know the void scale). Default: scan.")
    p.add_argument("--n_eps", type=int, default=8,
                   help="Number of scales scanned across the H^1 plateau; the "
                        "max cup rank over the scan is reported. Cost is "
                        "~linear in this (each scale rebuilds the 2-skeleton).")
    p.add_argument("--thresh", type=float, default=None,
                   help="ripser filtration cap (FS-distance units). MUST be "
                        "finite -- inf builds the complete complex and OOMs. "
                        "Default: thresh_factor x covering radius. For known "
                        "feature scales pass it directly (e.g. your witness "
                        "max_alpha); the scan needs thresh to reach the void.")
    p.add_argument("--thresh_factor", type=float, default=4.0,
                   help="Auto thresh = this x covering radius (max nearest-"
                        "landmark distance). Raise it if the cup rank stays 0 "
                        "(thresh not reaching the H^2/void scale); lower it (or "
                        "lower --landmarks) if you OOM.")
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
                                       thresh=args.thresh,
                                       thresh_factor=args.thresh_factor,
                                       n_eps=args.n_eps, verbose=True)
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
