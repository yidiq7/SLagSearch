"""Self-check for cup_product.py (Alexander-Whitney cup products + F_2 algebra).

Dependency-free deterministic validation of the bug-prone cup-product / F_2-rank
core, against hand-built complexes with known cohomology rings. Run directly:

    python persistent_homology/cup_product_selfcheck.py

The end-to-end check on *sampled* manifolds (ripser cocycles -> cup rank on a
T^2 / T^3 / S^2 point cloud) lives in persistent_homology_cup_products.py under
``--selftest`` (it needs ripser, so it runs on the cluster, not here).

A binary matrix is a list of *columns*, each a set of row indices where the
entry is 1 (F_2) -- the representation the cup-map linear algebra uses.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cup_product import (f2_rank, f2_independent_count, coboundary_1_columns,
                         aw_cup_cochain, cup_map_rank)


# ----------------------------------------------------------------- F_2 rank

def check_f2_rank_empty_is_zero():
    assert f2_rank([]) == 0


def check_f2_rank_identity_columns_full_rank():
    # 3x3 identity: three independent standard basis columns.
    assert f2_rank([{0}, {1}, {2}]) == 3


def check_f2_rank_duplicate_columns_collapse():
    # Two identical columns span a 1-dim space.
    assert f2_rank([{0, 1}, {0, 1}]) == 1


def check_f2_rank_xor_dependency_detected():
    # Over F_2: {0,1} + {1,2} + {0,2} = {} -> the three columns are dependent.
    assert f2_rank([{0, 1}, {1, 2}, {0, 2}]) == 2


# --------------------------------------- F_2 independence modulo a subspace

def check_independent_count_extra_in_base_contributes_zero():
    # {0} already lies in span(base); {1} is new -> exactly 1 new dimension.
    assert f2_independent_count(base=[{0}], extra=[{0}, {1}]) == 1


def check_independent_count_extra_self_dependency():
    # Empty base; two identical extras -> only 1 independent direction.
    assert f2_independent_count(base=[], extra=[{0}, {0}]) == 1


def check_independent_count_sum_lands_in_base():
    # {0,1} = {0} + {1} over F_2, both in base -> no new dimension.
    assert f2_independent_count(base=[{0}, {1}], extra=[{0, 1}]) == 0


def check_independent_count_empty_extra_is_zero():
    assert f2_independent_count(base=[{0}, {1}], extra=[]) == 0


# --------------------------------------------------------- coboundary delta^1

def check_coboundary_single_triangle_rank_one():
    # One filled triangle (0,1,2): all three boundary edges map to the same
    # 2-simplex over F_2, so im(delta^1) is 1-dimensional.
    assert f2_rank(coboundary_1_columns([(0, 1, 2)])) == 1


def check_coboundary_tetrahedron_boundary_is_S2():
    # Boundary of a tetrahedron = S^2 (4 triangles, 6 edges). H^1 = 0,
    # H^2 = F_2  =>  rank(delta^1) = 6 - dim(Z^1) = 6 - 3 = 3, giving
    # dim H^2 = #triangles - rank(delta^1) = 4 - 3 = 1.
    tetra = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    assert f2_rank(coboundary_1_columns(tetra)) == 3


# ----------------------------------------- Alexander-Whitney cup (deg 1 x 1)
# (alpha cup beta)([a,b,c]) = alpha(a,b) * beta(b,c) for a < b < c:
# front face = first two vertices, back face = last two.

def check_aw_cup_front_and_back_face_both_present():
    assert aw_cup_cochain({(0, 1)}, {(1, 2)}, [(0, 1, 2)]) == {0}


def check_aw_cup_requires_back_face_for_beta():
    # beta lives on (0,2), but the back face of (0,1,2) is (1,2) -> miss.
    assert aw_cup_cochain({(0, 1)}, {(0, 2)}, [(0, 1, 2)]) == set()


def check_aw_cup_requires_front_face_for_alpha():
    # alpha lives on (0,2), but the front face is (0,1) -> miss.
    assert aw_cup_cochain({(0, 2)}, {(1, 2)}, [(0, 1, 2)]) == set()


def check_aw_cup_beta_uses_back_not_front_face():
    # beta on the front edge (0,1) must NOT count: it needs the back face.
    assert aw_cup_cochain({(0, 1)}, {(0, 1)}, [(0, 1, 2)]) == set()


def check_aw_cup_indexes_triangles_correctly():
    # Only triangle 0 has back face (1,2); triangle 1's back face is (1,3).
    tris = [(0, 1, 2), (0, 1, 3)]
    assert aw_cup_cochain({(0, 1)}, {(1, 2)}, tris) == {0}


# ------------------------------------------------- cup_map_rank (assembly)

def check_cup_map_rank_disk_is_zero():
    # A single filled triangle is a disk: H^2 = 0, so the product (0,1)cup(1,2),
    # though a nonzero cochain, lies in im(delta^1) -> class 0.
    assert cup_map_rank([(0, 1, 2)], [{(0, 1)}, {(1, 2)}]) == 0


def check_cup_map_rank_no_two_cells_is_zero():
    # No triangles (e.g. a wedge of circles, b_1 = 2): the cup map lands in
    # H^2 = 0. This is the structural reason a connected sum of S^1 x S^2 gives
    # rank 0 -- the degree-1 products have nowhere nontrivial to go.
    assert cup_map_rank([], [{(0, 1)}, {(1, 2)}]) == 0


# --------------------------------------- nonzero cup products: the torus case
# A grid-triangulated flat torus T^2 has explicit F_2 cocycle reps for its two
# H^1 generators (the i- and j-cuts), whose product is the fundamental class.
# This pins the *nonzero* behaviour of cup_map_rank deterministically -- the
# property the ripser self-test exercises, but here with no ripser / no numpy.

def _grid_torus(m, n, off=0):
    """Staircase-triangulated T^2 on an m x n periodic grid (m, n >= 3).

    Returns (triangles, alpha, beta): triangles as sorted vertex tuples and
    alpha/beta as sets of sorted edge tuples -- F_2 cocycle reps of the two
    H^1 generators. alpha = 1 on edges crossing the i = m-1 -> 0 seam (vertical
    + diagonal), beta = 1 on edges crossing the j = n-1 -> 0 seam (horizontal +
    diagonal); each satisfies delta = 0 (every triangle meets the seam in an
    even number of marked edges). ``off`` shifts vertex indices for unions.
    """
    def v(i, j):
        return off + (i % m) * n + (j % n)

    def e(a, b):
        return (a, b) if a < b else (b, a)

    tris, alpha, beta = [], set(), set()
    for i in range(m):
        for j in range(n):
            a, b, c, d = v(i, j), v(i + 1, j), v(i, j + 1), v(i + 1, j + 1)
            tris.append(tuple(sorted((a, b, d))))   # lower triangle
            tris.append(tuple(sorted((a, c, d))))   # upper triangle (diag a-d)
            if i == m - 1:                            # i-seam crossings
                alpha.add(e(v(m - 1, j), v(0, j)))        # vertical
                alpha.add(e(v(m - 1, j), v(0, j + 1)))    # diagonal
            if j == n - 1:                            # j-seam crossings
                beta.add(e(v(i, n - 1), v(i, 0)))         # horizontal
                beta.add(e(v(i, n - 1), v(i + 1, 0)))     # diagonal
    return tris, alpha, beta


def check_cup_map_rank_torus_is_one():
    # T^2: /\^2 H^1 is 1-dimensional and alpha cup beta = fundamental class != 0.
    tris, alpha, beta = _grid_torus(5, 5)
    assert cup_map_rank(tris, [alpha, beta]) == 1


def check_cup_map_rank_two_disjoint_tori_is_two():
    # Two disjoint T^2 (offset vertex sets): cross products vanish, each torus
    # contributes 1 -> total rank 2. (Validates the additive/block structure
    # the cup map has across components, at the core level.)
    t1, a1, b1 = _grid_torus(5, 5, off=0)
    t2, a2, b2 = _grid_torus(5, 5, off=100)
    assert cup_map_rank(t1 + t2, [a1, b1, a2, b2]) == 2


# ----------------------------------------------------------------- runner

def _run_all():
    checks = [v for k, v in sorted(globals().items())
              if k.startswith("check_") and callable(v)]
    failures = []
    for c in checks:
        try:
            c()
            print(f"  PASS  {c.__name__}")
        except Exception as e:  # noqa: BLE001
            failures.append((c.__name__, e))
            print(f"  FAIL  {c.__name__}: {e!r}")
    print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
