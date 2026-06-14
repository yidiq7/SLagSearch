"""Alexander-Whitney cup products + F_2 linear algebra on a simplicial complex.

Pure-Python / numpy-free core, in the spirit of ``hermitian_coeffs.py``: no JAX,
no ripser. Consumed by ``persistent_homology_cup_length.py`` (the CLI that feeds
it a ripser-derived complex + H^1 cocycles) and validated in isolation against
hand-built triangulations in ``cup_product_selfcheck.py``.

The decisive quantity for distinguishing T^3 from #^k(S^1 x S^2): the rank over
F_2 of the cup-product map mu : Lambda^2 H^1 -> H^2.  For a 3-torus mu is injective
(rank = C(b_1, 2)); for a connected sum of S^1 x S^2 every product of degree-1
classes vanishes, so rank mu = 0.

Representation: a binary matrix is a list of *columns*, each column a ``set`` of
the row indices where the entry is 1 (arithmetic over F_2).
"""
from __future__ import annotations

from typing import Iterable


def _reduce_into(pivots: dict[int, set[int]], col: Iterable[int]) -> bool:
    """Reduce ``col`` against the current ``pivots`` (mutated in place).

    Pivot on the highest set row index, XOR out collisions. If a nonzero
    remainder survives, register it as a new pivot and return True (the column
    added a dimension); otherwise it lay in the existing span -> return False.
    """
    col = set(col)
    while col:
        p = max(col)
        if p in pivots:
            col ^= pivots[p]
        else:
            pivots[p] = col
            return True
    return False


def f2_rank(columns: Iterable[Iterable[int]]) -> int:
    """Rank over F_2 of a binary matrix given as a list of columns.

    Each column is an iterable of the row indices where it is 1. Uses the
    standard greedy column reduction (the same primitive as persistence matrix
    reduction).
    """
    pivots: dict[int, set[int]] = {}
    return sum(_reduce_into(pivots, col) for col in columns)


def coboundary_1_columns(triangles: Iterable[tuple[int, int, int]]
                         ) -> list[set[int]]:
    """Columns of the coboundary delta^1 : C^1 -> C^2, indexed by triangle.

    Row index = position of a triangle in ``triangles``. Over F_2,
    (delta^1 f)([a,b,c]) = f(ab) + f(ac) + f(bc), so the indicator cochain of a
    single edge e maps to the sum of the triangles having e as a face. The
    column for edge e is therefore { triangle index : e is a face }. Edges that
    bound no triangle give zero columns and are omitted (irrelevant to the span
    im delta^1). Triangle vertices are sorted so faces are canonical.
    """
    edge_to_tris: dict[tuple[int, int], set[int]] = {}
    for t_idx, tri in enumerate(triangles):
        a, b, c = sorted(tri)
        for e in ((a, b), (a, c), (b, c)):
            edge_to_tris.setdefault(e, set()).add(t_idx)
    return list(edge_to_tris.values())


def aw_cup_cochain(alpha: set[tuple[int, int]],
                   beta: set[tuple[int, int]],
                   triangles: Iterable[tuple[int, int, int]]) -> set[int]:
    """Alexander-Whitney cup product of two 1-cochains, as a 2-cochain over F_2.

    ``alpha`` / ``beta`` are sets of sorted edge tuples ``(lo, hi)`` carrying
    value 1. For a triangle with sorted vertices a < b < c the AW formula gives
    (alpha cup beta)([a,b,c]) = alpha(front face) * beta(back face)
                              = alpha(a,b) * beta(b,c).
    Returns the set of triangle indices on which the product is 1.
    """
    out: set[int] = set()
    for t_idx, tri in enumerate(triangles):
        a, b, c = sorted(tri)
        if (a, b) in alpha and (b, c) in beta:
            out.add(t_idx)
    return out


def f2_independent_count(base: Iterable[Iterable[int]],
                         extra: Iterable[Iterable[int]]) -> int:
    """Number of ``extra`` columns independent modulo span(``base``) over F_2.

    Equivalently rank([base | extra]) - rank(base). This is exactly rank(mu):
    feed the columns of the coboundary delta^1 as ``base`` (their span is
    im delta^1 = the cup products that are cohomologically trivial) and the
    cup-product 2-cochains as ``extra``; the count is the number of independent
    classes [alpha_i cup alpha_j] in H^2.
    """
    pivots: dict[int, set[int]] = {}
    for col in base:
        _reduce_into(pivots, col)
    return sum(_reduce_into(pivots, col) for col in extra)


def cup_map_rank(triangles: Iterable[tuple[int, int, int]],
                 h1_cocycles: list[set[tuple[int, int]]]) -> int:
    """Rank over F_2 of the cup map mu : Lambda^2 H^1 -> H^2.

    ``triangles`` is the 2-skeleton (sorted-vertex tuples) of the complex at a
    chosen scale; ``h1_cocycles`` is a list of 1-cocycle representatives of the
    H^1 generators (each a set of sorted edge tuples), e.g. ripser's persistent
    H^1 cocycles. Forms all wedge products alpha_i cup alpha_j for i < j as
    2-cochains and counts how many are linearly independent in
    H^2 = ker(delta^2) / im(delta^1) -- i.e. independent modulo im(delta^1).

    Interpretation for a closed orientable 3-manifold component:
        rank == C(b_1, 2)  (mu injective on Lambda^2 H^1)  -> torus T^3
        rank == 0          (all degree-1 products vanish) -> #^k (S^1 x S^2)
    """
    triangles = list(triangles)  # fix one ordering shared by base + products
    base = coboundary_1_columns(triangles)
    n = len(h1_cocycles)
    products = [aw_cup_cochain(h1_cocycles[i], h1_cocycles[j], triangles)
                for i in range(n) for j in range(i + 1, n)]
    return f2_independent_count(base, products)
