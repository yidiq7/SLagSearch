"""Render the symmetry-projected sLag candidate equations (degrees 1-3) as
LaTeX, in complex-bilinear form: terms are monomials v_A conj(v_B) with
complex coefficients H_AB, plus '+/- c.c.' for the off-diagonal half, grouped
by degree, with each equation rescaled so its leading coefficient is 1.

Reads a (projected) coeffs pkl directly (bare (3, w) array or a checkpoint
dict with a 'coeffs' key) via hermitian_coeffs.extract_hermitians, and writes
<coeffs_stem>_equations.tex next to it (then compiles to PDF if pdflatex is
available). Intended input: the G-equivariant d=4 candidate produced by
`python -m symmetry.project_to_symmetric` (width 6375). The character labels
and title are hardcoded for that Z_2 x S_3 canonical candidate.

Conventions (round-trip exact against helper.py / hermitian_coeffs.py):
  E_k(z) = sum_d (1/||z||^{2d}) * [ sum_A H_AA |v_A|^2
            + ( sum_{A<B} H_AB v_A conj(v_B) + c.c. ) ]
with v the degree-d monomial vector in combinations_with_replacement order.

Usage:
    python -m symmetry.coeffs_to_latex --coeffs <projected.pkl> [--thr 0.02]
        [--out <file.tex>] [--no_pdf]
"""
import argparse
import datetime
import pickle
import shutil
import subprocess
from collections import Counter
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from hermitian_coeffs import _load_coeffs, extract_hermitians

CHAR_DESC = {
    0: r"odd under $z_0 \leftrightarrow z_4$, invariant under $S_3$",
    1: r"invariant under $z_0 \leftrightarrow z_4$, sign of $S_3$",
    2: r"fully invariant",
}


def complex_terms(H):
    """Yield (kind, A, B, coeff) with kind in {'diag','off'}: the diagonal and
    strict-upper entries of H, labelled by monomial index tuples."""
    N = H.shape[0]
    d = {5: 1, 15: 2, 35: 3, 70: 4}[N]
    tuples = list(combinations_with_replacement(range(5), d))
    out = []
    for A in range(N):
        out.append(("diag", tuples[A], tuples[A], complex(H[A, A].real)))
        for B in range(A + 1, N):
            out.append(("off", tuples[A], tuples[B], complex(H[A, B])))
    return out


def mass_weight(kind):
    """Squared-norm weight in the real coefficient basis: an off-diagonal
    entry contributes 4|H|^2 (Re+Im coeff pair), a diagonal entry H^2."""
    return 1.0 if kind == "diag" else 4.0


def term_latex(kind, A, B):
    """LaTeX for the monomial v_A conj(v_B), with common |z_i|^2 factors pulled
    out. Also returns an estimated width (units)."""
    cA, cB = Counter(A), Counter(B)
    common = cA & cB
    rA, rB = cA - common, cB - common

    parts, width = [], 0.0
    for i in sorted(common):
        e = 2 * common[i]
        parts.append("|z_%d|^%d" % (i, e))
        width += 2.4
    if kind == "diag":
        return " ".join(parts), max(width, 1.0)

    core = []
    for i in sorted(rA):
        e = rA[i]
        core.append("z_%d" % i + ("^%d" % e if e > 1 else ""))
        width += 1.4 + (0.3 if e > 1 else 0)
    for i in sorted(rB):
        e = rB[i]
        core.append("\\bar z_%d" % i + ("^%d" % e if e > 1 else ""))
        width += 1.4 + (0.3 if e > 1 else 0)
    body = (" ".join(parts) + " \\, " if parts else "") + " ".join(core)
    return body, width


def fmt_coeff(c):
    """Signed prefactor for a complex coefficient: '{}- 0.2594\\,',
    '{}+ 0.2594i\\,', '{}- (0.2688 + 0.0787i)\\,'. Coefficients equal to
    +/-1 or +/-i (at 4-decimal rounding) print bare (sign / i only).
    Components that round to 0 at 4 decimals are dropped.
    Returns (latex, width)."""
    re, im = c.real, c.imag
    if abs(im) < 5e-5:
        if abs(abs(re) - 1.0) < 5e-5:
            return ("{}- " if re < 0 else "{}+ "), 1.5
        return ("{}- " if re < 0 else "{}+ ") + "%.4f\\," % abs(re), 7.0
    if abs(re) < 5e-5:
        if abs(abs(im) - 1.0) < 5e-5:
            return ("{}- " if im < 0 else "{}+ ") + "i\\,", 2.5
        return ("{}- " if im < 0 else "{}+ ") + "%.4fi\\," % abs(im), 7.5
    sgn = -1 if re < 0 else 1
    b = im * sgn
    inner = "%.4f %s %.4fi" % (abs(re), "-" if b < 0 else "+", abs(b))
    return ("{}- " if sgn < 0 else "{}+ ") + "(" + inner + ")\\,", 13.0


def cluster_terms(terms, tol=1e-6):
    """Group terms (kind, A, B, c) by equal coefficient up to sign.

    Terms are first sorted by decreasing |c| and grouped into equal-|c|
    clusters (G-orbit multiplets); each cluster is then split by coefficient
    direction (c vs -c stay together with relative signs; conjugates split).
    Returns [(rep_coeff, [(sign, kind, A, B), ...]), ...] with the first
    member's sign folded into rep_coeff (bracket opens positive).
    """
    terms = sorted(terms, key=lambda t: -abs(t[3]))
    mag_clusters = []
    for kind, A, B, c in terms:
        if mag_clusters and abs(abs(c) - mag_clusters[-1][0]) <= tol:
            mag_clusters[-1][1].append((kind, A, B, c))
        else:
            mag_clusters.append((abs(c), [(kind, A, B, c)]))

    out = []
    kind_ord = {"diag": 0, "off": 1}
    for _, members in mag_clusters:
        subs = []  # [rep_c, [(sign, kind, A, B), ...]]
        for kind, A, B, c in members:
            for s in subs:
                if abs(c - s[0]) <= tol:
                    s[1].append((1, kind, A, B))
                    break
                if abs(c + s[0]) <= tol:
                    s[1].append((-1, kind, A, B))
                    break
            else:
                subs.append([c, [(1, kind, A, B)]])
        for rep, mem in subs:
            mem.sort(key=lambda m: (kind_ord[m[1]], m[2], m[3]))
            if mem[0][0] < 0:  # flip so the bracket opens with +
                rep = -rep
                mem = [(-s, k, A, B) for s, k, A, B in mem]
            out.append((rep, mem))
    return out


def emit_block(eq, d, clusters, cc_sign="+", budget=38.0):
    """LaTeX align* rows for one degree block: sorted clusters, then +/- c.c."""
    rows = []
    lhs = "\\hat E^{(%d)}_%d \\;\\approx\\; " % (d, eq)
    for ci, (rep, members) in enumerate(clusters):
        pre, pw = fmt_coeff(rep)
        if ci == 0:  # no leading '+' on the first cluster
            pre = pre.replace("{}+ ", "", 1).replace("{}- ", "-", 1)
        line = ("& " if ci > 0 else lhs + "& ") + pre
        used = pw
        if len(members) == 1:
            body, _ = term_latex(members[0][1], members[0][2], members[0][3])
            rows.append(line + body + "\\\\")
            continue
        line += "\\bigl["
        for idx, (s, kind, A, B) in enumerate(members):
            body, w = term_latex(kind, A, B)
            if idx == 0:
                line += body  # sign folded into rep
                used += w
            elif used + w + 1.0 > budget:
                rows.append(line + "\\\\")
                line = "& \\qquad " + ("{}- " if s < 0 else "{}+ ") + body
                used = 5.0 + w
            else:
                line += (" - " if s < 0 else " + ") + body
                used += w + 1.0
        rows.append(line + "\\bigr]\\\\")
    if any(m[1] != "diag" for _, mem in clusters for m in mem):
        rows.append("& {}%s \\mathrm{c.c.}\\\\" % cc_sign)
    if rows:
        rows[-1] = rows[-1][:-2]  # strip trailing \\ on last row
    return "\\begin{align*}\n" + "\n".join(rows) + "\n\\end{align*}"


PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage{url}
\allowdisplaybreaks
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\begin{document}
\begin{center}
{\Large\bfseries Symmetry-projected $d=4$ sLag candidate:\\[2pt]
 defining equations through degree 3}\\[6pt]
{\small Fermat quintic ($\psi=0$) --- generated from
\path{@SOURCE@}, @DATE@}
\end{center}
"""

SETUP = r"""\subsection*{Setup and conventions}

The candidate special-Lagrangian submanifold is the locus
$E_0 = E_1 = E_2 = 0$ on the Fermat quintic $\sum_i z_i^5 = 0$ in
$\mathbb{P}^4$. Each equation is a sum of degree-$(d,d)$ bilinear pieces,
\[
E_k(z) \;=\; \sum_{d=1}^{4} \frac{E_k^{(d)}(z)}{\lVert z\rVert^{2d}},
\qquad
E_k^{(d)}(z) \;=\; \sum_{A} H^{(d)}_{k,AA}\, |v^{(d)}_A|^2
 \;+\; \Bigl( \sum_{A<B} H^{(d)}_{k,AB}\, v^{(d)}_A \overline{v^{(d)}_B}
 \;+\; \mathrm{c.c.} \Bigr),
\]
with $\lVert z\rVert^2 = \sum_i |z_i|^2$, where $v^{(d)}$ runs over the plain
degree-$d$ monomials of $\operatorname{Sym}^d(\mathbb{C}^5)$ and $H^{(d)}_k$
is Hermitian, so each $E_k$ is real-valued and invariant under
$z \to \lambda z$, $\lambda \in \mathbb{C}^*$. \textbf{Only the numerator
polynomials $E_k^{(d)}$ are displayed below (rescaled as described next);
the division by $\lVert z\rVert^{2d}$ is suppressed throughout and is
restored when the blocks are summed into $E_k$.} The trailing
``$\pm\,\mathrm{c.c.}$'' in each block completes the off-diagonal sum; it
applies to the terms containing a genuine bilinear (those with a
$z_i \bar z_j$ core) --- pure modulus terms $|z_{i_1}|^2 \cdots$ appear once.
Common factors are pulled out of each monomial pair, e.g.\
$|z_1|^2\, z_0\bar z_4$ means $v_A \bar v_B = (z_0 z_1)\overline{(z_1 z_4)}$.

These coefficients are assumed to be the output of
\texttt{symmetry.project\_to\_symmetric} --- character projection onto
$G = \mathbb{Z}_2 \times S_3$, where $\mathbb{Z}_2$ swaps
$z_0 \leftrightarrow z_4$ and $S_3$ permutes $(z_1,z_2,z_3)$ --- so that they
are \emph{exactly} $G$-equivariant.@PROVENANCE@ The equations carry the characters
\[
E_0:\ \chi_{\mathbb{Z}_2}\otimes\mathbf{1}, \qquad
E_1:\ \mathbf{1}\otimes\operatorname{sgn}_{S_3}, \qquad
E_2:\ \mathbf{1}\otimes\mathbf{1}.
\]

\textbf{Display normalization.} Each equation is rescaled by a positive
constant $s_k$ and a unit phase $\mu_k$ (neither changes its zero set):
\[
\hat E^{(d)}_k \;:=\; \frac{\mu_k}{s_k}\, E^{(d)}_k
 \;=\; \sum_A c_{AA}\, |v_A|^2
 \;+\; \Bigl(\sum_{A<B} c_{AB}\, v_A \bar v_B \;+\; \mu_k^2\,\mathrm{c.c.}\Bigr),
\qquad c_{AB} = \frac{\mu_k}{s_k} H_{AB},
\]
where $s_k$ is the largest coefficient magnitude $|H_{AB}|$ of $E_k$ among
the displayed degrees $d \le 3$ and $\mu_k \in \{\pm 1, \pm i\}$ rotates that
leading coefficient to exactly $+1$, so the leading term carries no
prefactor:
\[
s_0 = @S0@, \qquad s_1 = @S1@, \qquad s_2 = @S2@,
\qquad \mu_0 = @MU0@, \qquad \mu_1 = @MU1@, \qquad \mu_2 = @MU2@.
\]
As indicated above, the off-diagonal completion inherits the phase:
blocks end in $+\,\mathrm{c.c.}$ when $\mu_k^2 = +1$ and in
$-\,\mathrm{c.c.}$ when $\mu_k^2 = -1$. For $\mu_k = \pm i$ the displayed
expression equals $i$ times a real function (zero locus unaffected), and any
modulus terms $|v_A|^2$ would carry an explicit factor $i$ in their
coefficients.@DIAGNOTE@

\textbf{Noise threshold.} Coefficients with $|c| < @THR@$ (i.e.\ @THRPCT@\%
of each equation's largest coefficient magnitude, after the rescaling above)
are omitted. After the symmetry projection, true numerical noise sits at
$\sim 10^{-19}$, so everything dropped here is small-but-genuine structure;
the table below reports exactly how much. Terms are sorted by decreasing
$|c|$. Square brackets group monomials whose coefficients are equal up to
sign at machine precision --- these are $G$-orbit multiplets; the prefactor
is the coefficient of the bracket's first term, and an orbit may split into
two brackets with conjugate prefactors (e.g.\ partners under
$z_0 \leftrightarrow z_4$).

\subsection*{Coverage at threshold $|c| \ge @THR@$}

\begin{center}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{c c ccc ccc cccc}
\toprule
 & & \multicolumn{3}{c}{entries kept / nonzero} &
\multicolumn{3}{c}{dropped $\sum c^2$ mass} &
\multicolumn{4}{c}{block share of $\sum c^2$ ($d=1\ldots4$)} \\
$E_k$ & $s_k$ & $d{=}1$ & $d{=}2$ & $d{=}3$
 & $d{=}1$ & $d{=}2$ & $d{=}3$
 & $d{=}1$ & $d{=}2$ & $d{=}3$ & $d{=}4$ \\
\midrule
"""

TABLE_TAIL = r"""\bottomrule
\end{tabular}
\end{center}

Entries are counted as Hermitian-matrix entries (diagonal + strict upper
triangle); mass fractions are computed in the equivalent real coefficient
basis ($4|H_{AB}|^2$ off-diagonal, $H_{AA}^2$ diagonal), so they match the
fitness pipeline's normalization. The degree-4 blocks $E_k^{(4)}$ are not
displayed; their share of each equation's total $\sum c^2$ is given in the
last column above. Source coefficients: \path{@SOURCE@}.
"""


def load_provenance(coeffs_path):
    """LaTeX provenance sentence from a sibling projection_metadata.pkl, or ''."""
    meta_path = coeffs_path.with_name("projection_metadata.pkl")
    if not meta_path.exists():
        return ""
    try:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        kept, resid = meta.get("overall_kept"), meta.get("max_residual_after")
        if kept is None or resid is None:
            return ""
        mant, exp = ("%.0e" % resid).split("e")
        return (" The projection kept $%.1f\\%%$ of the coefficient norm "
                "(equivariance residual $%s\\times 10^{%d}$)."
                % (100 * kept, mant, int(exp)))
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coeffs", required=True, type=Path,
                    help="projected coeffs pkl (bare (3, w) array or a dict "
                         "with a 'coeffs' key).")
    ap.add_argument("--thr", type=float, default=0.02,
                    help="relative threshold: drop |c| < thr * max|c| per "
                         "equation (max over displayed degrees d<=3).")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .tex path (default: <coeffs_stem>_equations.tex "
                         "next to --coeffs).")
    ap.add_argument("--no_pdf", action="store_true",
                    help="skip the pdflatex compile step.")
    args = ap.parse_args()

    if args.out is None:
        args.out = args.coeffs.with_name(args.coeffs.stem + "_equations.tex")
    source = str(args.coeffs)
    date = datetime.date.today().isoformat()
    provenance = load_provenance(args.coeffs)

    coeffs = _load_coeffs(args.coeffs)
    H_by_d = extract_hermitians(coeffs)
    terms = {}
    for d, H_list in H_by_d.items():
        for k in range(3):
            terms[(d, k)] = complex_terms(H_list[k])
    missing = [d for d in (1, 2, 3, 4) if (d, 0) not in terms]
    if missing:
        raise SystemExit("coeffs (width %d) is missing degree(s) %s; this tool "
                         "expects the full d=4 (width-6375) projected coeffs."
                         % (coeffs.shape[1], missing))

    # per-equation normalization scale: max |H entry| over d<=3
    scale = {}
    for k in range(3):
        scale[k] = max(abs(c) for d in (1, 2, 3) for _, _, _, c in terms[(d, k)])

    # per-equation unit phase mu rotating the leading coefficient to +1
    mu_units = [(1 + 0j, "1"), (-1 + 0j, "-1"), (1j, "i"), (-1j, "-i")]
    mu, mu_name, cc_sign = {}, {}, {}
    for k in range(3):
        lead = max((c for d in (1, 2, 3) for _, _, _, c in terms[(d, k)]),
                   key=abs)
        target = scale[k] / lead  # exact unit phase with mu * lead / s = 1
        u, name = min(mu_units, key=lambda un: abs(un[0] - target))
        resid = abs(u * lead / scale[k] - 1)
        if resid > 1e-6:
            print("WARNING: E_%d leading coefficient is %.2e off the "
                  "{1,-1,i,-i} phase lattice; leading term will not be "
                  "exactly 1" % (k, resid))
        mu[k], mu_name[k] = u, name
        cc_sign[k] = "+" if abs(u.imag) < 0.5 else "-"  # sign of mu^2

    stats = {}   # (d,k) -> dict
    blocks = {}  # (d,k) -> latex
    kept_diag_imag_mu = False  # any modulus term kept where mu is imaginary?
    for k in range(3):
        s = scale[k]
        for d in (1, 2, 3):
            tl = [(kind, A, B, mu[k] * c / s) for kind, A, B, c in terms[(d, k)]]
            total = sum(mass_weight(kind) * abs(c) ** 2
                        for kind, _, _, c in tl)
            nonzero = [t for t in tl if abs(t[3]) > 1e-12]
            kept = [t for t in nonzero if abs(t[3]) >= args.thr]
            kept_mass = sum(mass_weight(kind) * abs(c) ** 2
                            for kind, _, _, c in kept)
            stats[(d, k)] = dict(
                n_nonzero=len(nonzero), n_kept=len(kept),
                mass_dropped=1.0 - kept_mass / total if total > 0 else 0.0,
            )
            if cc_sign[k] == "-" and any(t[0] == "diag" for t in kept):
                kept_diag_imag_mu = True
            blocks[(d, k)] = emit_block(k, d, cluster_terms(kept),
                                        cc_sign=cc_sign[k])

    # per-degree share of total squared coeff norm (d=1..4)
    share = {}
    for k in range(3):
        tot = {d: sum(mass_weight(kind) * abs(c) ** 2
                      for kind, _, _, c in terms[(d, k)]) for d in (1, 2, 3, 4)}
        z = sum(tot.values())
        share[k] = {d: tot[d] / z for d in (1, 2, 3, 4)}

    # ---------------- stdout summary ----------------
    print("Noise threshold: drop |c| < %g * max|c| (per equation, max over "
          "d<=3; c = Hermitian matrix entries)" % args.thr)
    for k in range(3):
        print("  E_%d: scale s_%d = %.6f, phase mu_%d = %s (displayed c = "
              "mu * H / s; completion sign '%s c.c.'; raw threshold = %.2e)"
              % (k, k, scale[k], k, mu_name[k], cc_sign[k],
                 args.thr * scale[k]))
        for d in (1, 2, 3):
            st = stats[(d, k)]
            print("    d=%d: kept %4d / %4d nonzero entries; dropped mass = "
                  "%.4g%% of block sum c^2"
                  % (d, st["n_kept"], st["n_nonzero"],
                     100 * st["mass_dropped"]))
        print("    block shares of sum c^2 (d=1,2,3,4): "
              + ", ".join("%.1f%%" % (100 * share[k][d]) for d in (1, 2, 3, 4)))

    # ---------------- LaTeX document ----------------
    L = [PREAMBLE.replace("@SOURCE@", source).replace("@DATE@", date)]
    diag_note = ("" if kept_diag_imag_mu else
                 " (At the chosen threshold no such terms survive, and the "
                 "real-$\\mu$ equations keep real modulus coefficients.)")
    setup = (SETUP
             .replace("@S0@", "%.6f" % scale[0])
             .replace("@S1@", "%.6f" % scale[1])
             .replace("@S2@", "%.6f" % scale[2])
             .replace("@MU0@", mu_name[0])
             .replace("@MU1@", mu_name[1])
             .replace("@MU2@", mu_name[2])
             .replace("@PROVENANCE@", provenance)
             .replace("@DIAGNOTE@", diag_note)
             .replace("@THRPCT@", "%g" % (100 * args.thr))
             .replace("@THR@", "%g" % args.thr))
    L.append(setup)
    for k in range(3):
        row = ["$E_%d$" % k, "%.4f" % scale[k]]
        row += ["%d/%d" % (stats[(d, k)]["n_kept"], stats[(d, k)]["n_nonzero"])
                for d in (1, 2, 3)]
        row += ["%.3g\\%%" % (100 * stats[(d, k)]["mass_dropped"])
                for d in (1, 2, 3)]
        row += ["%.1f\\%%" % (100 * share[k][d]) for d in (1, 2, 3, 4)]
        L.append(" & ".join(row) + r" \\")
    L.append(TABLE_TAIL.replace("@SOURCE@", source))

    for d in (1, 2, 3):
        L.append("\\section*{Degree $(%d,%d)$ parts}\n" % (d, d))
        L.append("{\\small These blocks enter the full equations as "
                 "$\\hat E_k^{(%d)} / \\lVert z\\rVert^{%d}$; the division "
                 "is suppressed in the display.}\n" % (d, 2 * d))
        for k in range(3):
            st = stats[(d, k)]
            L.append("\\subsection*{$\\hat E_%d^{(%d)}$ --- %s}\n"
                     % (k, d, CHAR_DESC[k]))
            L.append(blocks[(d, k)])
            L.append("\n{\\small (%d of %d nonzero entries shown; omitted "
                     "entries carry %.3g\\%% of this block's $\\sum c^2$.)}\n"
                     % (st["n_kept"], st["n_nonzero"],
                        100 * st["mass_dropped"]))

    L.append(r"\end{document}")

    args.out.write_text("\n".join(L))
    print("\nWrote %s" % args.out)

    if args.no_pdf:
        return
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        print("pdflatex not on PATH; skipping compile. Build with:\n"
              "  pdflatex -interaction=nonstopmode %s" % args.out.name)
        return
    out_dir = args.out.resolve().parent
    proc = subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", args.out.name],
        cwd=out_dir, capture_output=True, text=True)
    pdf = args.out.with_suffix(".pdf")
    if proc.returncode == 0 and pdf.exists():
        print("Wrote %s" % pdf)
    else:
        tail = "\n".join(proc.stdout.splitlines()[-15:])
        print("pdflatex failed (rc=%d). Last output:\n%s" % (proc.returncode, tail))


if __name__ == "__main__":
    main()
