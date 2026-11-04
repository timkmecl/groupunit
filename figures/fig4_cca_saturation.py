"""Figure 4: CCA cannot see the annotations once they span all group differences.

Simulated as in Section 8.1, at the reference scenario with M = 21 groups (as
Figure 5): log-normal group sizes with Kish's effective number at 40% of M,
heterogeneity ratio h = 1, p = 10 features, q = M - 1 = 20 annotations. The
first annotation drives two features (group-level R² = 0.6); the others are
unrelated.

Panels
  a  at full rank (r = M - 1), canonical correlations are identical for the real annotation
     table, the same profiles shuffled among groups, random numbers, and group indicators
  b  first canonical correlation for the real table against 500 group-level shuffles,
     as the annotation ratio kappa = r / (M - 1) grows; the line marks discriminant analysis (LDA)

The seed is the first for which the relation is detected at r = 1, so that
panel b shows a real relation being lost as kappa grows.

    python figures/fig4_cca_saturation.py
"""
import os
import sys

import numpy as np

from svg import ACCENT, INK, INK2, MUTED, RULE, SVG, save_png  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "simulations"))
from common import Design  # noqa: E402

M, P = 21, 10
QMAX = M - 1
DESIGN = Design(M=M, kappa=1.0, p=P, h=1, signal=0.6)
SEED = 1

def basis(A, tol=1e-10):
    A = A - A.mean(0)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > tol * s.max()]


def cancorr(Xi, Yi):
    return np.clip(np.linalg.svd(basis(Xi).T @ basis(Yi), compute_uv=False), 0, 1)


def cc(Yt):                       # canonical correlations of X with the inherited table Yt[G]
    return np.clip(np.linalg.svd(QX.T @ basis(Yt[G]), compute_uv=False), 0, 1)


def first_p(r, n_null=500):
    obs = cc(Yt_full[:, :r])[0]
    null = np.array([cc(Yt_full[rng.permutation(M), :r])[0] for _ in range(n_null)])
    return obs, null, (1 + (null >= obs - 1e-9).sum()) / (1 + n_null)   # tolerance: ties at full rank


while True:   # first seed at which the relation is detected with r = 1
    rng = np.random.default_rng(SEED)
    X, G, Yt_full, n = DESIGN.draw(rng)
    N = len(G)
    QX = basis(X)
    if first_p(1, 199)[2] <= 0.05:
        break
    SEED += 1


# ---------------------------------------------------------------- panel a data
tables = {
    "real annotations": Yt_full,
    "shuffled among groups": Yt_full[rng.permutation(M)],
    "random numbers": rng.standard_normal((M, QMAX)),
    "group indicators (LDA)": np.eye(M),
}
rho = {k: cc(v) for k, v in tables.items()}
maxdiff = max(np.abs(rho[k] - rho["real annotations"]).max() for k in rho)

# ---------------------------------------------------------------- panel b data
R_LIST = [1, 4, 10, 18, 20]
lda1 = rho["group indicators (LDA)"][0]
res = []
for r in R_LIST:
    obs, null, pval = first_p(r)
    res.append((r, obs, null, pval))

# ---------------------------------------------------------------- draw
W, H = 1400, 580
s = SVG(W, H)
MARK = {"real annotations": ("#1d5d7f", "circle", 7), "shuffled among groups": ("#CC79A7", "square", 6),
        "random numbers": ("#E69F00", "diamond", 6), "group indicators (LDA)": ("#1b2128", "cross", 6)}

def marker(x, y, kind, col, size):
    if kind == "circle":
        s.circle(x, y, size, "none", stroke=col)
        s.o[-1] = s.o[-1].replace('stroke="' + col + '"', f'stroke="{col}" stroke-width="2"')
    elif kind == "square":
        s.rect(x - size, y - size, 2 * size, 2 * size, "none", stroke=col, sw=2)
    elif kind == "diamond":
        s.path(f"M{x},{y-size-1} L{x+size+1},{y} L{x},{y+size+1} L{x-size-1},{y} Z", col, 2)
    else:
        s.line(x - size, y - size, x + size, y + size, col, 2)
        s.line(x - size, y + size, x + size, y - size, col, 2)

# panel a
s.label(20, 40, "a", "At full rank, any annotation table gives the same CCA")
s.text(46, 62, f"simulated as in Section 8.1: M = {M} groups, N = {N:,} individuals, p = {P} features, q = {QMAX} annotations (r = M − 1)", 12, INK2)
px, py, pw, ph = 80, 100, 520, 330
s.rect(px, py, pw, ph, "none", stroke=RULE)
K = P
def X_(i): return px + 30 + i * (pw - 60) / (K - 1)
YMAX = 0.5   # canonical correlations stay below 0.5 at the reference scenario's between-group share
def Y_(v): return py + ph - v / YMAX * ph
for v in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    s.line(px, Y_(v), px + pw, Y_(v), RULE, 1, opacity=0.6)
    s.text(px - 8, Y_(v) + 4, f"{v:g}", 11, INK2, anchor="end")
for i in range(K):
    s.text(X_(i), py + ph + 18, f"{i+1}", 11, INK2, anchor="middle")
s.text(px + pw / 2, py + ph + 40, "canonical dimension", 11.5, INK2, anchor="middle")
s.text(px - 44, py + ph / 2, "ρ", 14, INK2, anchor="middle", style="italic", fam="Georgia, serif")
for k in tables:
    col, kind, size = MARK[k]
    for i in range(K):
        marker(X_(i), Y_(rho[k][i]), kind, col, size)
lx, ly = px + pw + 30, py + 20
for j, k in enumerate(tables):
    col, kind, size = MARK[k]
    marker(lx + 8, ly + j * 28, kind, col, size)
    s.text(lx + 24, ly + j * 28 + 4, k, 12, INK)
s.text(lx, ly + 4 * 28 + 18, "all four overlap exactly:", 12, INK, weight="700")
s.text(lx, ly + 4 * 28 + 36, f"largest difference {maxdiff:.0e}", 12, INK2)
s.text(lx, ly + 4 * 28 + 66, "Canonical correlations depend on the", 11.5, INK2)
s.text(lx, ly + 4 * 28 + 82, "annotations only through the space", 11.5, INK2)
s.text(lx, ly + 4 * 28 + 98, "they span. At r = M − 1 that space is", 11.5, INK2)
s.text(lx, ly + 4 * 28 + 114, "every between-group difference.", 11.5, INK2)

# panel b
bx = 900 + 20
s.label(bx - 20 + 20, 40, "b", "The real table stands out only when κ is small")
s.text(bx + 26, 62, "first canonical correlation: real table vs 500 shuffles of its rows among groups", 12, INK2)
s.text(bx + 26, 77, "r = 1: the annotation that drives the features; larger r adds unrelated annotation columns", 11, MUTED)
qx, qy, qw, qh = bx + 30, 100, 420, 330
s.rect(qx, qy, qw, qh, "none", stroke=RULE)
lo, hi = 0.0, 1.0
lo = min(min(r[2].min() for r in res), min(r[1] for r in res)) - 0.03
hi = max(lda1, max(r[1] for r in res)) + 0.03
def Yb(v): return qy + qh - (v - lo) / (hi - lo) * qh
for v in np.arange(np.ceil(lo * 10) / 10, hi, 0.1):
    s.line(qx, Yb(v), qx + qw, Yb(v), RULE, 1, opacity=0.6)
    s.text(qx - 8, Yb(v) + 4, f"{v:.1f}", 11, INK2, anchor="end")
s.line(qx, Yb(lda1), qx + qw, Yb(lda1), INK, 1.6, dash="5 4")
s.text(qx + 6, Yb(lda1) - 6, "LDA on group labels", 11, INK)
colw = qw / len(res)
for i, (r, obs, null, pval) in enumerate(res):
    cxm = qx + colw * (i + 0.5)
    q5, q50, q95 = np.quantile(null, [0.05, 0.5, 0.95])
    jit = np.random.default_rng(i).uniform(-colw * 0.28, colw * 0.28, len(null))
    for v, j in zip(null[::2], jit[::2]):
        s.circle(cxm + j, Yb(v), 1.6, MUTED, 0.35)
    s.rect(cxm - colw * 0.3, Yb(q95), colw * 0.6, Yb(q5) - Yb(q95), "none", stroke=INK2, sw=1)
    s.line(cxm - colw * 0.3, Yb(q50), cxm + colw * 0.3, Yb(q50), INK2, 1.4)
    s.circle(cxm, Yb(obs), 7, "#1d5d7f", 0.95)
    kappa = r / (M - 1)
    s.text(cxm, qy + qh + 18, f"{kappa:.2f}", 11.5, INK2, anchor="middle")
    s.text(cxm, qy + qh + 33, f"r = {r}", 10.5, MUTED, anchor="middle")
    ptxt = f"p = {pval:.3f}" if pval > 1.5 / 501 else "p ≤ 0.002"
    s.text(cxm, qy - 8, ptxt, 10.5, INK2 if pval >= 0.05 else "#1d5d7f", anchor="middle",
           weight="400" if pval >= 0.05 else "700")
s.text(qx + qw / 2, qy + qh + 54, "annotation ratio κ = r / (M − 1) (categories, not to scale)", 11.5, INK2, anchor="middle")
s.text(qx - 44, qy + qh / 2, "ρ₁", 14, INK2, anchor="middle", style="italic", fam="Georgia, serif")
ly2 = qy + qh + 78
s.circle(qx + 8, ly2, 6, "#1d5d7f", 0.95)
s.text(qx + 20, ly2 + 4, "real table", 11.5, INK2)
s.circle(qx + 110, ly2, 2.5, MUTED, 0.6)
s.rect(qx + 120, ly2 - 6, 14, 12, "none", stroke=INK2)
s.text(qx + 142, ly2 + 4, "shuffles (box: 5th–95th percentile)", 11.5, INK2)
s.text(qx, ly2 + 30, "The data contain a real relation with one annotation. As κ → 1 the null rises", 11.5, INK)
s.text(qx, ly2 + 46, "to the LDA value and the test loses it; at κ = 1 every shuffle equals the real result.", 11.5, INK)

os.chdir(HERE)
s.save("figure4_cca_saturation.svg")
save_png("figure4_cca_saturation.svg")
print("seed", SEED, "n =", n.tolist(), "N =", N)
print(f"max difference at full rank: {maxdiff:.2e}; LDA rho1 = {lda1:.3f}")
for r, obs, null, pval in res:
    print(f"r={r:2d} kappa={r/(M-1):.2f} obs={obs:.3f} null median={np.median(null):.3f} p={pval:.4f}")
