"""Figure 6: typicality versus pattern in a similarity comparison.

Toy data: M = 12 groups (A-L). Each group's features and annotations deviate from
a common profile by an amount u_m (how atypical the group is) in directions that
are independent between the two tables. The two similarity matrices therefore
share typicality but no pattern.

Similarities are negative squared Euclidean distances, for which the additive
model is exact, and the pattern is compared through residuals standardised for
each group's residual scale (`groupunit.compare_similarity`). Squared distances
amplify differences in u_m, so its spread is narrower than in the original
figure (log-sd 0.2 instead of 0.55); otherwise a single group dominates every panel.

Panels
  a  the two group similarity matrices (groups ordered by feature typicality), with group effects
  b  the usual Mantel comparison of all pairs
  c  typicality: group effects a_m, features vs annotations
  d  pattern: standardised residual similarities, features vs annotations

    python figures/fig6_typicality.py
"""
import os

import numpy as np

import groupunit as gu
from svg import ACCENT, INK, INK2, MUTED, RULE, SVG, save_png

M, DIM, NPERM = 12, 30, 5000
LET = [chr(65 + i) for i in range(M)]
rng = np.random.default_rng(3)


def make_table(u):
    base = rng.standard_normal(DIM)
    return base[None, :] + u[:, None] * rng.standard_normal((M, DIM))


u = np.exp(rng.normal(0, 0.2, M))                         # atypicality, shared by both tables
F = make_table(u)                                         # features (group means)
A = make_table(u * np.exp(rng.normal(0, 0.15, M)))        # annotations: same u, independent directions
SF, SA = gu.distance_similarity(F), gu.distance_similarity(A)

dF, dA = gu.typicality_pattern(SF), gu.typicality_pattern(SA)
aF, aA = dF.typicality, dA.typicality
EF, EA = gu.standardised_pattern(dF.pattern), gu.standardised_pattern(dA.pattern)
res = gu.compare_similarity(SF, SA, n_perm=NPERM, rng=0)
iu = np.triu_indices(M, 1)


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def ramp(v):
    c0, c1 = np.array([238, 242, 246]), np.array([22, 64, 96])
    c = (c0 + (c1 - c0) * np.clip(v, 0, 1) ** 1.2).astype(int)
    return f"rgb({c[0]},{c[1]},{c[2]})"


W, H = 1400, 465
s = SVG(W, H)

# ---------------------------------------------------------------- panel a
s.label(20, 40, "a", "Two similarity matrices")
s.text(46, 62, "similarity between groups, from features and from annotations", 12, INK2)
s.text(46, 77, "toy data; groups ordered by typicality in features", 11, MUTED, style="italic")
order = np.argsort(-aF)
cs = 15
amax = max(np.abs(aF).max(), np.abs(aA).max())
for k, (S, a, title) in enumerate([(SF, aF, "features"), (SA, aA, "annotations")]):
    lo, hi = S[iu].min(), S[iu].max()          # each matrix on its own scale (distances differ in size)
    gx, gy = 60 + k * 255, 120
    s.text(gx + M * cs / 2, gy - 12, title, 12, INK, anchor="middle", weight="700")
    for ii, i in enumerate(order):
        s.text(gx - 5, gy + ii * cs + 11, LET[i], 9.5, INK2, anchor="end")
        for jj, j in enumerate(order):
            if i == j:
                continue
            s.rect(gx + jj * cs, gy + ii * cs, cs - 1, cs - 1, ramp((S[i, j] - lo) / (hi - lo)))
        # group effect (typicality) bar to the right
        bx0 = gx + M * cs + 30
        w = 22 * a[i] / amax
        s.rect(bx0 if w >= 0 else bx0 + w, gy + ii * cs + 2, abs(w), cs - 5, ACCENT, 0.85)
    s.line(gx + M * cs + 30, gy - 2, gx + M * cs + 30, gy + M * cs, INK2)
    s.text(gx + M * cs + 30, gy + M * cs + 16, "group effect", 10.5, INK2, anchor="middle")
# colour key
ky = 120 + M * cs + 34
for i in range(60):
    s.rect(60 + i * 3, ky, 3.2, 10, ramp(i / 59))
s.text(60, ky + 24, "less similar", 10.5, INK2)
s.text(60 + 180, ky + 24, "more similar", 10.5, INK2, anchor="end")
s.text(300, ky + 8, "Groups that are typical in one table", 11, INK2)
s.text(300, ky + 23, "are typical in the other.", 11, INK2)


# ---------------------------------------------------------------- scatter helper
def scatter(x0, letter, title, sub, xs, ys, corr, ylab_text, xlab_text, labels=None, big=False):
    s.label(x0, 40, letter, title)
    s.text(x0 + 26, 62, sub, 12, INK2)
    px, py, pw, ph = x0 + 40, 104, 200, 200
    s.rect(px, py, pw, ph, "none", stroke=RULE)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    padx, pady = 0.1 * (xmax - xmin), 0.1 * (ymax - ymin)
    X = lambda v: px + pw * (v - xmin + padx) / (xmax - xmin + 2 * padx)
    Y = lambda v: py + ph - ph * (v - ymin + pady) / (ymax - ymin + 2 * pady)
    # the two most atypical groups and the most typical one; typical groups cluster tightly
    label_set = set(np.argsort(xs)[[0, 1, -1]])
    b1, b0 = np.polyfit(xs, ys, 1)
    s.line(X(xmin), Y(b0 + b1 * xmin), X(xmax), Y(b0 + b1 * xmax), MUTED, 1, dash="4 3")
    for k in range(len(xs)):
        s.circle(X(xs[k]), Y(ys[k]), 5.5 if big else 3.6, ACCENT, 0.8 if big else 0.55)
        if labels is not None and k in label_set:
            s.text(X(xs[k]) + 7, Y(ys[k]) + 4, labels[k], 10, INK2)
    strong = corr.p < 0.05
    s.text(px, py - 9, f"r = {corr.r:.2f}", 13, INK, weight="700")
    s.text(px + 62, py - 9, fmt_p(corr.p), 11.5, ACCENT if strong else INK2, weight="700" if strong else "400")
    s.text(px + pw / 2, py + ph + 20, xlab_text, 11, INK2, anchor="middle")
    s.text(px - 10, py + ph / 2, ylab_text, 11, INK2, anchor="end")


scatter(580, "b", "Mantel comparison", "all 66 pairs of groups", SF[iu], SA[iu], res.mantel,
        "annot.", "feature similarity")
scatter(840, "c", "Typicality", "group effects, 12 groups", aF, aA, res.typicality,
        "annot.", "feature group effect", labels=LET, big=True)
scatter(1100, "d", "Pattern", "standardised residuals, 66 pairs", EF[iu], EA[iu], res.pattern,
        "annot.", "feature residual")

ny = 372
s.text(620, ny, "Similarities are negative squared distances between group profiles. Permutation p-values", 12, INK)
s.text(620, ny + 17, "reassign groups (5,000 permutations). The Mantel correlation in b is carried by typicality (c):", 12, INK)
s.text(620, ny + 34, "atypical groups are atypical in both tables. Which groups resemble which (d) is not shared.", 12, INK)
s.text(620, ny + 60, "Decomposition: s_ij = μ + a_i + a_j + e_ij, a = group effect (typicality), e = pattern,", 11, INK2)
s.text(620, ny + 75, "here divided by each group's residual scale.", 11, INK2)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
s.save("figure6_typicality.svg")
save_png("figure6_typicality.svg")
print(f"Mantel r={res.mantel.r:.3f} {fmt_p(res.mantel.p)}; typicality r={res.typicality.r:.3f} "
      f"{fmt_p(res.typicality.p)}; pattern r={res.pattern.r:.3f} {fmt_p(res.pattern.p)}")
