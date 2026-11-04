"""Figure 5: regimes of a design with group-level annotations.

Simulated as in Section 8.1, at the reference scenario with M = 21 groups (as
Figure 4): log-normal group sizes with Kish's effective number at 40% of M,
heterogeneity ratio h = 1, no relation (β = 0), group means weighted by size.

Panels
  a  group-level CCA: mean first canonical correlation between the first k principal
     components of the group means and r annotation dimensions, among M - 1 = 20
     between-group dimensions. Above k + r = M - 1 at least k + r - (M - 1)
     correlations equal 1 exactly.
  b  the annotation ratio kappa = r / (M - 1): the Beta(r/2, (M-1-r)/2) null of the
     group-level R² of a score on the annotations (Result 3b, band and mean), and the
     simulated mean R² of the first principal component of the group means under size
     weights (dots); published designs marked. Result 3b assumes the weighted group
     means point in a random direction. The optimal weights of Section 5 meet that
     (printed, on the line); size weights with unequal sizes do not, and the null R²
     then exceeds kappa.

    python figures/fig5_regimes.py
"""
import os
import sys
from math import exp, lgamma  # noqa: F401

import numpy as np

import groupunit as gu
from svg import ACCENT, INK, INK2, MUTED, RULE, SVG, save_png  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "simulations"))
from common import Design  # noqa: E402

rng = np.random.default_rng(5)
DEGEN = "#0b1117"   # degenerate cells (rho = 1): black, the end of the colour range
M = 21
D = M - 1                   # between-group dimensions
SIMS = 1000
DESIGN = Design(M=M, kappa=1.0, p=D, h=1)


def weighted_bases(Xbar, Y, w):
    """Principal components of the group means and a basis of the annotations, both weighted by w."""
    sw = np.sqrt(w / w.sum())[:, None]
    U = np.linalg.svd(sw * gu.wcentre(Xbar, w), full_matrices=False)[0][:, :D]
    Q = np.linalg.qr(sw * gu.wcentre(Y, w))[0]          # nested: Q[:, :r] spans the first r annotations
    return U, Q


n = DESIGN.sizes()
WEIGHTS = {"size": n.astype(float), "optimal": gu.optimal_weights(n, DESIGN.tau2, 1.0),
           "exchangeable": gu.exchangeable_weights(n, DESIGN.tau2, 1.0)}
rho = np.zeros((D, D))
r2 = {name: np.zeros((SIMS, D)) for name in WEIGHTS}
for i in range(SIMS):
    X, groups, Y, _ = DESIGN.draw(rng)
    Xbar = gu.group_means(X, groups)
    for name, w in WEIGHTS.items():
        U, Q = weighted_bases(Xbar, Y, w)
        r2[name][i] = np.cumsum((Q.T @ U[:, 0]) ** 2)    # R² of the first PC on r = 1..D annotations
    U, Q = weighted_bases(Xbar, Y, WEIGHTS["size"])       # panel a: size weights, as individual-level CCA
    if i < 300:
        for k in range(1, D + 1):
            for r in range(1, D + 1 - k + 1):
                rho[k - 1, r - 1] += np.linalg.svd(U[:, :k].T @ Q[:, :r], compute_uv=False)[0] / 300
grid = rho


def ramp(v):
    """light grey-blue -> deep blue for 0..1"""
    c0, c1 = np.array([238, 242, 246]), np.array([22, 64, 96])
    c = (c0 + (c1 - c0) * v ** 1.3).astype(int)
    return f"rgb({c[0]},{c[1]},{c[2]})"


def beta_ppf(a, b, qs, n=4000):
    x = np.linspace(1e-6, 1 - 1e-6, n)
    logpdf = (a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - (lgamma(a) + lgamma(b) - lgamma(a + b))
    cdf = np.cumsum(np.exp(logpdf)); cdf /= cdf[-1]
    return [x[np.searchsorted(cdf, q)] for q in qs]


W, H = 1400, 640
s = SVG(W, H)

# ---------------------------------------------------------------- panel a
s.label(20, 40, "a", "Group-level CCA: where chance alone gives large correlations")
s.text(46, 62, f"mean first canonical correlation under no association, simulated as in Section 8.1 (M = {M}, size weights)", 12, INK2)
cs = 22
gx, gy = 150, 100
for ki in range(D):
    for ri in range(D):
        v = grid[ki, ri]
        x, y = gx + ri * cs, gy + (D - 1 - ki) * cs
        if ki + 1 + ri + 1 > D:
            continue
        else:
            s.rect(x, y, cs - 1, cs - 1, ramp(v))
# boundary k + r = D
pts = []
for ri in range(D):
    k_max = D - (ri + 1)              # last non-degenerate k
    pts.append((gx + ri * cs, gy + (D - k_max) * cs))
    pts.append((gx + (ri + 1) * cs, gy + (D - k_max) * cs))
poly = [(gx, gy), (gx + D * cs, gy)] + pts[::-1]
s.path("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in poly) + " Z", "none", 0, fill=DEGEN)   # degenerate region, solid
s.path("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts), "#ffffff", 2.6)
for i in [1, 5, 10, 15, 20]:
    s.text(gx + (i - 0.5) * cs, gy + D * cs + 16, f"{i}", 10.5, INK2, anchor="middle")
    s.text(gx - 8, gy + (D - i + 0.5) * cs + 4, f"{i}", 10.5, INK2, anchor="end")
s.text(gx + D * cs / 2, gy + D * cs + 36, "annotation dimensions r", 11.5, INK2, anchor="middle")
s.text(gx + D * cs / 2, gy + D * cs + 52, f"(κ = r / {D})", 10.5, MUTED, anchor="middle")
s.text(gx - 30, gy + D * cs / 2 - 8, "retained", 11.5, INK2, anchor="end")
s.text(gx - 30, gy + D * cs / 2 + 8, "feature dims k", 11.5, INK2, anchor="end")
s.text(gx + D * cs - 8, gy + 30, "k + r > M − 1:", 12, "#ffffff", anchor="end", weight="700")
s.text(gx + D * cs - 8, gy + 46, "some ρ = 1 exactly", 12, "#ffffff", anchor="end")
# examples
def mark(k, r, label, dx=10, dy=4, col="#E69F00"):
    x, y = gx + (r - 0.5) * cs, gy + (D - k + 0.5) * cs
    s.circle(x, y, 6.5, "none", stroke=col)
    s.o[-1] = s.o[-1].replace(f'stroke="{col}"', f'stroke="{col}" stroke-width="2.5"')
    return x, y
x1, y1 = mark(3, 3, "")
x2, y2 = mark(5, 19, "")
# colour legend
lx = gx + D * cs + 30
for i in range(100):
    s.rect(lx, gy + 200 - i * 2, 16, 2.2, ramp(i / 99))
s.text(lx + 22, gy + 204, "0", 10.5, INK2)
s.text(lx + 22, gy + 6, "1", 10.5, INK2)
s.text(lx, gy - 10, "mean first ρ", 11, INK2)
s.rect(lx, gy + 220, 16, 16, DEGEN)
s.text(lx + 22, gy + 232, "degenerate (ρ = 1)", 10.5, INK2)
ex_y = gy + 280
s.circle(lx + 8, ex_y, 6.5, "none", stroke="#E69F00")
s.o[-1] = s.o[-1].replace('stroke="#E69F00"', 'stroke="#E69F00" stroke-width="2.5"')
s.text(lx + 22, ex_y - 6, "3 feature PCs, 3 annotation PCs:", 10.5, INK2)
s.text(lx + 22, ex_y + 8, f"mean first ρ = {grid[2, 2]:.2f}", 10.5, INK, weight="700")
s.text(lx + 22, ex_y + 34, "5 feature PCs, 19 receptors:", 10.5, INK2)
s.text(lx + 22, ex_y + 48, "first ρ = 1 by construction", 10.5, INK, weight="700")
s.circle(lx + 8, ex_y + 40, 6.5, "none", stroke="#E69F00")
s.o[-1] = s.o[-1].replace('stroke="#E69F00"', 'stroke="#E69F00" stroke-width="2.5"')

# ---------------------------------------------------------------- panel b
bx = 760
s.label(bx, 40, "b", "The annotation ratio decides what can be detected")
s.text(bx + 26, 62, "null distribution of the group-level R² of any score on the annotations", 12, INK2)
px, py, pw, ph = bx + 50, 110, 430, 330
s.rect(px, py, pw, ph, "none", stroke=RULE)
def X_(v): return px + v * pw
def Y_(v): return py + ph - v * ph
# regime labels (no hard boundaries: the regimes shade into one another)
s.text(X_(0.03), py + 18, "informative (small κ)", 11, INK2)
s.text(X_(0.97), py + ph - 30, "near-saturated", 11, INK2, anchor="end")
s.line(X_(1), py, X_(1), py + ph, INK, 2)
s.text(X_(1), py - 10, "saturated at κ = 1: CCA = LDA", 11, INK, anchor="end", weight="700")
for v in [0, 0.25, 0.5, 0.75, 1]:
    s.line(px, Y_(v), px + pw, Y_(v), RULE, 1, opacity=0.6)
    s.text(px - 8, Y_(v) + 4, f"{v:g}", 11, INK2, anchor="end")
    s.text(X_(v), py + ph + 18, f"{v:g}", 11, INK2, anchor="middle")
s.text(px + pw / 2, py + ph + 40, "annotation ratio κ = r / (M − 1)", 11.5, INK2, anchor="middle")
s.text(px - 40, py + ph / 2, "R²", 13, INK2, anchor="middle", style="italic", fam="Georgia, serif")
# band from Beta(r/2,(D-r)/2), D = 20
upper, lower, xs = [], [], []
for r in range(1, D):
    lo_, hi_ = beta_ppf(r / 2, (D - r) / 2, [0.05, 0.95])
    xs.append(r / D); lower.append(lo_); upper.append(hi_)
xs = [0] + xs + [1]; lower = [0] + lower + [1]; upper = [0] + upper + [1]
band = "M" + " L".join(f"{X_(x):.1f},{Y_(u):.1f}" for x, u in zip(xs, upper)) + " L" + \
       " L".join(f"{X_(x):.1f},{Y_(l):.1f}" for x, l in zip(xs[::-1], lower[::-1])) + " Z"
s.path(band, "none", 0, fill=ACCENT, opacity=1)
s.o[-1] = s.o[-1].replace('fill="#1d5d7f"', 'fill="#1d5d7f" fill-opacity="0.18"')
s.line(X_(0), Y_(0), X_(1), Y_(1), ACCENT, 2.2)
for j, t in enumerate(["Result 3b: line, mean (= κ);", "band, 5th–95th percentile"]):
    s.text(X_(0.97), Y_(0.36) + 15 * j, t, 11, ACCENT, anchor="end")
# simulated means under size weights; optimal weights fall on the theory line (printed only)
s.path("M" + " L".join(f"{X_(r / D):.1f},{Y_(r2['size'][:, r - 1].mean()):.1f}" for r in range(1, D)), "#D55E00", 1.4)
for r in range(1, D):
    s.circle(X_(r / D), Y_(r2["size"][:, r - 1].mean()), 3.2, "#D55E00", 1, stroke="#ffffff")
s.text(X_(0.03), Y_(0.82), "simulated, size weights", 11, "#D55E00", weight="700")
s.text(X_(0.03), Y_(0.82) + 15, "(nearer r / M_eff)", 10.5, "#D55E00")
# published designs, labelled directly
def pt(k, v):
    s.circle(X_(k), Y_(v), 6, "#E69F00", 0.95, stroke="#ffffff")
pt(3 / 20, 0.15)
s.line(X_(3 / 20) + 5, Y_(0.15) + 5, X_(0.24), Y_(0.04), INK2, 1)
s.text(X_(0.245), Y_(0.04) + 4, "Suresh et al. 2026, receptors reduced to 3 PCs", 11, INK)
pt(19 / 20, 0.95)
s.text(X_(0.95) - 12, Y_(0.95) + 4, "Suresh et al. 2026 (r = 19, M − 1 = 20)", 11, INK, anchor="end")
pt(1.0, 1.0)
s.line(X_(1), Y_(1), X_(1) + 18, Y_(1) + 40, INK2, 1)
s.text(X_(1) + 22, Y_(1) + 38, "Ballentine et al. 2022", 11, INK)
s.text(X_(1) + 22, Y_(1) + 53, "(r = M − 1 = 26)", 10.5, INK2)
s.text(X_(1) + 22, Y_(1) + 74, "Friedman and", 11, INK)
s.text(X_(1) + 22, Y_(1) + 89, "Ballentine 2024", 11, INK)
s.text(X_(1) + 22, Y_(1) + 104, "(r = M − 1 = 51)", 10.5, INK2)
s.text(px, py + ph + 70, "At κ near 1, a group-level fit explains almost all between-group variance of any score.", 11.5, INK)
s.text(px, py + ph + 86, "With size weights and unequal sizes the null mean is nearer r / M_eff than κ (dots, mean of", 11.5, INK)
s.text(px, py + ph + 102, f"{SIMS:,} data sets): the large groups dominate both the score and the annotations.", 11.5, INK)

os.chdir(HERE)
s.save("figure5_regimes.svg")
save_png("figure5_regimes.svg")
print(f"E[rho1] k=3,r=3: {grid[2,2]:.3f}; k=1,r=19: {grid[0,18]:.3f}")

print("Kish M_eff", round(gu.effective_number(n), 2))
for r in (1, 3, 10, 19):
    print(f"r={r}: size {r2['size'][:, r - 1].mean():.3f}, optimal {r2['optimal'][:, r - 1].mean():.3f}, exchangeable {r2['exchangeable'][:, r - 1].mean():.3f}, "
          f"Beta mean {r / D:.3f}")
