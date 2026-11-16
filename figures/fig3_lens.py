"""Figure 3 (methods paper): the lens.

Loadings are q = Ỹᵀ D t̄. The annotation table Ỹ and group sizes D are fixed by the
design, so even random group means t̄ produce loadings concentrated in a few directions.

Panels
  a  the lens as a machine: random group means in, loadings out
  b  loadings from many random draws, with the lens's leading direction
  c  eigenvalues of the lens matrix and its participation ratio
  d  congruence of an observed loading with the random-draw loadings
"""
import numpy as np
from toy import *

rng = np.random.default_rng(1)
M, Q = Y_RAW.shape
Yc, ev, evec, nu = lens(Y_RAW, N_REPORTS)
e1 = evec[:, 0]
R = random_loadings(Y_RAW, N_REPORTS, 5000, rng)
R = R * np.sign(R @ e1)[:, None]            # align signs with the leading direction
obs = decompose(Y_RAW, T_RAW, N_REPORTS)["q"]
obs = obs / np.linalg.norm(obs)
obs = obs * np.sign(obs @ e1)
cong = np.abs(R @ obs)
cong_med, cong_95 = np.median(cong), np.quantile(cong, 0.95)

W, H = 1400, 880
s = SVG(W, H)

# ---------------------------------------------------------------- panel a
s.label(20, 40, "a", "The lens: random group means in, loadings out")
s.text(46, 62, "loading = centred annotation table (transposed) × group sizes × group means", 12, INK2)
s.text(46, 77, "toy data", 11, MUTED, style="italic")
y0 = 130
t_demo = wcentre(np.random.default_rng(4).standard_normal(M), N_REPORTS)
# input: random means as dots on short axes
s.text(46, y0 - 18, "random group means", 11.5, INK2)
s.text(46, y0 - 4, "(no relation to annotations)", 10.5, MUTED)
heights = N_REPORTS * 0.6 + 4
ys, y = [], y0 + 8
for m in range(M):
    ys.append(y)
    y += heights[m] + 4
for m, d in enumerate(DRUGS):
    yc = ys[m] + heights[m] / 2
    s.line(60, yc, 170, yc, RULE)
    s.circle(115 + 40 * t_demo[m] / np.abs(t_demo).max(), yc, 5, color(d))
# lens box
lx = 250
s.line(178, (ys[0] + ys[-1] + heights[-1]) / 2, lx - 16, (ys[0] + ys[-1] + heights[-1]) / 2, INK2, 1.6, arrow=True)
box_top, box_bot = ys[0] - 10, ys[-1] + heights[-1] + 10
s.rect(lx - 10, box_top, Q * 16 + 20, box_bot - box_top, "#f4f6f8", stroke=RULE, rx=6)
for m, d in enumerate(DRUGS):
    for j in range(Q):
        s.rect(lx + j * 16, ys[m], 15, heights[m], color(d), 0.08 + 0.85 * Y_RAW[m, j])
s.text(lx + Q * 8, box_top - 26, "lens", 13, INK, anchor="middle", weight="700")
s.text(lx + Q * 8, box_top - 10, "annotation table, row height = group size", 10.5, INK2, anchor="middle")
s.text(lx + Q * 8, box_bot + 18, "fixed by the design", 11, MUTED, anchor="middle")
# output: one loading
ox = lx + Q * 16 + 60
mid = (box_top + box_bot) / 2
s.line(lx + Q * 16 + 14, mid, ox - 10, mid, INK2, 1.6, arrow=True)
q_demo = Yc.T @ (N_REPORTS * t_demo)
q_demo = q_demo / np.abs(q_demo).max()
s.text(ox, box_top - 10, "loading", 11.5, INK2)
s.line(ox, mid, ox + Q * 20, mid, INK2)
for j in range(Q):
    s.rect(ox + j * 20 + 2, mid, 16, -90 * q_demo[j], ACCENT, 0.9)
for j in range(Q):
    s.text(ox + j * 20 + 10, box_bot - 4, RECEPTORS[j], 10.5, INK2, anchor="middle")
s.text(ox, box_bot + 18, "one draw", 11, MUTED)

# ---------------------------------------------------------------- panel b
bx = 760
s.label(bx, 40, "b", "Noise in, structure out")
s.text(bx + 26, 62, "loadings from 40 random draws of the group means", 12, INK2)
px, py, pw, ph = bx + 40, 110, 560, 300
zero = py + ph / 2
s.line(px, zero, px + pw, zero, INK2)
step = pw / (Q - 1)
def pt(j, v): return px + j * step, zero - (ph / 2 - 10) * v
for k in range(40):
    d = " ".join(f"{'M' if j == 0 else 'L'}{pt(j, R[k, j])[0]:.1f},{pt(j, R[k, j])[1]:.1f}" for j in range(Q))
    s.path(d, ACCENT, 1, opacity=0.18)
d = " ".join(f"{'M' if j == 0 else 'L'}{pt(j, e1[j])[0]:.1f},{pt(j, e1[j])[1]:.1f}" for j in range(Q))
s.path(d, INK, 2.6)
for j in range(Q):
    s.text(px + j * step, py + ph + 22, RECEPTORS[j], 11, INK2, anchor="middle")
s.line(px + 10, py + ph + 46, px + 34, py + ph + 46, INK, 2.6)
s.text(px + 40, py + ph + 50, "leading direction of the lens", 11.5, INK2)
s.line(px + 250, py + ph + 46, px + 274, py + ph + 46, ACCENT, 1.5, opacity=0.4)
s.text(px + 280, py + ph + 50, "random draws (unit length, sign-aligned)", 11.5, INK2)

# ---------------------------------------------------------------- panel c
cy = 520
s.label(20, cy, "c", "How many directions the lens allows")
s.text(46, cy + 22, "eigenvalues of the lens matrix (share of total)", 12, INK2)
share = ev / ev.sum()
bxs, base = 60, cy + 230
for j in range(Q):
    h = 170 * share[j]
    s.rect(bxs + j * 46, base, 34, -h, ACCENT if j == 0 else MUTED, 0.85)
    s.text(bxs + j * 46 + 17, base + 18, f"{j + 1}", 11, INK2, anchor="middle")
    if share[j] > 0.005:
        s.text(bxs + j * 46 + 17, base - h - 6, f"{100 * share[j]:.0f}%", 11, INK2, anchor="middle")
s.line(bxs - 6, base, bxs + Q * 46, base, INK2)
s.text(bxs + Q * 46 / 2, base + 38, "lens direction", 11, INK2, anchor="middle")
s.text(470, cy + 90, f"participation ratio ν = {nu:.1f}", 14, INK, weight="700")
s.text(470, cy + 112, "effective number of annotation", 12, INK2)
s.text(470, cy + 128, f"directions, out of {Q} annotations", 12, INK2)
s.text(470, cy + 156, f"first direction: {100 * share[0]:.0f}% of the lens", 12, INK2)

# ---------------------------------------------------------------- panel d
dx = 760
s.label(dx, cy, "d", "An observed loading against the lens null")
s.text(dx + 26, cy + 22, "congruence (|cosine|) of the loading in panel 2b with 5,000 random-draw loadings", 12, INK2)
hx, hy, hw, hh = dx + 40, cy + 50, 560, 170
bins = np.linspace(0, 1, 41)
cnt, _ = np.histogram(cong, bins)
bw = hw / (len(bins) - 1)
for i, c in enumerate(cnt):
    h = hh * c / cnt.max()
    s.rect(hx + i * bw, hy + hh, bw - 1, -h, MUTED, 0.7)
s.line(hx, hy + hh, hx + hw, hy + hh, INK2)
for v in [0, 0.25, 0.5, 0.75, 1.0]:
    s.text(hx + hw * v, hy + hh + 18, f"{v:g}", 10.5, INK2, anchor="middle")
for v, lab in [(cong_med, f"median {cong_med:.2f}"), (cong_95, f"95th pct {cong_95:.2f}")]:
    s.line(hx + hw * v, hy - 4, hx + hw * v, hy + hh, INK, 1.6, dash="4 3")
s.text(hx + hw * cong_med - 6, hy + 8, f"median {cong_med:.2f}", 11, INK, anchor="end")
s.text(hx + hw * cong_95 - 6, hy - 8, f"95th pct {cong_95:.2f}", 11, INK, anchor="end")
s.rect(hx + hw * 0.85, hy + hh + 30, hw * 0.10, 8, "#c9a227", 0.5)
s.rect(hx + hw * 0.95, hy + hh + 30, hw * 0.05, 8, "#2e7d4f", 0.5)
s.text(hx + hw * 0.84, hy + hh + 54, "0.85–0.95 fair similarity · ≥ 0.95 near-equal", 10.5, INK2, anchor="end")
s.text(hx, hy + hh + 72, "benchmarks: Lorenzo-Seva and ten Berge (2006)", 10.5, MUTED)
s.text(hx, hy + hh + 100, f"Many random draws come close to the observed loading (5% exceed {cong_95:.2f}),", 12, INK)
s.text(hx, hy + hh + 116, "so the loading's shape alone is no evidence of a relation with the annotations.", 12, INK)

s.save("figure3_lens.svg")
print(f"nu = {nu:.2f}; first share = {share[0]:.3f}; congruence median {cong_med:.3f}, 95th {cong_95:.3f}")
