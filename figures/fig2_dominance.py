"""Figure 2 (methods paper): a component's annotation loading rests on a few groups.

Panels
  a  each group's term n_m * t_m * y_m, shown as its three factors and the result
  b  the loading q as the stack of those terms, coloured by group
  c  absolute group shares and the effective number of groups behind the component
  d  the loading with and without the dominant group (component held fixed)
  e  the loading against the contrast between the two dominant groups
"""
import numpy as np
from toy import *

r = decompose(Y_RAW, T_RAW, N_REPORTS)
M, Q = Y_RAW.shape
i_mdma, i_dmt = DRUGS.index("Group C"), DRUGS.index("Group D")

# leave out the dominant group, component fixed (same drug means for the rest)
keep = [i for i in range(M) if i != i_mdma]
r_lo = decompose(Y_RAW[keep], T_RAW[keep], N_REPORTS[keep])

# contrast of standardised profiles of the two dominant groups
Yc = wcentre(Y_RAW, N_REPORTS)
sd = np.sqrt((N_REPORTS[:, None] * Yc ** 2).sum(0) / N_REPORTS.sum())
Z = Yc / sd
contrast = Z[i_dmt] - Z[i_mdma]
r_contrast = np.corrcoef(r["q"], contrast)[0, 1]

W, H = 1400, 840
s = SVG(W, H)

# ---------------------------------------------------------------- panel a
s.label(20, 40, "a", "Each group contributes one term")
s.text(46, 62, "term = group size × group mean score on the component × centred annotation profile", 12, INK2)
s.text(46, 77, "toy data", 11, MUTED, style="italic")
y0, rh = 128, 38
cx = dict(name=20, n=118, t=262, yrow=410, term=572)
s.text(cx["n"], y0 - 18, "n (individuals)", 11.5, INK2)
s.text(cx["t"] + 60, y0 - 18, "mean score on component", 11.5, INK2, anchor="middle")
s.text(cx["yrow"], y0 - 18, "ỹ (centred profile)", 11.5, INK2)
s.text(cx["term"], y0 - 18, "term", 11.5, INK2)
cw = 14
nmax = N_REPORTS.max()
tmax = np.abs(r["tc"]).max()
ymax = np.abs(r["Yc"]).max()
term_max = np.abs(r["terms"]).max()
# axis line for t
s.line(cx["t"] + 60, y0 - 8, cx["t"] + 60, y0 + M * rh - 4, RULE)
for m, d in enumerate(DRUGS):
    y = y0 + m * rh
    c = color(d)
    s.text(cx["name"], y + 19, d, 12, c if d in COLOR else INK2, weight="700" if d in COLOR else "400")
    bw = 90 * N_REPORTS[m] / nmax
    s.rect(cx["n"], y + 6, bw, 18, c, 0.45)
    s.text(cx["n"] + bw + 4, y + 19, f"{int(N_REPORTS[m])}", 10.5, INK2)
    s.text(cx["t"] - 18, y + 19, "×", 13, MUTED, anchor="middle")
    s.circle(cx["t"] + 60 + 50 * r["tc"][m] / tmax, y + 15, 5.5, c)
    s.text(cx["yrow"] - 18, y + 19, "×", 13, MUTED, anchor="middle")
    for j in range(Q):
        v = r["Yc"][m, j] / ymax
        fill = c if v > 0 else "#5f6b78"
        s.rect(cx["yrow"] + j * cw, y + 4, cw - 1, 22, fill, 0.08 + 0.85 * abs(v))
    s.text(cx["term"] - 22, y + 19, "=", 13, MUTED, anchor="middle")
    # term as mini bar chart around a zero line
    base = y + 15
    s.line(cx["term"], base, cx["term"] + Q * cw, base, RULE)
    for j in range(Q):
        h = -14 * r["terms"][m, j] / term_max
        s.rect(cx["term"] + j * cw + 2, base, cw - 4, h, c, 0.9)
s.text(cx["t"] + 60 - 55, y0 + M * rh + 12, "−", 11, MUTED, anchor="middle")
s.text(cx["t"] + 60 + 55, y0 + M * rh + 12, "+", 11, MUTED, anchor="middle")
s.text(cx["yrow"], y0 + M * rh + 12, "darker = further from the average profile; grey = below it", 10.5, MUTED)

# arrow to panel b
ay = y0 + M * rh / 2
s.line(cx["term"] + Q * cw + 14, ay, 760, ay, INK2, 1.6, arrow=True)
s.text((cx["term"] + Q * cw + 14 + 760) / 2, ay - 8, "sum over", 11, INK2, anchor="middle")
s.text((cx["term"] + Q * cw + 14 + 760) / 2, ay + 18, "groups", 11, INK2, anchor="middle")

# ---------------------------------------------------------------- panel b
bx, by = 790, 40
s.label(bx, by, "b", "The loading is the stack of terms")
s.text(bx + 26, by + 22, "annotation loading = sum of the terms in a, coloured by contributing group", 12, INK2)
cx0, cy0, bw, gap = bx + 40, 300, 50, 18
scale = 150 / max(np.abs(r["terms"]).clip(0).sum(0).max(), (-r["terms"]).clip(0).sum(0).max())
s.line(cx0 - 10, cy0, cx0 + Q * (bw + gap), cy0, INK2)
for j in range(Q):
    x = cx0 + j * (bw + gap)
    up, down = 0.0, 0.0
    order = np.argsort(-np.abs(r["terms"][:, j]))
    for m in order:
        v = r["terms"][m, j]
        hgt = v * scale
        if v >= 0:
            s.rect(x, cy0 - up - hgt, bw, hgt, color(DRUGS[m]), 0.9)
            up += hgt
        else:
            s.rect(x, cy0 + down, bw, -hgt, color(DRUGS[m]), 0.9)
            down += -hgt
    # net loading tick
    net = r["q"][j] * scale
    s.line(x - 4, cy0 - net, x + bw + 4, cy0 - net, INK, 2.2)
    s.text(x + bw / 2, cy0 + 168, RECEPTORS[j], 11.5, INK2, anchor="middle")
s.line(cx0 + 4, 124, cx0 + 26, 124, INK, 2.2)
s.text(cx0 + 32, 128, "net loading", 11, INK2)
s.text(cx0 - 14, cy0 - 130, "+", 12, MUTED, anchor="middle")
s.text(cx0 - 14, cy0 + 140, "−", 12, MUTED, anchor="middle")

# ---------------------------------------------------------------- panel c
c_y = 520
s.label(20, c_y, "c", "Group shares")
s.text(46, c_y + 22, "absolute shares of the loading", 12, INK2)
x, yb, tot_w = 46, c_y + 46, 380
order = np.argsort(-r["abs_share"])
for m in order:
    w = tot_w * r["abs_share"][m]
    s.rect(x, yb, w, 30, color(DRUGS[m]), 0.9, stroke="#ffffff", sw=1)
    if w > 34:
        s.text(x + w / 2, yb + 20, f"{100 * r['abs_share'][m]:.0f}%", 11.5, "#ffffff", anchor="middle", weight="700")
    x += w
lx = 46
for m in order[:3]:
    s.rect(lx, yb + 46, 10, 10, color(DRUGS[m]))
    s.text(lx + 14, yb + 55, DRUGS[m], 11.5, INK2)
    lx += 90
s.rect(lx, yb + 46, 10, 10, GREY_DRUG)
s.text(lx + 14, yb + 55, "others", 11.5, INK2)
s.text(46, yb + 96, f"Effective number of groups behind", 13, INK, weight="700")
s.text(46, yb + 114, f"this component: {r['m_eff']:.1f} (of {M})", 13, INK, weight="700")
s.text(46, yb + 136, f"Effective number from group sizes alone: {kish(N_REPORTS):.1f}", 11.5, INK2)

# ---------------------------------------------------------------- panel d
d_x = 490
s.label(d_x, c_y, "d", "Leave the dominant group out")
s.text(d_x + 26, c_y + 22, "same component, Group C removed (group means of the others unchanged)", 12, INK2)
base = c_y + 140
s.line(d_x + 20, base, d_x + 20 + Q * 52, base, INK2)
qa = r["q"] / np.abs(r["q"]).max()
qb = r_lo["q"] / np.abs(r["q"]).max()
for j in range(Q):
    x = d_x + 24 + j * 52
    s.rect(x, base, 20, -80 * qa[j], ACCENT, 0.9)
    s.rect(x + 22, base, 20, -80 * qb[j], ACCENT, 0.3, stroke=ACCENT, sw=1)
    s.text(x + 21, base + 104, RECEPTORS[j], 10.5, INK2, anchor="middle")
s.rect(d_x + 24, c_y + 276, 12, 12, ACCENT, 0.9)
s.text(d_x + 42, c_y + 286, "all groups", 11.5, INK2)
s.rect(d_x + 24, c_y + 296, 12, 12, ACCENT, 0.3, stroke=ACCENT)
s.text(d_x + 42, c_y + 306, f"without Group C (r = {np.corrcoef(r['q'], r_lo['q'])[0,1]:.2f} with all groups; "
       f"norm {np.linalg.norm(r_lo['q']) / np.linalg.norm(r['q']):.0%} of original)", 11.5, INK2)

# ---------------------------------------------------------------- panel e
e_x = 970
s.label(e_x, c_y, "e", "A contrast between two groups")
s.text(e_x + 26, c_y + 22, "loading vs the difference of the standardised profiles of D and C", 12, INK2)
px, py, pw, ph = e_x + 50, c_y + 44, 250, 190
s.rect(px, py, pw, ph, "none", stroke=RULE)
cmin, cmax = contrast.min(), contrast.max()
qmin, qmax = r["q"].min(), r["q"].max()
padc, padq = 0.12 * (cmax - cmin), 0.12 * (qmax - qmin)
def X(v): return px + pw * (v - cmin + padc) / (cmax - cmin + 2 * padc)
def Yp(v): return py + ph - ph * (v - qmin + padq) / (qmax - qmin + 2 * padq)
lab_idx = set(np.argsort(contrast)[[0, -2, -1]])
b1, b0 = np.polyfit(contrast, r["q"], 1)
s.line(X(cmin - padc / 2), Yp(b0 + b1 * (cmin - padc / 2)), X(cmax + padc / 2), Yp(b0 + b1 * (cmax + padc / 2)),
       MUTED, 1, dash="4 3")
for j in range(Q):
    s.circle(X(contrast[j]), Yp(r["q"][j]), 5, ACCENT, 0.85)
    if j in lab_idx:
        s.text(X(contrast[j]) + 8, Yp(r["q"][j]) + 4, RECEPTORS[j], 10.5, INK2)
s.text(px + pw / 2, py + ph + 20, "profile of D − profile of C (standardised)", 11, INK2, anchor="middle")
s.text(px - 12, py + ph / 2, "loading", 11, INK2, anchor="end")
s.text(px + 10, py + 18, f"r = {r_contrast:.2f}", 13, INK, weight="700")

s.save("figure2_dominance.svg")
print(f"shares {dict(zip(DRUGS, np.round(r['abs_share'], 3)))}")
print(f"M_eff,k = {r['m_eff']:.2f}; Kish = {kish(N_REPORTS):.2f}; r(contrast) = {r_contrast:.3f}")
