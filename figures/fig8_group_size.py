"""Figure 8: power depends on the number of groups, not on the number of individuals.

From simulation outcome 2 (second part): the population is fixed (τ² = 0.02,
group-level R² of the population means 0.5, κ = 0.5, Kish 40%) and only the
number of groups M and the mean group size n̄ vary; 300 data sets per cell.

Panels
  a  power of the PLS test (exchangeable weights) against individuals per group
  b  the same against the total number of individuals: equal power needs far
     fewer individuals when they are spread over more groups
  c  the same test with size weights: power barely grows with group size

    python simulations/run_all.py s2      # if missing
    python figures/fig8_group_size.py
"""
import os

import numpy as np
import pandas as pd

from svg import INK, INK2, MUTED, RULE, SVG, save_png

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "simulations", "results")
ALPHA = 0.05

# one hue, light to dark, for an ordered quantity (the number of groups)
SHADE = {10: "#8fb8d6", 20: "#4a8fc0", 50: "#1d5d7f", 100: "#0b2a3e"}
MARK = {10: "circle", 20: "square", 50: "triangle", 100: "diamond"}

df = pd.read_csv(os.path.join(RESULTS, "s2_group_size_replicates.csv"))
df["N"] = df.M * df.nbar


def rates(test):
    """(M, n̄, N, rate, low, high) with a 95% Wilson interval per cell."""
    rows = []
    for (M, nbar), d in df.groupby(["M", "nbar"]):
        m, p, z = len(d), (d[test] < ALPHA).mean(), 1.96
        centre = (p + z**2 / (2 * m)) / (1 + z**2 / m)
        half = z * np.sqrt(p * (1 - p) / m + z**2 / (4 * m**2)) / (1 + z**2 / m)
        rows.append((M, nbar, M * nbar, p, max(centre - half, 0), min(centre + half, 1)))
    return pd.DataFrame(rows, columns=["M", "nbar", "N", "rate", "low", "high"])


class Panel:
    """A plot box with a log x-axis and power (0-1) on the y-axis."""

    def __init__(self, s, x0, letter, title, sub, xlabel, xlo, xhi, ticks):
        self.s, self.px, self.py, self.pw, self.ph = s, x0 + 62, 104, 300, 330
        self.xlo, self.xhi = np.log(xlo), np.log(xhi)
        s.label(x0 + 20, 40, letter, title)
        s.text(x0 + 46, 62, sub, 12, INK2)
        px, py, pw, ph = self.px, self.py, self.pw, self.ph
        for v in np.arange(0, 1.01, 0.2):
            s.line(px, self.Y(v), px + pw, self.Y(v), RULE, 1, opacity=0.6)
            s.text(px - 8, self.Y(v) + 4, f"{v:.1f}", 11, INK2, anchor="end")
        for v in ticks:
            s.line(self.X(v), py + ph, self.X(v), py + ph + 4, INK2)
            s.text(self.X(v), py + ph + 18, f"{v:,}", 11, INK2, anchor="middle")
        s.rect(px, py, pw, ph, "none", stroke=RULE)
        s.line(px, self.Y(ALPHA), px + pw, self.Y(ALPHA), INK2, 1.2, dash="3 3")
        s.text(px + pw / 2, py + ph + 40, xlabel, 11.5, INK2, anchor="middle")
        s.text(px - 44, py + ph / 2, "power", 11.5, INK2, anchor="middle")
        s.o[-1] = s.o[-1].replace("<text ", f'<text transform="rotate(-90 {px - 44} {py + ph / 2})" ', 1)

    def X(self, v):
        return self.px + self.pw * (np.log(v) - self.xlo) / (self.xhi - self.xlo)

    def Y(self, v):
        return self.py + self.ph * (1 - v)

    def series(self, r, x, M, dy=0):
        r = r[r.M == M].sort_values(x)
        color = SHADE[M]
        xs, ys = [self.X(v) for v in r[x]], [self.Y(v) for v in r.rate]
        band = [f"{self.X(a):.1f},{self.Y(b):.1f}" for a, b in zip(r[x], r.high)]
        band += [f"{self.X(a):.1f},{self.Y(b):.1f}" for a, b in zip(r[x][::-1], r.low[::-1])]
        self.s.o.append(f'<polygon points="{" ".join(band)}" fill="{color}" fill-opacity="0.12" stroke="none"/>')
        d = " ".join(f"{'M' if i == 0 else 'L'}{a:.1f},{b:.1f}" for i, (a, b) in enumerate(zip(xs, ys)))
        self.s.o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2" '
                        'stroke-linejoin="round" stroke-linecap="round"/>')
        for a, b in zip(xs, ys):
            self.marker(a, b, MARK[M], color)
        self.s.text(xs[-1] + 10, ys[-1] + 4 + dy, f"M = {M}", 11, INK2)

    def marker(self, x, y, kind, color):
        style = f'fill="{color}" stroke="#ffffff" stroke-width="1.2"'
        o = self.s.o
        if kind == "circle":
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" {style}/>')
        elif kind == "square":
            o.append(f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" {style}/>')
        elif kind == "triangle":
            o.append(f'<path d="M{x:.1f},{y - 5:.1f} L{x + 4.6:.1f},{y + 3.6:.1f} L{x - 4.6:.1f},{y + 3.6:.1f} Z" {style}/>')
        else:
            o.append(f'<path d="M{x:.1f},{y - 5.4:.1f} L{x + 5:.1f},{y:.1f} L{x:.1f},{y + 5.4:.1f} '
                     f'L{x - 5:.1f},{y:.1f} Z" {style}/>')


exch, size = rates("PLS: exchangeable"), rates("PLS: size weights")
NBAR = dict(xlo=7, xhi=1500, ticks=(10, 25, 50, 100, 250, 1000))
DY = {10: 0, 20: 0, 50: 0, 100: 0}

W, H = 1400, 600
s = SVG(W, H)

a = Panel(s, 0, "a", "More individuals per group", "PLS test, exchangeable weights",
          "individuals per group n̄ (log scale)", **NBAR)
for M in SHADE:
    a.series(exch, "nbar", M, dy={100: -4, 50: 6}.get(M, 0))

b = Panel(s, 465, "b", "Or more groups", "the same results against the total number of individuals",
          "individuals in total N (log scale)", xlo=70, xhi=150_000, ticks=(100, 1000, 10_000, 100_000))
for M in SHADE:
    b.series(exch, "N", M)
# equal power, about 0.5, at 20 × 1,000 and at 100 × 25 individuals
p20 = exch[(exch.M == 20) & (exch.nbar == 1000)].iloc[0]
p100 = exch[(exch.M == 100) & (exch.nbar == 25)].iloc[0]
for p in (p20, p100):
    b.s.circle(b.X(p.N), b.Y(p.rate), 8, "none", stroke=INK)
b.s.line(b.X(p100.N) + 8, b.Y(p100.rate), b.X(p20.N) - 8, b.Y(p20.rate), INK, 1.2, dash="4 3")
tx, ty = b.px + 12, b.py + 22
for i, line in enumerate(["Ringed: equal power (≈ 0.5)", f"20 groups × {p20.nbar:,.0f} and",
                          f"{p100.M:.0f} groups × {p100.nbar:.0f}:", f"{p20.N / p100.N:.0f}× fewer individuals"]):
    b.s.text(tx, ty + 15 * i, line, 11, INK, weight="700" if i == 3 else "400")

c = Panel(s, 930, "c", "Size weights waste the extra individuals", "the PLS test of a, with size weights",
          "individuals per group n̄ (log scale)", **NBAR)
for M in SHADE:
    c.series(size, "nbar", M, dy={10: 12, 20: -5, 50: -2, 100: 0}.get(M, 0))

ny = 512
s.text(82, ny, "Simulated data with a fixed population: groups differ by τ² = 0.02, and their population means relate to "
               "the annotations with R² = 0.5 (κ = 0.5, Kish 40%).", 12, INK)
s.text(82, ny + 18, f"{len(df):,} data sets, 300 per cell. Dashed line: the nominal rate, 0.05. Bands: 95% intervals. "
                    "Darker lines: more groups.", 12, INK)
s.text(82, ny + 44, "More individuals sharpen each group mean but add no observations of the annotations; once the "
                    "means are precise (n̄ well above σ²_w/τ² = 50), only more groups add power.", 11, INK2)

os.chdir(HERE)
s.save("figure8_group_size.svg")
save_png("figure8_group_size.svg")
print(exch.pivot(index="M", columns="nbar", values="rate").round(2))
