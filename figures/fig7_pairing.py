"""Figure 7: false-positive rate of group permutation tests against the pairing ratio.

Pools every null data set of simulation outcome 5 (M = 20; one group's profile
1.5-4 times as far out as the others, on the largest, the smallest or a random
group; 7,200 data sets).

Panels
  a  each test against the pairing ratio of its own weights: the ratio predicts
     failure for size and equal weights alike
  b  the same data sets against the pairing ratio of the design under size
     weights (what `describe` reports), with and without exchangeable weights

Colour is the weighting, line style and marker the test; every line is also
labelled directly.

    python simulations/run_all.py s5      # if the results are missing
    python figures/fig7_pairing.py
"""
import os

import numpy as np
import pandas as pd

from svg import INK, INK2, MUTED, RULE, SVG, save_png

HERE = os.path.dirname(os.path.abspath(__file__))
REPLICATES = os.path.join(HERE, "..", "simulations", "results", "s5_dominance_replicates.csv")
EDGES = np.array([0.45, 0.6, 0.75, 0.9, 1.1, 1.35, 1.7, 2.2, 3.0, 4.0, 6.0])
MIN_COUNT = 40

WEIGHTING = {"size": "#0072B2", "equal": "#E69F00", "exchangeable": "#009E73"}   # Okabe-Ito, as in Figures 2-4
TEST = {"PLS": (None, "circle"), "correlation": ("6 4", "square"), "Mantel": ("1.5 3.5", "triangle")}

XLO, XHI, YHI = 0.45, 6.0, 0.85


def binned(ratio, reject):
    """Rejection rate with a 95% Wilson interval per bin of the ratio, bins with enough data only."""
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        inside = (ratio > lo) & (ratio <= hi)
        k, m = reject[inside].sum(), inside.sum()
        if m < MIN_COUNT:
            continue
        p, z = k / m, 1.96
        centre = (p + z**2 / (2 * m)) / (1 + z**2 / m)
        half = z * np.sqrt(p * (1 - p) / m + z**2 / (4 * m**2)) / (1 + z**2 / m)
        rows.append((np.median(ratio[inside]), p, max(centre - half, 0), centre + half))
    return np.array(rows)


class Panel:
    """A plot box with a log x-axis (pairing ratio) and a linear y-axis (rejection rate)."""

    def __init__(self, s, x0, letter, title, sub, xlabel):
        self.s, self.px, self.py, self.pw, self.ph = s, x0 + 60, 100, 470, 330
        s.label(x0 + 20, 40, letter, title)
        s.text(x0 + 46, 62, sub, 12, INK2)
        px, py, pw, ph = self.px, self.py, self.pw, self.ph
        s.rect(self.X(0.8), py, self.X(1.1) - self.X(0.8), ph, RULE, 0.45)       # describe(): no warning
        s.text((self.X(0.8) + self.X(1.1)) / 2, py + 14, "no warning", 10, MUTED, anchor="middle")
        for v in np.arange(0, YHI, 0.1):
            s.line(px, self.Y(v), px + pw, self.Y(v), RULE, 1, opacity=0.6)
            s.text(px - 8, self.Y(v) + 4, f"{v:.1f}", 11, INK2, anchor="end")
        for v in (0.5, 0.75, 1, 1.5, 2, 3, 4):
            s.line(self.X(v), py + ph, self.X(v), py + ph + 4, INK2)
            s.text(self.X(v), py + ph + 18, f"{v:g}", 11, INK2, anchor="middle")
        s.rect(px, py, pw, ph, "none", stroke=RULE)
        s.line(px, self.Y(0.05), px + pw, self.Y(0.05), INK2, 1.2, dash="3 3")   # nominal rate
        s.text(px + pw / 2, py + ph + 40, xlabel, 11.5, INK2, anchor="middle")
        s.text(px - 44, py + ph / 2, "false-positive rate", 11.5, INK2, anchor="middle")
        s.o[-1] = s.o[-1].replace("<text ", f'<text transform="rotate(-90 {px - 44} {py + ph / 2})" ', 1)

    def X(self, v):
        return self.px + self.pw * (np.log(v) - np.log(XLO)) / (np.log(XHI) - np.log(XLO))

    def Y(self, v):
        return self.py + self.ph - self.ph * v / YHI

    def series(self, ratio, reject, weighting, test, label, dy=0):
        b = binned(np.asarray(ratio), np.asarray(reject))
        color, (dash, marker) = WEIGHTING[weighting], TEST[test]
        xs, ys = [self.X(v) for v in b[:, 0]], [self.Y(v) for v in b[:, 1]]
        band = [f"{self.X(x):.1f},{self.Y(hi):.1f}" for x, hi in zip(b[:, 0], b[:, 3])]
        band += [f"{self.X(x):.1f},{self.Y(lo):.1f}" for x, lo in zip(b[::-1, 0], b[::-1, 2])]
        self.s.o.append(f'<polygon points="{" ".join(band)}" fill="{color}" fill-opacity="0.12" stroke="none"/>')
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.s.o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"{dash_attr} '
                        'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in zip(xs, ys):
            self.marker(x, y, marker, color)
        self.s.text(xs[-1] + 10, ys[-1] + 4 + dy, label, 11, INK2)

    def marker(self, x, y, kind, color):
        o = self.s.o
        if kind == "circle":
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}" stroke="#ffffff" stroke-width="1.2"/>')
        elif kind == "square":
            o.append(f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" fill="{color}" '
                     'stroke="#ffffff" stroke-width="1.2"/>')
        else:
            o.append(f'<path d="M{x:.1f},{y - 5:.1f} L{x + 4.6:.1f},{y + 3.6:.1f} L{x - 4.6:.1f},{y + 3.6:.1f} Z" '
                     f'fill="{color}" stroke="#ffffff" stroke-width="1.2"/>')


df = pd.read_csv(REPLICATES)
null = df[df.signal == 0]
ratio_size, ratio_equal = null["pairing ratio (size)"], null["pairing ratio (equal)"]

W, H = 1400, 600
s = SVG(W, H)

a = Panel(s, 0, "a", "Tests fail when the pairing is unusual",
          "each test against the pairing ratio of its own weights", "pairing ratio of the test's weights")
a.series(ratio_size, null["PLS: size"] < 0.05, "size", "PLS", "PLS, size weights")
a.series(ratio_size, null["corr: size"] < 0.05, "size", "correlation", "correlation, size weights")
a.series(ratio_equal, null["PLS: equal"] < 0.05, "equal", "PLS", "PLS, equal weights", dy=8)
a.series(ratio_equal, null["Mantel: equal"] < 0.05, "equal", "Mantel", "Mantel, equal weights", dy=-8)

b = Panel(s, 700, "b", "Exchangeable weights restore the level",
          "the same data sets against the pairing ratio of the design (size weights)",
          "pairing ratio of the design")
b.series(ratio_size, null["PLS: size"] < 0.05, "size", "PLS", "PLS, size weights")
b.series(ratio_size, null["corr: size"] < 0.05, "size", "correlation", "correlation, size weights")
b.series(ratio_size, null["corr: exchangeable"] < 0.05, "exchangeable", "correlation", "correlation, exchangeable",
         dy=-6)
b.series(ratio_size, null["PLS: exchangeable"] < 0.05, "exchangeable", "PLS", "PLS, exchangeable", dy=8)

ny = 512
s.text(80, ny, f"Simulated null data: M = 20 groups, one group's annotation profile 1.5–4 times as far out as the others, "
               f"on the largest, the smallest or a random group ({len(null):,} data sets).", 12, INK)
s.text(80, ny + 18, "Dashed line: the nominal rate, 0.05. Bands: 95% intervals per bin of the ratio. Grey: the range "
                    "in which describe() gives no warning. Exchangeable weights ∝ 1/√(τ² + σ²_w / n_m).", 12, INK)
s.text(80, ny + 44, "Colour: weighting (blue size, orange equal, green exchangeable). Line: test (solid PLS, dashed "
                    "correlation, dotted Mantel).", 11, INK2)

os.chdir(HERE)
s.save("figure7_pairing.svg")
save_png("figure7_pairing.svg")
print(f"{len(null)} null data sets; bins need at least {MIN_COUNT}")
