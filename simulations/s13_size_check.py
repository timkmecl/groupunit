"""Outcome 13: does `size_check` predict the size of a test for the design at hand?

`groupunit.size_check` simulates null data with the observed group sizes and
annotation table and with variance components estimated from the data, and
reports how often a test rejects. Its target is the conditional size of the
test given that table. For each design we draw an annotation table and one
data set, run the size check with variance components estimated from that
data set (as a user would), and compare it with the same check run with the
true components and four times as many simulated data sets, which estimates
the conditional size itself. Both PLS tests are checked: size weights, which
are liberal in pairing designs, and exchangeable weights.
"""
import numpy as np

import groupunit as gu
from common import Design, run, save

N_PERM_INNER = 99
LIBERAL = 0.075          # a size above this is flagged as liberal


def _pls(weights):
    def test(X, g, Y, rng):
        return gu.pls_test(X, g, Y, n_components=1, n_perm=N_PERM_INNER, rng=rng, weights=weights).p_value[0]
    return test


TESTS = {"PLS size": _pls("size"), "PLS exchangeable": _pls("exchangeable")}


def cells(n_sim):
    grid = [Design(M=20, h=1),
            Design(M=20, h=1, structure="extreme", extreme_at="largest", extremity=1.5),
            Design(M=20, h=1, structure="extreme", extreme_at="largest", extremity=4.0),
            Design(M=20, h=10, structure="extreme", extreme_at="largest", extremity=2.0),
            Design(M=20, h=1, structure="extreme", extreme_at="smallest", extremity=4.0),
            Design(M=20, h=1, structure="several", n_atypical=3, extremity=2.0),
            Design(M=20, h=1, structure="continuous", gamma=0.5),
            Design(M=50, h=1, structure="extreme", extreme_at="largest", extremity=2.0)]
    return [(d, {"n_sim": n_sim}) for d in grid]


def replicate(d, rng, n_sim):
    X, groups, Y, n = d.draw(rng)
    vc = gu.pooled_variance_components(X, groups)
    row = {"tau2 estimate / true": vc.tau2 / d.tau2,
           "pairing ratio (size)": gu.pairing_ratio(Y, n, tau2=d.tau2, sigma2_w=1.0)}
    for name, test in TESTS.items():
        row[f"{name}: predicted"] = gu.size_check(test, n, Y, vc.tau2, vc.sigma2_w, p=d.p, n_sim=n_sim, rng=rng).rate
        row[f"{name}: true"] = gu.size_check(test, n, Y, d.tau2, 1.0, p=d.p, n_sim=4 * n_sim, rng=rng).rate
    return row


def summarise(df):
    keys = [df.structure, df.extreme_at, df.extremity, df.M, df.h]
    out = df[["tau2 estimate / true", "pairing ratio (size)"]].groupby(keys).median()
    for name in TESTS:
        pred, true = df[f"{name}: predicted"], df[f"{name}: true"]
        g = df.assign(pred=pred, true=true, err=(pred - true).abs(),
                      agree=((pred > LIBERAL) == (true > LIBERAL)).astype(float)).groupby(keys)
        out[f"{name}: true size"] = g.true.mean()
        out[f"{name}: predicted"] = g.pred.mean()
        out[f"{name}: mean |error|"] = g.err.mean()
        out[f"{name}: flag agrees"] = g.agree.mean()
    corr = {name: np.corrcoef(df[f"{name}: predicted"], df[f"{name}: true"])[0, 1] for name in TESTS}
    print("correlation of predicted with true size over all tables:", corr)
    return out


def main(reps=40, n_perm=None, seed=14, n_sim=100):
    df = run(replicate, cells(n_sim), reps, seed, chunk=2)
    save(df, summarise(df), "s13_size_check")


if __name__ == "__main__":
    main()
