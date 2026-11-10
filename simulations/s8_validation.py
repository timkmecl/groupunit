"""Outcome 8: optimism of individual-split versus group-split cross-validation (Section 4.5).

A ridge regression predicts each individual's (inherited) annotation from its
features. Splitting individuals into folds leaves every held-out group in the
training set; leaving groups out does not.
"""
import numpy as np

from common import Design, run, save

LAMBDA = 1.0


def cells():
    return [(Design(M=M, kappa=1 / (M - 1), h=h, signal=s), {})
            for M in (10, 20, 50) for h in (1, 10) for s in (0.0, 0.3)]


def ridge_r2(X, y, folds):
    """Out-of-sample R² pooled over folds, against the training mean of each fold."""
    sse = sst = 0.0
    for test in folds:
        train = np.ones(len(y), bool)
        train[test] = False
        mx, my = X[train].mean(0), y[train].mean()
        Xt = X[train] - mx
        b = np.linalg.solve(Xt.T @ Xt + LAMBDA * np.eye(X.shape[1]), Xt.T @ (y[train] - my))
        pred = (X[test] - mx) @ b + my
        sse += ((y[test] - pred) ** 2).sum()
        sst += ((y[test] - my) ** 2).sum()
    return 1 - sse / sst


def replicate(d, rng):
    X, groups, Y, n = d.draw(rng)
    y = Y[groups, 0]
    individual_folds = np.array_split(rng.permutation(len(y)), 5)
    group_folds = [np.flatnonzero(groups == m) for m in range(d.M)]
    return {"R2 individual split": ridge_r2(X, y, individual_folds),
            "R2 leave one group out": ridge_r2(X, y, group_folds)}


def summarise(df):
    out = df.groupby(["M", "h", "signal"]).mean(numeric_only=True)[["R2 individual split", "R2 leave one group out"]]
    out["optimism"] = out["R2 individual split"] - out["R2 leave one group out"]
    return out


def main(reps=300, n_perm=None, seed=8):
    df = run(replicate, cells(), reps, seed)
    save(df, summarise(df), "s8_validation")


if __name__ == "__main__":
    main()
