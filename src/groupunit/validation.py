"""Leave-one-group-out validation against a mean baseline (Section 7.4).

A claim that a relation holds for groups not yet studied is a prediction. It
is validated by predicting each group from the others, with every selection
step inside the loop, and comparing with predicting the mean of the others.
"""
import copy
from dataclasses import dataclass

import numpy as np


def linear_model(Y_train, t_train, w_train, Y_test):
    """Weighted least squares of a group-level target on the annotations, with an intercept.

    Uses the minimum-norm solution when the annotations outnumber the training
    groups; such a fit interpolates and will usually predict badly.
    """
    Z = np.column_stack([np.ones(len(Y_train)), Y_train])
    sw = np.sqrt(w_train)
    coef, *_ = np.linalg.lstsq(sw[:, None] * Z, sw.reshape((-1,) + (1,) * (t_train.ndim - 1)) * t_train,
                               rcond=None)
    return np.concatenate([[1.0], Y_test]) @ coef


def pca_regression(k, standardise=False):
    """A model that reduces the annotations to k principal components fitted on the training groups only.

    An example of nesting a selection step inside validation: the components
    are recomputed in every fold, never from the held-out group.
    """
    def model(Y_train, t_train, w_train, Y_test):
        mean = w_train @ Y_train / w_train.sum()
        Yc = Y_train - mean
        scale = np.ones(Yc.shape[1])
        if standardise:
            sd = np.sqrt(w_train @ Yc**2 / w_train.sum())
            scale = np.where(sd > 0, sd, 1.0)
        _, _, Vt = np.linalg.svd(np.sqrt(w_train)[:, None] * (Yc / scale), full_matrices=False)
        V = Vt[:k].T
        return linear_model((Yc / scale) @ V, t_train, w_train, ((Y_test - mean) / scale) @ V)
    return model


def estimator_model(estimator):
    """Wrap a scikit-learn style regressor (``fit``/``predict``) as a model for `leave_one_group_out`.

    The estimator is copied and refitted in every fold, so a pipeline that
    standardises or reduces the annotations does so on the training groups only.
    Unequal weights are passed as ``sample_weight``, which the estimator must
    accept. A Pipeline does not, so use equal weights with pipelines.
    """
    def model(Y_train, t_train, w_train, Y_test):
        fitted = copy.deepcopy(estimator)
        if np.allclose(w_train, w_train[0]):
            fitted.fit(Y_train, t_train)
        else:
            fitted.fit(Y_train, t_train, sample_weight=w_train)
        return fitted.predict(Y_test[None, :])[0]
    return model


@dataclass(frozen=True)
class LOGOResult:
    """Leave-one-group-out predictions and out-of-sample R².

    Attributes
    ----------
    predictions : prediction of each group from the other groups, (M,) or (M, k).
    baseline : weighted mean of the other groups, same shape.
    r2 : 1 - Σw(t - prediction)² / Σw(t - baseline)²; negative if the model
        predicts worse than the mean of the other groups.
    """

    predictions: np.ndarray
    baseline: np.ndarray
    r2: float | np.ndarray


def leave_one_group_out(target, Y, weights=None, model=linear_model):
    """Predict each group's ``target`` from its annotations, fitting on the other groups.

    ``target`` is group-level, (M,) or (M, k): for example group means of a
    component or of features. ``weights`` default to equal (the population of
    groups); pass group sizes to weight by individuals. ``model`` is either a
    function ``model(Y_train, t_train, w_train, Y_test)`` returning the
    prediction for the held-out group, or a scikit-learn style regressor (see
    `estimator_model`). Either way it must contain every data-dependent choice.
    """
    if hasattr(model, "fit") and hasattr(model, "predict"):
        model = estimator_model(model)
    target = np.asarray(target, float)
    Y = np.asarray(Y, float).reshape(len(target), -1)
    M = len(target)
    w = np.ones(M) if weights is None else np.asarray(weights, float)

    predictions = np.empty_like(target)
    baseline = np.empty_like(target)
    for m in range(M):
        train = np.arange(M) != m
        predictions[m] = model(Y[train], target[train], w[train], Y[m])
        baseline[m] = w[train] @ target[train] / w[train].sum()

    w_ = w.reshape((-1,) + (1,) * (target.ndim - 1))
    r2 = 1 - (w_ * (target - predictions) ** 2).sum(0) / (w_ * (target - baseline) ** 2).sum(0)
    return LOGOResult(predictions, baseline, float(r2) if target.ndim == 1 else r2)
