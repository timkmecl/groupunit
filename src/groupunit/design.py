"""What a design can show before any analysis: counts, ranks and regimes (Sections 4.5 and Box 1)."""
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .core import annotation_rank, check_groups, group_means, wcentre
from .diagnostics import effective_number
from .inference import pairing_ratio
from .scores import feature_between_share
from .weights import pooled_variance_components


@dataclass(frozen=True)
class DesignSummary:
    """Counts and ratios that bound what a design with group-level annotations can show.

    Attributes
    ----------
    N, M, q : numbers of individuals, groups and annotations.
    r : rank of the size-weighted centred annotation table.
    kappa : annotation ratio r / (M - 1); 1 means saturated.
    kish_groups : Kish effective number of groups under size weights.
    max_components : number of non-trivial PLS or CCA components, min(rank X̄, r).
    pairing_ratio : `groupunit.inference.pairing_ratio` for size weights, with
        variance components pooled over the features if they were given, and
        group means taken as equally variable otherwise. In the paper's
        simulations, size-weighted permutation tests were liberal above about
        1.1 and conservative below about 0.8.
    feature_between_share : tr S_B / tr S_T, if features were given.
    single_group : indices of annotations that single out one group (Section 4.1).
    """

    N: int
    M: int
    q: int
    r: int
    kappa: float
    kish_groups: float
    max_components: int
    pairing_ratio: float
    feature_between_share: float | None = None
    single_group: list = field(default_factory=list)
    annotation_names: list | None = None

    def __str__(self):
        lines = [
            f"individuals N = {self.N}, groups M = {self.M}, annotations q = {self.q}",
            f"annotation rank r = {self.r}, annotation ratio kappa = r/(M-1) = {self.kappa:.2f}",
            f"effective number of groups under size weights (Kish) = {self.kish_groups:.1f}",
            f"at most {self.max_components} non-trivial PLS/CCA components",
        ]
        if self.feature_between_share is not None:
            lines.append(f"between-group share of the features = {self.feature_between_share:.3f}")
        if self.kappa >= 1:
            lines.append("SATURATED: CCA equals discriminant analysis of group labels and cannot "
                         "respond to the annotation values; reduce the annotations first.")
        elif self.kappa >= 0.8:
            lines.append("NEAR-SATURATED: the null mean of R² is close to 1, and PLS loadings "
                         "are dominated by the lens; consider reducing the annotations.")
        lines.append(f"pairing ratio of sizes with annotation profiles = {self.pairing_ratio:.2f}")
        if self.pairing_ratio > 1.1:
            lines.append("UNUSUAL PAIRING: large groups carry atypical profiles, so size-weighted group "
                         "permutation tests (weights='size', cca_test, estimator_test) are liberal here; "
                         "keep the default exchangeable weights.")
        elif self.pairing_ratio < 0.8:
            lines.append("UNUSUAL PAIRING: size-weighted group permutation tests are conservative here; "
                         "the default exchangeable weights recover power.")
        if self.single_group:
            names = self.annotation_names or list(range(self.q))
            flagged = ", ".join(str(names[j]) for j in self.single_group)
            lines.append(f"annotations that single out one group: {flagged}")
        return "\n".join(lines)


def single_group_annotations(Y, atol=1e-12):
    """Indices of annotation columns in which exactly one group differs from all others.

    Such an annotation correlates with any score exactly as that group's mean
    does, so it describes one group rather than a relation.
    """
    Y = np.asarray(Y, float)
    flagged = []
    for j, col in enumerate(Y.T):
        values, counts = np.unique(col, return_counts=True)
        mode = values[np.argmax(counts)]
        if np.sum(~np.isclose(col, mode, rtol=0, atol=atol)) == 1:
            flagged.append(j)
    return flagged


def describe(groups, Y, X=None, annotation_names=None):
    """Summarise a design: N, M, q, r, kappa, effective groups and flagged annotations (Box 1, items 1-4)."""
    Y = np.asarray(Y, float)
    groups, n = check_groups(groups, len(Y))
    M, q = Y.shape
    r = annotation_rank(Y, n)
    max_components = r
    share = None
    ratio = pairing_ratio(Y, n)
    if X is not None:
        X = np.asarray(X, float)
        Xbar = group_means(X - X.mean(0), groups, M)
        max_components = min(r, annotation_rank(Xbar, n))
        share = feature_between_share(X, groups)
        vc = pooled_variance_components(X, groups)
        ratio = pairing_ratio(Y, n, tau2=vc.tau2, sigma2_w=vc.sigma2_w)
    return DesignSummary(
        N=len(groups), M=M, q=q, r=r, kappa=r / (M - 1),
        kish_groups=effective_number(n),
        max_components=max_components,
        pairing_ratio=ratio,
        feature_between_share=share,
        single_group=single_group_annotations(Y),
        annotation_names=None if annotation_names is None else list(annotation_names),
    )


def r2_null(r, M):
    """Null distribution of the group-level R² of any score on r annotations (Result 3b).

    Beta(r/2, (M-1-r)/2), with mean r/(M-1). Returns a frozen ``scipy.stats``
    distribution. At r = M - 1 the distribution collapses to 1 for every score.

    It assumes the weighted group means point in a random direction, which
    `groupunit.weights.optimal_weights` achieve. With size weights and unequal
    group sizes the largest groups dominate both the score and the
    annotations, and the null R² is larger, about r / M_eff for small r
    (Kish's number); treat this distribution as a lower bound there.
    """
    if not 0 < r < M - 1:
        raise ValueError(f"need 0 < r < M - 1; at r = M - 1 every score has R² = 1 (r={r}, M={M})")
    return stats.beta(r / 2, (M - 1 - r) / 2)


def reduce_annotations(Y, w, k, standardise=False):
    """First k weighted principal components of the annotation table (Box 1, item 3).

    Reducing the annotations in advance moves a design out of the saturated
    regime. Returns ``(scores, explained)``: the (M, k) component scores and the
    share of the weighted annotation variance each one holds. Inside a
    permutation test or cross-validation, call this on the training groups only.
    """
    Y = np.asarray(Y, float)
    w = np.asarray(w, float)
    Yc = wcentre(Y, w)
    if standardise:
        sd = np.sqrt(w @ Yc**2 / w.sum())
        Yc = Yc / np.where(sd > 0, sd, 1.0)
    _, s, Vt = np.linalg.svd(np.sqrt(w)[:, None] * Yc, full_matrices=False)
    explained = s**2 / (s**2).sum()
    return Yc @ Vt[:k].T, explained[:k]
