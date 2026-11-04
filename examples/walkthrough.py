"""A walk through the checklist of the paper (Box 1) on simulated data.

Run with:  python examples/walkthrough.py
"""
import numpy as np

import groupunit as gu

rng = np.random.default_rng(42)

# 20 groups of unequal size, 15 annotations, 8 features. The group means of the
# features depend on the first two annotations; the rest are unrelated.
n = gu.lognormal_sizes(20, 60, sigma=0.8, rng=rng)
beta = np.zeros((15, 8))
beta[:2, :3] = 1.0
X, groups, Y = gu.simulate(n, p=8, q=15, beta=beta, tau2=0.5, sigma2_w=4.0, rng=rng)

print("1. The design")
print(gu.describe(groups, Y, X), "\n")

print("2. Reduce the annotations in advance (here, to 3 weighted principal components)")
Y3, explained = gu.reduce_annotations(Y, n, 3, standardise=True)
print(f"   variance held: {np.round(explained, 2)}; kappa now {3 / (len(n) - 1):.2f}\n")

print("3. PLS with a sequential test over groups, refitting in every permutation")
test = gu.pls_test(X, groups, Y3, n_perm=999, rng=0)
for c, (s, p) in enumerate(zip(test.strength, test.p_value), 1):
    print(f"   component {c}: singular value {s:8.1f}   p = {p:.3f}")
print()

print("   Are size-weighted permutation tests trustworthy here?")
vc = gu.pooled_variance_components(X, groups)
ratio = gu.pairing_ratio(Y3, n, tau2=vc.tau2, sigma2_w=vc.sigma2_w)
w = gu.exchangeable_weights(n, vc.tau2, vc.sigma2_w)
p_exchangeable = gu.pls_test(X, groups, Y3, n_perm=999, rng=0, weights=w).p_value[0]
print(f"   pairing ratio {ratio:.2f} (near 1 is fine); with exchangeable weights, component 1: p = {p_exchangeable:.3f}\n")

print("4. Which groups carry the first component?")
fit = test.fit
shares = gu.group_shares(fit.group_scores[:, 0], Y3, n)
top = shares.top(3)
print(f"   largest shares: groups {top.tolist()} with {np.round(shares.share[top], 2).tolist()}")
print(f"   effective number of groups behind it: {shares.m_eff:.1f} of {len(n)}")
dropped = gu.leave_out(fit.group_scores[:, 0], Y3, n, top[0])
print(f"   without group {top[0]}: congruence {dropped.congruence:.2f} with the original loading, "
      f"and keeps {dropped.norm_ratio:.0%} of its length\n")

print("5. The lens of the full annotation table, and its null for PLS on all 15 annotations")
full = gu.pls(X, groups, Y)
L = gu.lens(Y, n)
null = gu.lens_congruence_null(full.y_loadings[:, 0], Y, n, rng=0)
print(f"   participation ratio nu = {L.nu:.1f} of {Y.shape[1]} annotations")
print(f"   congruence of component 1 with random-group-mean loadings: median {np.median(null):.2f}\n")

print("6. Between-group share, separately from any relation to the annotations")
print(f"   features overall: {gu.feature_between_share(X, groups):.3f}; "
      f"first PLS score: {gu.between_share(fit.scores(X)[:, 0], groups):.3f}\n")

print("7. Saturated CCA cannot see the annotations")
Y19 = np.column_stack([Y, rng.standard_normal((len(n), 4))])   # rank 19 = M - 1
same = np.allclose(gu.cca(X, groups, Y19).strength, gu.cca(X, groups, np.eye(len(n))).strength)
print(f"   CCA with 19 annotations equals LDA: {same}\n")

print("8. Leave-one-group-out prediction of the first feature's group means, PCs refitted per fold")
target = gu.group_scores(X[:, 0], groups)
logo = gu.leave_one_group_out(target, Y, weights=n, model=gu.pca_regression(3, standardise=True))
print(f"   out-of-sample R2 against the mean of the other groups: {logo.r2:.2f}\n")

print("9. Weights and effective numbers")
vc = gu.variance_components(X[:, 0], groups, Y3)
print(f"   sigma2_w = {vc.sigma2_w:.2f}, tau2 = {vc.tau2:.2f}")
print(f"   effective groups: Kish {gu.effective_groups(n):.1f}, "
      f"model-based {gu.effective_groups(n, n, vc.tau2, vc.sigma2_w):.1f}, "
      f"best possible {gu.max_effective_groups(n, vc.tau2, vc.sigma2_w):.1f} of {len(n)}")
