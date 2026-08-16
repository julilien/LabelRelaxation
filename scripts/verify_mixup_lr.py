"""Numerical verification of the mixup-LR closed form against a convex-solver oracle.

Runs the four-case closed-form projection loss against cvxpy's KL projection onto
S = {q : q_i >= lam(1-alpha), q_j >= (1-lam)(1-alpha)} over many random instances, and reports
per-case coverage so we know every branch of the derivation is exercised. This is the
before-proof-writing check referenced in docs/mixup_lr_derivation.md; it is not part of the
shipped package (kept under scripts/).

    uv run python scripts/verify_mixup_lr.py
"""

import cvxpy as cp
import numpy as np
import torch

from label_relaxation import mixup_label_relaxation_loss


def oracle(phat, i, j, a, b):
    """KL projection loss via cvxpy. Returns None when the solver flags an inaccurate solution
    (occasional on kl_div), so solver noise is skipped rather than mistaken for a formula error."""
    q = cp.Variable(len(phat))
    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.kl_div(q, phat))),
        [cp.sum(q) == 1, q >= 0, q[i] >= a, q[j] >= b],
    )
    problem.solve(solver=cp.CLARABEL)  # exponential-cone accurate; SCS is unreliable on kl_div
    if problem.status != "optimal" or problem.value is None:
        return None
    return float(problem.value)


def classify(phat, i, j, a, b):
    pi, pj = phat[i], phat[j]
    di, dj = pi < a, pj < b
    if not di and not dj:
        return "none"
    Ci = (1 - a) / (1 - pi)
    Cj = (1 - b) / (1 - pj)
    if (di and dj) or (di and not dj and Ci * pj < b) or (dj and not di and Cj * pi < a):
        return "both"
    return "i_only" if di else "j_only"


def main(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    counts, worst = {"none": 0, "i_only": 0, "j_only": 0, "both": 0}, 0.0
    for _ in range(n):
        k = int(rng.integers(2, 12))
        scale = float(rng.uniform(0.5, 4.0))
        logits = rng.normal(size=k) * scale
        i, j = rng.choice(k, size=2, replace=False)
        alpha = float(rng.uniform(0.02, 0.6))
        lam = float(rng.uniform(0.0, 1.0))
        phat = np.exp(logits - logits.max())
        phat /= phat.sum()
        a, b = lam * (1 - alpha), (1 - lam) * (1 - alpha)

        want = oracle(phat, i, j, a, b)
        if want is None:  # solver flagged inaccurate; skip (not a formula discrepancy)
            continue
        counts[classify(phat, i, j, a, b)] += 1
        got = float(mixup_label_relaxation_loss(
            torch.tensor(logits), torch.tensor(i), torch.tensor(j), lam, alpha=alpha,
            reduction="none",
        ))
        worst = max(worst, abs(got - max(want, 0.0)))

    print(f"instances (solved): {sum(counts.values())} / {n}")
    print(f"case coverage: {counts}")
    print(f"max |closed-form - cvxpy|: {worst:.2e}")
    assert all(v > 0 for v in counts.values()), f"a case was never hit: {counts}"
    assert worst < 1e-4, f"closed form disagrees with the oracle: {worst:.2e}"
    print("OK — closed form matches the convex-solver oracle across all four cases.")


if __name__ == "__main__":
    main()
