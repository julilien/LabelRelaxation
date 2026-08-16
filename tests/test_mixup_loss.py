import math

import pytest
import torch
import torch.nn.functional as F

from label_relaxation import (
    LabelRelaxationLoss,
    MixupLabelRelaxationLoss,
    label_relaxation_loss,
    mixup_label_relaxation_loss,
)

torch.manual_seed(0)


def random_inputs(batch=64, classes=10, dtype=torch.float64, scale=2.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    logits = scale * torch.randn(batch, classes, dtype=dtype, generator=gen)
    ya = torch.randint(0, classes, (batch,), generator=gen)
    yb = torch.randint(0, classes, (batch,), generator=gen)
    return logits, ya, yb


# --- Equivalence with an independent convex-solver oracle -------------------------------


def test_matches_cvxpy_oracle():
    cp = pytest.importorskip("cvxpy")

    def oracle(phat, i, j, a, b):
        q = cp.Variable(len(phat))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.kl_div(q, phat))),
                          [cp.sum(q) == 1, q >= 0, q[i] >= a, q[j] >= b])
        prob.solve(solver=cp.CLARABEL)
        return None if prob.status != "optimal" else max(float(prob.value), 0.0)

    gen = torch.Generator().manual_seed(3)
    checked = 0
    for _ in range(60):
        k = int(torch.randint(2, 10, (1,), generator=gen))
        logits = 2.5 * torch.randn(k, dtype=torch.float64, generator=gen)
        i, j = torch.randperm(k, generator=gen)[:2].tolist()
        alpha = float(torch.empty(1).uniform_(0.03, 0.5, generator=gen))
        lam = float(torch.empty(1).uniform_(0.0, 1.0, generator=gen))
        phat = logits.softmax(-1).numpy()
        want = oracle(phat, i, j, lam * (1 - alpha), (1 - lam) * (1 - alpha))
        if want is None:
            continue
        got = float(mixup_label_relaxation_loss(
            logits, torch.tensor(i), torch.tensor(j), lam, alpha=alpha, reduction="none"))
        assert abs(got - want) < 1e-4
        checked += 1
    assert checked > 20


# --- Limiting cases (the propositions' sanity constraints) ------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3])
def test_lambda_one_recovers_plain_lr_on_target_a(alpha):
    logits, ya, yb = random_inputs(seed=1)
    mixed = mixup_label_relaxation_loss(logits, ya, yb, lam=1.0, alpha=alpha, reduction="none")
    plain = label_relaxation_loss(logits, ya, alpha=alpha, reduction="none")
    torch.testing.assert_close(mixed, plain, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3])
def test_lambda_zero_recovers_plain_lr_on_target_b(alpha):
    logits, ya, yb = random_inputs(seed=2)
    mixed = mixup_label_relaxation_loss(logits, ya, yb, lam=0.0, alpha=alpha, reduction="none")
    plain = label_relaxation_loss(logits, yb, alpha=alpha, reduction="none")
    torch.testing.assert_close(mixed, plain, rtol=1e-9, atol=1e-9)


def test_same_class_reduces_to_plain_lr():
    logits, ya, _ = random_inputs(seed=5)
    mixed = mixup_label_relaxation_loss(logits, ya, ya, lam=0.4, alpha=0.2, reduction="none")
    plain = label_relaxation_loss(logits, ya, alpha=0.2, reduction="none")
    torch.testing.assert_close(mixed, plain, rtol=1e-9, atol=1e-9)


def test_alpha_to_zero_matches_mixup_ce_gradient():
    logits, ya, yb = random_inputs(batch=32, seed=8)
    logits = logits.clamp(-3, 3)
    lam = 0.3

    x1 = logits.clone().requires_grad_()
    mixup_label_relaxation_loss(x1, ya, yb, lam=lam, alpha=1e-6).backward()

    x2 = logits.clone().requires_grad_()
    (lam * F.cross_entropy(x2, ya) + (1 - lam) * F.cross_entropy(x2, yb)).backward()

    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-3, atol=1e-3)


# --- Loss properties -------------------------------------------------------------------


def test_zero_loss_and_gradient_inside_credal_set():
    # lam=0.5, alpha=0.3 -> both bounds are 0.35; put ~0.44 on each mixed class.
    logits = torch.tensor([[2.0, 2.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    ya, yb = torch.tensor([0]), torch.tensor([1])
    loss = mixup_label_relaxation_loss(logits, ya, yb, lam=0.5, alpha=0.3)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.all(logits.grad == 0)


def test_gradcheck_both_active_region():
    gen = torch.Generator().manual_seed(5)
    # Suppress the two mixed classes so both constraints are active (smooth interior).
    logits = torch.randn(4, 6, dtype=torch.float64, generator=gen).clamp(-1, 1)
    logits[:, 0] -= 4.0
    logits[:, 1] -= 4.0
    logits = logits.requires_grad_()
    ya, yb = torch.tensor([0, 0, 0, 0]), torch.tensor([1, 1, 1, 1])
    assert torch.autograd.gradcheck(
        lambda x: mixup_label_relaxation_loss(x, ya, yb, lam=0.5, alpha=0.2), (logits,)
    )


def test_per_sample_lambda():
    logits, ya, yb = random_inputs(batch=4, seed=6)
    lam = torch.tensor([0.0, 1.0, 0.5, 0.3], dtype=logits.dtype)
    mixed = mixup_label_relaxation_loss(logits, ya, yb, lam=lam, alpha=0.1, reduction="none")
    # lam=0 and lam=1 rows must equal the plain-LR values on yb, ya respectively.
    torch.testing.assert_close(mixed[0], label_relaxation_loss(logits[0:1], yb[0:1], alpha=0.1))
    torch.testing.assert_close(mixed[1], label_relaxation_loss(logits[1:2], ya[1:2], alpha=0.1))


def test_reduction_modes():
    logits, ya, yb = random_inputs(seed=7)
    none = mixup_label_relaxation_loss(logits, ya, yb, lam=0.4, alpha=0.1, reduction="none")
    assert none.shape == ya.shape
    total = mixup_label_relaxation_loss(logits, ya, yb, lam=0.4, alpha=0.1, reduction="sum")
    mean = mixup_label_relaxation_loss(logits, ya, yb, lam=0.4, alpha=0.1, reduction="mean")
    torch.testing.assert_close(total, none.sum())
    torch.testing.assert_close(mean, none.mean())


# --- Stability & API -------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_is_finite(dtype):
    logits, ya, yb = random_inputs(dtype=torch.float32, seed=2)
    loss = mixup_label_relaxation_loss(logits.to(dtype), ya, yb, lam=0.5, alpha=0.1)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


def test_extreme_logits_stable():
    logits = torch.zeros(3, 8, requires_grad=True)
    with torch.no_grad():
        logits[0, 0] = 1e4  # confident on ya
        logits[1, 5] = 1e4  # confident off both mixed classes
        logits[2] = -1e4
    ya, yb = torch.tensor([0, 0, 0]), torch.tensor([1, 1, 1])
    loss = mixup_label_relaxation_loss(logits, ya, yb, lam=0.5, alpha=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.all(torch.isfinite(logits.grad))


def test_module_wrapper_matches_functional():
    logits, ya, yb = random_inputs(seed=1)
    module = MixupLabelRelaxationLoss(alpha=0.25)
    torch.testing.assert_close(
        module(logits, ya, yb, 0.6),
        mixup_label_relaxation_loss(logits, ya, yb, 0.6, alpha=0.25),
    )
    assert "alpha=0.25" in repr(module)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_invalid_alpha_rejected(alpha):
    logits, ya, yb = random_inputs(seed=0)
    with pytest.raises(ValueError, match="alpha"):
        mixup_label_relaxation_loss(logits, ya, yb, 0.5, alpha=alpha)


def test_invalid_lambda_rejected():
    logits, ya, yb = random_inputs(seed=0)
    with pytest.raises(ValueError, match="lam"):
        mixup_label_relaxation_loss(logits, ya, yb, 1.5, alpha=0.1)
