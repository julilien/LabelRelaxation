import math

import pytest
import torch
import torch.nn.functional as F

from label_relaxation import LabelRelaxationLoss, label_relaxation_loss

# The original AAAI 2021 implementation (kept frozen in legacy/) is the equivalence oracle.
from lr_torch.lr_torch import LabelRelaxationLoss as ReferenceLoss

torch.manual_seed(0)


def random_inputs(batch=64, classes=10, dtype=torch.float64, scale=2.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    logits = scale * torch.randn(batch, classes, dtype=dtype, generator=gen)
    target = torch.randint(0, classes, (batch,), generator=gen)
    return logits, target


# --- Equivalence with the original implementation -------------------------------------

# The original implementation computes `torch.ones_like(target) - alpha` on the int64
# one-hot tensor, which type-promotes to float32: its `1 - alpha` credal-set entry is
# rounded to float32 precision even in a float64 computation. Equivalence with the
# reference therefore only holds to ~1e-7; the exact-oracle tests below check the paper
# formula at full float64 precision.


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.3, 0.6])
@pytest.mark.parametrize("classes", [2, 10, 100])
def test_value_matches_reference(alpha, classes):
    logits, target = random_inputs(classes=classes, seed=classes)
    reference = ReferenceLoss(alpha=alpha, num_classes=classes)
    expected = reference(logits, target)
    actual = label_relaxation_loss(logits, target, alpha=alpha)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3])
def test_gradient_matches_reference(alpha):
    logits, target = random_inputs(seed=7)
    reference = ReferenceLoss(alpha=alpha, num_classes=logits.shape[-1])

    logits_ref = logits.clone().requires_grad_()
    reference(logits_ref, target).backward()

    logits_new = logits.clone().requires_grad_()
    label_relaxation_loss(logits_new, target, alpha=alpha).backward()

    torch.testing.assert_close(logits_new.grad, logits_ref.grad, rtol=1e-6, atol=1e-7)


def exact_oracle(logits: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """Paper formula in full float64: explicit KL projection onto the credal set."""
    pred = logits.softmax(dim=-1)
    one_hot = F.one_hot(target, num_classes=logits.shape[-1]).to(logits.dtype)
    s = ((1 - one_hot) * pred).sum(dim=-1, keepdim=True)
    projection = torch.where(one_hot.bool(), 1.0 - alpha, alpha * pred / s)
    divergence = (projection * (projection.log() - pred.log())).sum(dim=-1)
    p_y = (pred * one_hot).sum(dim=-1)
    return torch.where(p_y >= 1 - alpha, torch.zeros_like(divergence), divergence).mean()


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.3, 0.6])
@pytest.mark.parametrize("classes", [2, 10, 100])
def test_value_matches_exact_oracle(alpha, classes):
    logits, target = random_inputs(classes=classes, seed=classes)
    expected = exact_oracle(logits, target, alpha)
    actual = label_relaxation_loss(logits, target, alpha=alpha)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


# --- Loss properties -------------------------------------------------------------------


def test_zero_loss_and_zero_gradient_inside_credal_set():
    # Target-class probability is pushed far above 1 - alpha for every sample.
    logits = torch.full((8, 5), -10.0, dtype=torch.float64)
    target = torch.arange(8) % 5
    logits[torch.arange(8), target] = 10.0
    logits.requires_grad_()

    loss = label_relaxation_loss(logits, target, alpha=0.1)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.all(logits.grad == 0)


def test_converges_to_cross_entropy_for_small_alpha():
    logits, target = random_inputs(seed=3)
    logits = logits.clamp(-3, 3)
    loss = label_relaxation_loss(logits, target, alpha=1e-6)
    ce = F.cross_entropy(logits, target)
    torch.testing.assert_close(loss, ce, rtol=1e-4, atol=1e-4)


def test_loss_is_zero_iff_inside_credal_set():
    alpha = 0.2
    logits, target = random_inputs(batch=512, seed=11)
    per_sample = label_relaxation_loss(logits, target, alpha=alpha, reduction="none")
    p_y = logits.softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    inside = p_y >= 1 - alpha
    assert inside.any() and (~inside).any(), "test inputs should cover both branches"
    assert torch.all(per_sample[inside] == 0)
    assert torch.all(per_sample[~inside] > 0)


def test_gradcheck():
    gen = torch.Generator().manual_seed(5)
    logits = torch.randn(4, 6, dtype=torch.float64, generator=gen).clamp(-1, 1).requires_grad_()
    target = torch.tensor([0, 1, 2, 3])
    # Logits clamped to [-1, 1] keep p_y well below the 1 - alpha boundary, so the
    # loss is smooth in the perturbation neighborhood used by gradcheck.
    assert torch.autograd.gradcheck(
        lambda x: label_relaxation_loss(x, target, alpha=0.1), (logits,)
    )


# --- Numerical stability ---------------------------------------------------------------


def test_extreme_logits_are_stable():
    # Saturated predictions: confident-correct (inside the set) and confident-wrong.
    logits = torch.zeros(4, 10, requires_grad=True)
    with torch.no_grad():
        logits[0, 0] = 1e4    # p_y ~ 1, inside the credal set
        logits[1, 5] = 1e4    # confidently wrong, p_y ~ 0
        logits[2] = -1e4      # all logits equal and extreme
        logits[3, 3] = 1e4
    target = torch.tensor([0, 0, 0, 0])

    loss = label_relaxation_loss(logits, target, alpha=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.all(torch.isfinite(logits.grad))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_inputs(dtype):
    logits, target = random_inputs(dtype=torch.float32, seed=2)
    loss = label_relaxation_loss(logits.to(dtype), target, alpha=0.1)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    # Half-precision result close to the float32 computation.
    expected = label_relaxation_loss(logits, target, alpha=0.1)
    torch.testing.assert_close(loss, expected, rtol=2e-2, atol=2e-2)


# --- API surface ------------------------------------------------------------------------


def test_one_hot_targets_equal_index_targets():
    logits, target = random_inputs(seed=9)
    one_hot = F.one_hot(target, num_classes=logits.shape[-1]).to(logits.dtype)
    expected = label_relaxation_loss(logits, target, alpha=0.1)
    actual = label_relaxation_loss(logits, one_hot, alpha=0.1)
    torch.testing.assert_close(actual, expected)


def test_soft_targets_rejected():
    logits, target = random_inputs(seed=4)
    soft = F.one_hot(target, num_classes=logits.shape[-1]).to(logits.dtype)
    soft = 0.7 * soft + 0.3 / logits.shape[-1]
    with pytest.raises(ValueError, match="one-hot"):
        label_relaxation_loss(logits, soft, alpha=0.1)


def test_reduction_modes():
    logits, target = random_inputs(seed=6)
    none = label_relaxation_loss(logits, target, alpha=0.1, reduction="none")
    assert none.shape == target.shape
    total = label_relaxation_loss(logits, target, alpha=0.1, reduction="sum")
    mean = label_relaxation_loss(logits, target, alpha=0.1, reduction="mean")
    torch.testing.assert_close(total, none.sum())
    torch.testing.assert_close(mean, none.mean())


def test_multidimensional_inputs_match_flattened():
    gen = torch.Generator().manual_seed(8)
    logits = torch.randn(2, 3, 7, dtype=torch.float64, generator=gen)
    target = torch.randint(0, 7, (2, 3), generator=gen)
    nd = label_relaxation_loss(logits, target, alpha=0.1, reduction="none")
    flat = label_relaxation_loss(logits.reshape(6, 7), target.reshape(6), alpha=0.1, reduction="none")
    assert nd.shape == (2, 3)
    torch.testing.assert_close(nd.reshape(6), flat)


def test_module_wrapper_matches_functional():
    logits, target = random_inputs(seed=1)
    module = LabelRelaxationLoss(alpha=0.25)
    torch.testing.assert_close(
        module(logits, target), label_relaxation_loss(logits, target, alpha=0.25)
    )
    assert "alpha=0.25" in repr(module)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_invalid_alpha_rejected(alpha):
    logits, target = random_inputs(seed=0)
    with pytest.raises(ValueError, match="alpha"):
        label_relaxation_loss(logits, target, alpha=alpha)
    with pytest.raises(ValueError, match="alpha"):
        LabelRelaxationLoss(alpha=alpha)


def test_invalid_shapes_rejected():
    logits, target = random_inputs(seed=0)
    with pytest.raises(ValueError, match="shape"):
        label_relaxation_loss(logits, target.unsqueeze(-1), alpha=0.1)
    with pytest.raises(ValueError, match="reduction"):
        label_relaxation_loss(logits, target, alpha=0.1, reduction="bogus")
