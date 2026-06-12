"""Label relaxation loss (Lienen & Hüllermeier, AAAI 2021).

Instead of a precise (possibly smoothed) target distribution, label relaxation
trains against the credal set of distributions

    Q_{alpha,y} = { p in simplex : p_y >= 1 - alpha },

i.e. all distributions assigning probability at least ``1 - alpha`` to the
observed class ``y``. The loss is the optimistic superset loss

    L(p_hat, y) = 0                  if p_hat in Q_{alpha,y}
                  KL(pr || p_hat)    otherwise,

where ``pr`` is the member of the credal set minimizing the KL divergence to
the prediction: ``pr_y = 1 - alpha`` and ``pr_k = alpha * p_hat_k / (1 - p_hat_y)``
for ``k != y``. Note the KL direction: the projected target is the *first*
argument, matching Eq. (7) of the paper and the original implementation.

For one-hot targets the divergence collapses to a closed form depending only
on the predicted probability of the true class:

    KL(pr || p_hat) = (1 - alpha) * log((1 - alpha) / p_hat_y)
                      + alpha * log(alpha / (1 - p_hat_y)).

This module computes that closed form directly from ``log_softmax`` outputs
(``log(1 - p_hat_y)`` via a masked ``logsumexp``), which is numerically stable
for extreme logits and avoids the ``softmax().log()`` round trip of the
original reference implementation. Gradients are identical to the original
formulation (the projection ``pr`` is the KL minimizer over the credal set, so
detaching it does not change the gradient).

As ``alpha -> 0`` the loss converges to standard cross-entropy; for predictions
inside the credal set both the loss and its gradient are exactly zero.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["LabelRelaxationLoss", "label_relaxation_loss"]

_REDUCTIONS = ("mean", "sum", "none")


def _targets_to_indices(logits: Tensor, target: Tensor) -> Tensor:
    """Normalize class-index or one-hot targets to an index tensor of shape ``logits.shape[:-1]``."""
    if not target.is_floating_point():
        if target.shape != logits.shape[:-1]:
            raise ValueError(
                f"Index targets must have shape {tuple(logits.shape[:-1])} for logits of shape "
                f"{tuple(logits.shape)}, got {tuple(target.shape)}."
            )
        return target.long()

    if target.shape != logits.shape:
        raise ValueError(
            f"Probability-vector targets must have the same shape as the logits "
            f"{tuple(logits.shape)}, got {tuple(target.shape)}."
        )
    one_hot_mass = target.max(dim=-1).values
    if not bool(torch.all((one_hot_mass == 1) & (target.sum(dim=-1) == 1)).item()):
        raise ValueError(
            "Probability-vector targets must be exactly one-hot. Soft targets (e.g. from "
            "mixup) require a credal-set combination rule that is not implemented yet."
        )
    return target.argmax(dim=-1)


def label_relaxation_loss(
    logits: Tensor,
    target: Tensor,
    alpha: float = 0.1,
    reduction: str = "mean",
) -> Tensor:
    """Functional label relaxation loss.

    Args:
        logits: Unnormalized scores of shape ``(..., num_classes)``. The class
            dimension is the last one, so token-level inputs ``(batch, seq, vocab)``
            work without reshaping.
        target: Class indices of shape ``(...,)`` (any integer dtype), or an
            exactly one-hot float tensor of shape ``(..., num_classes)``.
        alpha: Imprecisiation degree in ``(0, 1)``; the credal set requires
            ``p_y >= 1 - alpha``. ``alpha -> 0`` recovers cross-entropy.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Returns:
        Scalar loss for ``"mean"``/``"sum"``, or a tensor of shape ``(...,)``
        (the logits shape without the class dimension) for ``"none"``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}.")
    if logits.dim() < 1 or logits.shape[-1] < 2:
        raise ValueError(f"logits must have a class dimension of size >= 2, got shape {tuple(logits.shape)}.")

    index = _targets_to_indices(logits, target)

    # Upcast half-precision inputs to float32 (fp16/bf16-safe under autocast);
    # float32/float64 inputs are kept as-is.
    if logits.dtype not in (torch.float32, torch.float64):
        logits = logits.float()
    log_z = torch.logsumexp(logits, dim=-1)
    log_p_y = logits.gather(-1, index.unsqueeze(-1)).squeeze(-1) - log_z

    # log(1 - p_y) via logsumexp over the non-target logits; stable even when p_y ~ 1.
    target_mask = F.one_hot(index, num_classes=logits.shape[-1]).bool()
    log_s = torch.logsumexp(logits.masked_fill(target_mask, float("-inf")), dim=-1) - log_z

    divergence = (1.0 - alpha) * (math.log(1.0 - alpha) - log_p_y) + alpha * (math.log(alpha) - log_s)
    loss = torch.where(log_p_y >= math.log(1.0 - alpha), divergence.new_zeros(()), divergence)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class LabelRelaxationLoss(nn.Module):
    """Label relaxation loss module; see :func:`label_relaxation_loss`.

    Drop-in replacement for ``nn.CrossEntropyLoss(label_smoothing=...)`` in the
    common ``(batch, num_classes)`` + index-target setting, except that the
    class dimension is the last one.

    Example:
        >>> criterion = LabelRelaxationLoss(alpha=0.1)
        >>> loss = criterion(model(x), y)
    """

    def __init__(self, alpha: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if reduction not in _REDUCTIONS:
            raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}.")
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return label_relaxation_loss(logits, target, alpha=self.alpha, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, reduction={self.reduction!r}"
