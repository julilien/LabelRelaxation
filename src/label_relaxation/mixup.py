"""Mixup-compatible label relaxation loss.

Label relaxation for a mixed (mixup / CutMix) target ``lam * e_i + (1 - lam) * e_j``: the
credal set becomes ``S = { p in simplex : p_i >= lam*(1-alpha), p_j >= (1-lam)*(1-alpha) }``
and the loss is the KL projection ``min_{q in S} KL(q || p_hat)`` (zero inside ``S``). The
projection has a four-case closed form; see ``docs/mixup_lr_derivation.md`` for the derivation,
the KKT case analysis, and the limiting cases (lam -> 0/1 recover plain LR; alpha -> 0 recovers
mixup cross-entropy; i == j reduces to plain LR).

All log-domain quantities come from ``log_softmax`` / masked ``logsumexp``, so every branch is
finite for finite logits and the ``torch.where`` selection does not poison gradients — the same
stability contract as :mod:`label_relaxation.loss`.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["MixupLabelRelaxationLoss", "mixup_label_relaxation_loss"]

_REDUCTIONS = ("mean", "sum", "none")


def mixup_label_relaxation_loss(
    logits: Tensor,
    target_a: Tensor,
    target_b: Tensor,
    lam,
    alpha: float = 0.1,
    reduction: str = "mean",
) -> Tensor:
    """Functional mixup label relaxation loss.

    Args:
        logits: ``(..., num_classes)`` unnormalized scores.
        target_a, target_b: class indices ``(...,)`` (the two mixup components).
        lam: mixing weight in ``[0, 1]``; python scalar or a tensor broadcastable to ``(...,)``.
        alpha: imprecisiation degree in ``(0, 1)``.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}.")
    if logits.dim() < 1 or logits.shape[-1] < 2:
        raise ValueError(f"logits must have a class dimension of size >= 2, got {tuple(logits.shape)}.")

    lead = logits.shape[:-1]
    k = logits.shape[-1]
    flat = logits.reshape(-1, k)
    ia = target_a.reshape(-1).long()
    ja = target_b.reshape(-1).long()
    if ia.shape != (flat.shape[0],) or ja.shape != (flat.shape[0],):
        raise ValueError("target_a/target_b must have shape logits.shape[:-1].")

    if flat.dtype not in (torch.float32, torch.float64):
        flat = flat.float()
    lam_t = torch.as_tensor(lam, dtype=flat.dtype, device=flat.device).reshape(-1)
    if lam_t.numel() == 1:
        lam_t = lam_t.expand(flat.shape[0])
    if torch.any((lam_t < 0) | (lam_t > 1)):
        raise ValueError("lam must be in [0, 1].")

    a = lam_t * (1.0 - alpha)
    b = (1.0 - lam_t) * (1.0 - alpha)

    logp = F.log_softmax(flat, dim=-1)
    log_z = torch.logsumexp(flat, dim=-1)  # == 0 after log_softmax offset; kept for clarity
    n = flat.shape[0]
    rows = torch.arange(n, device=flat.device)
    logpi = logp[rows, ia]
    logpj = logp[rows, ja]
    pi, pj = logpi.exp(), logpj.exp()

    # Stable log(1 - p_i), log(1 - p_j), and log(rest) = log(1 - p_i - p_j) via masked logsumexp.
    i_mask = F.one_hot(ia, k).bool()
    j_mask = F.one_hot(ja, k).bool()
    log1mi = torch.logsumexp(flat.masked_fill(i_mask, float("-inf")), dim=-1) - log_z
    log1mj = torch.logsumexp(flat.masked_fill(j_mask, float("-inf")), dim=-1) - log_z
    log_rest = torch.logsumexp(flat.masked_fill(i_mask | j_mask, float("-inf")), dim=-1) - log_z
    log_rest = torch.clamp(log_rest, min=-60.0)  # -inf only when it is not the selected branch

    xlogy = torch.special.xlogy
    # Single-coordinate raises.
    Ci = (1.0 - a) / (1.0 - pi)
    Cj = (1.0 - b) / (1.0 - pj)
    di, dj = pi < a, pj < b  # coordinate deficient (constraint violated)

    both_active = (di & dj) | (di & ~dj & (Ci * pj < b)) | (~di & dj & (Cj * pi < a))
    i_only = di & ~dj & ~both_active
    j_only = ~di & dj & ~both_active

    loss_i = xlogy(a, a) - a * logpi + xlogy(1.0 - a, 1.0 - a) - (1.0 - a) * log1mi
    loss_j = xlogy(b, b) - b * logpj + xlogy(1.0 - b, 1.0 - b) - (1.0 - b) * log1mj
    loss_both = (xlogy(a, a) - a * logpi + xlogy(b, b) - b * logpj
                 + alpha * math.log(alpha) - alpha * log_rest)

    zero = flat.new_zeros(n)
    loss = torch.where(i_only, loss_i, zero)
    loss = torch.where(j_only, loss_j, loss)
    loss = torch.where(both_active, loss_both, loss)

    # Same-class mix (i == j): single constraint p_i >= 1 - alpha, i.e. plain label relaxation.
    plain = ((1.0 - alpha) * (math.log(1.0 - alpha) - logpi) + alpha * (math.log(alpha) - log1mi))
    plain = torch.where(logpi >= math.log(1.0 - alpha), zero, plain)
    loss = torch.where(ia == ja, plain, loss)

    loss = loss.reshape(lead)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class MixupLabelRelaxationLoss(nn.Module):
    """Module form of :func:`mixup_label_relaxation_loss`.

    Example:
        >>> criterion = MixupLabelRelaxationLoss(alpha=0.1)
        >>> loss = criterion(logits, y_a, y_b, lam)
    """

    def __init__(self, alpha: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if reduction not in _REDUCTIONS:
            raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}.")
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: Tensor, target_a: Tensor, target_b: Tensor, lam) -> Tensor:
        return mixup_label_relaxation_loss(
            logits, target_a, target_b, lam, alpha=self.alpha, reduction=self.reduction
        )

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, reduction={self.reduction!r}"
