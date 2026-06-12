# Label Relaxation

A modern, tested PyTorch implementation of the **label relaxation** loss from

> Julian Lienen and Eyke Hüllermeier. **From Label Smoothing to Label Relaxation.** AAAI 2021.
> [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17041)

Label relaxation replaces the precise (possibly smoothed) target distribution with a
**credal set** of distributions — all distributions assigning at least `1 - alpha` to the
observed class. The loss is zero whenever the prediction lies inside this set, and otherwise
penalizes the KL divergence to the set's nearest member. Compared to label smoothing, this
avoids penalizing confident-correct predictions and yields better-calibrated classifiers.

This repository contains the maintained `label-relaxation` package (PyTorch) and, under
[`legacy/`](legacy/), the original code of the AAAI 2021 paper (TensorFlow 2), kept frozen
for reproducibility — see [`legacy/README.md`](legacy/README.md) for the paper experiments
and the [supplementary material](Lienen_AAAI21_LabelRelaxation_Supplement.pdf).

## Installation

```bash
pip install label-relaxation
```

## Usage

```python
from label_relaxation import LabelRelaxationLoss

criterion = LabelRelaxationLoss(alpha=0.1)  # drop-in for nn.CrossEntropyLoss
loss = criterion(model(x), y)               # logits (..., C), integer targets (...)
```

A functional form is also available:

```python
from label_relaxation import label_relaxation_loss

loss = label_relaxation_loss(logits, targets, alpha=0.1, reduction="mean")
```

Inputs are unnormalized logits with the class dimension **last**, so token-level inputs of
shape `(batch, seq_len, vocab)` work without reshaping. Targets are class indices of shape
`(...)` (the logits shape without the class dimension) or exactly one-hot float vectors of
the same shape as the logits. Soft targets (e.g. from mixup) are intentionally rejected for
now — mixing credal targets requires a set-combination rule that is future work.

## What's different from the original implementation?

The package is a from-scratch reimplementation, numerically equivalent to the original paper
code (the test suite checks values and gradients against the frozen `legacy/` implementation),
but:

- **Closed form.** For one-hot targets the projected KL divergence collapses to
  `(1-α)·log((1-α)/p_y) + α·log(α/(1-p_y))` — it depends only on the predicted probability
  of the true class. The implementation computes this directly from `log_softmax` outputs,
  with `log(1-p_y)` obtained via a masked `logsumexp`.
- **Numerically stable.** No `softmax().log()` round trip; safe for extreme logits and for
  fp16/bf16 inputs under autocast (the loss is computed in float32 internally).
- **No magic constants.** The original identified the positive class via a hardcoded
  `target > 0.1` threshold; targets are handled explicitly here.
- **Exactly zero loss *and* gradient** for predictions inside the credal set, by
  construction (covered by tests).

Note on gradients: the credal projection is the KL minimizer over the set, so detaching it
(as the original does) yields the same gradient as differentiating through it — the two
implementations agree in both value and gradient (see `tests/test_loss.py`).

## Development

```bash
uv sync        # installs CPU torch + dev dependencies
uv run pytest  # 50 tests, including equivalence with the legacy implementation
```

## Citation

```bibtex
@inproceedings{lienen2021label,
  author    = {Julian Lienen and Eyke H{\"{u}}llermeier},
  title     = {From Label Smoothing to Label Relaxation},
  booktitle = {Thirty-Fifth {AAAI} Conference on Artificial Intelligence},
  pages     = {8583--8591},
  year      = {2021}
}
```

## License

Apache 2.0
