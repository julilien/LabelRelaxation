"""torch-label-relaxation: label relaxation loss for PyTorch.

Reference: Lienen & Hüllermeier, "From Label Smoothing to Label Relaxation",
AAAI 2021. https://ojs.aaai.org/index.php/AAAI/article/view/17041
"""

from label_relaxation.loss import LabelRelaxationLoss, label_relaxation_loss

__version__ = "0.1.0"

__all__ = ["LabelRelaxationLoss", "label_relaxation_loss", "__version__"]
