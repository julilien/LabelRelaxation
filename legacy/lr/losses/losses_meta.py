from lr.models.models_meta import StringEnum


class LossType(StringEnum):
    LR = "LabelRelaxation"
    CATEGORICAL_CROSSENTROPY = "CategoricalCrossentropy"
    FOCAL = "FocalLoss"
    CONFIDENCE_PENALTY = "ConfidencePenalty"


def get_loss_type_by_name(loss_name, case_sensitive=False):
    if not case_sensitive:
        loss_name = loss_name.lower()
    if loss_name in ["lr", "label_relaxation"]:
        return LossType.LR
    elif loss_name in ["categoricalcrossentropy", "categorical_cross_entropy", "cross_entropy",
                       "categorical_crossentropy", "crossentropy"]:
        return LossType.CATEGORICAL_CROSSENTROPY
    elif loss_name in ["focal", "focal_loss"]:
        return LossType.FOCAL
    elif loss_name in ["confidencepenalty", "confidence_penalty"]:
        return LossType.CONFIDENCE_PENALTY
    else:
        raise ValueError("Unknown loss type: {}".format(loss_name))
