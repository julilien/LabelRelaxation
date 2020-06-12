from lr.models.models_meta import StringEnum


class LossType(StringEnum):
    LR = "LabelRelaxation"
    CATEGORICAL_CROSSENTROPY = "CategoricalCrossentropy"


def get_loss_type_by_name(loss_name, case_sensitive=False):
    if not case_sensitive:
        loss_name = loss_name.lower()
    if loss_name == "lr" or loss_name == "label_relaxation":
        return LossType.LR
    elif loss_name == "categoricalcrossentropy" or loss_name == "categorical_cross_entropy" \
            or loss_name == "cross_entropy" or loss_name == "categorical_crossentropy" or \
            loss_name == "crossentropy":
        return LossType.CATEGORICAL_CROSSENTROPY
    else:
        raise ValueError("Unknown loss type: {}".format(loss_name))
