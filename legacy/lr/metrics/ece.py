import numpy as np
from scipy.special import softmax
import logging


def evaluate_ece(preds_test, y_test, n_bins, temp_scaling=False,
                 temp_scaling_vals=[0.25, 0.5, 0.75, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.5, 3.],
                 preds_val=None, y_val=None):
    if temp_scaling:
        assert temp_scaling_vals is not None, "Temperature scaling values must not be one!"
        assert preds_val is not None, "For temperature scaling, the predictions for the validation must be given!"
        assert y_val is not None, "For temperature scaling, the targets for the validation must be given!"
    else:
        ece = calculate_ece(preds_test, y_test, n_bins)
        return ece, None

    scores = np.zeros(len(temp_scaling_vals))
    for idx, t_val in enumerate(temp_scaling_vals):
        transformed_preds = apply_temperature_scaling(preds_val, t_val)
        scores[idx] = calculate_ece(transformed_preds, y_val, n_bins)
        logging.info("ECE score for T={}: {}".format(t_val, scores[idx]))

    opt_t = temp_scaling_vals[int(np.argmin(scores))]
    transformed_test_preds = apply_temperature_scaling(preds_test, opt_t)
    final_ece = calculate_ece(transformed_test_preds, y_test, n_bins)

    return final_ece, opt_t


def apply_temperature_scaling(preds, t):
    transformed_preds = np.copy(preds)
    transformed_preds /= t
    return softmax(transformed_preds, axis=-1)


def calculate_ece(preds, y_test, n_bins):
    interval_step = 1 / n_bins

    accs = np.zeros([n_bins])
    confs = np.zeros([n_bins])
    indices = []
    for i in range(n_bins):
        indices.append([])

    for i in range(preds.shape[0]):
        conf = np.max(preds[i])

        # probability of 1.0 has to belong to last bin
        bin_idx = min(int(conf // interval_step), n_bins - 1)
        indices[bin_idx].append(i)

        if np.argmax(preds[i]) == np.argmax(y_test[i]):
            accs[bin_idx] += 1
        confs[bin_idx] += conf

    for i in range(n_bins):
        accs[i] /= max(len(indices[i]), 1)
        confs[i] /= max(len(indices[i]), 1)

    ece = 0
    for i in range(n_bins):
        ece += (len(indices[i]) / preds.shape[0]) * np.abs(accs[i] - confs[i])

    return ece
