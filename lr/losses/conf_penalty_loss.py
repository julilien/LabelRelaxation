from tensorflow.keras import backend as K
from tensorflow.python.keras.losses import LossFunctionWrapper


class ConfidencePenaltyLoss(LossFunctionWrapper):
    def __init__(self, alpha=0.0):
        super(ConfidencePenaltyLoss, self).__init__(fn=self.loss, name="confidence_penalty")
        # alpha represents beta
        self.alpha = alpha

    def loss(self, y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        entropy = -K.sum(y_pred * K.log(y_pred), axis=-1)
        loss = K.sum(y_true * K.log(y_pred), axis=-1) - self.alpha * entropy

        return -loss
