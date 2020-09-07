from keras import backend as K
from tensorflow.python.keras.losses import LossFunctionWrapper


class FocalLoss(LossFunctionWrapper):
    def __init__(self, alpha=0.0):
        super(FocalLoss, self).__init__(fn=self.loss, name="focal_loss")
        # alpha represent gamma
        self.alpha = alpha

    def loss(self, y_true, y_pred):
        # Clipping for numerical stability
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Cross-entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Focal loss
        loss = K.pow(1 - y_pred, self.alpha) * cross_entropy

        return K.sum(loss, axis=-1)
