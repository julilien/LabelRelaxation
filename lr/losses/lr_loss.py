import tensorflow as tf

from keras.losses import LossFunctionWrapper, kullback_leibler_divergence
from keras import backend as K


class LabelRelaxationLoss(LossFunctionWrapper):
    """
    Optimistic superset loss for probability-based loss functions based on Q sets. This loss is parameterized by
    alpha, which indicates the possibility assigned to the classes that are not the actual target. For the actual class,
    a possibility value of 1 is given. The resulting possibility measure \pi^y is used to specify the bounds of the
    underlying new target set Q.
    The final loss is then given as
        L*(Q_{\pi^y}, \hat{p}) = min_{p \in Q_{\pi^y}} L(p, \hat{p}),
    where \hat{p} is the predicted probability vector and y indicates the actual (correct) target class.
    """

    def __init__(self, alpha=0.1, debug=False, uniform_penalty=False, beta=1., n_classes=None):
        super(LabelRelaxationLoss, self).__init__(fn=self.loss, name="lr_loss")
        self.debug = debug
        self.alpha = alpha
        self.uniform_penalty = uniform_penalty
        self.beta = beta
        self.n_classes = n_classes
        if self.uniform_penalty:
            assert n_classes is not None, "If the uniform penalty is used, the number of classes must be specified!"

    def loss(self, y_true, y_pred):
        if self.debug:
            y_true = tf.compat.v1.Print(y_true, [y_true], "y_true=", summarize=-1)
            y_pred = tf.compat.v1.Print(y_pred, [y_pred], "y_pred=", summarize=-1)

        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())

        sum_y_hat_prime = tf.reduce_sum((1. - y_true) * y_pred, axis=-1)

        y_pred_hat = self.alpha * y_pred / (tf.expand_dims(sum_y_hat_prime, axis=-1) + K.epsilon())

        y_true_credal = tf.where(tf.greater(y_true, 0.1), 1. - self.alpha, y_pred_hat)

        divergence = kullback_leibler_divergence(y_true_credal, y_pred)

        preds = tf.reduce_sum(y_pred * y_true, axis=-1)

        # Uniform penalty
        if self.uniform_penalty and self.alpha > 0:
            class_dim = self.n_classes
            uniform_dist = tf.where(tf.greater(y_true, 0.1), 1 - self.alpha, self.alpha / (class_dim - 1))

            if self.debug:
                uniform_dist = tf.compat.v1.Print(uniform_dist, [uniform_dist], "uniform_dist=", summarize=-1)

            penalty = self.beta * ((1 - self.alpha) - y_pred[y_true >= 0.1]) * kullback_leibler_divergence(uniform_dist,
                                                                                                           y_pred)
            if self.debug:
                penalty = tf.compat.v1.Print(penalty, [penalty], "penalty=", summarize=-1)

            divergence += penalty

        result = tf.where(tf.greater_equal(preds, 1. - self.alpha), tf.zeros_like(divergence),
                          divergence)
        if self.debug:
            result = tf.compat.v1.Print(result, [result], "result=")

        return result
