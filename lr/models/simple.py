from tensorflow.keras import layers
from tensorflow.keras import models


def get_simple_dense(input_shape, classes=10, final_activation='softmax', weights=None):
    if weights is not None:
        raise NotImplementedError('Weight load is not implemented.')

    input = layers.Input(input_shape)
    x = layers.Flatten()(input)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(1024, activation="relu")(x)
    out = layers.Dense(classes, activation=final_activation)(x)
    return models.Model(input, out)
