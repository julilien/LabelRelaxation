from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten


def get_simple_dense(input_shape, classes=10, final_activation='softmax', weights=None):
    if weights is not None:
        raise NotImplementedError('Weight load is not implemented.')

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(1024, activation="relu", input_shape=input_shape))
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(classes, activation=final_activation))
    return model