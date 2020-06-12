from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras import regularizers


def get_vgg16(input_shape, final_activation='softmax', weights=None, classes=10, weight_decay=0.0005):
    if weights is not None:
        raise NotImplementedError('Weight load is not implemented.')

    model = Sequential()

    model.add(layers.Conv2D(64, (3, 3), padding='same',
                            input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes))
    model.add(layers.Activation(final_activation))
    return model
