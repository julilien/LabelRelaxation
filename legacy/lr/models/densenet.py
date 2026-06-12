import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Convolution2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, ZeroPadding2D, Dense, Dropout, Activation, Convolution2D
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, BatchNormalization


def custDenseNet121_2(weights='imagenet', input_shape=None, classes=1000, final_activation='softmax', **kwargs):
    if weights is not None:
        raise NotImplementedError('Weight load is not implemented.')
    return densenet_model(shape=input_shape, final_activation=final_activation, classes=classes)


def densenet_model(growth_rate=12, nb_layers=[16, 16, 16], reduction=0.5, dropout_rate=0.0, classes=16,
                   shape=(32, 32, 3), final_activation='softmax'):
    # compute compression factor
    compression = 1.0 - reduction
    nb_dense_block = len(nb_layers)
    nb_filter = 2 * growth_rate

    img_input = Input(shape=shape, name='data')

    x = Convolution2D(2 * growth_rate, 3, 1, name='conv1', use_bias=False)(img_input)

    stage = 0
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate)

    x = BatchNormalization(name='conv_final_blk_bn', )(x)
    x = Activation('relu', name='relu_final_blk')(x)

    x = GlobalAveragePooling2D(name='pool_final')(x)
    output = Dense(classes, name='fc6', activation=final_activation)(x)

    return Model(inputs=img_input, outputs=output)


def conv_block(x, stage, branch, nb_filter, dropout_rate=None):
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(name=conv_name_base + '_x1_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Convolution2D(inter_channel, 1, 1, name=conv_name_base + '_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(name=conv_name_base + '_x2_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, 3, 1, name=conv_name_base + '_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None,
                grow_nb_filters=True):
    concat_feat = x
    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate)
        concat_feat = tf.concat([concat_feat, x], -1)

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None):
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(name=conv_name_base + '_bn')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), name=pool_name_base)(x)

    return x
