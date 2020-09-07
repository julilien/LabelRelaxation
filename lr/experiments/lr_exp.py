import logging
import os
import numpy as np

import mlflow
import tensorflow.keras as keras
from keras.callbacks import ReduceLROnPlateau, TerminateOnNaN
from keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from lr.data.io_utils import get_dataset_type_by_name, get_dataset_by_type
from lr.losses.conf_penalty_loss import ConfidencePenaltyLoss
from lr.losses.focal_loss import FocalLoss
from lr.losses.losses_meta import LossType
from lr.losses.lr_loss import LabelRelaxationLoss
from lr.metrics.ece import evaluate_ece
from lr.models.models_meta import get_backbone_model_fn_by_type
from lr.utils.tracking_utils import get_model_checkpoint_path, get_tensorboard_path
from lr.utils.training_utils import LearningRateScheduleProvider, conduct_gen_model_training, \
    conduct_simple_model_training


def get_hyperparameter_file_path(model_params):
    if model_params.get_parameter("loss_type") == LossType.LR:
        return 'misc/hyperparams_lr.json'
    elif model_params.get_parameter("loss_type") == LossType.FOCAL:
        return 'misc/hyperparams_focal.json'
    elif model_params.get_parameter("loss_type") == LossType.CONFIDENCE_PENALTY:
        return 'misc/hyperparams_conf_pen.json'
    else:
        return 'misc/hyperparams_ce.json'


def preprocess_data(x, y, num_classes, pixel_mean=None, train=False):
    # Normalize data
    x = x.astype('float32') / 255.

    if pixel_mean is None and train:
        pixel_mean = np.mean(x, axis=0)
    elif pixel_mean is None:
        pixel_mean = 0
    x -= pixel_mean

    # Convert class vectors to binary class matrices
    y = keras.utils.to_categorical(y, num_classes)

    return x, y, pixel_mean


def perform_run(model_params, cluster_job, model_checkpoints, config=None):
    # Load data
    dataset_type = get_dataset_type_by_name(model_params.get_parameter("dataset_name"))
    (x_train, y_train), (x_test, y_test), num_classes = get_dataset_by_type(dataset_type,
                                                                            model_params.get_parameter("seed"))
    x_val, y_val = None, None
    x_scaling, y_scaling = None, None

    if model_params.get_parameter("temp_scaling", False):
        x_train, x_scaling, y_train, y_scaling = train_test_split(x_train, y_train, test_size=0.15,
                                                                  random_state=model_params.get_parameter("seed"))

    if not model_params.get_parameter("test_run"):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          random_state=model_params.get_parameter("seed"))

    x_train, y_train, pixel_mean = preprocess_data(x_train, y_train, num_classes, None,
                                                   model_params.get_parameter("subtract_pixel_mean"))
    if not model_params.get_parameter("test_run"):
        x_val, y_val, _ = preprocess_data(x_val, y_val, num_classes, pixel_mean)
    else:
        x_test, y_test, _ = preprocess_data(x_test, y_test, num_classes, pixel_mean)

    if model_params.get_parameter("temp_scaling", False):
        x_scaling, y_scaling, _ = preprocess_data(x_scaling, y_scaling, num_classes, pixel_mean)

    input_shape = x_train.shape[1:]

    # Instantiate model
    model_type = model_params.get_parameter("model_type")
    model_fn = get_backbone_model_fn_by_type(model_type)

    if model_params.get_parameter("temp_scaling", False):
        final_activation = None
    else:
        final_activation = "softmax"

    model = model_fn(classes=num_classes, weights=None, input_shape=input_shape, final_activation=final_activation)

    if model_params.get_parameter("loss_type") == LossType.LR:
        loss_fn = LabelRelaxationLoss(alpha=model_params.get_parameter("alpha"), n_classes=num_classes)
    elif model_params.get_parameter("loss_type") == LossType.FOCAL:
        loss_fn = FocalLoss(alpha=model_params.get_parameter("alpha"))
    elif model_params.get_parameter("loss_type") == LossType.CONFIDENCE_PENALTY:
        loss_fn = ConfidencePenaltyLoss(alpha=model_params.get_parameter("alpha"))
    else:
        loss_fn = CategoricalCrossentropy(label_smoothing=model_params.get_parameter("alpha"),
                                          from_logits=model_params.get_parameter("temp_scaling", False))

    lr_sched_prov = LearningRateScheduleProvider(init_lr=model_params.get_parameter("initial_lr"),
                                                 steps=model_params.get_parameter("steps", [80, 120, 160, 180]),
                                                 multiplier=model_params.get_parameter("lr_sched_multipler", 0.1))

    opt_params = {}
    if model_params.get_parameter("gradient_clipping") > 0.0:
        opt_params["clipvalue"] = model_params.get_parameter("gradient_clipping")
    opt_params["decay"] = model_params.get_parameter("decay")

    optimizer = SGD(learning_rate=lr_sched_prov.get_lr_schedule(0), momentum=0.9, nesterov=True, **opt_params)

    model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Callbacks
    callbacks = []
    if model_params.get_parameter("reduce_on_plateau"):
        callbacks.append(ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6))
    callbacks.append(LearningRateScheduler(lr_sched_prov.get_lr_schedule))

    if model_checkpoints and not model_params.get_parameter("test_run"):
        save_dir = get_model_checkpoint_path(config)
        model_name = '{}_{}_model.h5'.format(dataset_type.value, model_type)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        callbacks.append(ModelCheckpoint(filepath=filepath,
                                         monitor='val_accuracy',
                                         verbose=1,
                                         save_best_only=True))

    if not cluster_job:
        callbacks.append(TensorBoard(log_dir=get_tensorboard_path(config)))

    callbacks.append(TerminateOnNaN())

    verbosity = 1
    if cluster_job:
        verbosity = 0

    val_data = None
    if not model_params.get_parameter("test_run"):
        val_data = (x_val, y_val)

    # Training
    if not model_params.get_parameter("data_augmentation"):
        logging.info('Train without data augmentation...')
        conduct_simple_model_training(model, x_train, y_train, validation_data=val_data,
                                      model_parameters=model_params, callbacks=callbacks, shuffle=True,
                                      verbose=verbosity)
    else:
        logging.info('Using data augmentation.')
        datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False,
                                     featurewise_std_normalization=False, samplewise_std_normalization=False,
                                     zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.1,
                                     height_shift_range=0.1, shear_range=0., zoom_range=0., channel_shift_range=0.,
                                     fill_mode='nearest', cval=0., horizontal_flip=True, vertical_flip=False,
                                     rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0)
        datagen.fit(x_train)

        conduct_gen_model_training(model,
                                   datagen.flow(x_train, y_train, batch_size=model_params.get_parameter("batch_size")),
                                   val_data, model_parameters=model_params, callbacks=callbacks, verbose=verbosity)
    # Score trained model
    if model_params.get_parameter("test_run"):
        scores = model.evaluate(x_test, y_test, verbose=verbosity)
        logging.info('Test accuracy: {}'.format(scores[1]))
        mlflow.log_metric("test_accuracy", scores[1])

        # ECE evaluation
        preds_test = model.predict(x_test)
        if not model_params.get_parameter("temp_scaling", False):
            ece, _ = evaluate_ece(preds_test, y_test, n_bins=15, temp_scaling=False)
        else:
            scaling_preds = model.predict(x_scaling)

            ece, opt_t = evaluate_ece(preds_test, y_test, n_bins=15, temp_scaling=True, preds_val=scaling_preds,
                                      y_val=y_scaling)
            mlflow.log_param("T", opt_t)

        mlflow.log_metric("ece", ece)

        return scores[1], ece
    else:
        # Return validation error
        scores = model.evaluate(x_val, y_val, verbose=verbosity)

        # ECE evaluation
        preds_val = model.predict(x_val)
        if not model_params.get_parameter("temp_scaling", False):
            ece, _ = evaluate_ece(preds_val, y_val, n_bins=15, temp_scaling=False)
        else:
            scaling_preds = model.predict(x_scaling)

            ece, opt_t = evaluate_ece(preds_val, y_val, n_bins=15, temp_scaling=True, preds_val=scaling_preds,
                                      y_val=y_scaling)
            mlflow.log_param("T", opt_t)

        mlflow.log_metric("ece", ece)

        return scores[1], ece
