import configparser
import os
import mlflow
import mlflow.tensorflow
import logging
import tensorflow as tf

from lr.utils.time_utils import get_curr_date_str

ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../.."
CONFIG_FILE = 'conf/run.ini'
CONFIG_FILE_ENCODING = 'utf-8-sig'


def get_config(path=None):
    config = configparser.ConfigParser()
    if path is None:
        path = os.path.join(ROOT_DIR, CONFIG_FILE)
    config.read(path, encoding=CONFIG_FILE_ENCODING)
    return config


def init_mlflow(config, tracking_uri=None, experiment_name=None):
    if tracking_uri is None:
        tracking_uri = config["MLFLOW"]["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)

    if experiment_name is None:
        today_str = get_curr_date_str()
        experiment_name = "{}_{}".format(today_str, config["MLFLOW"]["MLFLOW_EXP_PREFIX"])
    mlflow.set_experiment(experiment_name)


def init_tensorflow(seed, use_float16=False):
    if use_float16:
        import keras.backend as K

        dtype = 'float16'
        K.set_floatx(dtype)
        K.set_epsilon(1e-4)

    # Solves an issue with regard to the use of newest CUDA versions
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    _ = InteractiveSession(config=config)

    tf.random.set_seed(seed)


def init_env(tracking_uri=None, experiment_name=None, autolog_freq=100, seed=0, use_float16=False):
    config = get_config()
    init_mlflow(config, tracking_uri, experiment_name)

    if config["LOGGING"]["LOG_LEVEL"] == "DEBUG":
        log_level = logging.DEBUG
    elif config["LOGGING"]["LOG_LEVEL"] == "INFO":
        log_level = logging.INFO
    elif config["LOGGING"]["LOG_LEVEL"] == "WARNING":
        log_level = logging.WARNING
    elif config["LOGGING"]["LOG_LEVEL"] == "ERROR":
        log_level = logging.ERROR
    else:
        raise ValueError(
            "Unknown log level provided in the configuration file: {}".format(config["LOGGING"]["LOG_LEVEL"]))
    logging.basicConfig(level=log_level)

    init_tensorflow(seed, use_float16=use_float16)

    # Enable auto-logging to MLflow to capture TensorBoard metrics
    mlflow.tensorflow.autolog(every_n_iter=autolog_freq)

    return config
