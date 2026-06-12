import mlflow
import os


def log_parameter_dict(param_dict):
    for key in param_dict:
        mlflow.log_param(key, param_dict[key])


def get_model_checkpoint_path(config):
    save_dir = os.path.join(config["DATA"]["CACHE_PATH_PREFIX"], 'saved_models')
    return os.path.join(save_dir, mlflow.active_run().info.run_id)


def get_tensorboard_path(config):
    return os.path.join(config["LOGGING"]["TENSORBOARD_LOG_DIR"], "{}_{}".format(mlflow.active_run().info.experiment_id,
                                                                                 mlflow.active_run().info.run_id))
