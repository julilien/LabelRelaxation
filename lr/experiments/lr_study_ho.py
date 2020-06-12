import logging
import os
import numpy as np

import click
import mlflow
from tensorflow import random

from lr.experiments.lr_exp import perform_run
from lr.losses.losses_meta import get_loss_type_by_name, LossType
from lr.models.models_meta import ModelParameters, get_model_type_by_name
from lr.utils.env import init_env, ROOT_DIR


@click.command()
@click.option('--model_name', default='vgg', help='Backbone model',
              type=click.Choice(['resnet', 'vgg', 'simple_dense', 'densenet'], case_sensitive=False))
@click.option('--loss_name', default='lr', help='Loss function to be evaluated',
              type=click.Choice(['lr', 'cross_entropy'], case_sensitive=False))
@click.option('--epochs', default=200)
@click.option('--batch_size', default=64)
@click.option('--seed', default=0)
@click.option('--data_augmentation', default=True, type=click.BOOL)
@click.option('--subtract_pixel_mean', default=True, type=click.BOOL)
@click.option('--autolog_freq', default=10)
@click.option('--cluster_job', default=False, type=click.BOOL)
@click.option('--dataset_name', default='cifar100', help='Data set',
              type=click.Choice(['cifar100', 'imagenet', 'cifar10', 'mnist', 'fashion_mnist'], case_sensitive=False))
@click.option('--decay', default=0.0, help='Weight decay')
@click.option('--gradient_clipping', default=0.0, type=click.FLOAT)
@click.option('--model_checkpoints', default=False,
              help='Indicator whether the currently best performing model should be saved.')
@click.option('--warmup', default=0)
@click.option('--initial_lr', default=1e-3)
@click.option('--lr_sched_multipler', default=0.1)
@click.option('--reduce_on_plateau', default=False, type=click.BOOL)
@click.option('--load_ext_params', default=False, type=click.BOOL)
@click.option('--optimize_ece', default=False, type=click.BOOL)
def perform_credal_loss_ho_experiment(model_name, loss_name, epochs, batch_size, seed, data_augmentation,
                                      subtract_pixel_mean, autolog_freq, cluster_job, dataset_name, decay,
                                      gradient_clipping, model_checkpoints, warmup, initial_lr, lr_sched_multipler,
                                      reduce_on_plateau, load_ext_params, optimize_ece):
    config = init_env(autolog_freq=autolog_freq, seed=seed)

    # Reduce logging output
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    with mlflow.start_run():
        # Meta parameters
        loss_type = get_loss_type_by_name(loss_name)
        model_params = ModelParameters()
        model_params.set_parameter("seed", seed)
        random.set_seed(model_params.get_parameter("seed"))
        model_params.set_parameter("batch_size", batch_size)
        model_params.set_parameter("epochs", epochs)
        model_params.set_parameter("model_type", get_model_type_by_name(model_name))
        model_params.set_parameter("loss_type", loss_type)
        model_params.set_parameter("dataset_name", dataset_name)
        model_params.set_parameter("data_augmentation", data_augmentation)
        model_params.set_parameter("subtract_pixel_mean", subtract_pixel_mean)
        model_params.set_parameter("decay", decay)
        model_params.set_parameter("gradient_clipping", gradient_clipping)
        model_params.set_parameter("warmup", warmup)
        model_params.set_parameter("initial_lr", initial_lr)
        model_params.set_parameter("lr_sched_multipler", lr_sched_multipler)
        model_params.set_parameter("reduce_on_plateau", reduce_on_plateau)
        model_params.set_parameter("load_ext_params", load_ext_params)
        model_params.set_parameter("optimize_ece", optimize_ece)
        if load_ext_params:
            if model_params.get_parameter("loss_type") == LossType.LR:
                hyper_params_path = 'misc/hyperparams_lr.json'
            else:
                hyper_params_path = 'misc/hyperparams_ce.json'

            model_params.load_parameters_from_file(os.path.join(ROOT_DIR, hyper_params_path),
                                                   '{}_{}'.format(model_params.get_parameter("model_type"), dataset_name),
                                                   exclude_keys=["alpha"])

        model_params.log_parameters()

        alpha_vals = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4]
        scores = np.zeros(len(alpha_vals))
        for idx, alpha_val in enumerate(alpha_vals):
            with mlflow.start_run(nested=True):
                run_model_params = model_params.duplicate()
                run_model_params.set_parameter("alpha", alpha_val)
                run_model_params.log_parameters()

                local_acc, local_ece = perform_run(run_model_params, cluster_job, model_checkpoints, config)
                if optimize_ece:
                    scores[idx] = local_ece
                else:
                    scores[idx] = local_acc

        # Get argmax of the scores
        if optimize_ece:
            best_idx = int(np.argmin(scores))
        else:
            best_idx = int(np.argmax(scores))
        model_params.set_parameter("alpha", alpha_vals[best_idx])
        mlflow.log_param("alpha", model_params.get_parameter("alpha"))

        model_params.set_parameter("test_run", True)
        final_score, final_ece = perform_run(model_params, cluster_job, model_checkpoints, config)
        return final_score


if __name__ == "__main__":
    perform_credal_loss_ho_experiment()
