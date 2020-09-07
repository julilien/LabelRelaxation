# From Label Smoothing to Label Relaxation

A (Tensorflow 2) implementation of the novel *label relaxation* approach as presented in the corresponding paper "From Label Smoothing to Label Relaxation" submitted to AAAI 2021.

## Requirements

A detailed list of requirements can be found in `requirements.txt`. To install all required packages, one simply has to call
```
pip install -r requirements.txt
```

Note that the code has been tested with Python 3.7 on Ubuntu 18.04 and Python 3.8 on Ubuntu 20.04. Since we tried to avoid using any system-dependent call or library, we expect the code to be running also on other systems, such as Windows and MacOS.

In some cases, we've experienced issues with the MySQL adapter package for Python 3.*, for which the pip package install was not sufficient to run the code. On Linux systems, this package may require to install additional system-dependent sources (e.g., for Ubuntu, we also had to run `sudo apt install build-essential python-dev libmysqlclient-dev`).

## Repository structure

This repository provides a reimplementation of the models, losses, etc., that were used for the empirical evaluation of our _label relaxation_ approach.

The following components can be used:

- `lr.data` provides a data loader for the used datasets MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100
- `lr.experiments` provides the implementation of experiment runs (incl. hyperparameter optimization)
- `lr.losses` provides our label relaxation loss and (re-)implementations of the confidence penalty and focal loss
- `lr.metrics` provides the implementation of the ECE calculation
- `lr.models` provides all our model adaptions for our study
- `lr.utils` provides means to initialize the environment, and further time, tracking and time utils

The hyperparameters used within our studies are provided in the corresponding JSON file in `misc/`.

## Training and Evaluation

Our implementation evaluates every model after finishing the training with regard to the classification rate and expected calibration error. To start a simple run without hyperparameter optimization, you just need to run 

```python3 lr/experiments/lr_study.py [args]```

For hyperparameter runs, you need to run

```python3 lr/experiments/lr_study_ho.py [args]```

For our implementation, we used [Click](https://click.palletsprojects.com/en/7.x/) to provide a convenient CLI for passing parameters. Therefore, you can print out all possible program arguments with the `--help` parameter.

As an example, to run our loss on fashion_mnist using the simple dense architecture for the seed 42, you have to run the following command:

```python3 lr/experiments/lr_study_ho.py --model_name simple_dense --loss_name lr --load_ext_params True --dataset_name fashion_mnist --seed 42```

Here, `load_ext_params` enables loading the hyperparameters from the corresponding JSON file in `misc/`. Make sure to activate this in order to retrieve the correct hyperparameters as used within the experiments.

## Results

The exhaustive result tables can be found in the evaluation section of the paper. Due to space limitations, we refer to this for an overview. As described in the paper, we evaluated the simple dense model on MNIST and Fashion MNIST with 10 different seeds (the corresponding parameter can be set; we used seeds 0 to 10), while we trained the bigger models on CIFAR-10 and CIFAR-100 for 5 times each (seeds 0 to 5). 

As used within our code, [Mlflow](https://mlflow.org/) allows to efficiently aggregate the produced results. We used this framework to track our results.