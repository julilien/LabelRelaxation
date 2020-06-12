from sklearn.model_selection import train_test_split
import numpy as np
from keras import datasets

from lr.models.models_meta import StringEnum


class Dataset(StringEnum):
    CIFAR_10 = "cifar10"
    CIFAR_100 = "cifar100"
    IMAGENET = "imagenet"
    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"


def get_dataset_type_by_name(dataset_name):
    if dataset_name == Dataset.CIFAR_10.value:
        return Dataset.CIFAR_10
    elif dataset_name == Dataset.CIFAR_100.value:
        return Dataset.CIFAR_100
    elif dataset_name == Dataset.IMAGENET.value:
        return Dataset.IMAGENET
    elif dataset_name == Dataset.MNIST.value:
        return Dataset.MNIST
    elif dataset_name == Dataset.FASHION_MNIST.value:
        return Dataset.FASHION_MNIST
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))


def get_dataset_by_type(dataset_type, seed):
    if dataset_type == Dataset.CIFAR_10:
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        num_classes = 10
        test_size = 1 / 6
    elif dataset_type == Dataset.CIFAR_100:
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        num_classes = 100
        test_size = 1 / 6
    elif dataset_type == Dataset.IMAGENET:
        raise NotImplementedError("ImageNet not provided yet.")
    elif dataset_type == Dataset.MNIST:
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        # Add channel dimension
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        num_classes = 10

        test_size = 1 / 7
    elif dataset_type == Dataset.FASHION_MNIST:
        (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
        num_classes = 10

        # Add channel dimension
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        test_size = 1 / 7
    else:
        raise ValueError("Unknown dataset type: {}".format(dataset_type))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed,
                                                        shuffle=True)

    return (x_train, y_train), (x_test, y_test), num_classes
