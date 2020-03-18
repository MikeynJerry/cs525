import numpy as np  # Used for various array manipulations
import argparse  # Used to read in command line arguments
from models import *  # Contains task models
from utils import *  # Used for image preprocessing
from tensorflow.keras.datasets import fashion_mnist  # Fashion MNIST dataset
from constants import *
from plotting import *  # Plotting functions for data, confusion matrices, and clothes
from training import *  # Model training functions


def main(task):

    # Load Fashion MNIST
    train_data, test_data = preprocess(fashion_mnist.load_data())
    x_train, y_train = train_data
    x_test, y_test = test_data

    get_model = globals()[f"task_{task}_model"]

    if task == 1:
        train_data = (x_train.reshape(-1, 784), y_train)
        test_data = (x_test.reshape(-1, 784), y_test)

    config_dict = {"optimizer": [opt() for opt in optimizers]}

    if task == 5:
        train_data = (x_train, None)
        test_data = (x_test, None)
    else:
        config_dict["loss"] = losses

    configs = list(product_dict(**config_dict))
    data, models = config_trainer(get_model, train_data, test_data, configs)
    plot_data(data, task=task, configs=configs)

    if task == 5:
        encoder, decoder = best_vae(models, x_test)
        plot_clothes(encoder, decoder, kernel_size=3, kernels=16)
    else:
        test(models, test_data, configs=configs)
        plot_cm(models, test_data, task=task, configs=configs)

    # Change VAE for qualitative evaluation
    if task == 5:
        data, models = config_trainer(
            get_model,
            train_data,
            test_data,
            configs,
            filters=32,
            n_latent=10,
            kernel_size=5,
        )
        plot_data(data, task=task, configs=configs)
        encoder, decoder = best_vae(models, x_test)
        plot_clothes(encoder, decoder, kernel_size=5, kernels=32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=int, choices=range(1, 6))
    args = parser.parse_args()
    main(args.task)
