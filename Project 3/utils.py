from tensorflow.keras import backend as K  # Used for VAE reparameterization trick
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import (
    to_categorical,
)  # Used for min-max scaling preprocessing
import numpy as np  # Used for various array manipulations
from itertools import product


# Calculates the accuracy of a model based on predictions and truths
def accuracy(y_true, y_pred):
    return (
        np.sum(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)) / y_true.shape[0]
    )


def best_vae(models, x_test):

    min_loss, best_model = 1e10, None

    # Find model with highest accuracy
    for model in models:
        loss = model.evaluate(x_test)
        if loss < min_loss:
            min_loss, best_model = loss, model

    encoder = best_model.get_layer("encoder")
    decoder = best_model.get_layer("decoder")

    return encoder, decoder


# Generates config labels
def get_labels(configs):
    labels = []
    for config in configs:
        label = ""
        for key, val in config.items():
            label += f"{key.title()} - "
            if not hasattr(val, "__name__"):
                val = val.__class__
            label += f"{val.__name__.replace('_', ' ').title()}, "
        labels.append(label[:-2])
    return labels


# Preprocess data for model consumption
# Intended to be called with <dataset>.load_data() as input
def preprocess(data):
    x_train, y_train = data[0]
    x_test, y_test = data[1]

    # Min-Max scaling according to Project 3 specifications
    x_min, x_max = x_train.min(), x_train.max()
    x_train = (x_train - x_min) / (x_max - x_min)
    x_test = (x_test - x_min) / (x_max - x_min)

    # Convert data to (n_samples, width, height, depth) format for CNNs and VAE
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Convert labels to one-hot encodings
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)


# Generates config dicts for each possible combination of incoming arguments
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


# Adapted from Canvas VAE.pdf
# Reparameterization trick used for Variational Autoencoder
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
