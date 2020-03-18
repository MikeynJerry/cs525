from utils import sampling  # Used in VAE
from tensorflow.keras.models import Sequential  # Used in CNNs / NNs
from tensorflow.keras.layers import Dense  # Used in CNNs / NNs
from tensorflow.keras.layers import Dropout, Flatten, Conv2D  # Used in CNNs
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization  # Used in CNNs
from tensorflow.keras.models import Model  # Used in VAE
from tensorflow.keras.layers import Input, Lambda  # Used in VAE
from tensorflow.keras.layers import Reshape, Conv2DTranspose  # Used in VAE
from tensorflow.keras import backend as K  # Used in VAE
from tensorflow.keras.losses import mse  # Used in VAE
import numpy as np  # Used for various array manipulations

# Made according to Project 3 specifications
def task_1_model(x_train):
    return Sequential(
        [
            Dense(784, activation="tanh", input_shape=(np.prod(x_train.shape[1:3]),)),
            Dense(512, activation="sigmoid"),
            Dense(100, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )


# Made according to Project 3 specifications
def task_2_model(x_train):
    return Sequential(
        [
            Conv2D(40, (5, 5), activation="relu", input_shape=x_train.shape[1:]),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(100, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )


# Made according to Project 3 specifications
def task_3_model(x_train):
    return Sequential(
        [
            Conv2D(48, (3, 3), activation="relu", input_shape=x_train.shape[1:]),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(96, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(100, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )


# Made according to Project 3 specifications
def task_4_model(x_train):
    return Sequential(
        [
            Conv2D(
                32,
                (5, 5),
                activation="relu",
                input_shape=x_train.shape[1:],
                padding="same",
            ),
            Conv2D(32, (5, 5), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ]
    )


# Adapted from Canvas VAE.pdf
def task_5_model(x_train, filters=16, n_latent=5, kernel_size=3):
    inputs = Input(shape=x_train.shape[1:])
    x = inputs
    for i in range(2):
        filters *= 2
        x = Conv2D(filters, kernel_size, activation="relu", strides=2, padding="same")(
            x
        )

    shape = K.int_shape(x)

    x = Flatten()(x)
    x = Dense(filters, activation="relu")(x)
    z_mean = Dense(n_latent)(x)
    z_log_var = Dense(n_latent)(x)

    z = Lambda(sampling)([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = Input(shape=(n_latent,))
    x = Dense(shape[1] * shape[2] * shape[3], activation="relu")(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(2):
        x = Conv2DTranspose(
            filters, kernel_size, activation="relu", strides=2, padding="same"
        )(x)
        filters //= 2

    outputs = Conv2DTranspose(
        filters=1, kernel_size=kernel_size, activation="sigmoid", padding="same"
    )(x)

    decoder = Model(latent_inputs, outputs, name="decoder")

    outputs = decoder(encoder(inputs)[2])

    vae = Model(inputs, outputs)

    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= np.prod(x_train.shape[1:3])
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    return vae
