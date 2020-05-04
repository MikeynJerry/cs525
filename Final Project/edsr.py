import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model


def edsr(scale, dataset, nb_filters=64, nb_res=8, res_block_scaling=None):

    if dataset == "cinic":
        normalize = norm_cinic
        denormalize = denorm_cinic
    elif dataset == "div2k":
        normalize = norm_div2k
        denormalize = denorm_div2k

    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = b = Conv2D(nb_filters, 3, padding="same")(x)
    for i in range(nb_res):
        b = res_block(b, nb_filters, res_block_scaling)
    b = Conv2D(nb_filters, 3, padding="same")(b)
    x = add([x, b])

    x = upsample(x, scale, nb_filters)
    x = Conv2D(3, 3, padding="same")(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding="same", activation="relu")(x_in)
    x = Conv2D(filters, 3, padding="same")(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = add([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding="same", **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2)
    elif scale == 3:
        x = upsample_1(x, 3)
    elif scale == 4:
        x = upsample_1(x, 2)
        x = upsample_1(x, 2)

    return x


DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255
CINIC_RGB_MEAN = np.array([0.4704569, 0.46606101, 0.42049226]) * 255


def norm_div2k(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def norm_cinic(x, rgb_mean=CINIC_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denorm_div2k(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def denorm_cinic(x, rgb_mean=CINIC_RGB_MEAN):
    return x * 127.5 + rgb_mean


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
