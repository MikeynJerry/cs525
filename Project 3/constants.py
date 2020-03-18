from tensorflow.keras.optimizers import (
    SGD,
    RMSprop,
    Adagrad,
    Adadelta,
    Adam,
    Adamax,
    Nadam,
)
from tensorflow.keras.losses import categorical_crossentropy, mean_squared_error
import matplotlib

# All keras optimizers
optimizers = [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam]

# Subset of losses I'll test
losses = [categorical_crossentropy]

# Fashion MNIST classes
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Matplotlib default color cycle colors
colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
