from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import SGD

from utils import set_model_params

from losses import SquaredError
from activations import Logistic

from NeuralNetwork import NeuralNetwork

# example network specified in 2.5.a
def example_one(learning_rate):
    
    input_shape = (5, 5, 1)
    
    nn = NeuralNetwork(input_shape, SquaredError(), learning_rate)
    nn.add(layer_type='conv', num_kernels=1, kernel_size=3,
           activation=Logistic(), learning_rate=learning_rate)
    nn.add(layer_type='flat')
    nn.add(layer_type='dense', num_neurons=1,
           activation=Logistic(), learning_rate=learning_rate)
    
    
    model = Sequential([
        Conv2D(1, kernel_size=3, activation='sigmoid', input_shape=input_shape),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss=mean_squared_error, optimizer=SGD(lr=learning_rate))
    
    model = set_model_params(model, nn.weights, nn.biases)
    
    return nn, model


# example network specified in 2.5.b
def example_two(learning_rate):
    
    input_shape = (5, 5, 1)
    
    nn = NeuralNetwork(input_shape, SquaredError(), learning_rate)
    nn.add(layer_type='conv', num_kernels=1, kernel_size=3,
       activation=Logistic(), learning_rate=learning_rate)
    nn.add(layer_type='conv', num_kernels=1, kernel_size=3,
       activation=Logistic(), learning_rate=learning_rate)
    nn.add(layer_type='flat')
    nn.add(layer_type='dense', num_neurons=1, activation=Logistic(), learning_rate=learning_rate)

    model = Sequential([
        Conv2D(1, kernel_size=3, activation='sigmoid', input_shape=input_shape),
        Conv2D(1, kernel_size=3, activation='sigmoid'),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss=mean_squared_error, optimizer=SGD(lr=learning_rate))
    
    model = set_model_params(model, nn.weights, nn.biases)
    
    return nn, model
    

# example network specified in 2.5.c
def example_three(learning_rate):
    
    input_shape = (6, 6, 1)
    
    nn = NeuralNetwork(input_shape, SquaredError(), learning_rate)
    nn.add(layer_type='conv', num_kernels=2, kernel_size=3,
       activation=Logistic(), learning_rate=learning_rate)
    nn.add(layer_type='pool', kernel_size=2)
    nn.add(layer_type='flat')
    nn.add(layer_type='dense', num_neurons=1, activation=Logistic(), learning_rate=learning_rate)

    model = Sequential([
        Conv2D(2, kernel_size=3, activation='sigmoid', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss=mean_squared_error, optimizer=SGD(lr=learning_rate))
    
    model = set_model_params(model, nn.weights, nn.biases)
    
    return nn, model
