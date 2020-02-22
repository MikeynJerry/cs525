import sys
sys.path.append('..')

import numpy as np

from activations import Logistic

class Neuron:
    def __init__(self, activation, num_inputs, learning_rate, weights=None, bias=None):

        if not isinstance(activation, Logistic):
            raise ValueError('Unexpected activation function, expected Logistic.')

        if num_inputs < 1:
            raise ValueError('The number of inputs must be greater than 0.')

        if learning_rate < 10e-10:
            raise ValueError('The learning rate has to be greater than 0.')

        self._weights = np.array(weights) if weights is not None else np.random.randn(num_inputs)
        self._bias = bias if bias is not None else np.random.randn()

        self._activation = activation
        self._learning_rate = learning_rate
        self._delta = None
        self._inp = None
        self._out = None

    def activate(self, inp):
        return self._activation.activate(inp)

    def backprop(self, error):
        
        # calculate the per-neuron delta bases on last output and the backpropagated error
        self._delta = error * self._activation.prime(self._net)
        
        # calculate the per-neuron error to propagate backwards and return it
        ret = self._delta * self._weights
        
        # update the biases
        self._bias -= self._delta * self._learning_rate
        
        # update the weights
        self._weights -= self._learning_rate * self._inp * self._delta
        
        return ret

    def calculate(self, inp):
        
        # store input for backprop
        self._inp = inp

        # multiply the input by the incoming weights and add the bias
        self._net = np.dot(inp, self._weights) + self._bias

        return self._net

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights