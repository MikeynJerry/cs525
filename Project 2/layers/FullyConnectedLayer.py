import numpy as np

from .Neuron import Neuron

class FullyConnectedLayer:

    def __init__(self, num_neurons, activation, input_shape, learning_rate, weights=None, biases=None):

        self._num_neurons = num_neurons
        self._input_shape = input_shape
        self._activation = activation
        self._neurons = [
            Neuron(
                activation=activation,
                num_inputs=input_shape[0],
                learning_rate=learning_rate,
                weights=weights[neur_idx] if weights is not None else weights,
                bias=biases[neur_idx] if biases is not None else biases
            )
            for neur_idx in range(num_neurons)
        ]

    def backprop(self, error):

        # calculate the error that propagates through each neuron in the layer
        out = np.array([neuron.backprop(error[neur_idx]) for neur_idx, neuron in enumerate(self._neurons)])

        # sum the error before passing it to the next layer
        return np.sum(out, axis=0)

    def calculate(self, inp):
        return np.array([n.activate(n.calculate(inp)) for n in self._neurons])

    @property
    def biases(self):
        return np.array([neuron.bias for neuron in self._neurons])

    @biases.setter
    def biases(self, biases):
        for neur_idx, neuron in enumerate(self._neurons):
            neuron.bias = biases[neur_idx]

    @property
    def weights(self):
        weights = np.array([neuron.weights for neuron in self._neurons])
        return weights.reshape(weights.shape[::-1])

    @weights.setter
    def weights(self, weights):
        for neur_idx, neuron in enumerate(self._neurons):
            neuron.weights = weights[neur_idx]