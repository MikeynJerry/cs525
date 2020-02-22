import numpy as np

import argparse                                     # used to read in command line arguments

from activations import Logistic                    # contains activation functions
from losses import SquaredError                     # contains loss functions

from layers import Conv2D, MaxPooling2D, Flatten, Dense

class NeuralNetwork:
    def __init__(self, input_shape, loss, learning_rate):

        if not isinstance(loss, SquaredError):
            raise ValueError('Unexpected loss function, expected instance of SquaredError')

        self._input_shape = input_shape
        self._layers = []
        self._learning_rate = learning_rate
        self._loss = loss

    def add(self, layer_type, **kwargs):
        layer = None
        
        # send the layer the output shape of the last layer, or the input shape if it's the first
        kwargs['input_shape'] = self._layers[-1].output_shape if len(self._layers) else self._input_shape
        
        if layer_type == 'conv':
            layer = Conv2D(**kwargs)
        elif layer_type == 'dense':
            layer = Dense(**kwargs)
        elif layer_type == 'pool':
            layer = MaxPooling2D(**kwargs)
        elif layer_type == 'flat':
            layer = Flatten(**kwargs)
        
        self._layers.append(layer)

    def backprop(self, inp, out):
        
        # get the initial prediction
        pred = self.calculate(inp)
        
        # calculate the derivative of the error of that prediction w.r.t. the output
        error = self._loss.prime(pred, out)
        
        # pass the error through each layer backwards to calculate the error as it propagates backwards
        for i, layer in reversed(list(enumerate(self._layers))):
            error = layer.backprop(error)

    def calculate(self, inp):
        
        # pass the input through each layer to get the final prediction
        for layer in self._layers:
            inp = layer.calculate(inp)
        return np.array(inp)

    def calculateloss(self, inp, out):
        
        # get the prediction for each input
        preds = np.array([self.calculate(i) for i in inp])
        
        # return the total error based on the predictions and expected output
        return self._loss.error(preds, out) / out.shape[0]

    @property
    def biases(self):
        return np.array([layer.biases for layer in self._layers])

    @biases.setter
    def biases(self, biases):
        for layer_idx, layer in enumerate(self._layers):
            layer.biases = biases[layer_idx]

    @property
    def layers(self):
        return self._layers

    @property
    def weights(self):
        return np.array([layer.weights for layer in self._layers])

    @weights.setter
    def weights(self, weights):
        for layer_idx, layer in enumerate(self._layers):
            layer.weights = weights[layer_idx]

    def train(self, inp, out, epochs=10000, ret_error=False):
        errors = [self.calculateloss(inp, out)]
        
        # run backprop for each input a total of <epoch> times
        for _ in range(epochs):
            for i, o in zip(inp, out):
                self.backprop(i, o)
            if ret_error:
                errors.append(self.calculateloss(inp, out))

        if ret_error: return errors
        