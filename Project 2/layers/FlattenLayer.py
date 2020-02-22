import numpy as np

class FlattenLayer:
    def __init__(self, input_shape):
        self._input_shape = input_shape

    def backprop(self, error):
        
        # reshape the error so that it fits the output of the previous layer
        return error.reshape(self._input_shape)

    @property
    def biases(self):
        return None

    def calculate(self, inp):
        
        # squash the input to one dimension
        return inp.flatten()
    
    @property
    def output_shape(self):
        return (np.prod(self._input_shape),)
    
    @property
    def weights(self):
        return None