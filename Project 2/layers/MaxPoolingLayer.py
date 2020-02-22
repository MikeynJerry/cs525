import numpy as np

import itertools # used for indices combinations in `calculate`

class MaxPoolingLayer:
    def __init__(self, kernel_size, input_shape):
        self._kernel_size = kernel_size
        self._input_shape = input_shape

    def backprop(self, error):
        w, h, c = self._input_shape
        s = self._kernel_size
        out = np.zeros(self._input_shape)
        # iterate over over the input
        for row, col, chan in itertools.product(range(0, w, s), range(0, h, s), range(c)):
            
            # send the error to the indices from the mask
            out[row:row+s, col:col+s, chan] = \
                self.mask[row:row+s, col:col+s, chan] * \
                error[row // s, col // s, chan]
        
        return out

    @property
    def biases(self):
        return None

    def calculate(self, inp):
        w, h, c = inp.shape
        s = self._kernel_size
        out = np.zeros((w // s, h // s, c))
        self.mask = np.zeros((w, h, c))
        
        # iterate over over the input
        for row, col, chan in itertools.product(range(0, w, s), range(0, h, s), range(c)):
            
            # grab the current window we're looking at
            window = inp[row:row+s, col:col+s, chan]
            
            # send the max to the output
            out[row // s, col // s, chan] = np.max(window)
            
            # get the indices of the max element in the window
            row_max, col_max = np.unravel_index(np.argmax(window), window.shape)
            
            # set those indices in our mask for backprop
            self.mask[row + row_max, col + col_max, chan] = 1
        
        return out
    
    @property
    def output_shape(self):
        s = f = self._kernel_size
        return tuple(int(np.floor((dim - f) / s)) + 1 for dim in self._input_shape[:-1]) + (self._input_shape[-1],)
    
    @property
    def weights(self):
        return None
