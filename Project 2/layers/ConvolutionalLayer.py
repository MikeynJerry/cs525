import numpy as np

import itertools # used for indices combinations in `convolve`

from .Neuron import Neuron

class ConvolutionalLayer:
    def __init__(self, num_kernels, kernel_size, activation, input_shape, learning_rate, weights=None):
        self._activation = activation
        self._biases = np.random.randn(num_kernels)
        self._input_shape = input_shape
        self._kernel_shape = (kernel_size, kernel_size, input_shape[-1], num_kernels)
        self._learning_rate = learning_rate
        self._num_kernels = num_kernels

        self._kernels = np.array([
            Neuron(activation, 1, learning_rate, bias=0)
            for _ in range(np.prod(self._kernel_shape))
        ]).reshape(self._kernel_shape)

        if weights is not None:
            self.weights = weights

    
    def backprop(self, error):

        # calculate the delta w.r.t. the activation function of this layer
        deltas = error * self._activation.prime(self._net)

        weight_deltas = np.zeros(self._kernel_shape)
        input_deltas = np.zeros(self._input_shape)
        
        # we need to pad our deltas so we can convolve it with our kernel weights
        #    to get the error w.r.t. the input
        padded_deltas = np.pad(deltas, pad_width=((2, 2), (2, 2), (0, 0)))

        # iterate over the input
        for inp_row, inp_col, inp_chan in self.convolve(self._inp.shape, deltas.shape[:2]):
            
            # iterate over the deltas
            for delta_row, delta_col, delta_chan in self.convolve(deltas.shape):
                
                # our weight update is equal to the input convolved with our deltas
                weight_deltas[inp_row, inp_col, inp_chan, delta_chan] += \
                    self._inp[inp_row + delta_row, inp_col + delta_col, inp_chan] * \
                    deltas[delta_row, delta_col, delta_chan]
        

        # we need the width and height of our kernels so that we can flip them
        w_f, h_f = self._kernel_shape[:2]
        for inp_chan in range(self._input_shape[-1]):
            
            # iterate over the padded deltas
            for delta_row, delta_col, delta_chan in self.convolve(padded_deltas.shape, self._kernel_shape[:2]):
                
                # iterate over the kernel
                for kernel_row, kernel_col in self.convolve(self._kernel_shape[:2]):
                    
                    # our error w.r.t. the input is equal to the deltas convolved with our weights
                    input_deltas[delta_row, delta_col, inp_chan] += \
                        padded_deltas[delta_row + kernel_row, delta_col + kernel_col, delta_chan] * \
                        self._kernels[w_f - kernel_row - 1, h_f - kernel_col - 1, inp_chan, delta_chan].weights
    
        # update biases with the sum of all deltas across each kernel
        self._biases -= self._learning_rate * np.sum(deltas, axis=(0, 1))
        
        # update weights with the weight deltas calculate above
        self.weights -= self._learning_rate * weight_deltas
        

        return input_deltas

    # calculate an output after convolution by this layer's kernels
    def calculate(self, inp):
        
        # save last input for weight update in backprop
        self._inp = inp
        
        # start all net indices as the biases so they don't need to be added later
        net = np.ones(self.output_shape) * self._biases
        
        # convolve over the input
        for inp_row, inp_col, inp_chan in self.convolve(inp.shape, self._kernel_shape[:2]):
            
            # convolve over the kernels
            for kernel_row, kernel_col, kernel_chan in self.convolve(self._kernel_shape[:2] + (self._kernel_shape[-1],)):
                
                # calculate the input window * kernel
                net[inp_row, inp_col, kernel_chan] += \
                    self._kernels[kernel_row, kernel_col, inp_chan, kernel_chan] \
                        .calculate(inp[inp_row + kernel_row, inp_col + kernel_col, inp_chan])

        # save the net to calculate deltas in backprop
        self._net = net
        
        return self._activation.activate(net)

    # general purpose convolver, generates indices based on shape of array being convolved
    def convolve(self, inp_shape, kernel_shape=tuple()):
        ranges = [range(in_dim - k_dim + 1) for in_dim, k_dim in itertools.zip_longest(inp_shape, kernel_shape, fillvalue=1)]
        for combo in itertools.product(*ranges):
            yield combo
    
    @property
    def output_shape(self):
        w_i, h_i, _ = self._input_shape
        w_f, h_f, _, _ = self._kernel_shape
        # implied stride = 1
        # (w_i - w_f) / 1 + 1 = w_i - w_f + 1
        return (w_i - w_f + 1), (h_i - h_f + 1), self._num_kernels

    @property
    def biases(self):
        return self._biases

    @property
    def weights(self):
        weights = np.array([neuron.weights for neuron in self._kernels.flatten()])
        return weights.reshape(self._kernel_shape).astype(np.float32)

    @weights.setter
    def weights(self, weights):
        for neur, weight in zip(self._kernels.flatten(), weights.flatten()):
            neur.weights = weight