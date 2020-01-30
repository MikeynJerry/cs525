import numpy as np

# Logistic / Sigmoid activation function
class Logistic:
    @staticmethod
    def activate(x):
        return 1 / (1 + np.exp(-x))


# Linear activation function
class Linear:
    @staticmethod
    def activate(x):
        return x