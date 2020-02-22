import numpy as np

# Logistic / Sigmoid activation function
class Logistic:
    @staticmethod
    def activate(x):
        return 1 / (1 + np.exp(-x))
    
    def prime(self, x):
        out = self.activate(x)
        return out * (1 - out)
