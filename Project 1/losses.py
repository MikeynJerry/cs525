import numpy as np

# Squared Error loss function
class SquaredError:
    @staticmethod
    def error(preds, y):
        return np.sum((preds - y) ** 2) / 2

    @staticmethod
    def prime(preds, y):
        return preds - y

# Binary Cross Entropy loss function
class BinaryCrossEntropy:
    @staticmethod
    def error(preds, y):
        return np.sum(-(y * np.log(preds) + (1 - y) * np.log(1 - preds))) / y.shape[0]

    @staticmethod
    def prime(preds, y):
        return -(y / preds) + (1 - y) / (1 - preds)
