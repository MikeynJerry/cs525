import numpy as np

# Squared Error loss function
class SquaredError:
    @staticmethod
    def error(preds, y):
        
        # Dropped `/ 2` to match Keras
        return np.sum((preds - y) ** 2)

    @staticmethod
    def prime(preds, y):
        
        # Added `2 *` to match Keras
        return 2 * (preds - y)
