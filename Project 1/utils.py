import matplotlib.pyplot as plt
import pickle
from os.path import abspath

# loads a saved model from <filename>
def load_model(filename):
    if not filename.endswith('.pkl'):
        filename += '.pkl'

    filename = abspath(filename)

    with open(filename, 'rb') as f:
        return pickle.load(f)

# plot errors produced by neural network
def plot_errors(errors, lrs, title, n):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    styles = ['-', '--', '-.', ':']
    for i, lr in enumerate(lrs):
        plt.plot(range(len(errors[lr][:n])), errors[lr][:n], label=f'lr = {lr}', linestyle=styles[i % 4])
    
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()