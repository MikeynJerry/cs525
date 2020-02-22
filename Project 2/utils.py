import matplotlib.pyplot as plt

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
    
def set_model_params(model, weights, biases):
    # set all the model's weights to my neural net's for comparison
    for layer_idx, layer in enumerate(model.layers):
        if len(layer.get_weights()) != 0:
            layer.set_weights((weights[layer_idx], biases[layer_idx]))
            
    return model