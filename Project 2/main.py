import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import numpy as np

import argparse           # used to read in command line arguments

from examples import *   # contains example models

np.set_printoptions(precision=12, suppress=True)

learning_rate = 0.01

def main(opt):
    if opt == 'example1':
        nn, model = example_one(learning_rate)
    elif opt == 'example2':
        nn, model = example_two(learning_rate)
    elif opt == 'example3':
        nn, model = example_three(learning_rate)
        
    inp = np.random.randn(*((1,) + nn._input_shape))
    out = np.random.randn(1, 1)
    
    nn.backprop(inp[0], out[0])

    model.fit(inp, out)

    nn_pred = nn.calculate(inp[0])
    model_pred = model.predict(inp).flatten()
    
    print(f'Neural Network prediction: {nn_pred}, Keras model prediction: {model_pred}, Difference: {nn_pred - model_pred}')
    if not np.isclose(nn_pred, model_pred):
        print('Predictions differ by greater than 1e-5. This is bad and typically means backprop went wrong.')
    
    # Compare weights from my nn to keras's model
    for layer_idx, layer in enumerate(model.layers):
        if len(layer.get_weights()) != 0:
            nn_weights, nn_biases = nn.weights[layer_idx], nn.biases[layer_idx]
            model_weights, model_biases = layer.get_weights()
            if not np.allclose(nn_weights, model_weights):
                print(f"There is a weight difference (> 1e-5) between my neural net and keras's model at layer {layer_idx}.")
                print(f"Here's the difference in the weights", model_weights - nn_weights)

            if not np.allclose(model_biases, nn_biases):
                print(f"There is a bias difference (> 1e-5) between my neural net and keras's model at layer {layer_idx}.")
                print(f"Here's the difference in the biases:", model_biases - nn_biases)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('example_to_show', type=str, choices=['example1', 'example2', 'example3'])
    args = parser.parse_args()
    main(args.example_to_show)