import argparse
from ModelHelper import ModelHelper
from utils import product_dict
from tqdm import tqdm

# Multiple configuration trainer
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", nargs="+", type=str)
    parser.add_argument("--nb_hidden", nargs="+", type=int)
    parser.add_argument("--window_size", nargs="+", type=int)
    parser.add_argument("--stride", nargs="+", type=int)
    args = parser.parse_args()

    helper = ModelHelper("beatles.txt")
    configs = list(product_dict(**args.__dict__))
    for config in tqdm(configs):
        helper.build(config["window_size"], config["stride"])
        model = helper.create_model(config["model_name"], config["nb_hidden"])
        print(helper._label(config))
        history = helper.train(model, nb_epochs=100, verbose=0)
        helper.plot_history(history, config)

    helper.plot_history(configs=args.__dict__, show_plot=True)
