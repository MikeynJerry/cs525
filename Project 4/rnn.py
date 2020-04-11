import argparse
from ModelHelper import ModelHelper

# Single configuration trainer
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("nb_hidden", type=int)
    parser.add_argument("window_size", type=int)
    parser.add_argument("stride", type=int)
    args = parser.parse_args()
    helper = ModelHelper(args.filename)
    helper.build(args.window_size, args.stride)
    model = helper.create_model(args.model_name, args.nb_hidden)
    helper.train(model, verbose=0)
