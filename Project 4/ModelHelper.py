from collections import Counter
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, SimpleRNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np
import matplotlib.pyplot as plt
from constants import config_to_label, config_to_filename


class ModelHelper:
    def __init__(self, filename=None, save_file="data.txt"):
        self.built = False
        self.fig = None
        self.filename = filename
        self.save_file = save_file

    # Preprocesses and loads data into class variables for later use
    def build(self, window_size, stride):
        self.stride = stride
        self.window_size = window_size
        self.preprocess_data(self.filename, self.save_file)
        self.load_data(self.save_file)
        self.built = True

    # Creates a chracter to index and vice versa mapping
    def _build_vocab(self, data):
        vocab = Counter(data)
        self.vocab_size = len(vocab)
        self.ctoi = {}
        self.itoc = []
        for i, (key, cnt) in enumerate(vocab.most_common()):
            self.ctoi[key] = i
            self.itoc.append(key)

    def create_model(self, model_name, nb_hidden):
        if not self.built:
            raise ValueError(
                "The vocabulary has not yet been built. "
                "Build the model by calling `build()` first."
            )
        if model_name == "srnn":
            return self.srnn(nb_hidden)
        if model_name == "lstm":
            return self.lstm(nb_hidden)

        raise NotImplementedError(
            f"The {model_name} model hasn't been implemented yet."
        )

    # SimpleRNN with a single layer attached to a FCL for classification
    def srnn(self, nb_hidden):
        return Sequential(
            [
                SimpleRNN(nb_hidden, input_dim=self.vocab_size),
                Dense(self.vocab_size, activation="softmax"),
            ]
        )

    # LSTM with a single layer attached to a FCL for classification
    def lstm(self, nb_hidden):
        return Sequential(
            [
                LSTM(nb_hidden, input_dim=self.vocab_size),
                Dense(self.vocab_size, activation="softmax"),
            ]
        )

    # Reads in data file, chops it up into window_size + 1 partitions, and writes back to disk
    def preprocess_data(self, filename, save_file):
        with open(filename, "r") as f:
            raw = f.read().replace("\n", " ")

        self._build_vocab(raw)
        data = [
            raw[ind : ind + self.window_size + 1]
            for ind in range(0, len(raw), self.stride)
        ]
        data = [seq for seq in data if len(seq) == self.window_size + 1]

        with open(save_file, "w") as f:
            f.write("\n".join(data))

    # Reads in file of paritioned sequences and passes them through an embedding layer
    def load_data(self, filename):
        with open(filename, "r") as f:
            data = f.read().split("\n")

        train_x = np.array([self.embed(line[: self.window_size]) for line in data])
        train_y = self.embed(data, s=np.s_[self.window_size :])

        self.training_data = (train_x, train_y)

    # Takes a set of characters and converts it to indices
    def embed(self, chars, s=np.s_[:]):
        return np.array(
            [
                to_categorical(self.ctoi[char[s]], num_classes=self.vocab_size)
                for char in chars
            ]
        )

    # Predicts characters based on an initial sequence
    def predict(self, model, init_chars, nb_chars):
        pred = np.array([self.embed(init_chars)])
        for ind in range(nb_chars):
            pred = np.append(
                pred,
                np.expand_dims(
                    to_categorical(
                        np.argmax(model.predict(pred[:, ind:]), axis=1),
                        num_classes=self.vocab_size,
                    ),
                    axis=0,
                ),
                axis=1,
            )

        print("".join(self.itoc[i] for i in np.argmax(pred, axis=2)[0]))

    # Trains model and shows generated text every x epochs
    def train(self, model, training_data=None, nb_epochs=100, **kwargs):
        training_data = training_data or self.training_data
        gen_text = LambdaCallback(on_epoch_end=self.on_epoch_end(model))
        model.compile(
            loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01)
        )
        return model.fit(
            *training_data,
            batch_size=128,
            epochs=nb_epochs,
            callbacks=[gen_text],
            **kwargs,
        )

    # Keras callback to generate text every x epochs
    def on_epoch_end(self, model, interval=20):
        ind = np.random.choice(self.training_data[0].shape[0])
        init_chars = [
            self.itoc[i] for i in np.argmax(self.training_data[0][ind, :], axis=1)
        ]
        s = np.s_[ind : ind + int(np.ceil(self.window_size / self.stride)) * 6]
        correct_chars = [
            self.itoc[i]
            for window in np.argmax(self.training_data[0][s], axis=2)
            for i in window[: self.stride]
        ]
        print(f"Initial  chars: {''.join(init_chars)}")
        print(f"Correct   text: {''.join(correct_chars)}")

        def gen_text(epochs, _):
            if (epochs + 1) % interval == 0:
                print(f"Predicted text: ", end="")
                self.predict(model, init_chars, self.window_size * 5)

        return gen_text

    # Gathers losses across runs and then plots them
    def plot_history(self, history=None, config=None, configs=None, show_plot=False):
        self.fig = self.fig or plt.figure(figsize=(8, 6))
        if history:
            loss = history.history.get("loss")
            plt.plot(
                range(len(loss)),
                loss,
                linestyle="-" if config["model_name"] == "srnn" else "--",
                label=self._label(config) or "Loss",
            )
        if configs and show_plot:
            plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.savefig(self._gen_filename(configs), format="png", bbox_inches="tight")
            self.fig = None

    # Generates a label for plotting based on a config
    def _label(self, config):
        config = config or {}
        label = ""

        for key, val in config.items():
            label += f"{config_to_label[key]}"
            label += f"{str(val).upper()}, "
        return label[:-2]

    # Generates a filename for saving the plot based on a config
    def _gen_filename(self, configs):
        filename = ""

        for key, val in configs.items():
            filename += f"{config_to_filename[key]}_"
            if isinstance(val, list):
                filename += "{" + ",".join(str(v) for v in val) + "}_"
            else:
                filename += f"{str(val).upper()}_"
        return filename[:-1] + ".png"
