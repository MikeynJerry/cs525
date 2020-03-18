from tensorflow.keras.callbacks import Callback  # Used in TimingCallback as base class
from timeit import default_timer as timer  # Used in TimingCallback for epoch times
from utils import accuracy, get_labels


# Trains a set of models with different hyperparameters
def config_trainer(get_model, train_data, test_data, configs, **kwargs):
    data, models = [], []
    for config in configs:
        model = get_model(train_data[0], **kwargs)
        data.append(train(model, train_data=train_data, test_data=test_data, **config,))
        models.append(model)

    return data, models


# Keeps track of epoch timings
# Adapted from Github comment
# https://github.com/keras-team/keras/issues/5105#issuecomment-274182222
class TimingCallback(Callback):
    def __init__(self):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(timer() - self.start_time)


# Given a set of models, print the accuracy and configuration of all of them
def test(models, test_data, configs, epochs=50):
    x_test, y_test = test_data
    labels = get_labels(configs)
    for model, label in zip(models, labels):
        y_pred = model.predict(x_test)
        print("Model")
        print("    Configuration\n        ", end="")
        print(*label.replace(" -", ":").split(", "), sep="\n        ")
        print("    Accuracy")
        print(f"        Epoch {epochs}: {accuracy(y_test, y_pred) * 100:.2f}%")


# Compiles, trains, and gathers data from a model
def train(model, train_data, test_data, epochs=50, batch_size=200, **kwargs):

    x_train, y_train = train_data
    # gather time between epochs
    time = TimingCallback()

    # compile model
    model.compile(metrics=["accuracy"], **kwargs)

    # fit the training data and get the history for plotting data
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=test_data,
        callbacks=[time],
    )

    loss = history.history.get("loss")
    acc = history.history.get("accuracy") or [-1] * epochs
    val_loss = history.history.get("val_loss")
    val_acc = history.history.get("val_accuracy") or [-1] * epochs

    return loss, acc, val_loss, val_acc, time.times
