from sklearn.metrics import confusion_matrix  # Used to create confusion matrix
import matplotlib.pyplot as plt  # Used for plotting
import matplotlib  # Used for plotting
from constants import colors, classes  # Used for plotting
import numpy as np  # Used for various array manipulations
from utils import accuracy, get_labels

matplotlib.rcParams["figure.dpi"] = 300

# Adds labels to heatmap axes
# Adapted from Matplotlib "Creating annotated heatmaps"
# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def annotate_heatmap(im, threshold, textcolors=["white", "black"], **textkw):

    threshold = im.norm(threshold)

    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    valfmt = matplotlib.ticker.StrMethodFormatter("{x}")

    data = im.get_array()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            im.axes.text(j, i, valfmt(data[i, j], None), **kw)


# Creates a heatmap from an NxN array
# Adapted from Matplotlib "Creating annotated heatmaps"
# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, labels, ax, size=20, cbarlabel="", **kwargs):

    im = ax.imshow(data, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", size=size)
    cbar.ax.tick_params(labelsize=size)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        top=True, bottom=False, labeltop=True, labelbottom=False, labelsize=size
    )

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


# Given a set of models, plot the confusion matrix of the best performing one
def plot_cm(models, test_data, task, configs):

    x_test, y_test = test_data
    max_acc, max_config = 0, {}
    cm = None

    # Find model with highest accuracy
    for model, config in zip(models, configs):
        y_pred = model.predict(x_test)
        acc = accuracy(y_test, y_pred)
        if acc > max_acc:
            max_acc, max_config = acc, config
            cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    print(f"Max Accuracy: {max_acc * 100}, {get_labels([max_config])[0]}")

    fig, ax = plt.subplots(figsize=(16, 8))
    im = heatmap(
        cm, classes, ax=ax, cmap=plt.get_cmap("plasma", 1000), cbarlabel="Predictions"
    )
    annotate_heatmap(im, threshold=400, size=20)
    plt.savefig(f"task_{task}_cm.png", format="png", bbox_inches="tight", pad_inches=0)


# Adapted from Canvas VAE.pdf
# Iterate over each dimension in a latent vector and plot the resulting clothes generated from it
def plot_clothes(encoder, decoder, kernel_size, kernels, n_samples=11):

    n_latent = decoder.input_shape[1]
    clothes_size = encoder.input_shape[1]
    
    # Plot perturbed latent vector
    figure = np.zeros((clothes_size * n_latent, clothes_size * n_samples))
    grid_x = np.linspace(-5, 5, n_samples)

    for i in range(n_latent):
        for j, x in enumerate(grid_x):
            z_sample = np.zeros((1, n_latent))
            z_sample[:, i] = x
            x_decoded = decoder.predict(z_sample)
            clothes = x_decoded[0].reshape(clothes_size, clothes_size)
            figure[
                i * clothes_size : (i + 1) * clothes_size,
                j * clothes_size : (j + 1) * clothes_size,
            ] = clothes

    plt.figure(figsize=(10, 10))
    start_range = clothes_size // 2
    end_range = n_samples * clothes_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, clothes_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.arange(n_latent)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.title(f"Perturbing Latent Vector One Column at a Time (length = {n_latent})")
    plt.xlabel("z[y] value")
    plt.ylabel("y")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig(
        f"task_5_clothes_kernel_{kernel_size}_kernels_{kernels}_latent_{n_latent}.png"
    )
    
    # Plot random sampling of clothes
    figure = np.zeros((clothes_size, clothes_size * n_samples))

    for i in range(n_samples):
        z_sample = np.random.randn(1, n_latent)
        x_decoded = decoder.predict(z_sample)
        clothes = x_decoded[0].reshape(clothes_size, clothes_size)
        figure[
            :,
            i * clothes_size : (i + 1) * clothes_size
        ] = clothes

    plt.figure(figsize=(10, 2))
    plt.axis('off')
    plt.title(f"Random Latent Vectors (length = {n_latent})")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig(
        f"task_5_clothes_kernel_{kernel_size}_kernels_{kernels}_latent_{n_latent}_examples.png"
    )

# Plot losses vs epochs, losses vs time, and accuracy vs epochs
def plot_data(data, task, configs):

    labels = get_labels(configs)

    # Plot Loss vs Epochs
    plt.figure(figsize=(8, 6))
    plt.title("Loss vs Epochs")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    for (i, ((loss, acc, val_loss, val_acc, times), label),) in enumerate(
        zip(data, labels)
    ):
        epochs = range(1, len(loss) + 1)
        plt.plot(
            epochs, loss, color=colors[i], label=f"      Loss: {label}",
        )
        plt.plot(
            epochs,
            val_loss,
            color=colors[i],
            linestyle="--",
            label=f"Val Loss: {label}",
        )

    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig(f"task_{task}_loss_epochs.png", format="png", bbox_inches="tight")

    # Plot Loss vs Time
    plt.figure(figsize=(8, 6))
    plt.title("Loss vs Time")
    plt.ylabel("Loss")
    plt.xlabel("Time (in seconds)")
    for (i, ((loss, acc, val_loss, val_acc, times), label),) in enumerate(
        zip(data, labels)
    ):
        plt.plot(
            np.cumsum(times), loss, color=colors[i], label=f"      Loss: {label}",
        )
        plt.plot(
            np.cumsum(times),
            val_loss,
            color=colors[i],
            linestyle="--",
            label=f"Val Loss: {label}",
        )

    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig(f"task_{task}_loss_time.png", format="png", bbox_inches="tight")

    # Plot Accuracy vs Epochs
    if acc[0] == -1:
        return

    plt.figure(figsize=(8, 6))
    plt.title("Accuracy vs Epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    for (i, ((loss, acc, val_loss, val_acc, times), label),) in enumerate(
        zip(data, labels)
    ):
        epochs = range(1, len(acc) + 1)
        plt.plot(
            epochs, np.array(acc) * 100, color=colors[i], label=f"      Acc: {label}",
        )
        plt.plot(
            epochs,
            np.array(val_acc) * 100,
            color=colors[i],
            linestyle="--",
            label=f"Val Acc: {label}",
        )

    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig(f"task_{task}_acc_epochs.png", format="png", bbox_inches="tight")
