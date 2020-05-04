import argparse
import json
import numpy as np
import os
from random import sample
from PIL import Image
from imresize import imresize
from tqdm import tqdm


CINIC_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
CINIC_SUBSETS = ["train", "valid"]


def prep(
    classes=CINIC_CLASSES,
    dirname="cinic-10",
    load_file=None,
    nb_classes=10,
    nb_train=800,
    nb_valid=100,
    save_file=None,
    scale=1,
    subsets=CINIC_SUBSETS,
    upscale=False,
):
    # Select random image files
    if load_file:
        with open(load_file, "r") as f:
            data = json.load(f)
    else:
        data = {subset: {} for subset in subsets}
        for root, subdirs, files in os.walk(dirname):
            dirs = root.split(os.sep)
            classname = dirs[-1]
            if classname in classes:
                nb_keep = nb_train if "train" in root else nb_valid
                nb_keep //= nb_classes
                subset = dirs[-2]
                data[subset][classname] = list(sample(files, nb_keep))

    # Save filenames of images we use for reproducability
    if save_file:
        with open(save_file, "w") as f:
            json.dump(data, f)

    # Make scale directories
    if not os.path.exists(os.path.join(os.getcwd(), f".{dirname}")):
        os.mkdir(os.path.join(os.getcwd(), f".{dirname}"))
    for subset in subsets:
        if not os.path.exists(os.path.join(os.getcwd(), f".{dirname}", subset)):
            os.mkdir(os.path.join(os.getcwd(), f".{dirname}", subset))
        if not os.path.exists(
            os.path.join(os.getcwd(), f".{dirname}", subset, f"x{scale}")
        ):
            os.mkdir(os.path.join(os.getcwd(), f".{dirname}", subset, f"x{scale}"))
        if not os.path.exists(
            os.path.join(os.getcwd(), f".{dirname}", subset, f"x{scale}", "downscaled")
        ):
            os.mkdir(os.path.join(os.getcwd(), f".{dirname}", subset, f"x{scale}", "downscaled"))
        if upscale and not os.path.exists(
            os.path.join(os.getcwd(), f".{dirname}", subset, f"x{scale}", "upscaled")
        ):
            os.mkdir(os.path.join(os.getcwd(), f".{dirname}", subset, f"x{scale}", "upscaled"))

    # Resize images
    for subset in subsets:
        for classname in classes:
            for file in tqdm(data[subset][classname], desc="Resizing images"):
                src = os.path.join(os.getcwd(), f"{dirname}", subset, classname, file)
                dst = os.path.join(
                    os.getcwd(), f".{dirname}", subset, f"x{scale}", "downscaled", file
                )
                lr_image = np.asarray(Image.open(src))
                lr_image = imresize(lr_image, 1 / scale)
                lr_image = Image.fromarray(lr_image)
                lr_image.save(dst)
                # Upsample downsampled image for bicubic baseline
                if upscale and scale > 1:
                    dst = os.path.join(
                        os.getcwd(), f".{dirname}", subset, f"x{scale}", "upscaled", file
                    )
                    hr_image = np.asarray(lr_image)
                    hr_image = imresize(hr_image, scale)
                    hr_image = Image.fromarray(hr_image)
                    hr_image.save(dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=CINIC_CLASSES)
    parser.add_argument("--dirname", default="cinic-10")
    parser.add_argument("--load_file", default=None)
    parser.add_argument("--nb_classes", type=int, default=10)
    parser.add_argument("--nb_train", type=int, default=800)
    parser.add_argument("--nb_valid", type=int, default=100)
    parser.add_argument("--save_file", default=None)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--subsets", nargs="+", default=CINIC_SUBSETS)
    parser.add_argument("--upscale", action="store_true")
    args = parser.parse_args()

    prep(**args.__dict__)
