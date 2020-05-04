import os
import numpy as np
import tensorflow as tf

from tensorflow.python.data.experimental import AUTOTUNE


class Dataset:
    def __init__(self, scale, subset, downgrade, images_dir, caches_dir):

        _scales = [1, 2, 4]

        if scale in _scales:
            self.hr_scale = scale
            self.lr_scale = scale * 2
        else:
            raise ValueError(f"scale must be in ${_scales}")

        if subset == "train":
            self.image_ids = range(1, 801)
        elif subset == "valid":
            self.image_ids = range(801, 901)
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        self.subset = subset
        self.images_dir = images_dir
        self.caches_dir = caches_dir

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(caches_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    def shape(self, ds):
        avg_lr_shape = np.array((0, 0, 0))
        avg_hr_shape = np.array((0, 0, 0))
        total = 0
        for lr, hr in ds:
            avg_lr_shape += np.array(tf.shape(lr))
            avg_hr_shape += np.array(tf.shape(hr))
            total += 1

        return list(avg_lr_shape / total), list(avg_hr_shape / total)

    def dataset(
        self, batch_size=16, repeat_count=None, random_transform=True, upscale_lr=False
    ):
        ds = tf.data.Dataset.zip((self.lr_dataset(upscale_lr), self.hr_dataset()))
        shapes = self.shape(ds)
        if random_transform:
            ds = ds.map(lambda lr, hr: random_crop(lr, hr, scale=2), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds, shapes

    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds

    @staticmethod
    def _populate_cache(ds, cache_file):
        print(f"Caching decoded images in {cache_file} ...")
        for _ in ds:
            pass
        print(f"Cached decoded images in {cache_file}.")


class CINIC(Dataset):
    def __init__(
        self,
        scale=1,
        subset="train",
        downgrade="bicubic",
        images_dir=".cinic-10",
        caches_dir=".cinic-10/caches",
    ):
        super().__init__(
            scale=scale,
            subset=subset,
            downgrade=downgrade,
            images_dir=images_dir,
            caches_dir=caches_dir,
        )

    def hr_dataset(self):
        scale = self.hr_scale
        self.hr = self._image_files(scale)
        ds = self._images_dataset(self._image_files(scale)).cache(
            self._cache_file(scale)
        )

        if not os.path.exists(self._cache_index(scale)):
            self._populate_cache(ds, self._cache_file(scale))

        return ds

    def lr_dataset(self, upscale=False):
        scale = self.lr_scale
        self.lr = self._image_files(scale, upscale)
        ds = self._images_dataset(self._image_files(scale, upscale))#.cache(
#            self._cache_file(scale, upscale)
#        )

#        if not os.path.exists(self._cache_index(scale, upscale)):
#            self._populate_cache(ds, self._cache_file(scale, upscale))

        return ds

    def _cache_file(self, scale, upscale=False):
        if upscale:
            return os.path.join(self.caches_dir, f"cinic_x{scale}_upscaled.cache")
        else:
            return os.path.join(self.caches_dir, f"cinic_x{scale}_downscaled.cache")

    def _cache_index(self, scale, upscale=False):
        return f"{self._cache_file(scale, upscale)}.index"

    def _image_files(self, scale, upscale=False):
        images_dir = self._images_dir(scale, upscale)
        return [os.path.join(images_dir, image) for image in os.listdir(images_dir)]

    def _images_dir(self, scale, upscale=False):
        if upscale:
            return os.path.join(self.images_dir, self.subset, f"x{scale}", "upscaled")
        else:
            return os.path.join(self.images_dir, self.subset, f"x{scale}", "downscaled")


class DIV2K(Dataset):
    def __init__(
        self,
        scale=1,
        subset="train",
        downgrade="bicubic",
        images_dir=".div2k/images",
        caches_dir=".div2k/caches",
    ):
        super().__init__(
            scale=scale,
            subset=subset,
            downgrade=downgrade,
            images_dir=images_dir,
            caches_dir=caches_dir,
        )
        self._ntire_2018 = True

        if downgrade == "bicubic" and self.lr_scale == 8:
            self.hr_downgrade = downgrade
            self.lr_downgrade = "x8"
        else:
            self.hr_downgrade = downgrade
            self.lr_downgrade = downgrade
            self._ntire_2018 = False

    def hr_dataset(self):
        if not os.path.exists(self._hr_images_dir()):
            download_archive(self._hr_images_archive(), self.images_dir, extract=True)

        ds = self._images_dataset(self._hr_image_files()).cache(self._hr_cache_file())

        if not os.path.exists(self._hr_cache_index()):
            self._populate_cache(ds, self._hr_cache_file())

        return ds

    def lr_dataset(self, _):
        if not os.path.exists(self._lr_images_dir()):
            download_archive(self._lr_images_archive(), self.images_dir, extract=True)

        ds = self._images_dataset(self._lr_image_files()).cache(self._lr_cache_file())

        if not os.path.exists(self._lr_cache_index()):
            self._populate_cache(ds, self._lr_cache_file())

        return ds

    def _hr_cache_file(self):
        if self.hr_scale == 1:
            return os.path.join(self.caches_dir, f"DIV2K_{self.subset}_HR.cache")
        else:
            return os.path.join(
                self.caches_dir,
                f"DIV2K_{self.subset}_LR_{self.hr_downgrade}_X{self.hr_scale}.cache",
            )

    def _lr_cache_file(self):
        return os.path.join(
            self.caches_dir,
            f"DIV2K_{self.subset}_LR_{self.lr_downgrade}_X{self.lr_scale}.cache",
        )

    def _hr_cache_index(self):
        return f"{self._hr_cache_file()}.index"

    def _lr_cache_index(self):
        return f"{self._lr_cache_file()}.index"

    def _hr_image_files(self):
        images_dir = self._hr_images_dir()
        return [
            os.path.join(images_dir, self._hr_image_file(image_id))
            for image_id in self.image_ids
        ]

    def _hr_image_file(self, image_id):
        if self.hr_scale == 1:
            return f"{image_id:04}.png"
        elif not self._ntire_2018 or self.lr_scale == 8:
            return f"{image_id:04}x{self.hr_scale}.png"
        else:
            return f"{image_id:04}x{self.hr_scale}{self.hr_downgrade[0]}.png"

    def _lr_image_files(self):
        images_dir = self._lr_images_dir()
        return [
            os.path.join(images_dir, self._lr_image_file(image_id))
            for image_id in self.image_ids
        ]

    def _lr_image_file(self, image_id):
        if not self._ntire_2018 or self.lr_scale == 8:
            return f"{image_id:04}x{self.lr_scale}.png"
        else:
            return f"{image_id:04}x{self.lr_scale}{self.lr_downgrade[0]}.png"

    def _hr_images_dir(self):
        if self.hr_scale == 1:
            return os.path.join(self.images_dir, f"DIV2K_{self.subset}_HR")
        elif self._ntire_2018:
            return os.path.join(
                self.images_dir, f"DIV2K_{self.subset}_LR_{self.hr_downgrade}"
            )
        else:
            return os.path.join(
                self.images_dir,
                f"DIV2K_{self.subset}_LR_{self.hr_downgrade}",
                f"X{self.hr_scale}",
            )

    def _lr_images_dir(self):
        if self._ntire_2018:
            return os.path.join(
                self.images_dir, f"DIV2K_{self.subset}_LR_{self.lr_downgrade}"
            )
        else:
            return os.path.join(
                self.images_dir,
                f"DIV2K_{self.subset}_LR_{self.lr_downgrade}",
                f"X{self.lr_scale}",
            )

    def _hr_images_archive(self):
        if self.hr_scale == 1:
            return f"DIV2K_{self.subset}_HR.zip"
        elif self._ntire_2018:
            return f"DIV2K_{self.subset}_LR_{self.hr_downgrade}.zip"
        else:
            return f"DIV2K_{self.subset}_LR_{self.hr_downgrade}_X{self.hr_scale}.zip"

    def _lr_images_archive(self):
        if self._ntire_2018:
            return f"DIV2K_{self.subset}_LR_{self.lr_downgrade}.zip"
        else:
            return f"DIV2K_{self.subset}_LR_{self.lr_downgrade}_X{self.lr_scale}.zip"


def tf_resize(lr_img, hr_img, lr_scale=2, hr_scale=1):
    lr_shape = tf.shape(hr_img)[:2] // lr_scale
    lr_img = tf.image.resize(lr_img, lr_shape, method="bicubic")
    if hr_scale > 8:
        hr_shape = tf.shape(hr_img)[:2] // hr_scale
        hr_img = tf.image.resize(hr_img, hr_shape, method="bicubic")
    return lr_img, hr_img


def random_crop(lr_img, hr_img, hr_crop_size=96, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(
        shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32
    )
    lr_h = tf.random.uniform(
        shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32
    )

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h : lr_h + lr_crop_size, lr_w : lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h : hr_h + hr_crop_size, hr_w : hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(
        rn < 0.5,
        lambda: (lr_img, hr_img),
        lambda: (tf.image.flip_left_right(lr_img), tf.image.flip_left_right(hr_img)),
    )


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


def download_archive(file, target_dir, extract=True):
    source_url = f"http://data.vision.ee.ethz.ch/cvl/DIV2K/{file}"
    target_dir = os.path.abspath(target_dir)
    tf.keras.utils.get_file(file, source_url, cache_subdir=target_dir, extract=extract)
    os.remove(os.path.join(target_dir, file))
