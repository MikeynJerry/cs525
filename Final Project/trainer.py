import json
import time
import tensorflow as tf

from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


class Trainer:
    def __init__(self, model, loss, learning_rate, checkpoint_dir="./ckpt/edsr"):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            psnr=tf.Variable(-1.0),
            optimizer=Adam(learning_rate),
            model=model,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint, directory=checkpoint_dir, max_to_keep=3
        )

        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, args, shapes, save_best_only=False):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        lr_train_shape, hr_train_shape = shapes[0]
        lr_valid_shape, hr_valid_shape = shapes[1]

        info = {
            "losses": [],
            "psnr": [],
            "time": [],
            "every": args.every,
            "total": args.steps,
            "avg_lr_train_shape": lr_train_shape,
            "avg_hr_train_shape": hr_train_shape,
            "avg_lr_valid_shape": lr_valid_shape,
            "avg_hr_valid_shape": hr_valid_shape,
        }

        for lr, hr in train_dataset.take(args.steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            loss = self.train_step(lr, hr)
            loss_mean(loss)

            if step % args.every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                # Compute PSNR on validation dataset
                psnr_value = self.evaluate(valid_dataset)

                duration = time.perf_counter() - self.now

                info["losses"].append(float(loss_value.numpy()))
                info["psnr"].append(float(psnr_value.numpy()))
                info["time"].append(float(duration))

                filename = (
                    f"{args.dataset}_edsr_lr_x{args.scale * 2}_hr_x{args.scale}_"
                    f"res_{args.nb_res}_filt_{args.nb_filters}_batch_{args.batch_size}_"
                    f"transform_{args.transform}_every_{args.every}_"
                    f"steps_{args.steps}.json"
                )

                with open(filename, "w") as f:
                    json.dump(info, f)

                print(
                    f"{step}/{args.steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)"
                )

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(
            zip(gradients, self.checkpoint.model.trainable_variables)
        )

        return loss_value

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(
                f"Model restored from checkpoint at step {self.checkpoint.step.numpy()}."
            )


class EdsrTrainer(Trainer):
    def __init__(
        self,
        model,
        checkpoint_dir,
        learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]),
    ):
        super().__init__(
            model,
            loss=MeanAbsoluteError(),
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
        )

    def train(self, train_dataset, valid_dataset, args, shapes, save_best_only=True):
        super().train(train_dataset, valid_dataset, args, shapes, save_best_only)


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


def evaluate(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        sr_shape = tf.shape(sr)
        psnr_value = psnr(hr[:, : sr_shape[1], : sr_shape[2], :], sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch
