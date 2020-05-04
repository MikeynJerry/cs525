from datasets import CINIC, DIV2K
from edsr import edsr
from trainer import EdsrTrainer
import argparse


def train(args):

    if args.dataset == "cinic":
        DATA = CINIC
    elif args.dataset == "div2k":
        DATA = DIV2K

    train_loader = DATA(scale=args.scale, subset="train")

    train_ds, train_shapes = train_loader.dataset(
        batch_size=args.batch_size, random_transform=args.transform, repeat_count=None
    )

    valid_loader = DATA(scale=args.scale, subset="valid")

    valid_ds, valid_shapes = valid_loader.dataset(
        batch_size=1, random_transform=False, repeat_count=1
    )

    model = edsr(
        scale=2, dataset=args.dataset, nb_res=args.nb_res, nb_filters=args.nb_filters
    )
    checkpoint_dir = (
        f".ckpt/{args.dataset}-edsr-{args.nb_res}-x{args.scale}-{args.nb_filters}"
    )
    trainer = EdsrTrainer(model=model, checkpoint_dir=checkpoint_dir,)

    shapes = (train_shapes, valid_shapes)

    trainer.train(train_ds, valid_ds, args, shapes, save_best_only=True)

    trainer.restore()

    psnr = trainer.evaluate(valid_ds)
    print(f"Best PSNR = {psnr.numpy():3f}")

    trainer.model.save_weights(
        f"{args.dataset}-edsr-{args.nb_res}-x{args.scale}-{args.nb_filters}_weights.h5"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cinic", choices=["cinic", "div2k"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--nb_res", type=int, default=16)
    parser.add_argument("--nb_filters", type=int, default=64)
    parser.add_argument("--scale", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--every", type=int, default=1_000)
    parser.add_argument("--transform", action="store_true")
    args = parser.parse_args()
    train(args)
