"""Train a small CNN to tell cats from dogs from scratch: random starting weights, no
pretrained knowledge, so everything it knows it learns from your images. Takes hours on
a laptop CPU instead of minutes, and ends less accurate than train.py.

A checkpoint is written after every epoch. If training stops (Ctrl+C, sleep, crash),
run the same command again with --resume to continue where it left off.

Usage:
    python train_scratch.py                        # full run, see README for timings
    python train_scratch.py --resume               # continue an interrupted run
    python train_scratch.py --subset 500 --epochs 2   # quick test
"""
import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import MEAN, ROOT, STD, eval_transform, run_epoch, take_per_class


def conv_bn(cin, cout, stride=1):
    return [nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class SmallCNN(nn.Sequential):
    """9 conv layers, 1.2M parameters. The first layer halves the resolution so it trains fast on a CPU."""

    def __init__(self, num_classes=2):
        super().__init__(
            *conv_bn(3, 32, stride=2), *conv_bn(32, 32), nn.MaxPool2d(2),
            *conv_bn(32, 64), *conv_bn(64, 64), nn.MaxPool2d(2),
            *conv_bn(64, 128), *conv_bn(128, 128), nn.MaxPool2d(2),
            *conv_bn(128, 256), *conv_bn(256, 256),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes),
        )


def train_transform(size):
    """Stronger augmentation than train.py: a model learning from scratch overfits much more easily."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--out", type=Path, default=ROOT / "models" / "cat_dog_scratch.pt")
    p.add_argument("--img-size", type=int, default=128, help="160 is ~1.7x slower")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-3, help="peak learning rate")
    p.add_argument("--subset", type=int, default=0, help="use only N training images per class")
    p.add_argument("--workers", type=int, default=3, help="processes loading images in parallel")
    p.add_argument("--resume", action="store_true",
                   help="continue the last run (use the same options as that run)")
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = datasets.ImageFolder(args.data / "train", train_transform(args.img_size))
    val_ds = datasets.ImageFolder(args.data / "val", eval_transform(args.img_size))
    classes = train_ds.classes
    if args.subset:
        train_ds = take_per_class(train_ds, args.subset)
        val_ds = take_per_class(val_ds, max(args.subset // 4, 50))
    print(f"Classes {classes}: {len(train_ds)} train / {len(val_ds)} val images, "
          f"device {device}" + (f" ({torch.get_num_threads()} threads)" if device.type == "cpu" else ""))

    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, **loader_args)

    model = SmallCNN(len(classes)).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    # warm up, then slowly lower the learning rate over the whole run
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    total_steps=args.epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    last_path = args.out.with_name(args.out.stem + "_last.pt")
    done_epochs, best_acc = 0, 0.0
    if args.resume:
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        done_epochs, best_acc = state["epoch"], state["best_acc"]
        print(f"Resuming after epoch {done_epochs} (best val acc so far {best_acc:.2%})")

    start = time.time()
    for epoch in range(done_epochs + 1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        _, train_acc = run_epoch(model, train_loader, criterion, optimizer,
                                 f"Epoch {epoch}/{args.epochs} train", scheduler)
        model.eval()
        _, val_acc = run_epoch(model, val_loader, criterion, desc=f"Epoch {epoch}/{args.epochs} val")
        saved = val_acc > best_acc
        if saved:
            best_acc = val_acc
            torch.save({"arch": "small_cnn", "classes": classes, "img_size": args.img_size,
                        "state_dict": model.state_dict()}, args.out)
        # write to a temp file first so stopping mid-save can't corrupt the resume checkpoint
        tmp = last_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best_acc": best_acc}, tmp)
        tmp.replace(last_path)
        minutes = (time.time() - epoch_start) / 60
        print(f"Epoch {epoch}/{args.epochs}: train acc {train_acc:.2%}, val acc {val_acc:.2%}, "
              f"{minutes:.1f} min, ~{minutes * (args.epochs - epoch):.0f} min left" + ("  -> saved" if saved else ""))

    print(f"\nTraining took {(time.time() - start) / 60:.1f} min. Best val acc {best_acc:.2%}, model: {args.out}")

    test_dir = args.data / "test"
    if test_dir.exists() and not args.subset:
        model.load_state_dict(torch.load(args.out, map_location=device, weights_only=True)["state_dict"])
        model.eval()
        test_loader = DataLoader(datasets.ImageFolder(test_dir, eval_transform(args.img_size)), **loader_args)
        _, test_acc = run_epoch(model, test_loader, criterion, desc="Test")
        print(f"Test accuracy on {len(test_loader.dataset)} unseen images: {test_acc:.2%}")


if __name__ == "__main__":
    main()
