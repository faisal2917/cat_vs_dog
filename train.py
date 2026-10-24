"""Fine-tune an ImageNet-pretrained MobileNetV3 to tell cats from dogs (CPU, or GPU if present).

Phase 1 ("head"): the backbone is frozen and only the new 2-class layer learns. Fast.
Phase 2 ("finetune"): the last few backbone blocks are unfrozen too, at a lower
learning rate, for the last bit of accuracy.

The best model (by validation accuracy) is saved after every epoch, so stopping
early with Ctrl+C still leaves you a usable model.

Usage:
    python train.py                 # full run, see README for timings
    python train.py --subset 500    # quick test: 500 training images per class
"""
import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics the backbone expects


def build_model(num_classes=2, pretrained=True):
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def train_transform(size):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform(size):
    return transforms.Compose([
        transforms.Resize(size * 8 // 7),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def take_per_class(ds, n):
    """Random subset of an ImageFolder with at most n images per class."""
    order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0)).tolist()
    seen, keep = {}, []
    for i in order:
        c = ds.targets[i]
        if seen.get(c, 0) < n:
            seen[c] = seen.get(c, 0) + 1
            keep.append(i)
    return Subset(ds, keep)


def set_trainable(model, n_blocks):
    """Freeze the backbone except its last n_blocks. The classifier head always trains."""
    for p in model.features.parameters():
        p.requires_grad = False
    if n_blocks:
        for p in model.features[-n_blocks:].parameters():
            p.requires_grad = True


def train_mode(model, n_blocks):
    """Train mode, but frozen blocks stay in eval mode so their BatchNorm statistics don't drift."""
    model.train()
    model.features[:len(model.features) - n_blocks].eval()


def run_epoch(model, loader, criterion, optimizer=None, desc="", scheduler=None):
    """One pass over loader; trains if an optimizer is given, stepping the scheduler (if any)
    after every batch. Returns (loss, accuracy)."""
    device = next(model.parameters()).device
    total_loss = correct = seen = 0
    with torch.set_grad_enabled(optimizer is not None):
        bar = tqdm(loader, desc=desc, unit="batch", leave=False)
        for x, y in bar:
            x, y = x.to(device, memory_format=torch.channels_last), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            seen += len(y)
            bar.set_postfix(loss=f"{total_loss / seen:.3f}", acc=f"{correct / seen:.1%}")
    return total_loss / seen, correct / seen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--out", type=Path, default=ROOT / "models" / "cat_dog_mobilenetv3.pt")
    p.add_argument("--img-size", type=int, default=224, help="160 trains ~1.7x faster, slightly less accurate")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int, default=2, help="epochs with the backbone frozen")
    p.add_argument("--finetune-epochs", type=int, default=1, help="epochs with the top blocks unfrozen")
    p.add_argument("--finetune-blocks", type=int, default=4, help="backbone blocks to unfreeze in phase 2")
    p.add_argument("--subset", type=int, default=0, help="use only N training images per class")
    p.add_argument("--workers", type=int, default=3, help="processes loading images in parallel")
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

    model = build_model(len(classes)).to(device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    phases = [("head", args.head_epochs, 0, 1e-3),
              ("finetune", args.finetune_epochs, args.finetune_blocks, 1e-4)]
    total_epochs = args.head_epochs + args.finetune_epochs
    epoch, best_acc, start = 0, 0.0, time.time()

    for phase, n_epochs, n_blocks, lr in phases:
        if not n_epochs:
            continue
        set_trainable(model, n_blocks)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        for _ in range(n_epochs):
            epoch += 1
            epoch_start = time.time()
            train_mode(model, n_blocks)
            _, train_acc = run_epoch(model, train_loader, criterion, optimizer,
                                     f"Epoch {epoch}/{total_epochs} {phase} train")
            model.eval()
            _, val_acc = run_epoch(model, val_loader, criterion, desc=f"Epoch {epoch}/{total_epochs} val")
            saved = val_acc > best_acc
            if saved:
                best_acc = val_acc
                torch.save({"arch": "mobilenet_v3_large", "classes": classes, "img_size": args.img_size,
                            "state_dict": model.state_dict()}, args.out)
            print(f"Epoch {epoch}/{total_epochs} ({phase}): train acc {train_acc:.2%}, val acc {val_acc:.2%}, "
                  f"{(time.time() - epoch_start) / 60:.1f} min" + ("  -> saved" if saved else ""))

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
