"""Test a trained model on the held-out test images, which were never used in training:
overall and per-class accuracy, and the images it got most confidently wrong.

Usage:
    python evaluate.py                                  # newest model in models/
    python evaluate.py --model models/cat_dog_scratch.pt
    python evaluate.py --wrong 30                       # list 30 mistakes instead of 15
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from predict import find_model, load_model
from train import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, help="trained model (default: newest in models/)")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "test",
                   help="folder with cat/ and dog/ subfolders")
    p.add_argument("--wrong", type=int, default=15, help="how many mistakes to list")
    p.add_argument("--workers", type=int, default=3, help="processes loading images in parallel")
    args = p.parse_args()

    model_path = args.model or find_model()
    model, classes, transform = load_model(model_path)
    model = model.to(memory_format=torch.channels_last)
    ds = datasets.ImageFolder(args.data, transform)
    if ds.classes != classes:
        sys.exit(f"{args.data} has folders {ds.classes}, but the model knows {classes}")

    probs = []
    with torch.inference_mode():
        for x, _ in tqdm(DataLoader(ds, batch_size=64, num_workers=args.workers), desc="Testing", leave=False):
            probs.append(model(x.to(memory_format=torch.channels_last)).softmax(1))
    probs = torch.cat(probs)
    labels = torch.tensor(ds.targets)
    preds = probs.argmax(1)

    print(f"Model {model_path.name} on {len(ds)} test images: accuracy {(preds == labels).float().mean():.2%}\n")
    print("              predicted " + "".join(f"{c:>7}" for c in classes))
    for i, c in enumerate(classes):
        row = [(preds[labels == i] == j).sum().item() for j in range(len(classes))]
        print(f"actually {c:<4}          " + "".join(f"{n:>7}" for n in row) + f"   {row[i] / sum(row):.2%} right")

    wrong = (preds != labels).nonzero().flatten()
    wrong = wrong[probs[wrong].max(1).values.argsort(descending=True)][:args.wrong]
    if len(wrong):
        print(f"\nMost confident mistakes (open them to see what fooled the model):")
        for i in wrong.tolist():
            print(f"  said {classes[preds[i]]} {probs[i].max():6.1%}  {ds.samples[i][0]}")


if __name__ == "__main__":
    main()
