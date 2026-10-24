"""Classify images as cat or dog with the model trained by train.py or train_scratch.py.

Usage:
    python predict.py photo.jpg [more.jpg ...]
    python predict.py some_folder
    python predict.py photo.jpg --model models/cat_dog_scratch.pt

Without --model it uses the most recently trained model in models/.
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from train import ROOT, build_model, eval_transform
from train_scratch import SmallCNN

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_model():
    """The most recently trained model in models/ (resume checkpoints excluded)."""
    found = [p for p in (ROOT / "models").glob("*.pt") if not p.stem.endswith("_last")]
    if not found:
        sys.exit("No trained model in models/. Run train.py or train_scratch.py first.")
    return max(found, key=lambda p: p.stat().st_mtime)


def load_model(path):
    """The trained model in eval mode, its class names, and the preprocessing it expects."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["arch"] == "small_cnn":
        model = SmallCNN(len(ckpt["classes"]))
    else:
        model = build_model(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"], eval_transform(ckpt["img_size"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="image files or folders")
    p.add_argument("--model", type=Path, help="trained model (default: newest in models/)")
    args = p.parse_args()

    model_path = args.model or find_model()
    model, classes, transform = load_model(model_path)
    print(f"Model: {model_path.name}")

    files = []
    for path in args.paths:
        files += sorted(f for f in path.rglob("*") if f.suffix.lower() in IMG_EXTS) if path.is_dir() else [path]

    with torch.inference_mode():
        for f in files:
            x = transform(Image.open(f).convert("RGB")).unsqueeze(0)
            confidence, idx = model(x).softmax(1)[0].max(0)
            print(f"{classes[idx]:>4}  {confidence:6.1%}  {f}")


if __name__ == "__main__":
    main()
