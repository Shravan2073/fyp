"""Evaluate a trained checkpoint, broken down into image+text and image-only samples.

"image-only (native)" = samples whose title is empty; "image-only (text removed)" = every sample
with its title masked out, which shows how well the model classifies from the image alone.
"""
import argparse
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer

from dataset import LABEL_NAMES
from train import build_loader, build_model, predict


def report(name, preds, labels, names):
    if len(labels) == 0:
        print(f"\n== {name}: no samples")
        return
    print(f"\n== {name}: n={len(labels)} acc={accuracy_score(labels, preds):.4f} "
          f"macroF1={f1_score(labels, preds, average='macro'):.4f}")
    idx = list(range(len(names)))
    print(classification_report(labels, preds, labels=idx, target_names=names, digits=4, zero_division=0))
    print("confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(labels, preds, labels=idx))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", help="folder with multimodal_*.tsv (not needed with --cache_dir)")
    p.add_argument("--image_dir", help="folder with <id>.jpg images (not needed with --cache_dir)")
    p.add_argument("--cache_dir", default=None, help="evaluate on cached features instead of raw images")
    p.add_argument("--split", default="test", choices=["validate", "test"])
    p.add_argument("--val_subset", type=int, default=None, help="subsample the evaluated split")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    cli = p.parse_args()

    ckpt = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
    # Checkpoints from before the backbone option existed used ResNet-50.
    args = SimpleNamespace(**{"image_backbone": "resnet50", "txt_dim": None, **ckpt["args"],
                              "data_dir": cli.data_dir, "image_dir": cli.image_dir,
                              "cache_dir": cli.cache_dir, "val_subset": cli.val_subset,
                              "bs": cli.bs, "num_workers": cli.num_workers})
    if args.cache_dir and args.unfreeze_top:
        p.error("this checkpoint fine-tuned the backbones; evaluate on raw images (omit --cache_dir)")
    if not args.cache_dir and not (args.data_dir and args.image_dir):
        p.error("--data_dir and --image_dir are required unless --cache_dir is given")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args, args.label, cached=bool(args.cache_dir))
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    bad = [k for k in missing if not k.startswith(("cnn.", "bert."))] + list(unexpected)
    if bad:
        raise RuntimeError(f"checkpoint mismatch: {bad[:10]}")
    model.to(device)

    tokenizer = None if args.cache_dir else AutoTokenizer.from_pretrained(args.text_model)
    loader, _ = build_loader(args, cli.split, False, tokenizer)
    names = LABEL_NAMES[args.label]

    preds, labels, has_text = predict(model, loader, device)
    report(f"{cli.split} - all samples", preds, labels, names)
    report(f"{cli.split} - image+text", preds[has_text], labels[has_text], names)
    report(f"{cli.split} - image-only (native, empty title)", preds[~has_text], labels[~has_text], names)

    img_preds, _, _ = predict(model, loader, device, force_no_text=True)
    report(f"{cli.split} - image-only (text removed from every sample)", img_preds, labels, names)


if __name__ == "__main__":
    main()
