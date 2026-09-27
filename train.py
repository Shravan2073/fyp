"""Train the early-fusion Fakeddit classifier.

Raw mode (default) reads images + titles and runs the frozen backbones on the fly.
Cached mode (--cache_dir) reads features written by extract_features.py and is much faster --
recommended on free-tier Colab. A resumable last.pt is written every epoch (use --resume after a
disconnect), and best_<N>way.pt tracks the best validation macro-F1.
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import (EVAL_TF, LABEL_NAMES, TRAIN_TF, CachedFeatureDataset, Collator,
                     FakedditDataset, load_split)
from model import DEFAULT_IMAGE_BACKBONE, DEFAULT_TEXT_MODEL, IMAGE_BACKBONES, EarlyFusionClassifier


def apply_cache_meta(args):
    """Cached features fix the backbones/text length; adopt them from the cache's meta.json."""
    with open(os.path.join(args.cache_dir, "meta.json")) as fh:
        meta = json.load(fh)
    for key in ("image_backbone", "text_model", "max_text_len", "txt_dim"):
        if getattr(args, key, None) not in (None, meta[key]):
            print(f"note: using {key}={meta[key]!r} from the feature cache (was {getattr(args, key)!r})")
        setattr(args, key, meta[key])


def build_loader(args, split, train, tokenizer=None):
    """Returns (loader, labels). Cached mode needs only the cache (no TSVs or images on disk)."""
    if args.cache_dir:
        ds = CachedFeatureDataset(args.cache_dir, split, args.label)
        collate = None
    else:
        subset = args.subset if split == "train" else args.val_subset
        df = load_split(args.data_dir, split, args.image_dir, subset, args.seed)
        ds = FakedditDataset(df, args.image_dir, args.label, TRAIN_TF if train else EVAL_TF)
        collate = Collator(tokenizer, args.max_text_len)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=train, num_workers=args.num_workers,
                        collate_fn=collate, pin_memory=torch.cuda.is_available(), drop_last=train)
    return loader, np.asarray(ds.labels)


def build_model(args, n_classes, cached):
    return EarlyFusionClassifier(n_classes, d_model=args.d_model, n_layers=args.layers,
                                 n_heads=args.heads, max_text_len=args.max_text_len,
                                 dropout=args.dropout, image_backbone=args.image_backbone,
                                 text_model=args.text_model, txt_dim=getattr(args, "txt_dim", None),
                                 load_backbones=not cached, unfreeze_top=args.unfreeze_top)


def forward_batch(model, batch, device, text_drop=0.0, force_no_text=False):
    """Runs the model on a raw or cached batch. text_drop randomly turns samples image-only."""
    has_text = batch["has_text"].to(device)
    if force_no_text:
        has_text = torch.zeros_like(has_text)
    elif text_drop > 0:
        has_text = has_text & (torch.rand(has_text.shape, device=device) >= text_drop)
    with torch.autocast(device, enabled=device == "cuda"):
        if "img_feats" in batch:
            return model.fuse(batch["img_feats"].to(device), batch["txt_feats"].to(device),
                              batch["txt_mask"].to(device), has_text)
        return model(batch["pixels"].to(device), batch["input_ids"].to(device),
                     batch["attention_mask"].to(device), has_text)


@torch.no_grad()
def predict(model, loader, device, force_no_text=False):
    model.eval()
    preds, labels, has_text = [], [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        logits = forward_batch(model, batch, device, force_no_text=force_no_text)
        preds.append(logits.argmax(-1).cpu())
        labels.append(batch["labels"])
        has_text.append(batch["has_text"])
    return torch.cat(preds).numpy(), torch.cat(labels).numpy(), torch.cat(has_text).numpy()


def class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = len(labels) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(np.clip(w, 0.1, 10.0), dtype=torch.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", help="folder with multimodal_*.tsv (not needed with --cache_dir)")
    p.add_argument("--image_dir", help="folder with <id>.jpg images (not needed with --cache_dir)")
    p.add_argument("--cache_dir", default=None, help="use cached features from extract_features.py")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--label", type=int, choices=[2, 3, 6], default=6)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=float, default=0.05, help="fraction of steps for LR warmup")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_text_len", type=int, default=32)
    p.add_argument("--text_drop", type=float, default=0.15, help="modality dropout prob for text")
    p.add_argument("--image_backbone", default=DEFAULT_IMAGE_BACKBONE, choices=list(IMAGE_BACKBONES))
    p.add_argument("--text_model", default=DEFAULT_TEXT_MODEL)
    p.add_argument("--unfreeze_top", action="store_true",
                   help="fine-tune the last CNN stage + last 2 text layers (slower; raw mode only)")
    p.add_argument("--subset", type=int, default=None, help="subsample train split")
    p.add_argument("--val_subset", type=int, default=None, help="subsample validate split")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from <out_dir>/last.pt")
    args = p.parse_args()

    if args.cache_dir and args.unfreeze_top:
        p.error("--unfreeze_top needs raw images; cached features come from frozen backbones")
    if args.cache_dir:
        apply_cache_meta(args)
    elif not (args.data_dir and args.image_dir):
        p.error("--data_dir and --image_dir are required unless --cache_dir is given")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = args.label
    tokenizer = None if args.cache_dir else AutoTokenizer.from_pretrained(args.text_model)

    train_loader, train_labels = build_loader(args, "train", True, tokenizer)
    val_loader, _ = build_loader(args, "validate", False, tokenizer)

    model = build_model(args, n_classes, cached=bool(args.cache_dir)).to(device)
    args.txt_dim = model.txt_dim
    backbone_ids = {id(q) for q in model.backbone_parameters()}
    head_params = [q for q in model.parameters() if q.requires_grad and id(q) not in backbone_ids]
    bb_params = [q for q in model.backbone_parameters() if q.requires_grad]
    groups = [{"params": head_params, "lr": args.lr}]
    if bb_params:
        groups.append({"params": bb_params, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.wd)

    total = args.epochs * len(train_loader)
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    weights = class_weights(train_labels, n_classes)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    n_train = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"device={device} train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"classes={LABEL_NAMES[n_classes]} trainable_params={n_train / 1e6:.1f}M")

    os.makedirs(args.out_dir, exist_ok=True)
    last_path = os.path.join(args.out_dir, "last.pt")
    best_f1, start_epoch = -1.0, 1
    if args.resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        best_f1, start_epoch = state["best_f1"], state["epoch"] + 1
        print(f"resumed from epoch {state['epoch']} (best macro-F1 {best_f1:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in bar:
            labels = batch["labels"].to(device)
            logits = forward_batch(model, batch, device, text_drop=args.text_drop)
            loss = criterion(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            sched.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            bar.set_postfix(loss=f"{running / seen:.4f}")

        preds, labels, _ = predict(model, val_loader, device)
        acc, f1 = accuracy_score(labels, preds), f1_score(labels, preds, average="macro")
        img_preds, _, _ = predict(model, val_loader, device, force_no_text=True)
        img_acc = accuracy_score(labels, img_preds)
        print(f"epoch {epoch}: train_loss={running / seen:.4f} val_acc={acc:.4f} "
              f"val_macroF1={f1:.4f} val_acc_image_only={img_acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.trainable_state_dict(), "args": vars(args), "epoch": epoch,
                        "val_macro_f1": f1}, os.path.join(args.out_dir, f"best_{args.label}way.pt"))
            print(f"  saved new best (macro-F1 {f1:.4f})")
        torch.save({"model": model.trainable_state_dict(), "optimizer": optimizer.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "best_f1": best_f1, "args": vars(args)}, last_path)


if __name__ == "__main__":
    main()
