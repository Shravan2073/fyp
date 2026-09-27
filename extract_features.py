"""Cache frozen image-grid and text-token features so training only runs the small fusion model.

With the default ResNet-18 + MiniLM-L6 at max_text_len=32 this is ~75 KB per sample (fp16), i.e.
~6 GB for a 60k/10k/10k subset -- fine for free-tier Colab. Build it on local /content disk.
"""
import argparse
import json
import os

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import EVAL_TF, LABEL_COLS, Collator, FakedditDataset, load_split
from model import DEFAULT_IMAGE_BACKBONE, DEFAULT_TEXT_MODEL, IMAGE_BACKBONES, N_IMG_TOKENS, EarlyFusionClassifier


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "validate", "test"])
    p.add_argument("--subset", type=int, default=None, help="subsample train split to N rows")
    p.add_argument("--val_subset", type=int, default=None, help="subsample validate/test splits to N rows")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_text_len", type=int, default=32)
    p.add_argument("--image_backbone", default=DEFAULT_IMAGE_BACKBONE, choices=list(IMAGE_BACKBONES))
    p.add_argument("--text_model", default=DEFAULT_TEXT_MODEL)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    model = EarlyFusionClassifier(n_classes=2, image_backbone=args.image_backbone, text_model=args.text_model,
                                  max_text_len=args.max_text_len).to(device).eval()
    img_dim, txt_dim = model.img_dim, model.txt_dim

    os.makedirs(args.cache_dir, exist_ok=True)
    with open(os.path.join(args.cache_dir, "meta.json"), "w") as fh:
        json.dump({"image_backbone": args.image_backbone, "text_model": args.text_model,
                   "max_text_len": args.max_text_len, "img_dim": img_dim, "txt_dim": txt_dim}, fh, indent=2)

    for split in args.splits:
        subset = args.subset if split == "train" else args.val_subset
        df = load_split(args.data_dir, split, args.image_dir, subset, args.seed)
        n, L = len(df), args.max_text_len
        print(f"[{split}] {n} samples -> ~{n * (N_IMG_TOKENS * img_dim + L * txt_dim) * 2 / 1e9:.1f} GB")

        out = os.path.join(args.cache_dir, split)
        os.makedirs(out, exist_ok=True)
        img_mm = open_memmap(os.path.join(out, "img.npy"), "w+", np.float16, (n, N_IMG_TOKENS, img_dim))
        txt_mm = open_memmap(os.path.join(out, "txt.npy"), "w+", np.float16, (n, L, txt_dim))
        mask_mm = open_memmap(os.path.join(out, "txt_mask.npy"), "w+", np.bool_, (n, L))
        has_text = np.zeros(n, dtype=np.bool_)

        loader = DataLoader(FakedditDataset(df, args.image_dir, 6, EVAL_TF), batch_size=args.bs,
                            num_workers=args.num_workers,
                            collate_fn=Collator(tokenizer, L, pad_to_max=True))
        i = 0
        for batch in tqdm(loader, desc=split):
            b = batch["labels"].size(0)
            with torch.autocast(device, enabled=device == "cuda"):
                img = model.encode_image(batch["pixels"].to(device))
                txt = model.encode_text(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            img_mm[i:i + b] = img.half().cpu().numpy()
            txt_mm[i:i + b] = txt.half().cpu().numpy()
            mask_mm[i:i + b] = batch["attention_mask"].bool().numpy()
            has_text[i:i + b] = batch["has_text"].numpy()
            i += b

        img_mm.flush(); txt_mm.flush(); mask_mm.flush()
        np.save(os.path.join(out, "has_text.npy"), has_text)
        np.save(os.path.join(out, "labels.npy"), df[[LABEL_COLS[k] for k in (2, 3, 6)]].to_numpy(np.int64))
        np.save(os.path.join(out, "ids.npy"), df["id"].to_numpy(dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
