import os
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

SPLIT_FILES = {
    "train": "multimodal_train.tsv",
    "validate": "multimodal_validate.tsv",
    "test": "multimodal_test_public.tsv",
}

LABEL_COLS = {2: "2_way_label", 3: "3_way_label", 6: "6_way_label"}

# Class index meanings as defined by the Fakeddit paper.
LABEL_NAMES = {
    2: ["fake", "true"],
    3: ["true", "fake_with_true_text", "fake_with_false_text"],
    6: ["true", "satire/parody", "misleading", "imposter", "false_connection", "manipulated"],
}

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_split(data_dir, split, image_dir, subset=None, seed=0):
    """Read a Fakeddit TSV, keep only rows whose image exists locally, optionally subsample (seeded)."""
    df = pd.read_csv(os.path.join(data_dir, SPLIT_FILES[split]), sep="\t",
                     dtype={"id": str}, low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["clean_title"] = df["clean_title"].fillna("").astype(str).str.strip()

    available = {f[:-4] for f in os.listdir(image_dir) if f.endswith(".jpg")}
    keep = df["id"].isin(available)
    if (~keep).any():
        print(f"[{split}] skipping {(~keep).sum()} rows without a downloaded image")
    df = df[keep]

    if subset and subset < len(df):
        df = df.sample(n=subset, random_state=seed)
    return df.reset_index(drop=True)


class FakedditDataset(Dataset):
    """Raw samples: image tensor + title string. Empty titles become image-only samples."""

    def __init__(self, df, image_dir, label=6, transform=EVAL_TF):
        self.ids = df["id"].tolist()
        self.titles = df["clean_title"].tolist()
        self.labels = df[LABEL_COLS[label]].astype(int).tolist()
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        path = os.path.join(self.image_dir, f"{self.ids[i]}.jpg")
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            warnings.warn(f"unreadable image {path}, using a blank image")
            img = Image.new("RGB", (224, 224))
        title = self.titles[i]
        return {"pixels": self.transform(img), "title": title,
                "has_text": len(title) > 0, "label": self.labels[i]}


class Collator:
    """Tokenizes titles with the BERT tokenizer and stacks everything into a batch."""

    def __init__(self, tokenizer, max_len=64, pad_to_max=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.padding = "max_length" if pad_to_max else "longest"

    def __call__(self, items):
        enc = self.tokenizer([it["title"] for it in items], padding=self.padding,
                             truncation=True, max_length=self.max_len, return_tensors="pt")
        return {
            "pixels": torch.stack([it["pixels"] for it in items]),
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "has_text": torch.tensor([it["has_text"] for it in items], dtype=torch.bool),
            "labels": torch.tensor([it["label"] for it in items], dtype=torch.long),
        }


class CachedFeatureDataset(Dataset):
    """Pre-extracted frozen backbone features written by extract_features.py."""

    def __init__(self, cache_dir, split, label=6):
        d = os.path.join(cache_dir, split)
        self.img = np.load(os.path.join(d, "img.npy"), mmap_mode="r")
        self.txt = np.load(os.path.join(d, "txt.npy"), mmap_mode="r")
        self.txt_mask = np.load(os.path.join(d, "txt_mask.npy"), mmap_mode="r")
        self.has_text = np.load(os.path.join(d, "has_text.npy"))
        self.labels = np.load(os.path.join(d, "labels.npy"))[:, [2, 3, 6].index(label)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "img_feats": torch.from_numpy(np.array(self.img[i])),
            "txt_feats": torch.from_numpy(np.array(self.txt[i])),
            "txt_mask": torch.from_numpy(np.array(self.txt_mask[i])),
            "has_text": torch.tensor(bool(self.has_text[i])),
            "labels": torch.tensor(int(self.labels[i])),
        }
