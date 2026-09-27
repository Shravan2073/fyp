"""Download Fakeddit images referenced by the multimodal TSVs.

The TSVs themselves must be downloaded manually from the Google Drive link in
https://github.com/entitize/Fakeddit (folder `multimodal_only_samples`) and placed in --data_dir.

Each image is re-encoded to RGB JPEG and saved as <image_dir>/<id>.jpg. Images that fail to
download or decode are logged to <image_dir>/failed_<split>.txt and later skipped by dataset.py.
"""
import argparse
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from dataset import SPLIT_FILES

HEADERS = {"User-Agent": "Mozilla/5.0 (Fakeddit research downloader)"}


def fetch(row_id, url, image_dir, timeout):
    out = os.path.join(image_dir, f"{row_id}.jpg")
    if os.path.exists(out):
        return row_id, True
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.save(out, "JPEG", quality=90)
        return row_id, True
    except Exception:
        return row_id, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="folder containing the multimodal_*.tsv files")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--splits", nargs="+", default=list(SPLIT_FILES))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--timeout", type=float, default=10)
    p.add_argument("--limit", type=int, nargs="+", default=None,
                   help="random subset size per split (one value, or one per split), e.g. 60000 10000 10000")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.image_dir, exist_ok=True)
    limits = args.limit or [None]
    if len(limits) == 1:
        limits = limits * len(args.splits)
    for split, limit in zip(args.splits, limits):
        df = pd.read_csv(os.path.join(args.data_dir, SPLIT_FILES[split]), sep="\t",
                         dtype={"id": str}, low_memory=False)
        if "hasImage" in df.columns:
            df = df[df["hasImage"].astype(str).str.lower() == "true"]
        df = df.dropna(subset=["image_url"])
        if limit and limit < len(df):
            df = df.sample(n=limit, random_state=args.seed)

        failed = []
        with ThreadPoolExecutor(args.workers) as ex:
            futures = [ex.submit(fetch, i, u, args.image_dir, args.timeout)
                       for i, u in zip(df["id"], df["image_url"])]
            for f in tqdm(as_completed(futures), total=len(futures), desc=split):
                row_id, ok = f.result()
                if not ok:
                    failed.append(row_id)

        with open(os.path.join(args.image_dir, f"failed_{split}.txt"), "w") as fh:
            fh.write("\n".join(failed))
        print(f"{split}: {len(df) - len(failed)}/{len(df)} images available, {len(failed)} failed")


if __name__ == "__main__":
    main()
