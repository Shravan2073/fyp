"""Benchmark every best_<N>way.pt checkpoint under --ckpt_dir on one split and write a comparison.

For each checkpoint it reports accuracy / macro-F1 / weighted-F1 on:
  all samples, image+text samples, image-only samples (empty title), and all samples with the
  text removed (how much the model relies on the image alone),
plus a majority-class baseline, per-class F1, trainable parameters and inference throughput.

Results go to <out_dir>/benchmark.md and <out_dir>/benchmark.csv.
"""
import argparse
import csv
import glob
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from dataset import LABEL_NAMES
from evaluate import load_checkpoint
from train import build_loader, predict


def scores(preds, labels, n_classes):
    if len(labels) == 0:
        return {"n": 0, "acc": float("nan"), "macro_f1": float("nan"), "weighted_f1": float("nan")}
    idx = list(range(n_classes))
    return {"n": len(labels), "acc": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, labels=idx, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, preds, labels=idx, average="weighted", zero_division=0)}


def timed_predict(model, loader, device, **kw):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = predict(model, loader, device, **kw)
    if device == "cuda":
        torch.cuda.synchronize()
    return out, len(out[1]) / (time.perf_counter() - t0)


def benchmark(path, cli, device):
    model, args = load_checkpoint(path, cli, device)
    ckpt_meta = torch.load(path, map_location="cpu", weights_only=False)
    n = args.label
    tokenizer = None if cli.cache_dir else AutoTokenizer.from_pretrained(args.text_model)
    loader, labels_all = build_loader(args, cli.split, False, tokenizer)

    (preds, labels, has_text), throughput = timed_predict(model, loader, device)
    img_preds, _, _ = predict(model, loader, device, force_no_text=True)

    # Always predicting the split's most common class; a model must beat this to be useful.
    majority = np.bincount(labels_all, minlength=n).argmax()
    maj_preds = np.full_like(labels, majority)

    per_class = f1_score(labels, preds, labels=list(range(n)), average=None, zero_division=0)
    return {
        "checkpoint": os.path.relpath(path, cli.ckpt_dir),
        "label": f"{n}-way",
        "backbones": f"{args.image_backbone} + {args.text_model.split('/')[-1]}",
        "epoch": ckpt_meta.get("epoch"),
        "val_macro_f1": ckpt_meta.get("val_macro_f1", float("nan")),
        "trainable_params_M": sum(v.numel() for k, v in model.state_dict().items()
                                  if not k.startswith(("cnn.", "bert."))) / 1e6,
        "throughput_per_s": throughput,
        "mode": "cached features (fusion only)" if cli.cache_dir else "raw images (end to end)",
        "all": scores(preds, labels, n),
        "image+text": scores(preds[has_text], labels[has_text], n),
        "image-only (empty title)": scores(preds[~has_text], labels[~has_text], n),
        "text removed": scores(img_preds, labels, n),
        "majority baseline": scores(maj_preds, labels, n),
        "per_class_f1": dict(zip(LABEL_NAMES[n], per_class)),
    }


SUBSETS = ["all", "image+text", "image-only (empty title)", "text removed", "majority baseline"]


def fmt(x):
    return "-" if x != x else f"{x:.4f}"  # NaN -> dash


def write_reports(results, cli):
    os.makedirs(cli.out_dir, exist_ok=True)
    lines = [f"# Fakeddit early-fusion benchmark ({cli.split} split)", ""]

    lines += ["## Summary", "",
              "| Model | Samples | Accuracy | Macro-F1 | Weighted-F1 | Majority acc | Text-removed acc | Val macro-F1 | Params (M) | Samples/s |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        a = r["all"]
        lines.append(f"| {r['label']} ({r['backbones']}) | {a['n']} | {fmt(a['acc'])} | {fmt(a['macro_f1'])} | "
                     f"{fmt(a['weighted_f1'])} | {fmt(r['majority baseline']['acc'])} | "
                     f"{fmt(r['text removed']['acc'])} | {fmt(r['val_macro_f1'])} | "
                     f"{r['trainable_params_M']:.2f} | {r['throughput_per_s']:.0f} |")
    lines += ["", f"Throughput measured on {results[0]['mode']}.", ""]

    for r in results:
        lines += [f"## {r['label']} - `{r['checkpoint']}` (epoch {r['epoch']})", "",
                  "| Subset | n | Accuracy | Macro-F1 | Weighted-F1 |", "|---|---|---|---|---|"]
        for s in SUBSETS:
            m = r[s]
            lines.append(f"| {s} | {m['n']} | {fmt(m['acc'])} | {fmt(m['macro_f1'])} | {fmt(m['weighted_f1'])} |")
        lines += ["", "| Class | F1 |", "|---|---|"]
        lines += [f"| {c} | {fmt(v)} |" for c, v in r["per_class_f1"].items()]
        lines.append("")

    md = "\n".join(lines)
    with open(os.path.join(cli.out_dir, "benchmark.md"), "w", encoding="utf-8") as fh:
        fh.write(md)

    with open(os.path.join(cli.out_dir, "benchmark.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "label", "backbones", "epoch", "subset", "n", "acc", "macro_f1", "weighted_f1"])
        for r in results:
            for s in SUBSETS:
                m = r[s]
                w.writerow([r["checkpoint"], r["label"], r["backbones"], r["epoch"], s,
                            m["n"], m["acc"], m["macro_f1"], m["weighted_f1"]])
    print(md)
    print(f"\nwrote {cli.out_dir}/benchmark.md and benchmark.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="checkpoints", help="searched recursively for best_*way.pt")
    p.add_argument("--data_dir", help="folder with multimodal_*.tsv (not needed with --cache_dir)")
    p.add_argument("--image_dir", help="folder with <id>.jpg images (not needed with --cache_dir)")
    p.add_argument("--cache_dir", default=None, help="feature cache from extract_features.py")
    p.add_argument("--split", default="test", choices=["validate", "test"])
    p.add_argument("--val_subset", type=int, default=None, help="subsample the split (raw mode)")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_dir", default="benchmark_results")
    cli = p.parse_args()
    if not cli.cache_dir and not (cli.data_dir and cli.image_dir):
        p.error("no test data given. Use ONE of:\n"
                "  --cache_dir cache              (feature cache from extract_features.py / cache.tar)\n"
                "  --data_dir tsv --image_dir images   (TSVs + downloaded images, see download_data.py)")

    paths = sorted(glob.glob(os.path.join(cli.ckpt_dir, "**", "best_*way.pt"), recursive=True))
    if not paths:
        p.error(f"no best_*way.pt found under {cli.ckpt_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for path in paths:
        print(f"benchmarking {path} ...")
        results.append(benchmark(path, cli, device))
    write_reports(results, cli)


if __name__ == "__main__":
    main()
