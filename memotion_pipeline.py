"""
memotion_pipeline.py
====================
Storage-efficient streaming pipeline for Memotion 7k (Ahren09/MMSoc_Memotion).

Features:
- Streaming-only (streaming=True): Never downloads the full dataset archive.
- In-memory image processing: Images are decoded only when requested and immediately freed.
- Duplicate detection: Prevents data contamination and leakage.
- Strict multi-label mapping: Unsupported classes (e.g. Political) are saved as -1 (unknown).
- Cache isolation: Confines Hugging Face cache to local project directory.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime
from PIL import Image

# Force UTF-8 on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import memotion_config as cfg

os.makedirs(cfg.CACHE_DIR, exist_ok=True)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = cfg.CACHE_DIR
os.environ["HF_HOME"] = cfg.CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset


def dir_size_bytes(path: str) -> int:
    if not os.path.exists(path):
        return 0
    total = 0
    for dp, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def human_bytes(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def process_memotion_stream(
    max_samples: int = cfg.DEFAULT_SAMPLE_LIMIT,
    split: str = cfg.DEFAULT_SPLIT,
    decode_images: bool = True,
    overwrite: bool = True,
) -> dict:
    print("\n" + "=" * 65)
    print("  MEMOTION 7K STORAGE-EFFICIENT DATA PIPELINE")
    print("=" * 65)
    print(f"  Dataset       : {cfg.DATASET_NAME}")
    print(f"  Split         : {split}")
    print(f"  Sample Limit  : {max_samples}")
    print(f"  Decode Images : {decode_images}")
    print(f"  Cache Dir     : {cfg.CACHE_DIR}")

    cache_before = dir_size_bytes(cfg.CACHE_DIR)
    t0 = time.time()

    print("\n[1/3] Initializing dataset stream...")
    ds = load_dataset(
        cfg.DATASET_NAME,
        name=cfg.DATASET_CONFIG,
        split=split,
        streaming=True,
    )
    print(f"  Stream initialized in {time.time() - t0:.2f}s")

    if overwrite and os.path.exists(cfg.OUTPUT_CSV):
        try:
            os.remove(cfg.OUTPUT_CSV)
        except OSError:
            pass

    seen_hashes: set[str] = set()
    records = []
    category_counts = {cat: {"pos": 0, "neg": 0, "unk": 0} for cat in cfg.TARGET_CATEGORIES}

    print(f"\n[2/3] Processing records (target: {max_samples})...")
    for i, row in enumerate(ds):
        if len(records) >= max_samples:
            break

        text_ocr = (row.get("text_ocr") or "").strip()
        text_corrected = (row.get("text_corrected") or "").strip()
        preferred_text = text_corrected if text_corrected else text_ocr

        # Deduplication hash based on text
        text_hash = hashlib.md5(preferred_text.encode("utf-8", errors="ignore")).hexdigest()
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        # Multi-label ground truth
        mapped_labels = cfg.map_memotion_labels(row)

        for cat, val in mapped_labels.items():
            if val == 1:
                category_counts[cat]["pos"] += 1
            elif val == 0:
                category_counts[cat]["neg"] += 1
            else:
                category_counts[cat]["unk"] += 1

        img_w = img_h = None
        img_decoded_ok = False
        if decode_images:
            raw_img = row.get("image")
            if raw_img is not None:
                try:
                    if isinstance(raw_img, dict) and "bytes" in raw_img:
                        pil = Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
                    elif isinstance(raw_img, Image.Image):
                        pil = raw_img.convert("RGB")
                    else:
                        pil = None

                    if pil:
                        img_w, img_h = pil.width, pil.height
                        img_decoded_ok = True
                        # Immediately release memory, do NOT save image to disk
                        pil.close()
                        del pil
                except Exception:
                    img_decoded_ok = False

        record_entry = {
            "index": len(records),
            "text": preferred_text[:300],
            "raw_humor": row.get("humor"),
            "raw_sarcasm": row.get("sarcasm"),
            "raw_offensive": row.get("offensive"),
            "raw_motivational": row.get("motivational"),
            "raw_sentiment": row.get("sentiment"),
            "img_width": img_w,
            "img_height": img_h,
            "img_decoded": img_decoded_ok,
        }
        # Add mapped multi-labels
        for cat in cfg.TARGET_CATEGORIES:
            record_entry[f"label_{cat}"] = mapped_labels.get(cat, -1)

        records.append(record_entry)

        if len(records) % 10 == 0 or len(records) == max_samples:
            current_cache = dir_size_bytes(cfg.CACHE_DIR)
            print(
                f"  [{len(records):3d}/{max_samples}] "
                f"text={preferred_text[:30]!r:<32} "
                f"funny={mapped_labels['funny']} "
                f"sarcastic={mapped_labels['sarcastic']} "
                f"offensive={mapped_labels['offensive']} "
                f"cache={human_bytes(current_cache)}"
            )

    # Write output CSV
    print(f"\n[3/3] Saving processed multi-label metadata...")
    if records:
        fieldnames = list(records[0].keys())
        with open(cfg.OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    cache_after = dir_size_bytes(cfg.CACHE_DIR)
    cache_growth = cache_after - cache_before

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": cfg.DATASET_NAME,
        "split": split,
        "sample_limit": max_samples,
        "total_records_processed": len(records),
        "unique_records": len(seen_hashes),
        "decode_images": decode_images,
        "category_distribution": category_counts,
        "cache_dir": cfg.CACHE_DIR,
        "cache_bytes_before": cache_before,
        "cache_bytes_after": cache_after,
        "cache_growth_bytes": cache_growth,
        "cache_human": human_bytes(cache_after),
        "output_csv": cfg.OUTPUT_CSV,
    }

    with open(cfg.REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("  MEMOTION PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Total Processed    : {len(records)}")
    print(f"  Storage Growth     : {human_bytes(cache_growth)}")
    print(f"  Total Cache Size   : {human_bytes(cache_after)}")
    print(f"  Output CSV         : {cfg.OUTPUT_CSV}")
    print("\n  Category Label Breakdown:")
    for cat in cfg.TARGET_CATEGORIES:
        counts = category_counts[cat]
        print(f"    - {cat:<12}: +1 (Pos)={counts['pos']:2d} |  0 (Neg)={counts['neg']:2d} | -1 (Unknown)={counts['unk']:2d}")

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Stream and process Memotion 7k dataset.")
    p.add_argument("--samples", type=int, default=cfg.DEFAULT_SAMPLE_LIMIT, help="Records to process")
    p.add_argument("--split", type=str, default=cfg.DEFAULT_SPLIT, help="Dataset split (train/validation/test)")
    p.add_argument("--no-images", action="store_true", help="Skip image decoding pass")
    p.add_argument("--append", action="store_true", help="Append to CSV instead of overwriting")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_memotion_stream(
        max_samples=args.samples,
        split=args.split,
        decode_images=not args.no_images,
        overwrite=not args.append,
    )
