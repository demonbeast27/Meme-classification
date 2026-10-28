"""
stream_dataset.py
=================
Storage-efficient streaming experiment for:
  aliencaocao/multimodal_meme_classification_singapore

Run:
    python stream_dataset.py [--samples N] [--split SPLIT] [--no-images]

What this script does:
  1. Opens the dataset with streaming=True (no full Parquet download).
  2. Iterates up to MAX_SAMPLES records.
  3. For each record, extracts:
       - metadata.id, metadata.text, metadata.gpt_response,
         metadata.human_response, sub_dataset
       - The first image (decoded from the embedded binary blob in Parquet)
         only when DOWNLOAD_IMAGES_ONLY_WHEN_NEEDED is True.
  4. Maps gpt_response to {hateful / non-hateful} — the ONLY label this
     dataset explicitly supports for our 8-category classifier.
  5. Writes a metadata CSV and a JSON experiment report.
  6. Reports actual bytes consumed by the HF cache.

IMPORTANT CAVEATS (read before scaling up):
  - Images are embedded as binary blobs inside the Parquet shards.
    Streaming does NOT avoid downloading image data; it only avoids
    downloading the entire shard before processing starts.
  - The test split has 2 Parquet shards totalling ~612 MB.  If you
    process all 2 897 records in that split the full shard will be
    fetched.  Keep MAX_SAMPLES small during initial experiments.
  - License: GPL-3.0.  Check compliance before using for any published work.
  - Label reliability: gpt_response is GPT-generated; human_response is
    null for most records in the test split.  Treat labels as noisy.
"""

import argparse
import csv
import io
import json
import os
import shutil
import time
from datetime import datetime

from PIL import Image

import dataset_config as cfg

# ────────────────────────────────────────────────────────────────────────────
# Optional: redirect HF cache to the project directory so nothing
# writes to the global ~/.cache/huggingface folder.
# ────────────────────────────────────────────────────────────────────────────
os.makedirs(cfg.CACHE_DIR, exist_ok=True)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = cfg.CACHE_DIR
os.environ["HF_HOME"]           = cfg.CACHE_DIR

# Import AFTER setting env vars so the library picks up the custom cache path.
from datasets import load_dataset  # noqa: E402


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def dir_size_bytes(path: str) -> int:
    """Return total byte-size of all files under *path* (0 if not exists)."""
    if not os.path.exists(path):
        return 0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def decode_label(index) -> str:
    """Map ClassLabel integer to human string; return 'unknown' on failure."""
    if index is None:
        return "unknown"
    return cfg.LABEL_INDEX_TO_NAME.get(int(index), f"idx_{index}")


def decode_image(raw) -> Image.Image | None:
    """
    Convert the HF Image feature value to a PIL image.

    raw can be:
      - dict with 'bytes' key  (most common in streaming mode)
      - PIL.Image already decoded
      - None / empty
    """
    if raw is None:
        return None
    if isinstance(raw, Image.Image):
        return raw
    if isinstance(raw, dict):
        data = raw.get("bytes") or raw.get("path")
        if isinstance(data, (bytes, bytearray)):
            try:
                return Image.open(io.BytesIO(bytes(data))).convert("RGB")
            except Exception:
                return None
    return None


def resize_if_needed(img: Image.Image) -> Image.Image:
    """Resize so the longest edge is at most MAX_IMAGE_EDGE."""
    max_e = cfg.MAX_IMAGE_EDGE
    if max_e and max(img.width, img.height) > max_e:
        ratio = max_e / max(img.width, img.height)
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)),
            Image.LANCZOS
        )
    return img


# ────────────────────────────────────────────────────────────────────────────
# Main experiment
# ────────────────────────────────────────────────────────────────────────────

def run_experiment(
    max_samples: int,
    split: str,
    stream: bool,
    decode_images: bool,
    overwrite: bool = False,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  SG Meme Dataset Streaming Experiment")
    print(f"{'='*60}")
    print(f"  dataset  : {cfg.DATASET_NAME}")
    print(f"  split    : {split}")
    print(f"  streaming: {stream}")
    print(f"  samples  : {max_samples}")
    print(f"  images   : {decode_images}")
    print(f"  cache    : {cfg.CACHE_DIR}")
    print(f"{'='*60}\n")

    cache_before = dir_size_bytes(cfg.CACHE_DIR)

    # ── Step 1 – Load dataset (streaming, no full download) ────────────────
    print("[1/4] Opening dataset stream …")
    t0 = time.time()
    try:
        ds = load_dataset(
            cfg.DATASET_NAME,
            name=cfg.DATASET_CONFIG,
            split=split,
            streaming=stream,
            cache_dir=cfg.CACHE_DIR,
            trust_remote_code=False,
        )
    except Exception as exc:
        print(f"  ERROR loading dataset: {exc}")
        return {"error": str(exc)}
    print(f"  Dataset opened in {time.time()-t0:.1f}s")

    # ── Step 2 – Load existing IDs from CSV (skip duplicates) ─────────────
    if overwrite and os.path.exists(cfg.METADATA_CSV):
        try:
            os.remove(cfg.METADATA_CSV)
        except OSError:
            pass

    existing_ids: set[str] = set()
    if not overwrite and cfg.SKIP_DUPLICATES and os.path.exists(cfg.METADATA_CSV):
        with open(cfg.METADATA_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("id"):
                    existing_ids.add(row["id"])
        print(f"  Loaded {len(existing_ids)} existing IDs to skip")

    # ── Step 3 – Iterate stream ────────────────────────────────────────────
    print(f"\n[2/4] Streaming records (limit={max_samples}) …")
    csv_file_exists = os.path.exists(cfg.METADATA_CSV)
    csv_handle = open(cfg.METADATA_CSV, "a", newline="", encoding="utf-8")
    FIELDS = [
        "id", "text", "sub_dataset",
        "gpt_label", "human_label",
        "img_width", "img_height", "img_decoded",
        "msg_count",
    ]
    writer = csv.DictWriter(csv_handle, fieldnames=FIELDS)
    if not csv_file_exists:
        writer.writeheader()

    processed = 0
    skipped   = 0
    decode_ok = 0
    decode_fail = 0
    label_counts: dict[str, int] = {}
    sub_datasets: set[str] = set()

    try:
        for i, row in enumerate(ds):
            if processed >= max_samples:
                break

            meta     = row.get("metadata") or {}
            row_id   = str(meta.get("id", f"row_{i}"))
            text     = meta.get("text", "") or ""
            gpt_raw  = meta.get("gpt_response")
            hum_raw  = meta.get("human_response")
            sub_ds   = row.get("sub_dataset", "") or ""
            images   = row.get("images") or []
            messages = row.get("messages") or {}

            # Duplicate guard
            if row_id in existing_ids:
                skipped += 1
                continue

            gpt_label = decode_label(gpt_raw)
            hum_label = decode_label(hum_raw)
            label_counts[gpt_label] = label_counts.get(gpt_label, 0) + 1
            sub_datasets.add(sub_ds)

            # Image decoding (only when requested)
            img_w = img_h = None
            img_decoded = False
            if decode_images and images:
                first_img_raw = images[0] if isinstance(images, list) else images
                pil = decode_image(first_img_raw)
                if pil:
                    pil = resize_if_needed(pil)
                    img_w, img_h = pil.width, pil.height
                    img_decoded = True
                    decode_ok += 1
                    pil.close()
                else:
                    decode_fail += 1

            # How many message turns?
            msg_content = messages.get("content") if isinstance(messages, dict) else []
            msg_count   = len(msg_content) if isinstance(msg_content, list) else 0

            writer.writerow({
                "id":         row_id,
                "text":       text[:500],   # truncate long text for CSV
                "sub_dataset": sub_ds,
                "gpt_label":  gpt_label,
                "human_label": hum_label,
                "img_width":  img_w,
                "img_height": img_h,
                "img_decoded": img_decoded,
                "msg_count":  msg_count,
            })
            existing_ids.add(row_id)
            processed += 1

            if processed % 10 == 0:
                cache_used = dir_size_bytes(cfg.CACHE_DIR)
                print(
                    f"  [{processed:4d}/{max_samples}] "
                    f"id={row_id[:40]}  "
                    f"gpt={gpt_label:<12}  "
                    f"cache={human_bytes(cache_used)}"
                )
    finally:
        csv_handle.flush()
        csv_handle.close()

    # ── Step 4 – Write report ──────────────────────────────────────────────
    print(f"\n[3/4] Writing experiment report …")
    cache_after = dir_size_bytes(cfg.CACHE_DIR)

    report = {
        "timestamp":          datetime.now().isoformat(),
        "dataset":            cfg.DATASET_NAME,
        "split":              split,
        "streaming":          stream,
        "decode_images":      decode_images,
        "max_samples":        max_samples,
        "records_processed":  processed,
        "records_skipped_duplicate": skipped,
        "image_decode_ok":    decode_ok,
        "image_decode_fail":  decode_fail,
        "label_distribution": label_counts,
        "sub_datasets_seen":  sorted(sub_datasets),
        "cache_dir":          cfg.CACHE_DIR,
        "cache_bytes_before": cache_before,
        "cache_bytes_after":  cache_after,
        "cache_bytes_used_this_run": cache_after - cache_before,
        "cache_human":        human_bytes(cache_after),
        "output_csv":         cfg.METADATA_CSV,
        "label_note": (
            "Only 'hateful' and 'non-hateful' are explicitly annotated. "
            "Do NOT use these labels for the other 6 classifier categories."
        ),
        "license_note": "GPL-3.0 – check compliance before publishing.",
        "label_reliability_note": (
            "gpt_response is GPT-4 generated and noisy. "
            "human_response is null for most test-split records."
        ),
        "clear_cache_command": (
            "import shutil, dataset_config; "
            "shutil.rmtree(dataset_config.CACHE_DIR, ignore_errors=True)"
        ),
    }

    with open(cfg.REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Done.")
    print(f"\n  Records processed : {processed}")
    print(f"  Skipped (dups)    : {skipped}")
    print(f"  Images decoded    : {decode_ok} / {decode_ok + decode_fail}")
    print(f"  Label distribution: {label_counts}")
    print(f"  Sub-datasets seen : {sorted(sub_datasets)}")
    print(f"\n  Cache before      : {human_bytes(cache_before)}")
    print(f"  Cache after       : {human_bytes(cache_after)}")
    print(f"  Storage this run  : {human_bytes(max(0, cache_after - cache_before))}")
    print(f"\n  Metadata CSV      : {cfg.METADATA_CSV}")
    print(f"  Experiment report : {cfg.REPORT_JSON}")
    print(f"\n  To clear the cache:")
    print(f"    rd /s /q \"{cfg.CACHE_DIR}\"  (Windows)")
    print(f"    rm -rf \"{cfg.CACHE_DIR}\"  (Unix)")

    return report


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stream a sample from the SG Meme dataset.")
    p.add_argument("--samples",   type=int,  default=cfg.MAX_SAMPLES,
                   help=f"Records to process (default {cfg.MAX_SAMPLES})")
    p.add_argument("--split",     type=str,  default=cfg.DATASET_SPLIT,
                   help=f"Dataset split (default {cfg.DATASET_SPLIT})")
    p.add_argument("--no-stream", action="store_true",
                   help="Disable streaming (downloads full shard – not recommended)")
    p.add_argument("--no-images", action="store_true",
                   help="Skip image decoding (metadata pass only)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output CSV rather than appending")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        max_samples  = args.samples,
        split        = args.split,
        stream       = not args.no_stream,
        decode_images= not args.no_images,
        overwrite    = args.overwrite,
    )
