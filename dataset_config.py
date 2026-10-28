"""
dataset_config.py
=================
Central configuration for the SG Meme dataset streaming experiment.
Edit values here; all other scripts import from this file.
"""

import os

# ──────────────────────────────────────────────
# Dataset identification
# ──────────────────────────────────────────────
DATASET_NAME    = "aliencaocao/multimodal_meme_classification_singapore"
DATASET_CONFIG  = "default"

# Split to use for the experiment.
# Smallest usable split: "test"  (2 897 examples, ~545 MB raw Parquet)
# Full train splits are 15–21 GB each – do NOT use without storage budget.
DATASET_SPLIT   = "test"

# ──────────────────────────────────────────────
# Streaming / sampling
# ──────────────────────────────────────────────
# Use streaming=True to avoid downloading the full Parquet shard up front.
# NOTE: HF streaming for this dataset still fetches image bytes from the
# embedded Parquet column as each row is decoded.  Images are NOT
# downloaded separately from a remote URL – they are already embedded in
# the Parquet file as binary blobs.  Streaming will therefore download
# only the portion of the Parquet shard that is needed to yield
# MAX_SAMPLES rows, not the entire shard.  Expect ~150–250 kB per image.
STREAM_DATASET  = True

# Maximum number of records to process per run (keeps storage bounded).
MAX_SAMPLES     = 100          # start small; raise to 500 or 1000 later

# Skip records that have already been written to the output CSV.
# Avoids re-downloading the same images if the script is restarted.
SKIP_DUPLICATES = True

# ──────────────────────────────────────────────
# Image handling
# ──────────────────────────────────────────────
# Download / decode each image only when the pipeline actually needs it.
# Set to False to collect metadata-only in the first pass without decoding.
DOWNLOAD_IMAGES_ONLY_WHEN_NEEDED = True

# Maximum edge length (px) to which images are resized before encoding.
# Keeps per-image RAM / temp-disk usage small.
MAX_IMAGE_EDGE  = 512

# ──────────────────────────────────────────────
# Cache / storage
# ──────────────────────────────────────────────
# Project-local cache directory.  HF datasets and image temp files are
# confined here so nothing bleeds into the global HF cache at
# ~/.cache/huggingface.
CACHE_DIR       = os.path.join(os.path.dirname(__file__), ".cache", "hf_stream")

# How to clear the cache safely:
#   import shutil, dataset_config
#   shutil.rmtree(dataset_config.CACHE_DIR, ignore_errors=True)
# Or from the shell:
#   rd /s /q ".cache\hf_stream"          (Windows)
#   rm -rf .cache/hf_stream              (Unix)

# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "experiment_output")
METADATA_CSV    = os.path.join(OUTPUT_DIR, "sg_meme_metadata.csv")
REPORT_JSON     = os.path.join(OUTPUT_DIR, "experiment_report.json")

# ──────────────────────────────────────────────
# Label mapping
# ──────────────────────────────────────────────
# The dataset provides a BINARY label only:
#   metadata.gpt_response  -> ClassLabel(["Yes", "No"])
#   metadata.human_response -> ClassLabel(["Yes", "No"]) - often null
#
# "Yes" = offensive/harmful under Singapore standards  (index 0)
# "No"  = not offensive                                (index 1)
#
# Our BERT+CLIP classifier uses 8 categories.  Only ONE of those categories
# maps directly to this dataset's label:
#   "hateful"     <- gpt_response == "Yes"  (index 0)
#   "non-hateful" <- gpt_response == "No"   (index 1)
#
# The other six categories (funny, offensive, wholesome, sarcastic,
# political, neutral) are NOT annotated in this dataset.  We deliberately
# do NOT manufacture labels for them.
LABEL_INDEX_TO_NAME = {0: "hateful", 1: "non-hateful"}
LABEL_NAME_TO_INDEX = {v: k for k, v in LABEL_INDEX_TO_NAME.items()}
