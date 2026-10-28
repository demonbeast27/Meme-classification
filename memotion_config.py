"""
memotion_config.py
==================
Configuration and audited label mapping for Memotion 7k (Ahren09/MMSoc_Memotion).
Adheres strictly to scientific validation guidelines:
- Distinguishes genuine ground-truth annotations from proxies.
- Never manufactures ground truth for unsupported categories.
- Marks unsupported / unannotated categories as -1 (unknown) for masked loss & evaluation.
- Decouples stylistic categories from hate detection.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Dataset & Paths
# ─────────────────────────────────────────────────────────────────────────────
DATASET_NAME = "Ahren09/MMSoc_Memotion"
DATASET_CONFIG = "default"

# Standard splits in Ahren09/MMSoc_Memotion:
# - train: 5,593 records
# - validation: 699 records
# - test: 700 records (held-out)
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache", "memotion_stream")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "memotion_output")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 8 Target Categories Audit & Taxonomy
# ─────────────────────────────────────────────────────────────────────────────
# All 8 project categories:
ALL_TARGET_CATEGORIES = [
    "funny",
    "sarcastic",
    "offensive",
    "wholesome",
    "neutral",
    "political",
    "hateful",
    "non-hateful",
]

# Stylistic categories:
# Genuinely supported by Memotion 7k ground-truth:
SUPPORTED_STYLISTIC_CATEGORIES = [
    "funny",
    "sarcastic",
    "offensive",
]

# Unsupported in Memotion 7k (strictly marked as -1 Unknown, masked out):
UNSUPPORTED_STYLISTIC_CATEGORIES = [
    "wholesome",  # 'motivational' is not wholesome (user requirement: do not equate)
    "neutral",    # 'sentiment: neutral' != 'neutral meme' (user requirement: do not assume)
    "political",  # Zero political annotations in Memotion 7k
]

# Extra categories added via zero-shot CLIP inference (no training data; contrastive prompts only):
ZEROSHOT_EXTRA_CATEGORIES = [
    "political",
    "sad",
]

# Contrastive prompts for zero-shot CLIP inference (positive, negative):
ZEROSHOT_PROMPTS = {
    "political": ("a political meme", "a non-political meme"),
    "sad":       ("a sad or depressing meme", "a happy or lighthearted meme"),
    "wholesome": ("a wholesome or heartwarming meme", "a mean or cynical meme"),
}


# Decoupled hate detection (treated separately from stylistic classification):
HATE_CATEGORIES = [
    "hateful",
    "non-hateful",
]

# ─────────────────────────────────────────────────────────────────────────────
# Audited Label Mapping Function
# Output conventions:
#   1  = Positive instance (ground truth verified)
#   0  = Negative instance (ground truth verified)
#  -1  = Unknown / Unsupported / Ambiguous (MASK OUT during loss and evaluation)
# ─────────────────────────────────────────────────────────────────────────────
def map_memotion_record_labels(record: dict) -> dict[str, int]:
    """
    Audited mapping of raw Memotion 7k fields to target project categories.
    
    Raw fields in Ahren09/MMSoc_Memotion:
      - humor:        ['not_funny', 'funny', 'very_funny', 'hilarious']
      - sarcasm:      ['not_sarcastic', 'general', 'twisted_meaning', 'very_twisted']
      - offensive:    ['not_offensive', 'slight', 'very_offensive', 'hateful_offensive']
      - motivational: ['not_motivational', 'motivational']
      - sentiment:    ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
    """
    labels = {}

    # 1. Funny (Genuine Ground Truth from 'humor')
    humor = str(record.get("humor", "")).strip().lower()
    if humor in ["funny", "very_funny", "hilarious"]:
        labels["funny"] = 1
    elif humor == "not_funny":
        labels["funny"] = 0
    else:
        labels["funny"] = -1

    # 2. Sarcastic (Genuine Ground Truth from 'sarcasm')
    sarcasm = str(record.get("sarcasm", "")).strip().lower()
    if sarcasm in ["general", "twisted_meaning", "very_twisted"]:
        labels["sarcastic"] = 1
    elif sarcasm == "not_sarcastic":
        labels["sarcastic"] = 0
    else:
        labels["sarcastic"] = -1

    # 3. Offensive (Genuine Ground Truth from 'offensive')
    offensive = str(record.get("offensive", "")).strip().lower()
    if offensive in ["slight", "very_offensive", "hateful_offensive"]:
        labels["offensive"] = 1
    elif offensive == "not_offensive":
        labels["offensive"] = 0
    else:
        labels["offensive"] = -1

    # 4. Wholesome (NOT Ground Truth in Memotion)
    # User constraint: Do not treat 'motivational' as an exact equivalent of 'wholesome'.
    # Motivational memes often focus on hustle, fitness, or resilience, not wholesome warmth.
    # Therefore, wholesome is marked as unknown (-1) to avoid manufacturing labels.
    labels["wholesome"] = -1

    # 5. Neutral (NOT Ground Truth in Memotion)
    # User constraint: Do not assume that neutral text sentiment means a neutral meme.
    # A meme with neutral OCR text can still be heavily satirical, offensive, or hilarious.
    # Therefore, neutral meme status is marked as unknown (-1).
    labels["neutral"] = -1

    # 6. Political (NOT Annotated in Memotion 7k)
    # Zero political annotation in SemEval-2020 Task 8.
    labels["political"] = -1

    # 7 & 8. Hateful and Non-Hateful (Decoupled Hate Head)
    # In Memotion 7k, 'offensive' contains a 'hateful_offensive' sub-tier (~2% of records).
    # 'not_offensive' is clearly non-hateful (0 for hateful, 1 for non-hateful).
    # 'slight' and 'very_offensive' are offensive without confirmed hate speech definition.
    # We mark 'hateful_offensive' as 1, 'not_offensive' as 0, and ambiguous ones as -1.
    if offensive == "hateful_offensive":
        labels["hateful"] = 1
        labels["non-hateful"] = 0
    elif offensive == "not_offensive":
        labels["hateful"] = 0
        labels["non-hateful"] = 1
    else:
        # Ambiguous: slight/very_offensive without verified hate ground truth
        labels["hateful"] = -1
        labels["non-hateful"] = -1

    return labels


# Default training hyperparameters for lightweight fusion head
DEFAULT_HYPERPARAMS = {
    "clip_model_name": "openai/clip-vit-base-patch32",
    "fusion_dim": 256,
    "dropout": 0.3,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 16,
    "num_epochs": 3,
    "freeze_backbones": True,  # Keep heavy CLIP vision & text backbones frozen; train fusion head
}
