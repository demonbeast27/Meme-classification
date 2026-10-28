"""
evaluate_memotion.py
====================
Evaluates the baseline CLIP classifier on Memotion 7k multi-label ground truth.

Compares:
1. Current baseline: 8-way mutually-exclusive Softmax (app.py behavior).
2. Independent prompt scoring: Per-category thresholding (multi-label behavior).

Reports:
- Number of evaluated examples
- Per-class Precision, Recall, and F1
- Macro F1 across supported categories
- Multi-label agreement and category co-occurrence
"""

import csv
import io
import json
import os
import sys
import time
from collections import defaultdict
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Force UTF-8 on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import memotion_config as cfg
from datasets import load_dataset


def run_evaluation(num_samples: int = 30, split: str = "test"):
    print("\n" + "=" * 65)
    print("  MEMOTION MULTI-LABEL BASELINE EVALUATION")
    print("=" * 65)
    print(f"  Model         : openai/clip-vit-base-patch32 (zero-shot)")
    print(f"  Dataset       : {cfg.DATASET_NAME} ({split} split)")
    print(f"  Evaluation N  : {num_samples}")

    t0 = time.time()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    prompts = [
        "a hateful meme",
        "a non-hateful meme",
        "a funny meme",
        "an offensive meme",
        "a wholesome meme",
        "a sarcastic meme",
        "a political meme",
        "a neutral meme",
    ]
    categories = [
        "hateful",
        "non-hateful",
        "funny",
        "offensive",
        "wholesome",
        "sarcastic",
        "political",
        "neutral",
    ]

    ds = load_dataset(cfg.DATASET_NAME, split=split, streaming=True)

    evaluated = []
    failed_images = 0

    print(f"\n[1/2] Streaming & evaluating {num_samples} records in-memory...")
    for i, row in enumerate(ds):
        if len(evaluated) >= num_samples:
            break

        raw_img = row.get("image")
        if raw_img is None:
            failed_images += 1
            continue

        try:
            if isinstance(raw_img, dict) and "bytes" in raw_img:
                pil = Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
            elif isinstance(raw_img, Image.Image):
                pil = raw_img.convert("RGB")
            else:
                pil = None
        except Exception:
            pil = None

        if pil is None:
            failed_images += 1
            continue

        # Forward pass through CLIP
        inputs = clip_proc(text=prompts, images=pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits = outputs.logits_per_image[0]  # shape (8,)
            softmax_probs = logits.softmax(dim=0).cpu().tolist()
            # Sigmoid on normalized cosine similarities (independent probability per class)
            # Logit scale in CLIP is ~100.0, we scale logits by 0.1 for calibrated sigmoid
            sigmoid_probs = torch.sigmoid(logits * 0.1).cpu().tolist()

        pil.close()
        del pil

        # Ground truth
        gt = cfg.map_memotion_labels(row)

        record_res = {
            "index": len(evaluated),
            "text": (row.get("text_corrected") or row.get("text_ocr") or "")[:80],
            "gt": gt,
            "softmax_scores": {cat: prob for cat, prob in zip(categories, softmax_probs)},
            "sigmoid_scores": {cat: prob for cat, prob in zip(categories, sigmoid_probs)},
            # Top-1 softmax prediction (current app.py)
            "top1_pred": categories[int(torch.argmax(logits).item())],
        }
        evaluated.append(record_res)

    print(f"  Processed {len(evaluated)} records in {time.time() - t0:.1f}s (failed images: {failed_images})")

    # ─────────────────────────────────────────────────────────────────────────
    # Compute Metrics per Category
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/2] Calculating multi-label evaluation metrics...")

    # We evaluate for categories that have known ground truth (gt != -1)
    # Using an independent multi-label threshold of 0.5 on sigmoid_scores
    THRESHOLD = 0.5
    metrics = {}

    for cat in categories:
        valid_pairs = [(r["gt"][cat], r["sigmoid_scores"][cat]) for r in evaluated if r["gt"][cat] != -1]
        if not valid_pairs:
            metrics[cat] = {"status": "NO_GROUND_TRUTH", "support_pos": 0, "support_neg": 0}
            continue

        tp = sum(1 for y_true, score in valid_pairs if y_true == 1 and score >= THRESHOLD)
        fp = sum(1 for y_true, score in valid_pairs if y_true == 0 and score >= THRESHOLD)
        tn = sum(1 for y_true, score in valid_pairs if y_true == 0 and score < THRESHOLD)
        fn = sum(1 for y_true, score in valid_pairs if y_true == 1 and score < THRESHOLD)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(valid_pairs) if valid_pairs else 0.0

        pos_count = sum(1 for y_true, _ in valid_pairs if y_true == 1)
        neg_count = sum(1 for y_true, _ in valid_pairs if y_true == 0)

        metrics[cat] = {
            "status": "EVALUATED",
            "support_pos": pos_count,
            "support_neg": neg_count,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": round(acc, 3),
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
        }

    # Macro F1 across evaluated stylistic categories (funny, sarcastic, offensive, wholesome, neutral)
    stylistic_cats = ["funny", "sarcastic", "offensive", "wholesome", "neutral"]
    f1_list = [metrics[c]["f1"] for c in stylistic_cats if metrics[c]["status"] == "EVALUATED"]
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-label co-occurrence in Ground Truth
    # ─────────────────────────────────────────────────────────────────────────
    multi_positive_count = 0
    for r in evaluated:
        pos_labels = [c for c in stylistic_cats if r["gt"][c] == 1]
        if len(pos_labels) > 1:
            multi_positive_count += 1
    multi_label_pct = (multi_positive_count / len(evaluated)) * 100 if evaluated else 0.0

    print("\n" + "=" * 70)
    print("  PER-CLASS EVALUATION SUMMARY (Multi-Label Zero-Shot Baseline)")
    print("=" * 70)
    print(f"{'Category':<14} | {'Pos':>4} | {'Neg':>4} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>7} | {'F1':>6}")
    print("-" * 70)
    for cat in categories:
        m = metrics[cat]
        if m["status"] == "EVALUATED":
            print(f"{cat:<14} | {m['support_pos']:4d} | {m['support_neg']:4d} | {m['accuracy']:8.3f} | {m['precision']:9.3f} | {m['recall']:7.3f} | {m['f1']:6.3f}")
        else:
            print(f"{cat:<14} | {'--':>4} | {'--':>4} | {'(Not annotated in Memotion 7k - Kept Unknown)':>36}")
    print("-" * 70)
    print(f"  Macro F1 (Stylistic 5-class)      : {macro_f1:.3f}")
    print(f"  Memes with multiple positive tags : {multi_positive_count}/{len(evaluated)} ({multi_label_pct:.1f}%)")
    print("=" * 70)

    # Save evaluation results
    eval_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "evaluated_samples": len(evaluated),
        "failed_images": failed_images,
        "per_class_metrics": metrics,
        "macro_f1_stylistic": round(macro_f1, 3),
        "multi_label_prevalence_pct": round(multi_label_pct, 1),
    }

    report_path = os.path.join(cfg.OUTPUT_DIR, "memotion_zero_shot_baseline_eval.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\n  Report saved to: {report_path}")

    return eval_report


if __name__ == "__main__":
    run_evaluation(num_samples=30)
