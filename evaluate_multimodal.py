"""
evaluate_multimodal.py
======================
Side-by-side comparative evaluation script:
Compares the trained MultimodalMemeClassifier against the existing Zero-Shot CLIP baseline (app.py)
on the exact same dataset split and records.

Metrics:
- Per-class Precision, Recall, F1, Accuracy, Confusion Matrix (TP, FP, TN, FN)
- Macro F1 across supported stylistic categories (funny, sarcastic, offensive)
- Excludes unknown / unannotated categories (-1) from metric calculations
"""

import argparse
import io
import json
import os
import sys
import time
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import memotion_config as cfg
from data_loader import get_memotion_dataloader
from multimodal_model import MultimodalMemeClassifier
from train_multimodal import tune_thresholds_for_classes



def run_comparative_evaluation(
    split: str = "validation",
    num_samples: int = 30,
    checkpoint_name: str = "best_multimodal_classifier.pt",
    device_str: str | None = None,
) -> dict:
    if num_samples is not None and num_samples <= 0:
        num_samples = None

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("\n" + "=" * 75)
    print("  SIDE-BY-SIDE COMPARATIVE EVALUATION")
    print("=" * 75)
    print(f"  Split                 : {split}")
    print(f"  Samples               : {num_samples}")
    print(f"  Device                : {device}")
    print(f"  Multimodal Checkpoint : {checkpoint_name}")

    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        print(f"  [ERROR] Checkpoint not found at {checkpoint_path}")
        print("  Please train the model first using: python train_multimodal.py")
        return {"error": "Checkpoint not found"}

    ckpt = torch.load(checkpoint_path, map_location=device)
    tuned_thresholds = ckpt.get("best_thresholds", {})
    stylistic_categories = ckpt.get("stylistic_categories", ["funny", "sarcastic", "offensive", "wholesome", "neutral", "political"])

    # Load new multimodal model
    multimodal_model = MultimodalMemeClassifier(
        clip_model_name=cfg.DEFAULT_HYPERPARAMS["clip_model_name"],
        stylistic_categories=stylistic_categories,
        fusion_dim=ckpt.get("fusion_dim", 256),
        freeze_backbone=True,
    ).to(device)
    state_dict = ckpt.get("trainable_state_dict", ckpt.get("model_state_dict"))
    multimodal_model.load_state_dict(state_dict, strict=False)
    multimodal_model.eval()

    # Load baseline CLIP model (from app.py)
    clip_processor = CLIPProcessor.from_pretrained(cfg.DEFAULT_HYPERPARAMS["clip_model_name"])
    baseline_clip = CLIPModel.from_pretrained(cfg.DEFAULT_HYPERPARAMS["clip_model_name"]).to(device)
    baseline_clip.eval()

    # Baseline prompts from app.py
    baseline_prompts = [f"a {cat} meme" if cat != "offensive" else "an offensive meme" for cat in stylistic_categories]

    dataloader = get_memotion_dataloader(
        split=split,
        max_samples=num_samples,
        batch_size=8,
        clip_processor=clip_processor,
        stylistic_categories=stylistic_categories,
    )

    all_targets = []
    all_mm_probs = []
    all_base_probs = []
    total_evaluated = 0

    t0 = time.time()
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["stylistic_targets"].cpu().numpy()

            # 1. New Multimodal Model Forward Pass
            mm_outputs = multimodal_model(pixel_values, input_ids, attention_mask)
            mm_probs = mm_outputs["stylistic_probs"].cpu().numpy()

            # 2. Existing Baseline Zero-Shot CLIP Forward Pass
            # Paired contrastive prompts for proper binary calibration:
            # P(cat) = softmax([pos_prompt, neg_prompt])[0]
            batch_base_probs = []
            for cat in stylistic_categories:
                pos_prompt = f"a {cat} meme" if cat != "offensive" else "an offensive meme"
                neg_prompt = f"a non-{cat} meme" if cat != "offensive" else "a non-offensive meme"
                pair_inputs = clip_processor(
                    text=[pos_prompt, neg_prompt],
                    return_tensors="pt",
                    padding=True
                ).to(device)
                pair_outputs = baseline_clip(
                    pixel_values=pixel_values,
                    input_ids=pair_inputs["input_ids"],
                    attention_mask=pair_inputs["attention_mask"]
                )
                # Softmax over the paired positive and negative prompts
                pair_probs = torch.softmax(pair_outputs.logits_per_image, dim=-1)  # (B, 2)
                batch_base_probs.append(pair_probs[:, 0:1])  # P(pos)

            baseline_probs = torch.cat(batch_base_probs, dim=-1).cpu().numpy()  # (B, 6)

            all_targets.append(targets)
            all_mm_probs.append(mm_probs)
            all_base_probs.append(baseline_probs)
            total_evaluated += len(pixel_values)

    all_targets = np.vstack(all_targets)
    all_mm_probs = np.vstack(all_mm_probs)
    all_base_probs = np.vstack(all_base_probs)

    # --- Fair threshold tuning: both models tune on the same validation data ---
    # Multimodal: use the thresholds saved in the checkpoint (already tuned on train-time val)
    # Baseline:   tune now on the same evaluation samples for a fair comparison
    baseline_tuned_thresholds = tune_thresholds_for_classes(
        all_base_probs, all_targets, stylistic_categories
    )
    print(f"  Baseline tuned thresholds : {baseline_tuned_thresholds}")
    print(f"  Multimodal tuned thresholds: {tuned_thresholds}")

    all_multimodal_preds = np.zeros_like(all_mm_probs)
    all_baseline_preds = np.zeros_like(all_base_probs)
    for j, cat in enumerate(stylistic_categories):
        all_multimodal_preds[:, j] = (all_mm_probs[:, j] >= tuned_thresholds.get(cat, 0.5)).astype(int)
        all_baseline_preds[:, j] = (all_base_probs[:, j] >= baseline_tuned_thresholds.get(cat, 0.5)).astype(int)

    # Compute metrics side by side
    comparisons = {}
    mm_supported_f1s = []
    base_supported_f1s = []

    for j, cat in enumerate(stylistic_categories):
        y_true = all_targets[:, j]
        valid_mask = (y_true != -1)

        if not np.any(valid_mask):
            comparisons[cat] = {
                "status": "UNKNOWN_MASKED",
                "support_pos": 0,
                "support_neg": 0,
                "note": "Not annotated in dataset; excluded from metrics"
            }
            continue

        y_valid = y_true[valid_mask]
        mm_pred_valid = all_multimodal_preds[:, j][valid_mask]
        base_pred_valid = all_baseline_preds[:, j][valid_mask]

        # Multimodal metrics
        tp_m = int(np.sum((y_valid == 1) & (mm_pred_valid == 1)))
        fp_m = int(np.sum((y_valid == 0) & (mm_pred_valid == 1)))
        tn_m = int(np.sum((y_valid == 0) & (mm_pred_valid == 0)))
        fn_m = int(np.sum((y_valid == 1) & (mm_pred_valid == 0)))
        prec_m = tp_m / (tp_m + fp_m) if (tp_m + fp_m) > 0 else 0.0
        rec_m = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0.0
        f1_m = (2 * prec_m * rec_m) / (prec_m + rec_m) if (prec_m + rec_m) > 0 else 0.0
        spec_m = tn_m / (tn_m + fp_m) if (tn_m + fp_m) > 0 else 0.0

        # Baseline metrics
        tp_b = int(np.sum((y_valid == 1) & (base_pred_valid == 1)))
        fp_b = int(np.sum((y_valid == 0) & (base_pred_valid == 1)))
        tn_b = int(np.sum((y_valid == 0) & (base_pred_valid == 0)))
        fn_b = int(np.sum((y_valid == 1) & (base_pred_valid == 0)))
        prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
        rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
        f1_b = (2 * prec_b * rec_b) / (prec_b + rec_b) if (prec_b + rec_b) > 0 else 0.0
        spec_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0

        comparisons[cat] = {
            "status": "EVALUATED",
            "support_pos": int(np.sum(y_valid == 1)),
            "support_neg": int(np.sum(y_valid == 0)),
            "multimodal": {
                "precision": round(prec_m, 3),
                "recall": round(rec_m, 3),
                "f1": round(f1_m, 3),
                "specificity": round(spec_m, 3),
                "tp": tp_m, "fp": fp_m, "tn": tn_m, "fn": fn_m,
                "threshold": round(tuned_thresholds.get(cat, 0.5), 3),
            },
            "baseline": {
                "precision": round(prec_b, 3),
                "recall": round(rec_b, 3),
                "f1": round(f1_b, 3),
                "specificity": round(spec_b, 3),
                "tp": tp_b, "fp": fp_b, "tn": tn_b, "fn": fn_b,
                "threshold": round(baseline_tuned_thresholds.get(cat, 0.5), 3),
            },
        }

        if cat in cfg.SUPPORTED_STYLISTIC_CATEGORIES:
            mm_supported_f1s.append(f1_m)
            base_supported_f1s.append(f1_b)

    mm_macro = float(np.mean(mm_supported_f1s)) if mm_supported_f1s else 0.0
    base_macro = float(np.mean(base_supported_f1s)) if base_supported_f1s else 0.0

    # Micro F1: aggregate TP/FP/FN across all supported classes
    tp_mm_total = fp_mm_total = fn_mm_total = 0
    tp_b_total = fp_b_total = fn_b_total = 0
    for cat in cfg.SUPPORTED_STYLISTIC_CATEGORIES:
        c = comparisons.get(cat, {})
        if c.get("status") != "EVALUATED":
            continue
        tp_mm_total += c["multimodal"]["tp"]
        fp_mm_total += c["multimodal"]["fp"]
        fn_mm_total += c["multimodal"]["fn"]
        tp_b_total += c["baseline"]["tp"]
        fp_b_total += c["baseline"]["fp"]
        fn_b_total += c["baseline"]["fn"]
    mm_micro_prec = tp_mm_total / (tp_mm_total + fp_mm_total) if (tp_mm_total + fp_mm_total) > 0 else 0.0
    mm_micro_rec = tp_mm_total / (tp_mm_total + fn_mm_total) if (tp_mm_total + fn_mm_total) > 0 else 0.0
    mm_micro_f1 = (2 * mm_micro_prec * mm_micro_rec) / (mm_micro_prec + mm_micro_rec) if (mm_micro_prec + mm_micro_rec) > 0 else 0.0
    b_micro_prec = tp_b_total / (tp_b_total + fp_b_total) if (tp_b_total + fp_b_total) > 0 else 0.0
    b_micro_rec = tp_b_total / (tp_b_total + fn_b_total) if (tp_b_total + fn_b_total) > 0 else 0.0
    b_micro_f1 = (2 * b_micro_prec * b_micro_rec) / (b_micro_prec + b_micro_rec) if (b_micro_prec + b_micro_rec) > 0 else 0.0

    print("\n" + "=" * 95)
    print("  COMPARISON SUMMARY: NEW MULTIMODAL vs BASELINE ZERO-SHOT CLIP")
    print("=" * 95)
    print(f"{'Category':<10} | {'Pos':>3} {'Neg':>3} | {'Baseline (Prec / Rec / F1 / Spec)':>34} | {'Multimodal (Prec / Rec / F1 / Spec)':>36}")
    print("-" * 95)
    for cat in stylistic_categories:
        c = comparisons[cat]
        if c["status"] == "EVALUATED":
            b = c["baseline"]
            m = c["multimodal"]
            b_str = f"{b['precision']:.2f} / {b['recall']:.2f} / {b['f1']:.2f} / {b['specificity']:.2f}"
            m_str = f"{m['precision']:.2f} / {m['recall']:.2f} / {m['f1']:.2f} / {m['specificity']:.2f}"
            print(f"{cat:<10} | {c['support_pos']:3d} {c['support_neg']:3d} | {b_str:>34} | {m_str:>36}")
        else:
            print(f"{cat:<10} | {'--':>3} {'--':>3} | {'(Unsupported in Memotion 7k - Masked)':>72}")
    print("-" * 95)
    print(f"  Macro F1 (Supported 3-Class) | Baseline: {base_macro:.3f} | New Multimodal: {mm_macro:.3f} | Delta: {mm_macro - base_macro:+.3f}")
    print(f"  Micro F1 (Supported 3-Class) | Baseline: {b_micro_f1:.3f} | New Multimodal: {mm_micro_f1:.3f} | Delta: {mm_micro_f1 - b_micro_f1:+.3f}")
    print("=" * 95)


    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "evaluated_samples": total_evaluated,
        "split": split,
        "baseline_thresholds": baseline_tuned_thresholds,
        "multimodal_thresholds": tuned_thresholds,
        "comparisons": comparisons,
        "macro_f1_baseline": round(base_macro, 3),
        "macro_f1_multimodal": round(mm_macro, 3),
        "macro_f1_delta": round(mm_macro - base_macro, 3),
        "micro_f1_baseline": round(b_micro_f1, 3),
        "micro_f1_multimodal": round(mm_micro_f1, 3),
        "micro_f1_delta": round(mm_micro_f1 - b_micro_f1, 3),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    if checkpoint_name == "best_multimodal_classifier.pt":
        report_filename = "comparative_evaluation_report.json"
    else:
        base_name = os.path.splitext(checkpoint_name)[0]
        report_filename = f"comparative_evaluation_{base_name}.json"
    report_path = os.path.join(cfg.OUTPUT_DIR, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Detailed report saved to: {report_path}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multimodal model vs baseline.")
    parser.add_argument("--split", type=str, default="validation", help="Evaluation split (validation/test)")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples to evaluate")
    parser.add_argument("--checkpoint", type=str, default="best_multimodal_classifier.pt", help="Checkpoint filename")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    run_comparative_evaluation(
        split=args.split,
        num_samples=args.samples,
        checkpoint_name=args.checkpoint,
        device_str=args.device,
    )
