"""
train_multimodal.py
===================
Training and validation script for the Decoupled Multimodal Meme Classifier.
- Uses MaskedBCEWithLogitsLoss to ignore unsupported/unknown labels (-1).
- Tunes per-class decision thresholds on the validation split.
- Saves model checkpoints, metrics, and tuned thresholds to disk.
- Runs in storage-efficient streaming mode (zero image files saved locally).
"""

import argparse
import io
import json
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from transformers import CLIPProcessor

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


import memotion_config as cfg
from data_loader import get_memotion_dataloader
from multimodal_model import MultimodalMemeClassifier, MaskedBCEWithLogitsLoss


def compute_metrics_from_arrays(
    probs: np.ndarray,
    targets: np.ndarray,
    categories: list[str],
    thresholds: dict[str, float],
) -> dict:
    """Computes comprehensive per-class metrics directly from probability and target arrays."""
    class_metrics = {}
    f1_scores_supported = []

    for j, cat in enumerate(categories):
        y_true = targets[:, j]
        valid_mask = (y_true != -1)
        if not np.any(valid_mask):
            class_metrics[cat] = {"status": "UNKNOWN_MASKED", "support_pos": 0, "support_neg": 0}
            continue

        y_true_valid = y_true[valid_mask]
        th = thresholds.get(cat, 0.5)
        y_pred_valid = (probs[:, j][valid_mask] >= th).astype(int)

        tp = int(np.sum((y_true_valid == 1) & (y_pred_valid == 1)))
        fp = int(np.sum((y_true_valid == 0) & (y_pred_valid == 1)))
        tn = int(np.sum((y_true_valid == 0) & (y_pred_valid == 0)))
        fn = int(np.sum((y_true_valid == 1) & (y_pred_valid == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(y_true_valid) if len(y_true_valid) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        pos_count = int(np.sum(y_true_valid == 1))
        neg_count = int(np.sum(y_true_valid == 0))

        class_metrics[cat] = {
            "status": "EVALUATED",
            "threshold": round(th, 3),
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
            "specificity": round(spec, 3),
        }
        if cat in cfg.SUPPORTED_STYLISTIC_CATEGORIES:
            f1_scores_supported.append(f1)

    macro_f1 = float(np.mean(f1_scores_supported)) if f1_scores_supported else 0.0
    return {"metrics": class_metrics, "macro_f1": round(macro_f1, 3)}


def tune_thresholds_for_classes(
    probs: np.ndarray,
    targets: np.ndarray,
    categories: list[str],
    threshold_range: np.ndarray = np.arange(0.20, 0.85, 0.02),
) -> dict[str, float]:
    """
    Tune per-class binary decision threshold on validation set.
    Maximizes balanced binary Macro F1: (F1_pos + F1_neg) / 2 to strictly prevent
    degenerate all-positive (Specificity=0) or all-negative (Sensitivity=0) threshold selection.
    """
    best_thresholds = {}
    for j, cat in enumerate(categories):
        y_true = targets[:, j]
        valid_mask = (y_true != -1)
        if not np.any(valid_mask):
            best_thresholds[cat] = 0.5
            continue

        y_true_valid = y_true[valid_mask]
        y_prob_valid = probs[:, j][valid_mask]

        pos_count = int(np.sum(y_true_valid == 1))
        neg_count = int(np.sum(y_true_valid == 0))
        if pos_count == 0 or neg_count == 0:
            best_thresholds[cat] = 0.5
            continue

        best_score = -1.0
        best_th = 0.5

        for th in threshold_range:
            y_pred = (y_prob_valid >= th).astype(int)
            tp = int(np.sum((y_true_valid == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true_valid == 0) & (y_pred == 1)))
            tn = int(np.sum((y_true_valid == 0) & (y_pred == 0)))
            fn = int(np.sum((y_true_valid == 1) & (y_pred == 0)))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_pos = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1_neg = (2 * prec_neg * rec_neg) / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0.0

            # Balanced score penalizes degenerate all-positive (tn=0) or all-negative (tp=0)
            balanced_score = 0.5 * f1_pos + 0.5 * f1_neg
            if tn == 0 or tp == 0:
                balanced_score *= 0.2  # heavily penalize degenerate operating points

            if balanced_score > best_score:
                best_score = balanced_score
                best_th = float(th)

        best_thresholds[cat] = best_th

    return best_thresholds


def evaluate_split(
    model: MultimodalMemeClassifier,
    dataloader,
    device: torch.device,
    thresholds: dict[str, float] | None = None,
) -> dict:
    model.eval()
    all_probs = []
    all_targets = []
    total_samples = 0

    categories = model.stylistic_categories
    if thresholds is None:
        thresholds = {cat: 0.5 for cat in categories}

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["stylistic_targets"].to(device)

            outputs = model(pixel_values, input_ids, attention_mask)
            probs = outputs["stylistic_probs"].cpu().numpy()
            targets_np = targets.cpu().numpy()

            all_probs.append(probs)
            all_targets.append(targets_np)
            total_samples += len(pixel_values)

    if not all_probs:
        return {"samples": 0, "macro_f1": 0.0, "metrics": {}}

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    # Compute metrics per category
    class_metrics = {}
    f1_scores_supported = []

    for j, cat in enumerate(categories):
        y_true = all_targets[:, j]
        valid_mask = (y_true != -1)
        if not np.any(valid_mask):
            class_metrics[cat] = {"status": "UNKNOWN_MASKED", "support_pos": 0, "support_neg": 0}
            continue

        y_true_valid = y_true[valid_mask]
        th = thresholds.get(cat, 0.5)
        y_pred_valid = (all_probs[:, j][valid_mask] >= th).astype(int)

        tp = int(np.sum((y_true_valid == 1) & (y_pred_valid == 1)))
        fp = int(np.sum((y_true_valid == 0) & (y_pred_valid == 1)))
        tn = int(np.sum((y_true_valid == 0) & (y_pred_valid == 0)))
        fn = int(np.sum((y_true_valid == 1) & (y_pred_valid == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(y_true_valid) if len(y_true_valid) > 0 else 0.0

        pos_count = int(np.sum(y_true_valid == 1))
        neg_count = int(np.sum(y_true_valid == 0))

        class_metrics[cat] = {
            "status": "EVALUATED",
            "threshold": round(th, 3),
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
        if cat in cfg.SUPPORTED_STYLISTIC_CATEGORIES:
            f1_scores_supported.append(f1)

    macro_f1 = float(np.mean(f1_scores_supported)) if f1_scores_supported else 0.0

    return {
        "samples": total_samples,
        "probs": all_probs,
        "targets": all_targets,
        "metrics": class_metrics,
        "macro_f1": round(macro_f1, 3),
        "thresholds_used": thresholds,
    }


def train_multimodal_model(
    num_epochs: int = 3,
    max_train_samples: int | None = 40,
    max_val_samples: int | None = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device_str: str | None = None,
    checkpoint_name: str = "best_multimodal_classifier_gpu.pt",
    use_amp: bool = False,
) -> dict:
    if device_str is None or device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Stopping per instruction: do not silently fall back to CPU."
            )
        device = torch.device("cuda")
    else:
        device = torch.device(device_str)

    if max_train_samples is not None and max_train_samples <= 0:
        max_train_samples = None
    if max_val_samples is not None and max_val_samples <= 0:
        max_val_samples = None

    print("\n" + "=" * 65)
    print("  MULTIMODAL MODEL TRAINING & VALIDATION (GPU PIPELINE)")
    print("=" * 65)
    print(f"  Device              : {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"  Max Train Samples   : {'FULL DATASET' if max_train_samples is None else max_train_samples}")
    print(f"  Max Val Samples     : {'FULL DATASET' if max_val_samples is None else max_val_samples}")
    print(f"  Batch Size          : {batch_size}")
    print(f"  Epochs              : {num_epochs}")
    print(f"  Learning Rate       : {learning_rate}")
    print(f"  Mixed Precision AMP : {use_amp}")
    print(f"  Checkpoint File     : {checkpoint_name}")

    t0 = time.time()
    clip_processor = CLIPProcessor.from_pretrained(cfg.DEFAULT_HYPERPARAMS["clip_model_name"])

    # Instantiate model
    model = MultimodalMemeClassifier(
        clip_model_name=cfg.DEFAULT_HYPERPARAMS["clip_model_name"],
        stylistic_categories=["funny", "sarcastic", "offensive", "wholesome", "neutral", "political"],
        fusion_dim=cfg.DEFAULT_HYPERPARAMS["fusion_dim"],
        dropout=cfg.DEFAULT_HYPERPARAMS["dropout"],
        freeze_backbone=True,
    ).to(device)

    # Optimizer & Masked Criterion
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    criterion_stylistic = MaskedBCEWithLogitsLoss()
    criterion_hate = MaskedBCEWithLogitsLoss()

    use_amp = (device.type == "cuda" and use_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_macro_f1 = -1.0
    best_thresholds = {cat: 0.5 for cat in model.stylistic_categories}
    best_val_metrics = {}
    epoch_timings = []

    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, checkpoint_name)

    for epoch in range(1, num_epochs + 1):
        epoch_t0 = time.time()
        model.train()
        epoch_seen_hashes = set()
        train_loader = get_memotion_dataloader(
            split="train",
            max_samples=max_train_samples,
            batch_size=batch_size,
            clip_processor=clip_processor,
            seen_hashes=epoch_seen_hashes,
            stylistic_categories=model.stylistic_categories,
        )

        epoch_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stylistic_targets = batch["stylistic_targets"].to(device)
            hate_targets = batch["hate_targets"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(pixel_values, input_ids, attention_mask)
                loss_stylistic = criterion_stylistic(outputs["stylistic_logits"], stylistic_targets)
                loss_hate = criterion_hate(outputs["hate_logits"], hate_targets)
                loss = loss_stylistic + 0.3 * loss_hate

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        epoch_dur = round(time.time() - epoch_t0, 1)
        epoch_timings.append(epoch_dur)
        print(f"\n[Epoch {epoch}/{num_epochs}] Train Loss: {avg_loss:.4f} ({batch_count} batches, {epoch_dur}s)")
        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f"  Peak VRAM Allocation: {peak_vram:.1f} MiB")

        # Validation pass (use seen_hashes=None for deterministic validation)
        val_loader = get_memotion_dataloader(
            split="validation",
            max_samples=max_val_samples,
            batch_size=batch_size,
            clip_processor=clip_processor,
            seen_hashes=None,
            stylistic_categories=model.stylistic_categories,
        )
        val_eval = evaluate_split(model, val_loader, device, thresholds=None)

        # Tune thresholds on validation probabilities
        if "probs" in val_eval and "targets" in val_eval and len(val_eval["probs"]) > 0:
            current_thresholds = tune_thresholds_for_classes(
                val_eval["probs"],
                val_eval["targets"],
                model.stylistic_categories,
            )
            # Re-evaluate in memory with tuned thresholds (avoids re-streaming generator)
            val_eval_tuned = compute_metrics_from_arrays(
                val_eval["probs"],
                val_eval["targets"],
                model.stylistic_categories,
                current_thresholds,
            )
            val_macro_f1 = val_eval_tuned["macro_f1"]
            print(f"  Val Macro F1 (Tuned Thresholds): {val_macro_f1:.3f} (thresholds: {current_thresholds})")
            for cat, m in val_eval_tuned["metrics"].items():
                if m.get("status") == "UNKNOWN_MASKED":
                    continue
                spec = m.get("specificity", m["tn"] / (m["tn"] + m["fp"]) if (m["tn"] + m["fp"]) > 0 else 0.0)
                print(f"    - {cat:10s}: Prec={m['precision']:.3f}, Rec={m['recall']:.3f}, F1={m['f1']:.3f}, Spec={spec:.3f}")

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_thresholds = current_thresholds
                best_val_metrics = val_eval_tuned["metrics"]

                # Save lightweight checkpoint (trainable fusion & heads only, ~2.8 MB)
                trainable_state_dict = {
                    k: v.cpu() for k, v in model.state_dict().items()
                    if not k.startswith("clip.")
                }
                torch.save({
                    "epoch": epoch,
                    "trainable_state_dict": trainable_state_dict,
                    "val_macro_f1": best_val_macro_f1,
                    "best_thresholds": best_thresholds,
                    "stylistic_categories": model.stylistic_categories,
                    "fusion_dim": cfg.DEFAULT_HYPERPARAMS["fusion_dim"],
                }, checkpoint_path)
                print(f"  --> Checkpoint saved to {checkpoint_path} ({os.path.getsize(checkpoint_path)/1024:.1f} KB)")

    peak_vram_total = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Best Val Macro F1 : {best_val_macro_f1:.3f}")
    print(f"  Tuned Thresholds  : {best_thresholds}")
    print(f"  Peak VRAM Used    : {peak_vram_total:.1f} MiB")
    print(f"  Elapsed Time      : {time.time() - t0:.1f}s")

    summary = {
        "status": "SUCCESS",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "peak_vram_mib": round(peak_vram_total, 1),
        "epochs": num_epochs,
        "epoch_timings_seconds": epoch_timings,
        "best_val_macro_f1": round(best_val_macro_f1, 3),
        "best_thresholds": best_thresholds,
        "best_val_metrics": best_val_metrics,
        "checkpoint_path": checkpoint_path,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    if checkpoint_name == "best_multimodal_classifier.pt":
        summary_filename = "train_summary.json"
    else:
        base_name = os.path.splitext(checkpoint_name)[0]
        summary_filename = f"{base_name}_summary.json"

    summary_file = os.path.join(cfg.OUTPUT_DIR, summary_filename)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Train Multimodal Meme Classifier on GPU/CPU.")
    p.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p.add_argument("--train-samples", type=int, default=0, help="Max training samples (0 or -1 for full dataset)")
    p.add_argument("--val-samples", type=int, default=0, help="Max validation samples (0 or -1 for full dataset)")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    p.add_argument("--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu')")
    p.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (FP16)")
    p.add_argument("--checkpoint", type=str, default="best_multimodal_classifier_gpu.pt", help="Checkpoint filename")
    p.add_argument("--smoke-test", action="store_true", help="Run quick smoke test with minimal steps")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        train_multimodal_model(
            num_epochs=1,
            max_train_samples=8,
            max_val_samples=4,
            batch_size=4,
            device_str=args.device,
            checkpoint_name="smoke_test_model.pt",
            use_amp=args.amp,
        )
    else:
        train_multimodal_model(
            num_epochs=args.epochs,
            max_train_samples=args.train_samples,
            max_val_samples=args.val_samples,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device_str=args.device,
            checkpoint_name=args.checkpoint,
            use_amp=args.amp,
        )
