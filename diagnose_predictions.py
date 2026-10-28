"""
diagnose_predictions.py
Diagnostic tool to inspect raw predicted probabilities from:
1. checkpoints/best_multimodal_classifier.pt
2. Zero-shot CLIP baseline (both raw logits, Softmax, and Sigmoids)
on the 20 validation examples.
"""
import io
import os
import sys
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import memotion_config as cfg
from multimodal_model import MultimodalMemeClassifier
from data_loader import get_memotion_dataloader

device = torch.device("cpu")
checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "best_multimodal_classifier.pt")
print(f"Loading checkpoint from: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location=device)

print("Checkpoint keys:", list(ckpt.keys()))
print("Checkpoint best_thresholds:", ckpt.get("best_thresholds"))
print("Checkpoint val_macro_f1:", ckpt.get("val_macro_f1"))

categories = ckpt.get("stylistic_categories", ["funny", "sarcastic", "offensive", "wholesome", "neutral", "political"])
multimodal_model = MultimodalMemeClassifier(
    clip_model_name=cfg.DEFAULT_HYPERPARAMS["clip_model_name"],
    stylistic_categories=categories,
    fusion_dim=ckpt.get("fusion_dim", 256),
    freeze_backbone=True,
).to(device)

state_dict = ckpt.get("trainable_state_dict", ckpt.get("model_state_dict"))
multimodal_model.load_state_dict(state_dict, strict=False)
multimodal_model.eval()

clip_processor = CLIPProcessor.from_pretrained(cfg.DEFAULT_HYPERPARAMS["clip_model_name"])
baseline_clip = CLIPModel.from_pretrained(cfg.DEFAULT_HYPERPARAMS["clip_model_name"]).to(device)
baseline_clip.eval()

# Also load app.py's exact 8 categories for baseline comparison
app_categories = [
    "a hateful meme",
    "a non-hateful meme", 
    "a funny meme",
    "an offensive meme",
    "a wholesome meme",
    "a sarcastic meme",
    "a political meme",
    "a neutral meme"
]

val_loader = get_memotion_dataloader(
    split="validation",
    max_samples=20,
    batch_size=8,
    clip_processor=clip_processor,
    seen_hashes=None,  # Do not filter out
    stylistic_categories=categories,
)

mm_probs_list = []
base_sig_probs_list = []
base_softmax_probs_list = []
targets_list = []
texts_list = []

with torch.no_grad():
    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["stylistic_targets"].cpu().numpy()
        texts = batch["texts"]

        # Multimodal forward
        outputs = multimodal_model(pixel_values, input_ids, attention_mask)
        mm_probs = outputs["stylistic_probs"].cpu().numpy()
        mm_logits = outputs["stylistic_logits"].cpu().numpy()

        # Baseline: with exact 8 prompts and softmax as in app.py
        app_text_inputs = clip_processor(text=app_categories, return_tensors="pt", padding=True).to(device)
        base_out = baseline_clip(
            pixel_values=pixel_values,
            input_ids=app_text_inputs["input_ids"],
            attention_mask=app_text_inputs["attention_mask"]
        )
        base_logits = base_out.logits_per_image # (B, 8)
        base_softmax = base_logits.softmax(dim=1).cpu().numpy()

        # Also 6 stylistic categories direct prompt
        base_prompts_6 = [f"a {c} meme" if c != "offensive" else "an offensive meme" for c in categories]
        text_6_inputs = clip_processor(text=base_prompts_6, return_tensors="pt", padding=True).to(device)
        base_out_6 = baseline_clip(
            pixel_values=pixel_values,
            input_ids=text_6_inputs["input_ids"],
            attention_mask=text_6_inputs["attention_mask"]
        )
        base_sig = torch.sigmoid(base_out_6.logits_per_image * 0.1).cpu().numpy()

        mm_probs_list.append(mm_probs)
        base_sig_probs_list.append(base_sig)
        base_softmax_probs_list.append(base_softmax)
        targets_list.append(targets)
        texts_list.extend(texts)

mm_probs_all = np.vstack(mm_probs_list)
base_sig_all = np.vstack(base_sig_probs_list)
base_softmax_all = np.vstack(base_softmax_probs_list)
targets_all = np.vstack(targets_list)

print(f"\nCollected {len(targets_all)} validation samples.")

# Supported categories: funny, sarcastic, offensive
supported_indices = [0, 1, 2] # funny, sarcastic, offensive

for idx in supported_indices:
    cat = categories[idx]
    y_true = targets_all[:, idx]
    probs = mm_probs_all[:, idx]
    base_probs = base_sig_all[:, idx]

    pos_count = int(np.sum(y_true == 1))
    neg_count = int(np.sum(y_true == 0))

    print(f"\n{'='*65}")
    print(f"CATEGORY: {cat.upper()} (Ground Truth Pos: {pos_count}, Neg: {neg_count})")
    print(f"{'='*65}")

    print(f"Multimodal Model Probabilities:")
    print(f"  Min   : {probs.min():.4f}")
    print(f"  Max   : {probs.max():.4f}")
    print(f"  Mean  : {probs.mean():.4f}")
    print(f"  Std   : {probs.std():.4f}")
    print(f"  Sample values: {np.round(probs[:8], 3)}")

    print(f"\nBaseline (CLIP 0.1*Logits Sigmoid) Probabilities:")
    print(f"  Min   : {base_probs.min():.4f}")
    print(f"  Max   : {base_probs.max():.4f}")
    print(f"  Mean  : {base_probs.mean():.4f}")
    print(f"  Std   : {base_probs.std():.4f}")

    print(f"\n--- Multimodal Model Threshold Sweep ---")
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for th in thresholds:
        y_pred = (probs >= th).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        pos_pred = tp + fp
        neg_pred = tn + fn
        print(f"  th={th:.1f} | PredPos={pos_pred:2d} PredNeg={neg_pred:2d} | TP={tp:2d} FP={fp:2d} TN={tn:2d} FN={fn:2d} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} Spec={spec:.3f}")
