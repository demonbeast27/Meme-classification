"""
test_paired_baseline.py
Tests calibrated paired-prompt zero-shot CLIP on 20 validation samples.
"""
import io
import sys
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import memotion_config as cfg
from data_loader import get_memotion_dataloader

device = torch.device("cpu")
clip_processor = CLIPProcessor.from_pretrained(cfg.DEFAULT_HYPERPARAMS["clip_model_name"])
baseline_clip = CLIPModel.from_pretrained(cfg.DEFAULT_HYPERPARAMS["clip_model_name"]).to(device)
baseline_clip.eval()

categories = ["funny", "sarcastic", "offensive"]
paired_prompts = {
    "funny": ("a funny meme", "a not funny meme"),
    "sarcastic": ("a sarcastic meme", "a non-sarcastic meme"),
    "offensive": ("an offensive meme", "a not offensive meme"),
}

val_loader = get_memotion_dataloader(
    split="validation",
    max_samples=20,
    batch_size=4,
    clip_processor=clip_processor,
    seen_hashes=None,
    stylistic_categories=categories,
)

all_probs = {c: [] for c in categories}
all_targets = {c: [] for c in categories}

with torch.no_grad():
    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device)
        targets = batch["stylistic_targets"].cpu().numpy()

        for j, cat in enumerate(categories):
            pos_p, neg_p = paired_prompts[cat]
            pair_inputs = clip_processor(text=[pos_p, neg_p], return_tensors="pt", padding=True).to(device)
            out = baseline_clip(
                pixel_values=pixel_values,
                input_ids=pair_inputs["input_ids"],
                attention_mask=pair_inputs["attention_mask"]
            )
            # Softmax over positive vs negative prompt
            pair_probs = out.logits_per_image.softmax(dim=1)[:, 0].cpu().numpy()
            all_probs[cat].extend(pair_probs)
            all_targets[cat].extend(targets[:, j])

print("=" * 65)
print("PAIRED-PROMPT ZERO-SHOT CLIP BASELINE RESULTS (20 Val Samples)")
print("=" * 65)
for cat in categories:
    y_true = np.array(all_targets[cat])
    probs = np.array(all_probs[cat])
    valid = (y_true != -1)
    y_v = y_true[valid]
    p_v = probs[valid]

    print(f"\n{cat.upper()}:")
    print(f"  Min={p_v.min():.4f}, Max={p_v.max():.4f}, Mean={p_v.mean():.4f}, Std={p_v.std():.4f}")

    # Threshold at 0.5
    y_pred = (p_v >= 0.5).astype(int)
    tp = np.sum((y_v == 1) & (y_pred == 1))
    fp = np.sum((y_v == 0) & (y_pred == 1))
    tn = np.sum((y_v == 0) & (y_pred == 0))
    fn = np.sum((y_v == 1) & (y_pred == 0))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"  At threshold 0.5:")
    print(f"    TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"    Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, Specificity={spec:.3f}")
