"""
verify_gradients_and_weights.py
Audits gradient flow, weight updates, and modality contribution in MultimodalMemeClassifier.
"""
import io
import sys
import torch
import torch.optim as optim

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import memotion_config as cfg
from multimodal_model import MultimodalMemeClassifier, MaskedBCEWithLogitsLoss
from data_loader import get_memotion_dataloader

device = torch.device("cpu")
print("=" * 65)
print("AUDITING MODEL WEIGHTS, GRADIENTS, AND MODALITY FLOW")
print("=" * 65)

# 1. Instantiate model
model = MultimodalMemeClassifier(
    clip_model_name=cfg.DEFAULT_HYPERPARAMS["clip_model_name"],
    stylistic_categories=["funny", "sarcastic", "offensive", "wholesome", "neutral", "political"],
    fusion_dim=256,
    freeze_backbone=True,
).to(device)

# Record initial weights of trainable layers
initial_params = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        initial_params[name] = param.detach().clone()
print(f"Total trainable parameter tensors: {len(initial_params)}")

# Check optimizer
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4) # test with lr 1e-3
criterion = MaskedBCEWithLogitsLoss()

# Get a mini-batch of 4 samples
loader = get_memotion_dataloader(
    split="train",
    max_samples=4,
    batch_size=4,
    stylistic_categories=model.stylistic_categories,
)
batch = next(iter(loader))

pixel_values = batch["pixel_values"].to(device)
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
targets = batch["stylistic_targets"].to(device)
hate_targets = batch["hate_targets"].to(device)

print(f"Batch targets shape: {targets.shape}")
print(f"Batch targets sample (values should be 0, 1, or -1):\n{targets.numpy()}")

# 2. Forward pass
outputs = model(pixel_values, input_ids, attention_mask)
loss_stylistic = criterion(outputs["stylistic_logits"], targets)
loss_hate = criterion(outputs["hate_logits"], hate_targets)
loss = loss_stylistic + 0.3 * loss_hate

print(f"Loss value: {loss.item():.4f}")

# 3. Backward pass
loss.backward()

# Check gradients
gradient_report = {}
has_zero_grad = []
has_nonzero_grad = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is None:
            has_zero_grad.append((name, "None"))
        else:
            grad_norm = param.grad.norm().item()
            if grad_norm == 0.0:
                has_zero_grad.append((name, "0.0"))
            else:
                has_nonzero_grad.append((name, grad_norm))

print(f"\nGradient Check:")
print(f"  Tensors with non-zero gradients: {len(has_nonzero_grad)} / {len(initial_params)}")
for name, gnorm in has_nonzero_grad[:6]:
    print(f"    {name:<45}: grad_norm = {gnorm:.6f}")
if has_zero_grad:
    print(f"  Tensors with ZERO / None gradients: {has_zero_grad}")

# 4. Optimizer Step & Weight Change Check
optimizer.step()

weight_changes = {}
max_delta = 0.0
for name, old_param in initial_params.items():
    new_param = dict(model.named_parameters())[name]
    diff = (new_param - old_param).abs()
    mean_diff = diff.mean().item()
    max_d = diff.max().item()
    weight_changes[name] = (mean_diff, max_d)
    if max_d > max_delta:
        max_delta = max_d

print(f"\nWeight Update Check after 1 optimizer step:")
print(f"  Max parameter delta: {max_delta:.8f}")
for name in list(weight_changes.keys())[:6]:
    mean_d, max_d = weight_changes[name]
    print(f"    {name:<45}: mean_delta={mean_d:.8f}, max_delta={max_d:.8f}")

# 5. Modality Ablation Test: Verify both Image and Text features affect outputs
with torch.no_grad():
    base_out = model(pixel_values, input_ids, attention_mask)["stylistic_probs"]

    # Zero out image:
    zero_img = torch.zeros_like(pixel_values)
    img_ablated_out = model(zero_img, input_ids, attention_mask)["stylistic_probs"]
    img_diff = (base_out - img_ablated_out).abs().mean().item()

    # Zero out text:
    zero_input_ids = torch.zeros_like(input_ids)
    zero_mask = torch.zeros_like(attention_mask)
    text_ablated_out = model(pixel_values, zero_input_ids, zero_mask)["stylistic_probs"]
    text_diff = (base_out - text_ablated_out).abs().mean().item()

print(f"\nModality Impact Check:")
print(f"  Mean prob change when Image is ablated : {img_diff:.6f} ({'Active' if img_diff > 1e-4 else 'INACTIVE'})")
print(f"  Mean prob change when Text is ablated  : {text_diff:.6f} ({'Active' if text_diff > 1e-4 else 'INACTIVE'})")
