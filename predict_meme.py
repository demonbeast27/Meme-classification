"""
predict_meme.py
===============
Phase 5 Standalone Inference Script for the Trained Multimodal Meme Classifier.

Usage:
    python predict_meme.py --image <path/to/meme.jpg> [--text "meme text caption"] [--checkpoint checkpoints/best_multimodal_classifier.pt]

Features:
- Loads the trained MultimodalMemeClassifier checkpoint with validation-tuned thresholds.
- Reuses the cached CLIP vision & text backbones (openai/clip-vit-base-patch32).
- Extracts text via OCR if pytesseract is available and no caption is provided.
- Outputs probabilities and binary decisions for supported classes: Funny, Sarcastic, Offensive.
- Reports experimental hate proxy separately with appropriate warnings.
- Explicitly flags unsupported categories (Wholesome, Neutral, Political) as unmodeled.
- Robust error handling for missing files, invalid image formats, and missing checkpoints.
"""

import argparse
import io
import json
import os
import sys
import torch
import numpy as np
from PIL import Image

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import memotion_config as cfg
from multimodal_model import MultimodalMemeClassifier
from transformers import CLIPProcessor


def extract_ocr_text(pil_img: Image.Image) -> str:
    """Attempt OCR extraction using pytesseract/cv2 if available; fail silently to empty string."""
    try:
        import cv2
        import pytesseract

        img_np = np.array(pil_img.convert("RGB"))
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        text = pytesseract.image_to_string(enhanced)
        return text.strip()
    except Exception:
        return ""


def run_inference(
    image_path: str,
    text: str | None = None,
    checkpoint_path: str = "checkpoints/best_multimodal_classifier.pt",
    device_str: str | None = None,
) -> dict:
    # 1. Validate image path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: '{image_path}'")
    
    if not os.path.isfile(image_path):
        raise ValueError(f"Path is not a regular file: '{image_path}'")

    # 2. Try loading and verifying image format
    try:
        pil_img = Image.open(image_path)
        pil_img.verify()  # Verify image integrity
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image file '{image_path}' as a valid image: {e}")

    # 3. Validate checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: '{checkpoint_path}'. "
            f"Please ensure Phase 4 training completed or specify a valid checkpoint."
        )

    # 4. Device setup
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # 5. Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    thresholds = ckpt.get("best_thresholds", {
        "funny": 0.80,
        "sarcastic": 0.80,
        "offensive": 0.66,
    })
    fusion_dim = ckpt.get("fusion_dim", 256)
    stylistic_categories = ckpt.get("stylistic_categories", ["funny", "sarcastic", "offensive", "wholesome", "neutral", "political"])

    # 6. Initialize model & load weights
    model = MultimodalMemeClassifier(
        clip_model_name=cfg.DEFAULT_HYPERPARAMS["clip_model_name"],
        stylistic_categories=stylistic_categories,
        fusion_dim=fusion_dim,
        freeze_backbone=True,
    ).to(device)

    state_dict = ckpt.get("trainable_state_dict", ckpt.get("model_state_dict"))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 7. Preprocess inputs
    processor = CLIPProcessor.from_pretrained(cfg.DEFAULT_HYPERPARAMS["clip_model_name"])

    ocr_extracted = False
    if text is None or not text.strip():
        extracted = extract_ocr_text(pil_img)
        if extracted:
            text = extracted
            ocr_extracted = True
        else:
            text = "meme"

    inputs = processor(
        text=[text],
        images=pil_img,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True,
    )

    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 8. Inference forward pass
    with torch.no_grad():
        outputs = model(pixel_values, input_ids, attention_mask)
        stylistic_probs = outputs["stylistic_probs"][0].cpu().numpy()
        hate_prob = float(outputs["hate_probs"][0].cpu().item())

    # Clean up PIL image
    pil_img.close()

    # 9. Format predictions
    supported_results = {}
    for cat in cfg.SUPPORTED_STYLISTIC_CATEGORIES:
        idx = stylistic_categories.index(cat)
        prob = float(stylistic_probs[idx])
        thresh = float(thresholds.get(cat, 0.5))
        decision = bool(prob >= thresh)
        supported_results[cat] = {
            "probability": round(prob, 4),
            "threshold": round(thresh, 3),
            "decision": decision,
            "prediction": "POSITIVE" if decision else "NEGATIVE",
        }

    unsupported_categories = [cat for cat in stylistic_categories if cat not in cfg.SUPPORTED_STYLISTIC_CATEGORIES]

    return {
        "image_path": os.path.abspath(image_path),
        "caption_used": text,
        "ocr_extracted": ocr_extracted,
        "device": str(device),
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "supported_categories": supported_results,
        "experimental_hate_proxy": {
            "probability": round(hate_prob, 4),
            "warning": (
                "EXPERIMENTAL PROXY ONLY: The hate head was trained on offensive meme labels as a weak proxy "
                "because Memotion 7k lacks dedicated hate-speech annotations. Do NOT use for production safety enforcement."
            ),
        },
        "unsupported_categories": {
            cat: {
                "status": "UNSUPPORTED_MASKED",
                "disclaimer": "This category was not annotated in Memotion 7k and is excluded from reliable classification.",
            }
            for cat in unsupported_categories
        },
    }


def print_prediction_report(results: dict):
    print("\n" + "=" * 70)
    print("  MULTIMODAL MEME CLASSIFIER — INFERENCE RESULT")
    print("=" * 70)
    print(f"  Image file   : {results['image_path']}")
    print(f"  Text / OCR   : \"{results['caption_used']}\" " + ("(auto-extracted via OCR)" if results["ocr_extracted"] else "(user provided or default)"))
    print(f"  Checkpoint   : {results['checkpoint_path']}")
    print(f"  Device       : {results['device']}")
    print("-" * 70)
    print(f"  {'CATEGORY':<12} | {'PROBABILITY':>12} | {'THRESHOLD':>10} | {'PREDICTION':>12}")
    print("-" * 70)

    for cat, info in results["supported_categories"].items():
        label = info["prediction"]
        print(f"  {cat.capitalize():<12} | {info['probability']:>12.4f} | {info['threshold']:>10.3f} | {label:>12}")

    print("-" * 70)
    print("  EXPERIMENTAL HATE SPEECH PROXY:")
    hate_info = results["experimental_hate_proxy"]
    print(f"  Hate Head Prob : {hate_info['probability']:.4f}")
    print(f"  [WARNING] {hate_info['warning']}")
    print("-" * 70)
    print("  UNSUPPORTED CATEGORIES (NOT RELIABLY CLASSIFIED):")
    for cat in results["unsupported_categories"]:
        print(f"  - {cat.capitalize()}: Masked in training (no annotations in Memotion 7k).")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict meme stylistic categories (Funny, Sarcastic, Offensive) using trained MultimodalMemeClassifier."
    )
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input meme image (jpg/png/webp)")
    parser.add_argument("--text", "-t", type=str, default=None, help="Meme caption/text (optional; OCR used if omitted)")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="checkpoints/best_multimodal_classifier.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--json", action="store_true", help="Print results as JSON instead of formatted table")

    args = parser.parse_args()

    try:
        results = run_inference(
            image_path=args.image,
            text=args.text,
            checkpoint_path=args.checkpoint,
            device_str=args.device,
        )
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_prediction_report(results)
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] Invalid input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during inference: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
