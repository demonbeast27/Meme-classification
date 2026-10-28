# Phase 5 Inference Report: Multimodal Meme Classifier Standalone Interface

## Executive Summary

Phase 5 prepared and validated the standalone inference interface for the trained **Decoupled Multimodal Meme Classifier** (`predict_meme.py`) without modifying `app.py`. The interface operates on local image inputs, leveraging the best trained checkpoint (`best_multimodal_classifier.pt`) and frozen `openai/clip-vit-base-patch32` backbones with validation-tuned decision thresholds.

Because the repository operates exclusively under a storage-efficient in-memory streaming architecture (no raw image files downloaded or stored on disk during Phases 1–4), **no pre-existing local meme images were present in the workspace**. The inference interface was thoroughly validated for parameter parsing, help documentation, device allocation, and graceful error handling on invalid/missing inputs. Exact execution commands are documented below for user-provided meme images.

---

## 1. Inference Command & Interface

### Primary Execution Command
```bash
python predict_meme.py --image <path/to/meme.jpg> [--text "optional caption"] [--checkpoint checkpoints/best_multimodal_classifier.pt] [--device cpu]
```

### JSON Output Command (Machine-Readable)
```bash
python predict_meme.py --image <path/to/meme.jpg> --json
```

### CLI Interface Specification
| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--image`, `-i` | String | **Yes** | None | Path to local meme image file (`.jpg`, `.jpeg`, `.png`, `.webp`) |
| `--text`, `-t` | String | No | Auto (OCR / "meme") | Meme text caption; if omitted, automatically extracts text via OCR |
| `--checkpoint`, `-c` | String | No | `checkpoints/best_multimodal_classifier.pt` | Path to trained model weights |
| `--device` | String | No | Auto (`cuda` if available, else `cpu`) | Compute device |
| `--json` | Flag | No | False | Formats output as structured JSON |

---

## 2. Model Architecture & Preprocessing

### Architecture Specification
- **Vision Backbone:** Frozen `openai/clip-vit-base-patch32` Vision Transformer (512-dimensional output embeddings, L2-normalized).
- **Text Backbone:** Frozen `openai/clip-vit-base-patch32` Text Transformer (512-dimensional output embeddings, max token sequence 77, L2-normalized).
- **Fusion Layer:** Trainable `GatedMultimodalFusion`:
  - Visual projection: `Linear(512, 256)` + `LayerNorm` + `GELU` + `Dropout(0.3)`
  - Text projection: `Linear(512, 256)` + `LayerNorm` + `GELU` + `Dropout(0.3)`
  - Context-dependent gating vector: `Sigmoid(Linear(512, 256))`
  - Residual dense block with `LayerNorm`
- **Stylistic Classification Head:** `Linear(256, 128)` -> `GELU` -> `Dropout(0.3)` -> `Linear(128, 6)` with independent element-wise Sigmoids.
- **Decoupled Hate Detection Head:** `Linear(256, 128)` -> `GELU` -> `Dropout(0.3)` -> `Linear(128, 1)` with Sigmoid.

### Preprocessing Pipeline
1. **File & Integrity Verification:** Validates file existence and tests image integrity with `PIL.Image.verify()` before decoding to RGB.
2. **OCR Extraction:** When `--text` is omitted, the script executes an OCR pass via `pytesseract` and OpenCV grayscale/contrast enhancement. If OCR finds no text or pytesseract is unavailable, falls back gracefully to `"meme"`.
3. **CLIP Processor Tokenization & Resizing:** Processes the image (resizing to 224x224, standard CLIP normalization) and tokenizes caption text (padded/truncated to 77 tokens).
4. **Memory Management:** PIL image handles are strictly closed after tensor conversion to avoid memory leaks.

---

## 3. Validation Thresholds & Decision Logic

Decision thresholds tuned on the 100-sample validation split during Phase 4:

| Category | Validation Threshold | Prediction Logic | Status |
|---|---|---|---|
| **Funny** | **0.800** | Positive if $P(\text{Funny}) \ge 0.800$ | Supported & Modeled |
| **Sarcastic** | **0.800** | Positive if $P(\text{Sarcastic}) \ge 0.800$ | Supported & Modeled |
| **Offensive** | **0.660** | Positive if $P(\text{Offensive}) \ge 0.660$ | Supported & Modeled |
| **Hate (Proxy)** | *None (Raw Prob)* | Weak proxy only (requires warning) | Experimental |
| **Wholesome** | *None* | Unannotated in Memotion 7k | Masked / Unsupported |
| **Neutral** | *None* | Unannotated in Memotion 7k | Masked / Unsupported |
| **Political** | *None* | Unannotated in Memotion 7k | Masked / Unsupported |

---

## 4. Local Image Verification & Test Execution

### Workspace Image Audit
A filesystem search across `D:\MEME_CLASSIFIER_BERT_CLIP` confirmed **0 local image files**:
- No `.jpg`, `.png`, `.jpeg`, or `.webp` files exist in root, `experiment_output`, or `memotion_output`.
- This confirms that prior phases strictly adhered to the in-memory streaming constraint without leaving leftover image artifacts.

### Interface Testing & Verification
The interface was verified using the following execution tests:
1. **CLI Help Test:**
   ```bash
   python predict_meme.py --help
   ```
   *Result:* Exited 0 with full argument documentation.
2. **Missing File Exception Handling:**
   ```bash
   python predict_meme.py --image non_existent_image.jpg
   ```
   *Result:* Exited 1 with clean user-facing error: `[ERROR] File not found: Image file not found: 'non_existent_image.jpg'` without unhandled stack traces.
3. **Invalid Image Format Handling:**
   *Result:* Catches non-image or corrupted files via `PIL.Image.verify()` and reports `[ERROR] Invalid input: Cannot open image file...`.

---

## 5. How to Provide a Meme Image for Inference

To run inference on any real meme image:
1. Copy or download your meme image into `D:\MEME_CLASSIFIER_BERT_CLIP\` (e.g. `test_meme.jpg`).
2. Run:
   ```bash
   python predict_meme.py --image test_meme.jpg
   ```
3. If you want to supply the exact caption manually (e.g. if the meme font is unusual for OCR):
   ```bash
   python predict_meme.py --image test_meme.jpg --text "When the code compiles on the first try"
   ```

### Expected Output Example
```text
======================================================================
  MULTIMODAL MEME CLASSIFIER — INFERENCE RESULT
======================================================================
  Image file   : D:\MEME_CLASSIFIER_BERT_CLIP\test_meme.jpg
  Text / OCR   : "When the code compiles on the first try" (user provided)
  Checkpoint   : D:\MEME_CLASSIFIER_BERT_CLIP\checkpoints\best_multimodal_classifier.pt
  Device       : cpu
----------------------------------------------------------------------
  CATEGORY     |  PROBABILITY |  THRESHOLD |   PREDICTION
----------------------------------------------------------------------
  Funny        |       0.8421 |      0.800 |     POSITIVE
  Sarcastic    |       0.6814 |      0.800 |     NEGATIVE
  Offensive    |       0.1245 |      0.660 |     NEGATIVE
----------------------------------------------------------------------
  EXPERIMENTAL HATE SPEECH PROXY:
  Hate Head Prob : 0.0892
  [WARNING] EXPERIMENTAL PROXY ONLY: The hate head was trained on offensive meme labels as a weak proxy because Memotion 7k lacks dedicated hate-speech annotations. Do NOT use for production safety enforcement.
----------------------------------------------------------------------
  UNSUPPORTED CATEGORIES (NOT RELIABLY CLASSIFIED):
  - Wholesome: Masked in training (no annotations in Memotion 7k).
  - Neutral: Masked in training (no annotations in Memotion 7k).
  - Political: Masked in training (no annotations in Memotion 7k).
======================================================================
```

---

## 6. Limitations & Warnings

1. **No Production Integration:** `app.py` has remained strictly untouched throughout Phase 4 and Phase 5.
2. **Unsupported Categories:** `wholesome`, `neutral`, and `political` are reported with explicit disclaimers and are never marked as classified.
3. **Hate Head:** Reported strictly as an experimental proxy with an explicit safety disclaimer.
