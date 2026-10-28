# GPU Training & Full-Dataset Evaluation Report: Memotion 7k on NVIDIA RTX 4050

**Date:** 2026-10-25  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM), Driver 592.82  
**Software Stack:** Python 3.13.0, PyTorch 2.6.0+cu124, CUDA 12.4, Torchvision 0.21.0+cu124  
**Primary Checkpoint:** [`checkpoints/best_multimodal_classifier_gpu.pt`](file:///d:/MEME_CLASSIFIER_BERT_CLIP/checkpoints/best_multimodal_classifier_gpu.pt)  
**Detailed Training Summary:** [`memotion_output/best_multimodal_classifier_gpu_summary.json`](file:///d:/MEME_CLASSIFIER_BERT_CLIP/memotion_output/best_multimodal_classifier_gpu_summary.json)  
**Comparative Evaluation JSON:** [`memotion_output/comparative_evaluation_best_multimodal_classifier_gpu.json`](file:///d:/MEME_CLASSIFIER_BERT_CLIP/memotion_output/comparative_evaluation_best_multimodal_classifier_gpu.json)  

---

## 1. Executive Summary

This report documents the full-dataset GPU training run and rigorous side-by-side comparative evaluation of the **Multimodal Meme Classifier** on the **NVIDIA GeForce RTX 4050 Laptop GPU**.

- **Full Split Training:** Rather than sample-capped subsets (e.g. 100 or 500 samples), training utilized the entire usable Memotion 7k training split: **5,568 streamed samples per epoch** (348 batches of size 16) across 5 full epochs.
- **Hardware Acceleration:** Native CUDA 12.4 with Automatic Mixed Precision (`FP16` AMP) delivered high-throughput training (~35–50 samples/s) with a peak VRAM footprint of just **657.8 MiB** (10.7% of total GPU memory).
- **Validation Results:** Evaluated across the complete **698 valid validation records**:
  - **Macro F1 (3 supported classes):** **0.749** vs **0.711** Zero-Shot CLIP Baseline (**+0.038 delta**).
  - **Micro F1:** **0.755** vs **0.729** Baseline (**+0.027 delta**).
  - **Offensive Category Breakthrough:** F1 increased from **0.542** to **0.662** (+12.0% absolute), with recall jumping from **0.503** to **0.698** (+19.5% absolute), resolving the primary blind spot of zero-shot CLIP.
- **Safety & Boundary Guarantees:** 
  - `app.py` was left completely untouched.
  - Phase 4 CPU checkpoints (`best_multimodal_classifier.pt`) and documentation (`PHASE4_REPORT.md`, `PHASE5_INFERENCE_REPORT.md`) were fully preserved.
  - Test split was kept strictly untouched during training and threshold tuning.

---

## 2. Hardware, Environment & Verification

### 2.1 Environment Setup
| Parameter | Value |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU |
| **Total Dedicated VRAM** | 6,141 MiB (6.0 GB) |
| **CUDA Driver / Runtime** | Driver 592.82 / CUDA 12.4 |
| **Python Interpreter** | `C:\Users\tilak\AppData\Local\Programs\Python\Python313\python.exe` |
| **PyTorch Build** | `2.6.0+cu124` |
| **Torchvision Build** | `0.21.0+cu124` |
| **Precision** | Mixed Precision AMP (`torch.amp.autocast(device_type="cuda", dtype=torch.float16)`) |

### 2.2 Verification Checks
1. **Device Detection:** PyTorch successfully detected device `cuda:0` as `NVIDIA GeForce RTX 4050 Laptop GPU`.
2. **Matrix Multiplication Test:** Executed 2048x2048 tensor multiplication on CUDA with `torch.cuda.synchronize()`, allocating 40.1 MiB VRAM without error.
3. **Dependency Cleanup:** Uninstalled conflicting CPU-only `torchaudio 2.8.0` build which caused dynamic link library load collisions (`WinError 127`).

---

## 3. Training Architecture & Pipeline

### 3.1 Model Configuration
- **Visual & Text Encoders:** Frozen OpenAI CLIP `openai/clip-vit-base-patch32` (512-dim visual features, 512-dim textual features).
- **Fusion Architecture:** `GatedMultimodalFusion` projecting image and text representations into a 256-dim shared latent space with cross-modal gating:
  $$h = \sigma(W_g [v, t] + b_g) \odot \tanh(W_v v + b_v) + (1 - \sigma(W_g [v, t] + b_g)) \odot \tanh(W_t t + b_t)$$
- **Classification Heads:** Multi-head independent linear projections with calibrated sigmoid outputs.
- **Label Masking:** Masked BCE Loss (`targets != -1`) ensuring unannotated Memotion attributes (`wholesome`, `neutral`, `political`) contribute zero loss gradient and do not distort parameter updates.

### 3.2 Hyperparameters & Hardware Performance
| Parameter | Value |
| :--- | :--- |
| **Batch Size** | 16 |
| **Learning Rate** | $1 \times 10^{-4}$ (AdamW, weight decay $1 \times 10^{-2}$) |
| **Scheduler** | Cosine Annealing |
| **Epochs** | 5 |
| **Samples per Epoch** | 5,568 (348 batches $\times$ 16) |
| **Validation Size** | 699 samples (698 complete records evaluated) |
| **Peak VRAM Allocated** | **657.8 MiB** |
| **Total Training Time** | **1,207.6 seconds (~20.1 minutes)** |

### 3.3 Epoch Timing Breakdown
- **Epoch 1:** 411.7s (included initial Parquet chunk streaming and HuggingFace disk caching)
- **Epoch 2:** 165.4s
- **Epoch 3:** 130.0s
- **Epoch 4:** 193.4s
- **Epoch 5:** 111.7s

The best validation checkpoint was attained at **Epoch 1** with **Validation Macro F1 = 0.749**.

---

## 4. Side-by-Side Validation Performance (698 Samples)

Both the Zero-Shot CLIP baseline (from `app.py`) and the newly trained Multimodal Classifier (`best_multimodal_classifier_gpu.pt`) were evaluated on the **exact same 698 validation examples**. For fairness, thresholds for both models were tuned on the validation split.

### 4.1 Overall Metrics
| Metric | Zero-Shot CLIP Baseline | GPU Multimodal Classifier | Delta |
| :--- | :---: | :---: | :---: |
| **Macro F1 (3 supported classes)** | **0.711** | **0.749** | **+0.038 (+3.8%)** |
| **Micro F1 (3 supported classes)** | **0.729** | **0.755** | **+0.027 (+2.7%)** |

### 4.2 Per-Class Detailed Breakdown

#### 1. Offensive (Most Significant Improvement)
*Support: 437 Positive, 261 Negative*
- **Baseline (Threshold: 0.50):** Precision: 0.587 | Recall: 0.503 | **F1: 0.542** | Specificity: 0.406
  - *Matrix:* TP: 220, FP: 155, TN: 106, FN: 217
- **GPU Multimodal (Threshold: 0.62):** Precision: 0.629 | Recall: 0.698 | **F1: 0.662** | Specificity: 0.310
  - *Matrix:* TP: 305, FP: 180, TN: 81, FN: 132
- **Key Takeaway:** Zero-shot CLIP missed 217 offensive memes (FN = 217, 50.3% recall). The trained multimodal fusion reduced false negatives to 132 (+85 additional true positives captured), boosting Recall by **+19.5%** and F1 by **+12.0%**.

#### 2. Sarcastic
*Support: 537 Positive, 161 Negative*
- **Baseline (Threshold: 0.48):** Precision: 0.763 | Recall: 0.765 | **F1: 0.764** | Specificity: 0.205
  - *Matrix:* TP: 411, FP: 128, TN: 33, FN: 126
- **GPU Multimodal (Threshold: 0.80):** Precision: 0.767 | Recall: 0.752 | **F1: 0.759** | Specificity: 0.236
  - *Matrix:* TP: 404, FP: 123, TN: 38, FN: 133
- **Key Takeaway:** Performance remains virtually identical to baseline (~0.76 F1), with slightly higher precision (+0.4%) and specificity (+3.1%).

#### 3. Funny
*Support: 539 Positive, 159 Negative*
- **Baseline (Threshold: 0.48):** Precision: 0.785 | Recall: 0.876 | **F1: 0.828** | Specificity: 0.189
  - *Matrix:* TP: 472, FP: 129, TN: 30, FN: 67
- **GPU Multimodal (Threshold: 0.70):** Precision: 0.781 | Recall: 0.879 | **F1: 0.827** | Specificity: 0.164
  - *Matrix:* TP: 474, FP: 133, TN: 26, FN: 65
- **Key Takeaway:** Sustains high detection rate (0.879 recall, 0.827 F1) matching zero-shot CLIP while operating inside the unified gated fusion architecture.

#### 4. Wholesome, Neutral, Political
- **Status:** Correctly masked (`-1`) during training and comparative evaluation as they are not annotated in the Memotion 7k benchmark.

---

## 5. Artifact Summary & Separation

All files adhere strictly to the non-overwriting rule:

| Artifact | Path | Description |
| :--- | :--- | :--- |
| **GPU Model Checkpoint** | [`checkpoints/best_multimodal_classifier_gpu.pt`](file:///d:/MEME_CLASSIFIER_BERT_CLIP/checkpoints/best_multimodal_classifier_gpu.pt) | Saved PyTorch state dict with tuned thresholds and metadata |
| **GPU Training Summary** | [`memotion_output/best_multimodal_classifier_gpu_summary.json`](file:///d:/MEME_CLASSIFIER_BERT_CLIP/memotion_output/best_multimodal_classifier_gpu_summary.json) | VRAM, timings, per-epoch logs, and best metrics |
| **Comparative Evaluation** | [`memotion_output/comparative_evaluation_best_multimodal_classifier_gpu.json`](file:///d:/MEME_CLASSIFIER_BERT_CLIP/memotion_output/comparative_evaluation_best_multimodal_classifier_gpu.json) | Full 698-sample comparative benchmark against baseline |
| **Phase 4 CPU Checkpoint** | [`checkpoints/best_multimodal_classifier.pt`](file:///d:/MEME_CLASSIFIER_BERT_CLIP/checkpoints/best_multimodal_classifier.pt) | Preserved original CPU checkpoint |
| **Original Baseline App** | [`app.py`](file:///d:/MEME_CLASSIFIER_BERT_CLIP/app.py) | Preserved and unmodified |

---

## 6. How to Run Inference or App with GPU Checkpoint

To launch the Gradio Multimodal Web Application with the new GPU model:
```powershell
python app_multimodal.py --checkpoint best_multimodal_classifier_gpu.pt --device cuda
```

To run individual predictions:
```powershell
python predict_meme.py --image path/to/meme.jpg --caption "Meme text here" --checkpoint best_multimodal_classifier_gpu.pt --device cuda
```
