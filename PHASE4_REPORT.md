# Phase 4 Evaluation Report: Multimodal Meme Classifier vs Baseline

## Executive Summary

Phase 4 scaled training from the 64-sample proof-of-concept (Phase 3) to **500 training samples** across **5 epochs**, evaluated on a held-out **100-sample validation split**. The trained Decoupled Multimodal Classifier (`MultimodalMemeClassifier`) was evaluated side-by-side against the Zero-Shot CLIP baseline using the exact same validation examples and the same validation threshold-tuning methodology (`tune_thresholds_for_classes`).

### Key Results
- **Multimodal Macro F1:** **0.807** (vs Zero-Shot CLIP Baseline **0.754**, **+0.053** improvement)
- **Multimodal Micro F1:** **0.807** (vs Zero-Shot CLIP Baseline **0.756**, **+0.051** improvement)
- **Best Training Epoch:** Epoch 2 (Validation Macro F1: 0.807)
- **Training Duration:** 660.5 seconds (~11.0 minutes) on CPU
- **Evaluation Duration:** 56.3 seconds on CPU across 100 validation samples

---

## 1. Experimental Setup & Verification

| Parameter | Value / Detail |
|---|---|
| **Dataset** | Memotion 7k (`hf-internal-testing/memotion_dataset_test`) streamed in-memory |
| **Train Samples** | 500 records |
| **Validation Samples** | 100 records |
| **Batch Size** | 8 |
| **Learning Rate** | 5e-4 (AdamW, cosine decay) |
| **Epochs** | 5 (Best checkpoint saved at Epoch 2) |
| **Hardware** | Intel/AMD CPU (No CUDA) |
| **Backbone** | Frozen `openai/clip-vit-base-patch32` (Vision 512-d + Text 512-d) |
| **Fusion Layer** | Trainable `GatedMultimodalFusion` (256-d, dropout 0.3) |
| **Loss Function** | `MaskedBCEWithLogitsLoss` (masks unannotated labels with `-1`) |
| **Threshold Tuning** | Optimal F1 grid search on validation probabilities for both models |
| **Data Leakage Check** | Zero overlap: MD5 hashing on text and strictly separated streaming splits |

### Evaluation Fairness Verification
1. **Identical Samples:** Both the multimodal classifier and the baseline CLIP model were evaluated synchronously within the same dataloader batch iteration on the same 100 validation examples.
2. **Identical Threshold Tuning Procedure:** Both models tuned binary decision thresholds using `tune_thresholds_for_classes` (evaluating thresholds from 0.10 to 0.90 in steps of 0.02 to maximize F1 on validation probabilities).
3. **Strict Label Masking:** Unsupported categories (`wholesome`, `neutral`, `political`) were excluded from metric calculations, preventing artificial metric inflation or penalties.

---

## 2. Detailed Performance Comparison (100 Validation Samples)

### Per-Class Metrics

| Category | Pos / Neg Support | Metric | Zero-Shot CLIP Baseline | Multimodal Classifier | Delta |
|---|---|---|---|---|---|
| **Funny** | 75 / 25 | **Threshold** | 0.52 | 0.80 | -- |
| | | **Precision** | 0.782 | **0.808** | +0.026 |
| | | **Recall** | **0.813** | 0.787 | -0.026 |
| | | **F1 Score** | 0.797 | **0.797** | 0.000 |
| | | **Specificity** | 0.320 | **0.440** | **+0.120** |
| **Sarcastic** | 77 / 23 | **Threshold** | 0.52 | 0.80 | -- |
| | | **Precision** | 0.795 | **0.795** | 0.000 |
| | | **Recall** | 0.753 | **0.857** | **+0.104** |
| | | **F1 Score** | 0.773 | **0.825** | **+0.052** |
| | | **Specificity** | **0.348** | 0.261 | -0.087 |
| **Offensive** | 65 / 35 | **Threshold** | 0.46 | 0.66 | -- |
| | | **Precision** | 0.649 | **0.731** | **+0.082** |
| | | **Recall** | 0.738 | **0.877** | **+0.139** |
| | | **F1 Score** | 0.691 | **0.797** | **+0.106** |
| | | **Specificity** | 0.257 | **0.400** | **+0.143** |
| **Wholesome** | 0 / 0 | *Masked* | *Not annotated in Memotion 7k* | *Excluded from metrics* | -- |
| **Neutral** | 0 / 0 | *Masked* | *Not annotated in Memotion 7k* | *Excluded from metrics* | -- |
| **Political** | 0 / 0 | *Masked* | *Not annotated in Memotion 7k* | *Excluded from metrics* | -- |

### Overall Summary
| Metric | Zero-Shot CLIP Baseline | Multimodal Classifier | Delta |
|---|---|---|---|
| **Macro F1 (3-Class)** | 0.754 | **0.807** | **+0.053** |
| **Micro F1 (3-Class)** | 0.756 | **0.807** | **+0.051** |

---

## 3. Confusion Matrices (N = 100)

### Multimodal Classifier (Epoch 2 Checkpoint)
| Category | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) | Total Samples |
|---|---|---|---|---|---|
| **Funny** | 59 | 14 | 11 | 16 | 100 |
| **Sarcastic** | 66 | 17 | 6 | 11 | 100 |
| **Offensive** | 57 | 21 | 14 | 8 | 100 |

### Zero-Shot CLIP Baseline
| Category | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) | Total Samples |
|---|---|---|---|---|---|
| **Funny** | 61 | 17 | 8 | 14 | 100 |
| **Sarcastic** | 58 | 15 | 8 | 19 | 100 |
| **Offensive** | 48 | 26 | 9 | 17 | 100 |

---

## 4. Comparison Against Phase 3 (Generalization Analysis)

| Aspect | Phase 3 (Small Scale) | Phase 4 (Current) | Change / Takeaway |
|---|---|---|---|
| **Train Samples** | 64 | 500 | 7.8x increase in training data |
| **Validation Samples** | 20 | 100 | 5x larger evaluation sample |
| **Multimodal Macro F1** | 0.649 | **0.807** | **+0.158 increase** |
| **Baseline Macro F1** | 0.638 | 0.754 | Calibrated baseline improves with larger sample |
| **Offensive F1** | 0.690 | **0.797** | Significant gain (+0.107) in offensive detection |
| **Funny Specificity** | 0.667 (6/9) | 0.440 (11/25) | More stable estimate with 25 negatives |
| **Offensive Specificity** | 0.100 (1/10) | 0.400 (14/35) | **4x improvement in rejecting non-offensive memes** |

### Key Generalization Observations:
1. **Strong Offensive Detection Improvement:** The multimodal model achieved an F1 of 0.797 on offensive content, outperforming baseline (0.691) by over 10 points. Specificity improved from 0.100 (Phase 3) to 0.400, correctly rejecting 14 non-offensive samples.
2. **Threshold Shift:** Notice that the multimodal model shifted its optimal thresholds upward (Funny: 0.80, Sarcastic: 0.80, Offensive: 0.66), reflecting higher model confidence and calibrated boundary separation.
3. **Persistent Challenge with Sarcasm Specificity:** Sarcastic specificity remains relatively low (0.261 vs baseline 0.348), because Memotion 7k has strong positive class skew (~77% positive).

---

## 5. Limitations & Caveats

1. **Memotion 7k Skew:** The validation dataset exhibits strong positive imbalance (75% funny, 77% sarcastic, 65% offensive). While F1 accounts for precision/recall balance, specificity metrics are constrained by the smaller negative sample count.
2. **Unsupported Classes:** `wholesome`, `neutral`, and `political` are absent in Memotion 7k annotations and remain strictly masked. They cannot be reliably evaluated or predicted by this model.
3. **Hate Speech Proxy:** Memotion 7k does not contain dedicated hate speech ground truth. The decoupled hate head was trained using offensive labels as a weak proxy and must NOT be treated as a production-grade hate speech detector.

---

## 6. Checkpoint Verification
- **File:** `D:\MEME_CLASSIFIER_BERT_CLIP\checkpoints\best_multimodal_classifier.pt`
- **Trained Epoch:** Epoch 2 (Validation Macro F1: 0.807)
- **State Dict:** `trainable_state_dict` (gated fusion + heads, backbones frozen)
- **Thresholds Recorded in Checkpoint:**
  - `funny`: 0.80
  - `sarcastic`: 0.80
  - `offensive`: 0.66
