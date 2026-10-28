# Phase 2: Decoupled Multimodal Meme Classifier

This directory contains the Phase 2 implementation for **`MEME_CLASSIFIER_BERT_CLIP`**, establishing a decoupled multimodal architecture and multi-label evaluation pipeline while keeping the original baseline application (`app.py`) intact.

---

## 1. Architecture Overview

```
Input Meme: Image + Extracted OCR Text
       │                         │
       ▼                         ▼
CLIP Vision Transformer    CLIP Text Transformer
 (openai/clip-vit-base)    (openai/clip-vit-base)
   [512-dim visual]          [512-dim text]
       │                         │
       └───────────┬─────────────┘
                   ▼
       Gated Multimodal Fusion
  - Modality Projections (LayerNorm + GELU + Dropout)
  - Dimension-wise Gating: g = σ(W_g [v; t])
  - Fused: f = g ⊙ v + (1 - g) ⊙ t
  - Residual Transformer MLP Block (256-dim)
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
Multi-Label Stylistic Head    Decoupled Hate Detection Head
  Independent Sigmoids          Binary Sigmoid
  Masked BCEWithLogitsLoss       Masked BCE Loss
  (funny, sarcastic, offensive)  (hateful vs non-hateful)
```

### Key Differences from Baseline `app.py`:
| Dimension | Existing Baseline (`app.py`) | Phase 2 Multimodal Model (`multimodal_model.py`) |
|---|---|---|
| **Architecture** | Decoupled BERT (sentiment) + Zero-Shot CLIP (prompts) | Unified **Gated Multimodal Fusion** combining visual & textual embeddings |
| **Output Activation** | 8-class mutually exclusive **Softmax** ($\sum P_i = 1$) | Independent **Sigmoids** $\sigma(z_i) \in [0, 1]$ per category |
| **Multi-label Support** | ❌ Fails (76.7% of memes carry multiple valid tags) | ✅ Native multi-label classification via tuned thresholds |
| **Loss Function** | None (Zero-shot) | **Masked `BCEWithLogitsLoss`** (strictly ignores unsupported targets `-1`) |
| **Hate Detection** | Derived by comparing `hateful` vs `non-hateful` slice | **Decoupled independent head** with dedicated binary evaluation |

---

## 2. Audited Label Mapping Taxonomy (Memotion 7k)

| Target Category | Ground Truth Status in Memotion 7k | Value Mapping & Handling |
|---|---|---|
| **Funny** | ✅ **Direct Ground Truth** | `humor` in `['funny', 'very_funny', 'hilarious']` $\rightarrow$ `1`<br>`humor == 'not_funny'` $\rightarrow$ `0` |
| **Sarcastic** | ✅ **Direct Ground Truth** | `sarcasm` in `['general', 'twisted_meaning', 'very_twisted']` $\rightarrow$ `1`<br>`sarcasm == 'not_sarcastic'` $\rightarrow$ `0` |
| **Offensive** | ✅ **Direct Ground Truth** | `offensive` in `['slight', 'very_offensive', 'hateful_offensive']` $\rightarrow$ `1`<br>`offensive == 'not_offensive'` $\rightarrow$ `0` |
| **Wholesome** | ❌ **Unsupported** | Marked as **`-1` (Unknown)**. *Do NOT equate 'motivational' to 'wholesome'.* Masked out. |
| **Neutral** | ❌ **Unsupported** | Marked as **`-1` (Unknown)**. *Do NOT assume neutral text sentiment = neutral meme.* Masked out. |
| **Political** | ❌ **Unsupported** | Marked as **`-1` (Unknown)**. *Zero political annotations in Memotion 7k.* Masked out. |
| **Hateful** | ⚠️ **Sparse / Proxy** | `hateful_offensive` $\rightarrow$ `1`, `not_offensive` $\rightarrow$ `0`, ambiguous $\rightarrow$ `-1`. Decoupled. |
| **Non-Hateful** | ⚠️ **Sparse / Proxy** | Inverse of Hateful. Decoupled from stylistic classification. |

---

## 3. Storage & Resource Profile

* **Streaming Execution**: Dataset streams directly from Hugging Face Parquet shards (`streaming=True`).
* **Zero Local Image Files**: Images are decoded in RAM and closed immediately.
* **Disk Footprint**: Total cache across all runs is **< 10 KB** (located in `.cache/memotion_stream`).
* **RAM / VRAM Requirements**:
  * With frozen backbones: ~1.2 GB system RAM, < 1.0 GB GPU VRAM (runs smoothly on CPU or GPU).
  * Checkpoint size: **~2.8 MB** (contains fusion layer and classification heads).

---

## 4. Execution Commands

### 1. Run Pipeline Extraction Test
Processes a sample of records with multi-label extraction and deduplication:
```bash
python memotion_pipeline.py --samples 30 --split test
```

### 2. Run Smoke Test (Fast Verification)
Runs 1 epoch on 8 training and 4 validation samples to verify forward/backward passes:
```bash
python train_multimodal.py --smoke-test
```

### 3. Run Standard Training & Threshold Tuning
Trains the gated fusion layer and searches optimal per-class F1 decision thresholds on the validation split:
```bash
python train_multimodal.py --epochs 3 --train-samples 64 --val-samples 32 --batch-size 8 --lr 1e-4
```

### 4. Side-by-Side Comparative Evaluation
Evaluates the trained model against the zero-shot baseline on the exact same records:
```bash
python evaluate_multimodal.py --split validation --samples 30
```

### 5. Run Existing Production Application (Baseline)
The original application remains completely intact and functional:
```bash
python app.py
```
