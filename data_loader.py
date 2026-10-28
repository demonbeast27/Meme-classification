"""
data_loader.py
==============
Storage-efficient streaming dataloader for Memotion 7k.
- Decodes images in-memory and releases them immediately.
- Tokenizes OCR text on-the-fly.
- Formats multi-label targets with -1 mask for unsupported classes.
- Detects and prevents cross-split data leakage via MD5 text hashing.
"""

import hashlib
import io
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import IterableDataset, DataLoader
from transformers import CLIPProcessor
import datasets

import memotion_config as cfg
from datasets import load_dataset


class StreamingMemotionDataset(IterableDataset):
    """
    Streaming PyTorch IterableDataset for Memotion 7k.
    Streams records on-demand directly from Hugging Face Parquet shards.
    """
    def __init__(
        self,
        split: str = "train",
        max_samples: int | None = None,
        clip_processor: CLIPProcessor | None = None,
        seen_hashes: set[str] | None = None,
        stylistic_categories: list[str] | None = None,
    ):
        super().__init__()
        self.split = split
        self.max_samples = max_samples
        if clip_processor is None:
            clip_processor = CLIPProcessor.from_pretrained(cfg.DEFAULT_HYPERPARAMS["clip_model_name"])
        self.processor = clip_processor
        self.seen_hashes = seen_hashes if seen_hashes is not None else set()
        if stylistic_categories is None:
            stylistic_categories = ["funny", "sarcastic", "offensive", "wholesome", "neutral", "political"]
        self.stylistic_categories = stylistic_categories

    def __iter__(self):
        ds = load_dataset(
            cfg.DATASET_NAME,
            name=cfg.DATASET_CONFIG,
            split=self.split,
            streaming=True,
        )
        ds = ds.cast_column("image", datasets.Image(decode=False))

        yielded = 0
        for row in ds:
            if self.max_samples is not None and self.max_samples > 0 and yielded >= self.max_samples:
                break

            text_ocr = (row.get("text_ocr") or "").strip()
            text_corrected = (row.get("text_corrected") or "").strip()
            text = text_corrected if text_corrected else text_ocr
            if not text:
                text = "meme"

            # Check duplication / cross-split leakage
            text_hash = hashlib.md5(text.lower().encode("utf-8", errors="ignore")).hexdigest()
            if text_hash in self.seen_hashes:
                continue
            self.seen_hashes.add(text_hash)

            # Image in-memory decode
            raw_img = row.get("image")
            pil_img = None
            if raw_img is not None:
                try:
                    if isinstance(raw_img, dict) and "bytes" in raw_img:
                        pil_img = Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
                    elif isinstance(raw_img, Image.Image):
                        pil_img = raw_img.convert("RGB")
                except Exception:
                    pil_img = None

            if pil_img is None:
                # Fallback to neutral black canvas if decoding fails
                pil_img = Image.new("RGB", (224, 224), color=(0, 0, 0))

            # Process text and image with CLIPProcessor
            try:
                inputs = self.processor(
                    text=[text],
                    images=pil_img,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                )
            finally:
                # Always close and delete PIL image to guarantee minimal RAM & storage usage
                pil_img.close()
                del pil_img

            # Map labels
            mapped = cfg.map_memotion_record_labels(row)
            stylistic_targets = torch.tensor(
                [mapped[cat] for cat in self.stylistic_categories],
                dtype=torch.float32
            )
            hate_target = torch.tensor(mapped["hateful"], dtype=torch.float32)

            yield {
                "pixel_values": inputs["pixel_values"][0],  # shape (3, 224, 224)
                "input_ids": inputs["input_ids"][0],        # shape (77,)
                "attention_mask": inputs["attention_mask"][0],  # shape (77,)
                "stylistic_targets": stylistic_targets,     # shape (num_stylistic,)
                "hate_targets": hate_target,                # scalar float
                "text": text,
            }
            yielded += 1


def collate_memotion_batch(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    stylistic_targets = torch.stack([item["stylistic_targets"] for item in batch])
    hate_targets = torch.stack([item["hate_targets"] for item in batch])
    texts = [item["text"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "stylistic_targets": stylistic_targets,
        "hate_targets": hate_targets,
        "texts": texts,
    }


def get_memotion_dataloader(
    split: str = "train",
    max_samples: int | None = None,
    batch_size: int = 16,
    clip_processor: CLIPProcessor | None = None,
    seen_hashes: set[str] | None = None,
    stylistic_categories: list[str] | None = None,
) -> DataLoader:
    dataset = StreamingMemotionDataset(
        split=split,
        max_samples=max_samples,
        clip_processor=clip_processor,
        seen_hashes=seen_hashes,
        stylistic_categories=stylistic_categories,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_memotion_batch,
    )
