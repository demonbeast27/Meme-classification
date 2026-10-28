"""
upload_model.py
===============
Uploads trained Multimodal Meme Classifier checkpoint to Hugging Face Model Hub.

Usage:
    # 1. Login to Hugging Face CLI:
    #    huggingface-cli login
    #
    # 2. Run upload script:
    #    python upload_model.py --repo-id your-username/meme-classifier-clip

Replace 'your-username/meme-classifier-clip' with your actual Hugging Face username and desired model repo name.
"""

import argparse
import os
import sys
from huggingface_hub import HfApi, login

DEFAULT_REPO_ID = "your-username/meme-classifier-clip"
DEFAULT_CHECKPOINT = "checkpoints/best_multimodal_classifier_gpu.pt"
FALLBACK_CHECKPOINT = "checkpoints/best_multimodal_classifier.pt"


def upload_checkpoint(
    repo_id: str,
    checkpoint_path: str,
    hf_token: str | None = None,
):
    if hf_token:
        login(token=hf_token)

    if not os.path.exists(checkpoint_path):
        if os.path.exists(FALLBACK_CHECKPOINT):
            print(f"[INFO] '{checkpoint_path}' not found, using fallback '{FALLBACK_CHECKPOINT}'")
            checkpoint_path = FALLBACK_CHECKPOINT
        else:
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    filename = os.path.basename(checkpoint_path)
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    print(f"[INFO] Preparing to upload {filename} ({file_size_mb:.2f} MB) to Hugging Face Model Hub: {repo_id}")

    if "your-username" in repo_id:
        print("\n[WARNING] You are using the placeholder repo ID 'your-username/meme-classifier-clip'.")
        print("Please pass your actual Hugging Face username via --repo-id <username>/<model-name>.")
        confirm = input("Continue anyway? (y/N): ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    api = HfApi()

    # Create model repository (idempotent if already exists)
    print(f"[1/2] Ensuring model repository exists: https://huggingface.co/{repo_id}...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )

    # Upload checkpoint file
    print(f"[2/2] Uploading '{checkpoint_path}' as '{filename}'...")
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload Multimodal Meme Classifier checkpoint ({filename})",
    )

    print("\n" + "=" * 60)
    print("SUCCESS: Checkpoint uploaded successfully!")
    print(f"Model Hub URL: https://huggingface.co/{repo_id}")
    print(f"File Path    : https://huggingface.co/{repo_id}/blob/main/{filename}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Upload checkpoint to Hugging Face Model Hub.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Target Hugging Face Model repo ID (e.g., 'your-username/meme-classifier-clip')",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to the checkpoint file (.pt)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face access token (or login beforehand via `huggingface-cli login`)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upload_checkpoint(
        repo_id=args.repo_id,
        checkpoint_path=args.checkpoint,
        hf_token=args.token,
    )
