#!/usr/bin/env python3
"""Download VGGT-1B-Commercial model with progress bar and fast transfer."""

import os
from huggingface_hub import snapshot_download

print("=" * 50)
print("Downloading VGGT-1B-Commercial model...")
print("=" * 50)
print()

# Download the model with progress bar
# Note: HF_HUB_ENABLE_HF_TRANSFER suppresses progress bars, so we use
# the default downloader which shows per-file tqdm bars.
snapshot_download(
    repo_id="facebook/VGGT-1B-Commercial",
    local_dir="./VGGT-1B-Commercial",
    local_dir_use_symlinks=False,
    resume_download=True,
)

print()
print("=" * 50)
print("Download complete!")
print("Model saved to: ./VGGT-1B-Commercial")
print("=" * 50)
