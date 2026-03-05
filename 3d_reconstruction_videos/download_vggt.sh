#!/bin/bash

# Download VGGT-1B-Commercial model with progress bar and fast transfer

echo "=========================================="
echo "Downloading VGGT-1B-Commercial model..."
echo "=========================================="

# Enable fast transfer with progress bar
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TRANSFER_MAX_CONCURRENT_DOWNLOADS=16
export HF_HUB_DOWNLOAD_PROGRESS_BAR=1

# Download the model
huggingface-cli download facebook/VGGT-1B-Commercial \
    --local-dir ./VGGT-1B-Commercial \
    --local-dir-use-symlinks False \
    --resume-download

echo ""
echo "=========================================="
echo "Download complete!"
echo "Model saved to: ./VGGT-1B-Commercial"
echo "=========================================="
