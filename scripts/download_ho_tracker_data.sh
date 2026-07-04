#!/bin/bash
# ============================================================================
# Download HO-Tracker-Challenge dataset
# ============================================================================
set -euo pipefail

# Use HF Mirror for faster download in China
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"

echo "=== Downloading HO-Tracker-Challenge dataset ==="
echo "HF Endpoint: $HF_ENDPOINT"
echo "Target: $DATA_DIR"
echo "Expected size: ~4.33 GB"
echo ""

# Step 1: Download HO-Tracker part (small, ~5 MB)
echo "[1/2] Downloading HO-Tracker reference data..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'kelvin34501/HO-Tracker-Challenge',
    repo_type='dataset',
    local_dir='${DATA_DIR}',
    allow_patterns=['HO-Tracker/**'],
    resume_download=True,
)
print('HO-Tracker data downloaded.')
"

# Step 2: Download human_demo part (large, ~4.33 GB)
echo ""
echo "[2/2] Downloading human_demo data (this will take a while)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'kelvin34501/HO-Tracker-Challenge',
    repo_type='dataset',
    local_dir='${DATA_DIR}',
    allow_patterns=['human_demo/**'],
    resume_download=True,
)
print('human_demo data downloaded.')
"

echo ""
echo "=== Download complete ==="
echo "Usage: du -sh ${DATA_DIR}/*/"
du -sh "${DATA_DIR}/"*/
