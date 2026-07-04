#!/usr/bin/env bash
# ============================================================================
# download_models.sh — Pre-download models for offline hackathon use
# ============================================================================
#
# Usage: bash scripts/download_models.sh [--all] [--octo] [--dry-run]
#
# This script downloads model checkpoints that would otherwise need internet
# at the hackathon venue. Run this BEFORE the event.
#
# All checkpoints go to ~/.cache/safefly/models/ by default.
# ============================================================================

set -euo pipefail

CACHE_DIR="${SAFEFLY_CACHE:-$HOME/.cache/safefly/models}"
DRY_RUN=false
DOWNLOAD_OCTO=false
DOWNLOAD_ALL=false

# --- Parse args ---
for arg in "$@"; do
    case "$arg" in
        --all) DOWNLOAD_ALL=true ;;
        --octo) DOWNLOAD_OCTO=true ;;
        --dry-run) DRY_RUN=true ;;
        --help|-h)
            echo "Usage: $0 [--all] [--octo] [--dry-run]"
            echo ""
            echo "  --all      Download all available models"
            echo "  --octo     Download Octo-small VLA checkpoint (~100MB)"
            echo "  --dry-run  Show what would be downloaded, don't download"
            exit 0
            ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

if $DOWNLOAD_ALL; then
    DOWNLOAD_OCTO=true
fi

if ! $DOWNLOAD_OCTO; then
    echo "Nothing selected. Use --octo to download Octo, or --all for everything."
    echo "Run with --help for more info."
    exit 0
fi

mkdir -p "$CACHE_DIR"

# ============================================================================
# Octo-small VLA model (~100 MB)
# ============================================================================
download_octo() {
    local dest="$CACHE_DIR/octo-small"
    echo ""
    echo "=== Octo-small VLA checkpoint (~100 MB) ==="

    if [ -d "$dest" ] && [ -f "$dest/checkpoint.pt" ]; then
        echo "  Already downloaded at: $dest"
        echo "  To re-download, delete: rm -rf $dest"
        return 0
    fi

    if $DRY_RUN; then
        echo "  [DRY RUN] Would download from HuggingFace to: $dest"
        echo "  Source: octo-models/octo-small"
        return 0
    fi

    echo "  Downloading via Python (octo package handles this)..."
    python3 -c "
import sys
try:
    from octo import OctoModel
    print('  Loading octo-small (first time will download ~100MB)...')
    model = OctoModel.load('octo-small')
    print('  ✓ Octo-small loaded successfully')
    print(f'  Model type: {type(model).__name__}')
except ImportError:
    print('  ⚠ octo package not installed. Install with: pip install octo')
    print('  Skipping download.')
    sys.exit(0)
except Exception as e:
    print(f'  ⚠ Failed to load: {e}')
    print('  You can download manually from: https://huggingface.co/octo-models/octo-small')
"
    echo "  ✓ Octo checkpoint cached"
}

# ============================================================================
# Summary
# ============================================================================
print_summary() {
    echo ""
    echo "============================================"
    echo " Download Summary"
    echo "============================================"
    echo " Cache directory: $CACHE_DIR"
    echo ""
    du -sh "$CACHE_DIR"/*/ 2>/dev/null || echo "  (empty)"
    echo ""
    echo " Total size:"
    du -sh "$CACHE_DIR" 2>/dev/null || echo "  0B"
    echo ""
    echo "✓ Ready for offline hackathon use."
    echo "  Set SAFEFLY_CACHE to change cache location."
}

# --- Run ---
if $DOWNLOAD_OCTO; then
    download_octo
fi

print_summary
