#!/usr/bin/env bash
# ============================================================================
# download_wheels.sh — Pre-download pip wheels for OFFLINE hackathon use
# ============================================================================
#
# Usage: bash scripts/download_wheels.sh [--all] [--tier0] [--tier1]
#
# Downloads all required Python packages as wheel files into a local
# directory. At the hackathon venue (no internet), install with:
#
#   pip install --no-index --find-links ./wheelhouse -r requirements.txt
#
# This is your INSURANCE against bad venue WiFi.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WHEELHOUSE="${WHEELHOUSE:-$PROJECT_DIR/wheelhouse}"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"

DRY_RUN=false
TIER="tier0"  # default: minimal

# --- Parse args ---
for arg in "$@"; do
    case "$arg" in
        --all)    TIER="all" ;;
        --tier0)  TIER="tier0" ;;
        --tier1)  TIER="tier1" ;;
        --dry-run) DRY_RUN=true ;;
        --help|-h)
            echo "Usage: $0 [--all] [--tier0] [--tier1] [--dry-run]"
            echo ""
            echo "  --all      Download all dependencies (including optional)"
            echo "  --tier0    Download only Tier 0 (numpy, matplotlib, scipy) — minimal"
            echo "  --tier1    Download Tier 0 + Tier 1 (pybullet, open3d, gymnasium etc)"
            echo "  --dry-run  Show what would be downloaded, don't download"
            echo ""
            echo "Wheels are saved to: $WHEELHOUSE"
            exit 0
            ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# --- Prepare tiered requirements ---
prepare_requirements() {
    local tier="$1"
    local tmpfile
    tmpfile=$(mktemp /tmp/safefly_reqs_XXXXXX.txt)

    case "$tier" in
        tier0)
            echo "# Tier 0 — Minimal survival" > "$tmpfile"
            grep -A100 "^# TIER 0" "$REQUIREMENTS" | grep -v "^#" | grep -v "^$" | head -3 >> "$tmpfile" || true
            ;;
        tier1)
            echo "# Tier 0+1 — Core demo" > "$tmpfile"
            awk '/^# TIER 0/,/^# TIER 2/' "$REQUIREMENTS" \
                | grep -v "^#" | grep -v "^$" >> "$tmpfile" || true
            ;;
        all)
            # Everything non-commented
            grep -v "^#" "$REQUIREMENTS" | grep -v "^$" > "$tmpfile" || true
            ;;
    esac

    echo "$tmpfile"
}

# --- Main ---
main() {
    echo "=== SafeFly Offline Wheel Downloader ==="
    echo "  Tier: $TIER"
    echo "  Wheelhouse: $WHEELHOUSE"
    echo ""

    mkdir -p "$WHEELHOUSE"

    local req_file
    req_file=$(prepare_requirements "$TIER")

    echo "Packages to download:"
    grep -v "^#" "$req_file" | grep -v "^$" | sed 's/^/  - /'
    echo ""

    if $DRY_RUN; then
        echo "[DRY RUN] Would download to: $WHEELHOUSE"
        rm -f "$req_file"
        return 0
    fi

    echo "Downloading wheels..."
    pip download \
        --dest "$WHEELHOUSE" \
        --only-binary :all: \
        --prefer-binary \
        --progress-bar on \
        -r "$req_file" 2>&1 || {
        echo ""
        echo "⚠ Some packages may not have binary wheels. Trying with source..."
        pip download \
            --dest "$WHEELHOUSE" \
            --progress-bar on \
            -r "$req_file" 2>&1
    }

    rm -f "$req_file"

    echo ""
    echo "============================================"
    echo " Download Complete"
    echo "============================================"
    echo " Location: $WHEELHOUSE"
    echo " Size: $(du -sh "$WHEELHOUSE" 2>/dev/null | cut -f1)"
    echo " Files: $(ls "$WHEELHOUSE"/*.whl 2>/dev/null | wc -l) wheels"
    echo ""
    echo "=== HOW TO USE OFFLINE ==="
    echo "  pip install --no-index --find-links $WHEELHOUSE -r requirements.txt"
    echo ""
    echo "=== VERIFY OFFLINE INSTALL ==="
    echo "  pip install --no-index --find-links $WHEELHOUSE --dry-run -r requirements.txt"
}

main
