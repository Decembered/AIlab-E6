#!/usr/bin/env bash
set -euo pipefail

scripts/setup_isaac_stack.sh prepare
scripts/setup_isaac_stack.sh check
