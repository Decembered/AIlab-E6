#!/usr/bin/env bash
set -euo pipefail

scripts/setup_isaac_stack.sh install-video2motion-deps
.venvs/isaacgym-py38/bin/python scripts/check_video2motion_isaacgym.py
