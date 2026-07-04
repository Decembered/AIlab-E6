#!/usr/bin/env bash
set -euo pipefail

# Isaac Lab local install for Ubuntu 22.04.
# Uses a pinned CPython 3.11 venv because the system python3.11 is 3.11.0rc1.

UV="${HOME}/.local/bin/uv"
VENV=".venvs/isaaclab-py311"

"${UV}" venv --seed "${VENV}" --python 3.11.13

"${VENV}/bin/python" -m pip --default-timeout=1000 --retries 10 install \
  "isaaclab[isaacsim,all]==2.3.2.post1" \
  --extra-index-url https://pypi.nvidia.com

"${VENV}/bin/python" - <<'PY'
import sys
import torch
import isaaclab
import isaacsim

print("python", sys.version)
print("torch", torch.__version__)
print("torch cuda available", torch.cuda.is_available())
print("torch cuda build", torch.version.cuda)
print("gpu count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu 0", torch.cuda.get_device_name(0))
print("isaaclab import ok", getattr(isaaclab, "__version__", "unknown"))
print("isaacsim import ok", getattr(isaacsim, "__version__", "unknown"))
PY

# Full SimulationApp smoke test. If this fails with "Failed to create change watch"
# and errno=28, raise Linux inotify limits before retrying.
OMNI_KIT_ACCEPT_EULA=YES "${VENV}/bin/python" - <<'PY'
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
print("SimulationApp started")
app.close()
print("SimulationApp closed OK")
PY
