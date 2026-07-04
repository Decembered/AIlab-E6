# Isaac Lab Local Install

## Goal

Install Isaac Lab for the Video2Motion2Action dexterous manipulation workflow on the local Ubuntu 22.04 machine.

## Result

Partial PASS.

- Isaac Lab installed in `.venvs/isaaclab-py311`.
- `isaaclab`, `isaacsim`, and `torch` imports pass.
- PyTorch CUDA is available on the local RTX 4060 Laptop GPU.
- A minimal Isaac Sim `SimulationApp({"headless": True})` reaches `app ready`, then exits with code 134.

## Environment

- OS: Ubuntu 22.04.5
- Python: 3.11.13
- Isaac Lab: 2.3.2.post1
- Isaac Sim: 5.1.0.0
- Torch: 2.7.0, CUDA 12.6 build
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM
- RAM: 22 GiB
- Swap: 0 GiB

## Diagnosis

The failure is not a missing package or a broken Isaac Lab install. The smoke test log shows many errors like:

```text
Failed to create change watch ... errno=28/No space left on device
```

Disk and inode space are available. The likely bottleneck is the Linux inotify watch limit:

```text
fs.inotify.max_user_watches=65536
fs.inotify.max_user_instances=128
```

Isaac Sim 5.1 loads a large Kit extension tree and needs many file watches. The machine also has limited free RAM and no swap, which increases fragility during startup and shutdown.

## Reproduce

```bash
bash experiments/2026-07-05_isaaclab_install/command.sh
```

## Recommended Fix

Requires user approval because it changes system-wide kernel parameters:

```bash
sudo sysctl -w fs.inotify.max_user_watches=1048576
sudo sysctl -w fs.inotify.max_user_instances=1024
sudo sysctl -w fs.inotify.max_queued_events=32768
```

Then rerun:

```bash
OMNI_KIT_ACCEPT_EULA=YES .venvs/isaaclab-py311/bin/python - <<'PY'
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
print("SimulationApp started")
app.close()
print("SimulationApp closed OK")
PY
```

If this becomes stable, persist the sysctl settings and add swap before running heavier manipulation scenes.

## Evidence

- `logs.txt`: package versions and host diagnostics.
- `simulationapp_smoke.log`: full Isaac Sim startup/crash log.
- `metrics.json`: machine-readable install result.
