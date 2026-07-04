# sim-env-debugger

## Purpose

Diagnose local environment issues for Physical AI simulation, VLA inference, RL baselines, and 3D pipelines.

Main targets include CUDA, PyTorch, NVIDIA driver, bitsandbytes, ManiSkill, Genesis, Isaac Lab, MuJoCo, OpenCV, Open3D, gsplat, PyTorch3D, transformers, and accelerate.

## When To Use

Use this skill when:

- Imports fail.
- CUDA is unavailable or mismatched.
- A simulator crashes during import, reset, or rendering.
- VLA model loading fails due to GPU, quantization, or dependency issues.
- 3D libraries fail due to OpenGL, EGL, CUDA, or compiler issues.

## Inputs

- Error log or traceback
- Environment name
- Python version
- Output from `scripts/check_physical_ai_env.py`
- Package versions, when available
- GPU and driver information

## Outputs

A diagnosis report with:

- Problem category
- Evidence from logs
- Likely root cause
- Minimal safe fix
- Commands to inspect versions
- Commands to install missing lightweight packages when appropriate
- Actions requiring user approval

## Steps

1. Run:

   ```bash
   python scripts/check_physical_ai_env.py
   ```

2. Inspect Python and package versions.
3. Check CUDA visibility and GPU name.
4. Check import failures for optional libraries.
5. For PyTorch/CUDA mismatch, compare PyTorch CUDA build, driver, and visible devices.
6. For simulator render errors, check display, EGL, headless mode, and OpenGL dependency.
7. For bitsandbytes errors, check CUDA version and whether CPU fallback is acceptable.
8. Recommend the smallest local fix first.

## Constraints

- Do not use `sudo` without explicit approval.
- Do not delete or recreate conda environments without approval.
- Do not run large installs as part of debugging.
- Do not upgrade core CUDA, drivers, or system packages casually.
- Prefer diagnostic commands and targeted package installs.

## Failure Debugging

- If `torch.cuda.is_available()` is false, check `nvidia-smi`, PyTorch CUDA build, `CUDA_VISIBLE_DEVICES`, and driver availability.
- If `bitsandbytes` fails, fall back to non-quantized or CPU inference for a tiny test when possible.
- If Open3D visualization fails on a server, use offscreen rendering or save geometry files.
- If Isaac Lab fails, verify Isaac Sim installation and version compatibility before changing Python packages.
- If ManiSkill / Genesis assets are missing, identify asset size before downloading.

## Minimum Runnable Demo

The minimum demo is a diagnostic report from:

```bash
python scripts/check_physical_ai_env.py
```

The report must distinguish installed, missing, and broken packages without fatal failure.

