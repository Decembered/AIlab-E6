# Video2Motion2Action IsaacGym Setup

## Goal

Install only the Isaac Gym-related baseline dependencies needed for the Video2Motion2Action: Dexterous Manipulation Skill Transfer challenge.

## Result Summary

- Used the challenge page to confirm the required simulator is Isaac Gym Preview 4.
- Reused `.venvs/isaacgym-py38`.
- Installed baseline Python dependencies for asset checks, video/mask processing, logging, and CPU imports.
- Installed CPU PyTorch 1.13.1 and torchvision 0.14.1.
- Did not install Isaac Sim 4.5.
- Did not download large datasets or simulator assets.

## Current Limitation

Isaac Gym Preview 4 itself is not present on this machine. The challenge page says the contest image provides it at `~/opt/isaac_gym/isaacgym`, but no such local path exists here. Install it from the NVIDIA package or use the contest DSW image.

## Next Step

Provide the Isaac Gym Preview 4 package locally, then run:

```bash
scripts/setup_isaac_stack.sh install-isaacgym ~/Downloads/IsaacGym_Preview_4_Package.tar.gz
.venvs/isaacgym-py38/bin/python scripts/check_video2motion_isaacgym.py
```
