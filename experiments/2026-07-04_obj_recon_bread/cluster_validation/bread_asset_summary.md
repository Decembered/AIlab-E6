# Bread IsaacGym Asset Load Test

## Result

PASS

## Asset

- URDF: /mnt/workspace/Hackthon/experiments/2026-07-04_obj_recon_bread/models/bread.urdf
- Mesh: /mnt/workspace/Hackthon/experiments/2026-07-04_obj_recon_bread/models/bread_extruded.obj

## Environment

- Platform: Aliyun PAI DSW
- GPU: 2 x RTX 4090
- IsaacGym: Preview 4 / isaacgym 1.0rc4
- Python: 3.8.20
- Torch: 1.13.1+cu117
- CUDA available in torch: True
- Physics: GPU PhysX, cuda:0

## Evidence

- URDF exists: True
- Mesh exists: True
- IsaacGym load_asset: OK
- create_actor: OK
- simulation steps: 120
- VHACD: finished convex decomposition
- VHACD hulls: 14

## Limitation

This is a headless asset loading smoke test. It verifies that the asset can be loaded and simulated in IsaacGym, but it does not yet verify physical interaction quality, scale accuracy against the real object, stable contact with Sharpa hand, or tracking task success.

## Drop Stability Test

PASS

Evidence:

- Initial height: z = 0.5 m
- Simulation steps: 240
- Final position: x = -0.0595, y = -0.00049, z = 0.06684
- Final linear velocity: vx = 0.00527, vy = -0.00291, vz = -0.00182
- Result flag: BREAD_DROP_TEST_OK

Interpretation:

The bread asset can be loaded, collide with the ground, and settle without obvious simulation explosion in a headless IsaacGym GPU PhysX run. This is still a smoke test; it does not yet validate real-world scale accuracy or hand-object contact quality.
