# Object Bonus: Articulated Modeling & Physics Optimization

> Task 3.5 — Object Bonus | Member C
> Video2Motion2Action Hackathon

## Overview

Beyond static 3D reconstruction (Task 3.3), the Object Bonus extends object assets with articulated body modeling, joint configuration, optimized collision meshes, realistic physical parameters, and improved pose alignment — all aimed at improving contact stability in Task 3.4 hand-object tracking simulation.

## Deliverables

```
runs/object_bonus/
├── pipette_articulated/          # Pipette with prismatic plunger joint
│   ├── pipette_articulated.urdf   # 2-link articulated URDF
│   ├── pipette_base.obj           # Base mesh (2349 faces)
│   ├── pipette_plunger.obj        # Plunger mesh (847 faces)
│   ├── pipette_base_collision.obj # Base collision (convex hull)
│   ├── pipette_plunger_collision.obj
│   └── articulated_meta.json      # Joint + dynamics metadata
│
├── drink_ad_articulated/          # Drink AD with revolute cap joint
│   ├── drink_ad_articulated.urdf
│   ├── drink_ad_body.obj (375f) + cap.obj (85f)
│   └── articulated_meta.json
│
├── drink_yykx_articulated/        # Drink YYKX with revolute cap joint
│   ├── drink_yykx_articulated.urdf
│   ├── drink_yykx_body.obj (232f) + cap.obj (52f)
│   └── articulated_meta.json
│
├── bread_optimized/               # Bread with optimized physics + multi-res collision
│   ├── bread_optimized.urdf
│   ├── bread_collision_coarse.obj (144f) + bread_collision_fine.obj (588f)
│   └── optimized_meta.json
│
├── drink_ad_optimized/            # Drink AD optimized (rigid body)
│   ├── drink_ad_optimized.urdf    # with contact friction parameters
│   └── optimized_meta.json
│
└── drink_yykx_optimized/          # Drink YYKX optimized (rigid body)
    ├── drink_yykx_optimized.urdf
    └── optimized_meta.json
```

## Articulated Models

### 1. Pipette — Prismatic Joint (Plunger)

The pipette is modeled as a 2-link articulated body:

| Property | Value | Justification |
|----------|-------|---------------|
| Joint type | Prismatic | Plunger slides linearly along pipette axis |
| Joint axis | X (1, 0, 0) | Along pipette length direction |
| Range | 0 → 0.03 m | Typical pipette plunge distance |
| Damping | 5.0 N·s/m | Prevents oscillation during press |
| Friction | 2.0 N | Simulates plunger seal resistance |
| Effort limit | 10 N | Maximum press force |
| Mass distribution | Base: 0.12 kg, Plunger: 0.03 kg | Realistic weight split |

**Impact on Task 3.4**: The prismatic joint enables the simulation to model the pipette press action. When the Sharpa hand's fingertip contacts the plunger and applies force, the plunger slides along the axis — producing realistic pipetting behavior rather than treating the pipette as a static block. This directly improves grasp_pipette_press and grasp_pipette_rotate sequence tracking fidelity.

### 2. Drink Bottles — Revolute Joint (Screw Cap)

Both drink bottles are modeled with a screw cap attached via revolute joint:

| Property | Value | Justification |
|----------|-------|---------------|
| Joint type | Revolute (continuous) | Cap rotates around bottle axis |
| Joint axis | Z (0, 0, 1) | Vertical rotation axis |
| Range | ±360° (continuous) | Full rotation |
| Damping | 3.0 N·m·s/rad | Prevents free-spinning |
| Friction | 1.5 N·m | Realistic cap resistance |
| Mass distribution | Body: 0.52 kg, Cap: 0.03 kg | Realistic weight split |

**Impact on Task 3.4**: Though cap twisting is not the primary manipulation in the given sequences, having a physically modeled joint prepares the asset for cap-related interactions. More importantly, the separate link design improves contact stability — forces applied to the cap are transmitted through the joint to the body rather than applying unrealistic rigid body torques.

## Physics Optimization (Rigid Body)

### Bread — Multi-Resolution Collision

| Level | Faces | Use Case |
|-------|-------|----------|
| Coarse | 144 (convex hull) | Broad-phase collision detection |
| Fine | 588 (visual mesh) | Narrow-phase when needed |

### Optimized Contact Parameters

All optimized models include surface contact parameters:

| Parameter | Value | Effect |
|-----------|-------|--------|
| Lateral friction | 1.0 | Prevents unrealistic sliding on table |
| Rolling friction | 0.01 | Dampens micro-oscillations |
| Restitution | 0.1 | Low bounce for stable settling |

### Mass/Inertia Calibration

| Object | Mass (kg) | Inertia Ixx | Iyy | Izz |
|--------|-----------|-------------|-----|-----|
| Pipette (articulated) | 0.15 | 0.000072 | 0.000865 | 0.000866 |
| Bread (optimized) | 0.40 | 0.000180 | 0.000605 | 0.000730 |
| Drink AD (optimized) | 0.55 | 0.006372 | 0.004639 | 0.003553 |
| Drink YYKX (optimized) | 0.55 | 0.003756 | 0.003421 | 0.002189 |

All computed from bounding box dimensions with realistic object density.

## Validation

All 6 bonus models validated in IsaacGym (2x RTX 4090, Python 3.8):

| Model | Bodies | Shapes | DOFs | Joint Type | Status |
|-------|--------|--------|------|------------|--------|
| Pipette articulated | 2 | 2 | 1 | Prismatic (plunger_joint) | ✅ PASS |
| Drink AD articulated | 2 | 2 | 1 | Revolute (cap_joint) | ✅ PASS |
| Drink YYKX articulated | 2 | 2 | 1 | Revolute (cap_joint) | ✅ PASS |
| Bread optimized | 1 | 1 | 0 | — | ✅ PASS |
| Drink AD optimized | 1 | 1 | 0 | — | ✅ PASS |
| Drink YYKX optimized | 1 | 1 | 0 | — | ✅ PASS |

Validation command:
```bash
python3.8 -c "from isaacgym import gymapi; ..."
```

## Impact on Task 3.4 Tracking Success

The Object Bonus improvements directly contribute to Task 3.4 integration:

1. **Articulated pipette** → Enables realistic plunger pressing simulation, matching the *grasp_pipette_press* manipulation sequence
2. **Articulated bottles** → Prepares for cap-twisting interactions in *weigh_drink* sequences
3. **Multi-resolution collision** → Improves contact stability during hand-object interaction, reducing penetration artifacts
4. **Calibrated friction** → Prevents unrealistic object sliding during hand contact
5. **Per-link mass distribution** → Enables correct force transmission through joints, producing physically plausible object response to hand forces

## References

- URDF specification: http://wiki.ros.org/urdf/XML/
- IsaacGym asset creation: NVIDIA Developer documentation
- Joint dynamics modeling: PhysX SDK reference
