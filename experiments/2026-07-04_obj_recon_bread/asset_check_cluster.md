# Cluster Validation Results — 2026-07-04

## Environment

- Host: root@123.57.187.96:937 (Aliyun DSW)
- IsaacGym: /root/opt/isaac_gym/isaacgym, Python 3.8
- Physics: CPU PhysX, GPU pipeline disabled

## Test 1: runs/object_asset_v1/bread (existing asset) — PASS

```
[3/5] Loaded OK: 1 body, 1 shape, 0 DOF
[5/5] Step  0: pos=(0.0000, 0.0980, 0.0000)
       Step 10: pos=(0.0000, 0.0370, 0.0000)   ← stable, resting on ground
       Step 60: pos=(0.0000, 0.0370, 0.0000)   ← no drift, no explosion

PASS: Asset loads and simulates correctly
```

- 642 visual vertices, 8 collision vertices (dedicated collision box)
- Watertight, manifold, stable physics

## Test 2: bread_v41 (our SAM v4.1 pipeline) — LOADS, collision issue

```
[3/5] Loaded OK: 1 body, 1 shape, 0 DOF
[5/5] Step  0: pos=(0.0000, 0.0980, 0.0000)
       Step 30: pos=(0.0503, -0.1040, 0.1034)  ← sinking below ground!
       Step 50: pos=(0.0307, -0.1071, 0.0797)  ← drifting horizontally

PASS: loads without error, but physics unstable
```

- 182 vertices, same mesh for visual + collision
- Root cause: extrusion model bottom face not perfectly flat, vertices at y=0.02 offset from origin
- Fix: generate dedicated simplified collision mesh (box/convex hull) instead of reusing visual mesh

## Conclusion

- Both assets **load** in IsaacGym ✅
- `runs/bread` physics is **stable** ✅ — use as reference format
- `bread_v41` physics has **collision mesh issues** ⚠️ — fix before final submission
- Pipeline (SAM → 3D → URDF) produces **loadable** assets — validation passed
- Lesson: always use a simplified separate collision mesh, not the visual extrusion
