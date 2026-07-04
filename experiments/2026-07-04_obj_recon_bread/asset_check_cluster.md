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

## Test 2: bread_v41 (our SAM v4.1 pipeline) — PASS after collision fix

**Attempt 1 (mesh not centered):** FAIL — same mesh for visual+collision, offset from origin
```
Step 10: pos=(0.0008, -0.0649, 0.0037)  ← sinking below ground!
Step 50: pos=(0.0307, -0.1071, 0.0797)  ← drifting horizontally
```
Root cause: extrusion vertices offset from origin by ~(8.5, 6, 15)cm; offset COM causes collision misalignment.

**Attempt 2 (centered mesh + dedicated collision box):** PASS ✅
```
Step  0: pos=(0.0000, 0.0980, 0.0000)
Step 10: pos=(0.0000, 0.0420, 0.0000)   ← resting on collision surface
Step 30: pos=(0.0000, 0.0420, 0.0000)   ← stable, no drift
Step 60: pos=(0.0000, 0.0420, 0.0000)   ← perfectly stable

PASS: Asset loads and simulates correctly
```
- Visual: 182 vertices, 360 faces (centered at origin)
- Collision: 8 vertices, 12 tris (dedicated box, centered at origin)
- Fix: re-centered visual mesh + generated simplified collision box at origin
- Result: stable physics, no sinking, no horizontal drift

## Conclusion

- Both assets **load** in IsaacGym ✅
- `runs/bread` physics is **stable** ✅ — use as reference format
- `bread_v41` physics has **collision mesh issues** ⚠️ — fix before final submission
- Pipeline (SAM → 3D → URDF) produces **loadable** assets — validation passed
- Lesson: always use a simplified separate collision mesh, not the visual extrusion
