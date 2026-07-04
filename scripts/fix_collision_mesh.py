#!/usr/bin/env python3
"""Fix bread_v41: center mesh at origin + generate centered collision box.

Root cause: the extrusion mesh vertices are offset from origin by ~(8.5cm, 6cm, 15cm).
Both visual and collision must be centered for stable IsaacGym physics.
"""
import sys
from pathlib import Path
import numpy as np


def load_obj(obj_path):
    """Load OBJ vertices and faces as numpy arrays."""
    verts = []
    faces = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()
                # Handle f v1 v2 v3 and f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(indices)
    return np.array(verts), faces


def write_centered_obj(path, verts, faces, comment=""):
    """Write OBJ with centered vertices."""
    with open(path, 'w') as f:
        if comment:
            f.write(f"# {comment}\n")
        f.write(f"# {len(verts)} verts, {len(faces)} faces, centered at origin\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"  Wrote: {path}")


def write_centered_collision(path, half_extents):
    """Write 8-vertex collision box centered at origin."""
    hx, hy, hz = half_extents
    corners = np.array([
        [-hx, -hy, -hz], [-hx, -hy, +hz], [-hx, +hy, -hz], [-hx, +hy, +hz],
        [+hx, -hy, -hz], [+hx, -hy, +hz], [+hx, +hy, -hz], [+hx, +hy, +hz],
    ])
    faces = [
        (0, 2, 1), (1, 2, 3),   # -X
        (4, 5, 6), (5, 7, 6),   # +X
        (0, 1, 4), (1, 5, 4),   # -Y
        (2, 6, 3), (3, 6, 7),   # +Y
        (0, 4, 2), (2, 4, 6),   # -Z
        (1, 3, 5), (3, 7, 5),   # +Z
    ]
    write_centered_obj(path, corners, faces,
                       f"Centered collision box — {len(corners)} verts, {len(faces)} tris")


def compute_centered_inertia(centered_verts, mass):
    """Compute box inertia approximation for centered mesh."""
    vmin = centered_verts.min(axis=0)
    vmax = centered_verts.max(axis=0)
    dims = vmax - vmin  # full length in each axis
    # Box formula: I = m/12 * (d2²+d3²)
    ixx = mass / 12.0 * (dims[1]**2 + dims[2]**2)
    iyy = mass / 12.0 * (dims[0]**2 + dims[2]**2)
    izz = mass / 12.0 * (dims[0]**2 + dims[1]**2)
    return (ixx, iyy, izz), dims


def write_urdf(path, visual_mesh_name, collision_mesh_name, mass, inertia, dims):
    """Write URDF with all origins at (0,0,0)."""
    ixx, iyy, izz = inertia
    dx, dy, dz = dims

    urdf = f'''<?xml version="1.0"?>
<robot name="bread">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia ixx="{ixx:.8f}" ixy="0.0" ixz="0.0" iyy="{iyy:.8f}" iyz="0.0" izz="{izz:.8f}"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{visual_mesh_name}" scale="1.0 1.0 1.0"/>
      </geometry>
      <material name="bread_mat">
        <color rgba="0.82 0.71 0.55 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{collision_mesh_name}" scale="1.0 1.0 1.0"/>
      </geometry>
    </collision>
  </link>
</robot>
'''
    with open(path, 'w') as f:
        f.write(urdf)
    print(f"  URDF: mass={mass:.3f}kg, dims=({dx*100:.1f}×{dy*100:.1f}×{dz*100:.1f})cm")
    print(f"  Inertia: ixx={ixx:.8f}, iyy={iyy:.8f}, izz={izz:.8f}")


def main():
    exp_dir = Path(__file__).resolve().parent.parent / 'experiments' / '2026-07-04_obj_recon_bread'
    models_dir = exp_dir / 'models'
    obj_path = models_dir / 'bread_v41.obj'

    if not obj_path.exists():
        print(f"ERROR: {obj_path} not found")
        sys.exit(1)

    # 1. Load original mesh
    verts, faces = load_obj(obj_path)
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    center = (vmin + vmax) / 2.0
    half_extents = (vmax - vmin) / 2.0

    print(f"Original mesh: {len(verts)} verts, {len(faces)} faces")
    print(f"  Center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print(f"  Dims: ({2*half_extents[0]*100:.1f}cm × {2*half_extents[1]*100:.1f}cm × {2*half_extents[2]*100:.1f}cm)")

    # 2. Center the visual mesh
    centered_verts = verts - center
    new_vmin = centered_verts.min(axis=0)
    new_vmax = centered_verts.max(axis=0)
    print(f"\nCentered mesh bounds:")
    print(f"  X: [{new_vmin[0]:.4f}, {new_vmax[0]:.4f}]")
    print(f"  Y: [{new_vmin[1]:.4f}, {new_vmax[1]:.4f}]")
    print(f"  Z: [{new_vmin[2]:.4f}, {new_vmax[2]:.4f}]")

    write_centered_obj(obj_path, centered_verts, faces,
                       f"Bread v4.1 — SAM mask contour extrusion, centered")

    # 3. Generate centered collision box
    collision_half = (new_vmax - new_vmin) / 2.0
    collision_path = models_dir / 'bread_v41_collision.obj'
    write_centered_collision(collision_path, collision_half)

    # 4. Compute inertia for centered mesh
    MASS = 0.24  # kg
    (ixx, iyy, izz), dims = compute_centered_inertia(centered_verts, MASS)

    # 5. Write URDF
    urdf_path = models_dir / 'bread_v41.urdf'
    write_urdf(urdf_path, 'bread_v41.obj', 'bread_v41_collision.obj',
               MASS, (ixx, iyy, izz), dims)

    print(f"\nDone! All assets centered at origin → ready for IsaacGym.")


if __name__ == '__main__':
    main()
