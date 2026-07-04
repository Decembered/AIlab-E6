#!/usr/bin/env python3
"""Headless mesh rendering using trimesh's built-in renderer.

trimesh.scene.Scene uses pyglet for rendering — works in pure CPU mode,
no EGL/OpenGL display needed. Falls back gracefully.
"""
import argparse, sys, os
from pathlib import Path
import numpy as np
import trimesh


def render_view(scene, cam_transform, resolution, out_path):
    """Render scene from a camera pose using trimesh's built-in renderer."""
    try:
        # trimesh uses pyglet behind the scenes for rasterization
        png_bytes = scene.save_image(resolution=resolution, visible=True)
        with open(out_path, 'wb') as f:
            f.write(png_bytes)
        return True
    except Exception as e:
        print(f"    trimesh render failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', required=True, help='Path to OBJ file')
    parser.add_argument('--out', default='render_output', help='Output directory')
    parser.add_argument('--res', type=int, default=1024, help='Output image resolution')
    args = parser.parse_args()

    obj_path = Path(args.obj).resolve()
    if not obj_path.exists():
        print(f"ERROR: {obj_path} not found")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    mesh = trimesh.load(str(obj_path))
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]

    # Ensure consistent color
    if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
        mesh.visual.vertex_colors = [209, 181, 140, 255]  # bread-like color

    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print(f"Bounds: {mesh.bounds[0]} → {mesh.bounds[1]}")

    # Create trimesh scene
    scene = trimesh.Scene(mesh)

    # Camera transforms: (name, (translation), (rotation_degrees_xyz))
    resolution = (args.res, args.res)
    views = [
        ("front", [0, 0, 0.25], [0, 0, 0]),
        ("side",  [0.25, 0, 0], [0, 90, 0]),
        ("top",   [0, 0.25, 0], [90, 0, 0]),
        ("angle", [0.15, 0.15, 0.15], [30, 45, 0]),
    ]

    for name, trans, rot in views:
        # Build camera-to-world transform
        # Convert degrees to rotation matrix
        rx = np.radians(rot[0])
        ry = np.radians(rot[1])
        rz = np.radians(rot[2])

        Rx = np.array([[1, 0, 0, 0],
                        [0, np.cos(rx), -np.sin(rx), 0],
                        [0, np.sin(rx), np.cos(rx), 0],
                        [0, 0, 0, 1]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry), 0],
                        [0, 1, 0, 0],
                        [-np.sin(ry), 0, np.cos(ry), 0],
                        [0, 0, 0, 1]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0, 0],
                        [np.sin(rz), np.cos(rz), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        rotation = Rz @ Ry @ Rx
        rotation[:3, 3] = trans

        out_path = out_dir / f"{name}.png"
        print(f"  Rendering {name}...")

        # Set camera
        scene.set_camera(angles=rot, distance=np.linalg.norm(trans), center=[0, 0, 0])

        success = render_view(scene, rotation, resolution, out_path)
        if not success:
            print(f"  ❌ {name} failed")
            continue

    # Check output
    for name in ["front", "side", "top", "angle"]:
        p = out_dir / f"{name}.png"
        if p.exists():
            size = p.stat().st_size
            print(f"  {name}.png: {size:,} bytes")
        else:
            print(f"  {name}.png: MISSING")

    print("Done!")


if __name__ == '__main__':
    main()
