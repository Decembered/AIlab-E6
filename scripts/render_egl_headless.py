#!/usr/bin/env python3
"""Headless mesh rendering using pyrender + EGL on the cluster.

No IsaacGym needed — loads OBJ directly, renders from multiple views
using the GPU via EGL (headless OpenGL).
"""
import argparse, sys, os
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Must be set before importing pyrender

from pathlib import Path
import numpy as np
import trimesh


def render_view(scene, cam_pose, width, height, out_path):
    """Render a single view and save as PNG."""
    import pyrender

    r = pyrender.OffscreenRenderer(width, height)
    try:
        color, depth = r.render(scene)
        from PIL import Image
        img = Image.fromarray(color)
        img.save(str(out_path))
        print(f"  ✅ {out_path.name} saved ({width}×{height})")
    finally:
        r.delete()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', required=True, help='Path to OBJ file')
    parser.add_argument('--out', default='render_output', help='Output directory')
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=600)
    args = parser.parse_args()

    import pyrender
    import trimesh

    obj_path = Path(args.obj).resolve()
    if not obj_path.exists():
        print(f"ERROR: {obj_path} not found")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    mesh = trimesh.load(str(obj_path))
    if isinstance(mesh, trimesh.Scene):
        # Extract first mesh
        mesh = list(mesh.geometry.values())[0]

    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print(f"Bounds: {mesh.bounds[0]} → {mesh.bounds[1]}")

    # Center the mesh at origin for rendering
    centroid = mesh.centroid
    mesh.vertices -= centroid
    extent = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    print(f"Centroid: {centroid}, Extent: {extent:.3f}")

    # Create pyrender mesh with bread-like color
    pymesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.82, 0.71, 0.55, 1.0],  # bread color
        metallicFactor=0.0,
        roughnessFactor=0.7,
    )

    # Set up scene
    scene = pyrender.Scene(bg_color=[0.2, 0.2, 0.25, 1.0], ambient_light=[0.3, 0.3, 0.3])
    node = scene.add(pymesh, pose=np.eye(4))

    # Add lights
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light, pose=np.eye(4))

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=args.width / args.height)
    cam_node = scene.add(camera)

    # Render from multiple views
    dist = extent * 1.8
    views = {
        "front":  (0, 0, dist),
        "side":   (dist, 0, 0),
        "top":    (0, dist, 0.001),
        "angle":  (dist * 0.7, dist * 0.5, dist * 0.7),
    }

    for name, (cx, cy, cz) in views.items():
        # Camera looking at origin from (cx, cy, cz)
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = [cx, cy, cz]
        # Look-at matrix: camera at (cx,cy,cz) looking at (0,0,0)
        z_axis = np.array([-cx, -cy, -cz])
        z_axis = z_axis / np.linalg.norm(z_axis)
        # up vector
        if abs(z_axis[1]) > 0.99:
            up = np.array([0, 0, 1])
        else:
            up = np.array([0, 1, 0])
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        cam_pose[:3, 0] = x_axis
        cam_pose[:3, 1] = y_axis
        cam_pose[:3, 2] = z_axis

        scene.set_pose(cam_node, pose=cam_pose)
        out_path = out_dir / f"{name}.png"
        render_view(scene, cam_pose, args.width, args.height, out_path)

    # Also save a combined comparison image
    from PIL import Image
    imgs = {}
    for name in ["front", "side", "top", "angle"]:
        p = out_dir / f"{name}.png"
        if p.exists():
            imgs[name] = Image.open(p)

    if len(imgs) == 4:
        # 2x2 grid
        grid = Image.new('RGB', (args.width * 2, args.height * 2))
        grid.paste(imgs["front"], (0, 0))
        grid.paste(imgs["side"], (args.width, 0))
        grid.paste(imgs["top"], (0, args.height))
        grid.paste(imgs["angle"], (args.width, args.height))
        grid_path = out_dir / "combined_views.png"
        grid.save(str(grid_path))
        print(f"  Combined: {grid_path}")

    print("Done!")


if __name__ == '__main__':
    main()
