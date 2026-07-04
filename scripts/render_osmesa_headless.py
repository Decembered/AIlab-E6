#!/usr/bin/env python3
"""Headless mesh rendering using pyrender + OSMesa (software OpenGL).

No GPU, no EGL, no display server needed — OSMesa renders entirely on CPU.
Reliable for headless cluster environments.
"""
import argparse, sys, os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # Software rendering, no GPU needed

from pathlib import Path
import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', required=True, help='Path to OBJ file')
    parser.add_argument('--out', default='render_output', help='Output directory')
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=600)
    args = parser.parse_args()

    import pyrender

    obj_path = Path(args.obj).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    mesh = trimesh.load(str(obj_path))
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]

    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    extent = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])

    # Create pyrender mesh with bread color
    pymesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.82, 0.71, 0.55, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.7,
    )

    # Scene setup
    scene = pyrender.Scene(
        bg_color=[0.25, 0.25, 0.28, 1.0],
        ambient_light=[0.4, 0.4, 0.4],
    )

    # Add mesh centered
    centroid = mesh.centroid
    mesh_pose = np.eye(4)
    mesh_pose[:3, 3] = -centroid  # Center at origin
    mesh_node = scene.add(pymesh, pose=mesh_pose)

    # Add key light (directional)
    dlight = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=8.0)
    scene.add(dlight, pose=[[1, 0, 0, 0],
                             [0, 0.707, -0.707, 0],
                             [0, 0.707, 0.707, 0],
                             [0, 0, 0, 1]])

    # Add fill light
    flight = pyrender.DirectionalLight(color=[0.7, 0.7, 0.8], intensity=3.0)
    scene.add(flight, pose=[[0.5, 0, 0.866, 0],
                              [0, 1, 0, 0],
                              [-0.866, 0, 0.5, 0],
                              [0, 0, 0, 1]])

    # Camera
    camera = pyrender.PerspectiveCamera(
        yfov=np.pi / 3.0,
        aspectRatio=args.width / args.height,
    )
    cam_node = scene.add(camera)

    # Views
    dist = 0.25
    views = {
        "front":  np.array([0.0,   0.0,    dist]),
        "side":   np.array([dist,   0.0,    0.0]),
        "top":    np.array([0.0,    dist,   0.001]),
        "angle":  np.array([dist,   dist*0.6, dist*0.8]),
    }

    # Look-at helper
    def look_at(eye, target, up):
        z = eye - target
        z = z / np.linalg.norm(z)
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.eye(4)
        R[:3, 0] = x
        R[:3, 1] = y
        R[:3, 2] = z
        R[:3, 3] = eye
        return R

    target = np.zeros(3)
    up = np.array([0, 1, 0])

    print(f"Rendering with OSMesa (CPU) at {args.width}×{args.height}...")
    for name, eye in views.items():
        cam_pose = look_at(eye, target, up)
        scene.set_pose(cam_node, pose=cam_pose)

        r = pyrender.OffscreenRenderer(args.width, args.height)
        try:
            color, depth = r.render(scene)
            from PIL import Image
            img = Image.fromarray(color)
            out_path = out_dir / f"{name}.png"
            img.save(str(out_path))
            print(f"  ✅ {name}.png")
        finally:
            r.delete()

    # Validation
    from PIL import Image
    for name in views:
        p = out_dir / f"{name}.png"
        if p.exists():
            img = np.array(Image.open(p))
            unique = len(np.unique(img.reshape(-1, 3), axis=0))
            print(f"  {name}: {p.stat().st_size:,} bytes, {unique} unique colors")
        else:
            print(f"  {name}: MISSING")

    print("Done!")


if __name__ == '__main__':
    main()
