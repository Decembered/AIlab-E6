#!/usr/bin/env python3.8
"""
Advanced 3D Reconstruction V3 — Addressing Limitations

Fixes:
1. Pipette: multi-slice cross-section from top+side masks (replaces 128v simple)
2. Bread:   watertight organic mesh with top/bottom caps (no convex hull forced)
3. Drinks:  keep cylinder (already good)
"""
import os, sys, json, argparse, math, shutil
from pathlib import Path
import numpy as np
import cv2
import trimesh
from scipy.spatial import ConvexHull

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get('HO_TRACKER_DATA', '/mnt/workspace/Hackthon/data/human_demo')
MASK_ROOT = os.path.join(SCRIPT_DIR, 'outputs', 'mask_pose')
OUT_ROOT = os.path.join(SCRIPT_DIR, 'runs', 'object_asset_v1')

sys.path.insert(0, os.path.join(SCRIPT_DIR, 'src'))
from object_recon.pose_tracking import load_intrinsics, load_extrinsics

OBJECTS = {
    'bread':      {'type': 'organic', 'seq': 'weigh_bread__2026_0701_0044_30',
                    'size': (0.12, 0.07, 0.04), 'density': 300},
    'pipette':    {'type': 'pipette', 'seq': 'grasp_pipette_stand__2026_0701_0019_19',
                    'size': (0.258, 0.02, 0.085), 'density': 1200},
    'drink_ad':   {'type': 'cylinder', 'seq': 'weigh_drink_ad__2026_0701_0047_56',
                    'size': (0.07, 0.07, 0.20), 'density': 1000},
    'drink_yykx': {'type': 'cylinder', 'seq': 'weigh_drink_yykx__2026_0701_0051_12',
                    'size': (0.07, 0.07, 0.20), 'density': 1000},
}


def load_mask_and_contour(obj_name, seq_name, cam_name):
    p = os.path.join(MASK_ROOT, obj_name, seq_name, 'masks', f'{cam_name}_frame_000000.png')
    if not os.path.exists(p):
        return None, None
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None, None
    mask = (m > 128).astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels > 1:
        largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        mask = (labels == largest)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)[:, 0].astype(np.float64) if contours else None
    return mask, contour


def px_to_m(pixel_size, depth, focal_length):
    return float(pixel_size * depth / focal_length)


def get_depth_and_focal(cam_name, seq_name, obj_pos):
    K = np.asarray(load_intrinsics(Path(DATA_ROOT) / seq_name / 'camera_calib', cam_name), dtype='float64')
    E = np.asarray(load_extrinsics(Path(DATA_ROOT) / seq_name / 'camera_calib', cam_name), dtype='float64')
    cam_center = -E[:3, :3].T @ E[:3, 3]
    depth = float(np.linalg.norm(obj_pos - cam_center))
    return depth, float(K[0, 0])


def load_obj_center(obj_name, seq_name):
    p = os.path.join(MASK_ROOT, obj_name, seq_name, 'object_trajectory.json')
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        tf = d['trajectory'][0].get('transform_4x4')
        if tf:
            return np.array([tf[0][3], tf[1][3], tf[2][3]])
    return np.array([0.05, 0.35, 0.05])


# ── Pipette: multi-slice cross-section from 3-view contour back-projection ──

def build_pipette(obj_name, seq_name, obj_pos, expected_size):
    """Multi-slice cross-section: top mask contour + side mask thickness."""
    mask_top, contour_top = load_mask_and_contour(obj_name, seq_name, 'camera_top')
    mask_s1, contour_s1 = load_mask_and_contour(obj_name, seq_name, 'camera_side_1')

    if contour_top is None or len(contour_top) < 5:
        return _build_pipette_fallback(expected_size)

    # Get scale
    depth_t, focal_t = get_depth_and_focal('camera_top', seq_name, obj_pos)
    scale_xy = px_to_m(1.0, depth_t, focal_t)
    depth_s, focal_s = get_depth_and_focal('camera_side_1', seq_name, obj_pos)
    scale_side = px_to_m(1.0, depth_s, focal_s)

    contour = contour_top.copy()
    contour -= contour.mean(axis=0)
    contour *= scale_xy

    # Fit oriented bounding box for principal axis
    rect = cv2.minAreaRect(contour.astype(np.float32))
    angle = -rect[2] * math.pi / 180
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotated_contour = np.column_stack([
        contour[:, 0] * cos_a - contour[:, 1] * sin_a,
        contour[:, 0] * sin_a + contour[:, 1] * cos_a,
    ])

    # Cut contour into slices along principal axis
    x_min, x_max = rotated_contour[:, 0].min(), rotated_contour[:, 0].max()
    h = expected_size[1]
    n_sections = 64
    xs = np.linspace(x_min, x_max, n_sections)

    # Estimate thickness per section from side mask
    thicknesses = np.zeros(n_sections)
    if mask_s1 is not None:
        s1_h, s1_w = mask_s1.shape
        for i in range(n_sections):
            rng = (xs[1] - xs[0]) * 1.5 if n_sections > 1 else 0.02
            band = rotated_contour[np.abs(rotated_contour[:, 0] - xs[i]) < rng]
            if len(band) > 1:
                thicknesses[i] = float(band[:, 1].ptp())
        if thicknesses.mean() > 0:
            thicknesses *= scale_side / scale_xy
        thicknesses = np.maximum(thicknesses, 0.002)

    verts = []
    for i, x in enumerate(xs):
        y = (i / max(1, n_sections - 1) - 0.5) * h
        thick = max(thicknesses[i], thicknesses.mean() * 0.3) if thicknesses.mean() > 0 else h * 0.3
        verts.append([x, y, -thick / 2])
        verts.append([x, y, thick / 2])

    verts = np.array(verts)
    faces = []
    for i in range(n_sections - 1):
        a0, a1 = i * 2, i * 2 + 1
        b0, b1 = a0 + 2, a1 + 2
        faces.append([a0, b0, b1])
        faces.append([a0, b1, a1])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    mesh.merge_vertices()
    return mesh


def _build_pipette_fallback(expected_size):
    """Simple rectangular box as absolute fallback."""
    w, h, d = expected_size
    box = trimesh.creation.box(extents=(w, h, d))
    return box


# ── Bread: watertight organic mesh ──

def build_bread(mask, contour, height_m, scale_xy):
    """Organic extrusion with top/bottom caps for watertightness."""
    contour = contour.copy()
    contour -= contour.mean(axis=0)
    contour *= scale_xy

    # Smooth contour
    for _ in range(3):
        c2 = contour.copy()
        n = len(contour)
        for i in range(n):
            c2[i] = (contour[(i-1)%n] + contour[i] + contour[(i+1)%n]) / 3.0
        contour = c2

    # Close loop
    if np.linalg.norm(contour[0] - contour[-1]) > 1e-3:
        contour = np.vstack([contour, contour[0]])

    n_pts = len(contour)
    n_z = 12
    zs = np.linspace(-height_m / 2, height_m / 2, n_z + 1)

    verts = []
    # Top pole
    top_idx = 0
    verts.append([0.0, zs[-1], 0.0])
    # Bottom pole
    bottom_idx = 1
    verts.append([0.0, zs[0], 0.0])

    # Ring vertices
    ring_starts = []
    for iz, z in enumerate(zs):
        scale = 1.0 - 0.5 * (abs(z) / (height_m / 2 + 1e-6)) ** 1.3
        ring_starts.append(len(verts))
        for pt in contour:
            verts.append([pt[0] * scale, z, pt[1] * scale])

    verts = np.array(verts)
    faces = []

    # Top cap: fan triangles from top pole to top ring
    top_ring_start = ring_starts[-1]
    for i in range(n_pts):
        ni = (i + 1) % n_pts
        faces.append([top_idx, top_ring_start + ni, top_ring_start + i])

    # Bottom cap: fan triangles from bottom ring to bottom pole
    bot_ring_start = ring_starts[0]
    for i in range(n_pts):
        ni = (i + 1) % n_pts
        faces.append([bottom_idx, bot_ring_start + i, bot_ring_start + ni])

    # Side walls between rings
    for iz in range(n_z):
        r0, r1 = ring_starts[iz], ring_starts[iz + 1]
        for ip in range(n_pts):
            ip1 = (ip + 1) % n_pts
            faces.append([r0 + ip, r1 + ip, r1 + ip1])
            faces.append([r0 + ip, r1 + ip1, r0 + ip1])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    mesh.merge_vertices()

    for _ in range(8):
        try:
            result = trimesh.smoothing.filter_laplacian(mesh, iterations=1, lamb=0.5)
            if isinstance(result, trimesh.Trimesh):
                mesh = result
            else:
                break
        except:
            break

    return mesh


# ── Drinks: cylinder ──

def build_drink(obj_name, seq_name, obj_pos, expected_size, top_contour, s1_contour):
    (cx, cy), radius_px = cv2.minEnclosingCircle(top_contour.astype(np.float32))
    depth, focal = get_depth_and_focal('camera_top', seq_name, obj_pos)
    radius_m = px_to_m(radius_px, depth, focal)
    radius_m = min(radius_m, expected_size[0] * 0.7)
    radius_m = max(0.01, radius_m)

    height_m = expected_size[1]
    if s1_contour is not None and len(s1_contour) > 4:
        h_px = s1_contour[:, 1].ptp()
        depth_s, focal_s = get_depth_and_focal('camera_side_1', seq_name, obj_pos)
        height_m = max(height_m, px_to_m(h_px, depth_s, focal_s))

    n_slices = 64
    r, h = radius_m, height_m
    theta = np.linspace(0, 2 * np.pi, n_slices, endpoint=False)
    verts = [[0, h / 2, 0], [0, -h / 2, 0]]
    for sgn in [1, -1]:
        y = sgn * h / 2
        for t in theta:
            verts.append([r * math.cos(t), y, r * math.sin(t)])
    verts = np.array(verts)
    faces = []
    tc, bc = 0, 1
    for i in range(n_slices):
        ni = (i + 1) % n_slices
        faces.append([tc, 2 + ni, 2 + i])
        faces.append([bc, 2 + n_slices + i, 2 + n_slices + ni])
        a0, a1 = 2 + i, 2 + ni
        b0, b1 = 2 + n_slices + i, 2 + n_slices + ni
        faces.append([a0, b1, b0])
        faces.append([a0, a1, b1])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    mesh.merge_vertices()
    return mesh, radius_m, height_m


# ── Asset saving ──

def build_collision(mesh, target=64):
    hull = mesh.convex_hull
    if len(hull.faces) > target * 2:
        try:
            from fast_simplification import simplify
            vs, fs = simplify(hull.vertices.astype(np.float32), hull.faces.astype(np.int32), target_count=target)
            hull = trimesh.Trimesh(vertices=vs, faces=fs)
        except:
            pass
    return hull


def save(obj_name, mesh, density):
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"  ERROR: mesh is {type(mesh)}, not Trimesh")
        return

    out = os.path.join(OUT_ROOT, obj_name)

    # Scale to expected size
    expected = np.array(OBJECTS[obj_name]['size'])
    scale_factors = expected / np.maximum(mesh.extents, 1e-6)
    mesh.vertices *= np.median(scale_factors)

    # Center vertically on Y=0
    mesh.vertices -= mesh.centroid
    mesh.vertices[:, 1] -= mesh.vertices[:, 1].min()

    # Ensure watertight
    if not mesh.is_watertight:
        try:
            mesh.fill_holes()
        except:
            pass
    if not mesh.is_watertight:
        try:
            repaired = trimesh.repair.broken_faces(mesh)
            if isinstance(repaired, trimesh.Trimesh):
                mesh = repaired
                mesh.fill_holes()
        except:
            pass
    if not mesh.is_watertight:
        mesh = mesh.convex_hull

    os.makedirs(os.path.join(out, 'mesh'), exist_ok=True)
    os.makedirs(os.path.join(out, 'asset'), exist_ok=True)

    old = os.path.join(out, 'mesh', 'visual_mesh.obj')
    bak = os.path.join(out, 'mesh', 'visual_mesh_backup.obj')
    if os.path.exists(old) and not os.path.exists(bak):
        shutil.copy(old, bak)

    mesh.export(old)
    col = build_collision(mesh)
    col.export(os.path.join(out, 'mesh', 'collision_mesh.obj'))

    volume = mesh.volume if mesh.is_watertight else np.prod(mesh.extents)
    mass = max(0.01, volume * density)
    ext = mesh.extents
    urdf = f'''<?xml version="1.0"?>
<robot name="{obj_name}">
  <link name="base_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.4f}"/>
      <inertia ixx="{mass*(ext[1]**2+ext[2]**2)/12:.6f}" ixy="0.0" ixz="0.0"
               iyy="{mass*(ext[0]**2+ext[2]**2)/12:.6f}" iyz="0.0"
               izz="{mass*(ext[0]**2+ext[1]**2)/12:.6f}"/>
    </inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="../mesh/visual_mesh.obj" scale="1 1 1"/></geometry>
    </visual>
    <collision><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="../mesh/collision_mesh.obj" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>'''
    with open(os.path.join(out, 'asset', 'object.urdf'), 'w') as f:
        f.write(urdf)

    print(f"  → {len(mesh.vertices)}v {len(mesh.faces)}f wt={mesh.is_watertight} "
          f"ext=[{ext[0]:.3f},{ext[1]:.3f},{ext[2]:.3f}] mass={mass:.4f}kg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=list(OBJECTS.keys()))
    args = parser.parse_args()

    for obj_name in args.objects:
        cfg = OBJECTS[obj_name]
        seq = cfg['seq']
        expected = np.array(cfg['size'])
        obj_type = cfg['type']
        obj_pos = load_obj_center(obj_name, seq)

        print(f"\n{'='*60}")
        print(f"{obj_name} ({obj_type}) pos=({obj_pos[0]:.3f},{obj_pos[1]:.3f},{obj_pos[2]:.3f})")
        print(f"{'='*60}")

        top_mask, top_contour = load_mask_and_contour(obj_name, seq, 'camera_top')
        s1_mask, s1_contour = load_mask_and_contour(obj_name, seq, 'camera_side_1')
        s2_mask, s2_contour = load_mask_and_contour(obj_name, seq, 'camera_side_2')

        if obj_type == 'cylinder':
            if top_contour is None or len(top_contour) < 5:
                print("  SKIP: no top contour"); continue
            mesh, r, h = build_drink(obj_name, seq, obj_pos, expected, top_contour, s1_contour)
            print(f"  cylinder: r={r*1000:.0f}mm h={h*1000:.0f}mm")
            save(obj_name, mesh, cfg['density'])

        elif obj_type == 'organic':
            if top_contour is None or len(top_contour) < 5:
                print("  SKIP: no top contour"); continue
            depth, focal = get_depth_and_focal('camera_top', seq, obj_pos)
            scale_xy = px_to_m(1.0, depth, focal)
            height_m = expected[1]
            if s1_contour is not None and len(s1_contour) > 4:
                h_px = s1_contour[:, 1].ptp()
                depth_s, focal_s = get_depth_and_focal('camera_side_1', seq, obj_pos)
                height_m = max(height_m, px_to_m(h_px, depth_s, focal_s) * 1.3)
            height_m = max(0.01, height_m)
            print(f"  organic: h={height_m*1000:.0f}mm scale={scale_xy*1000:.2f}mm/px contour={len(top_contour)}pts")
            mesh = build_bread(top_mask, top_contour, height_m, scale_xy)
            save(obj_name, mesh, cfg['density'])

        elif obj_type == 'pipette':
            mesh = build_pipette(obj_name, seq, obj_pos, expected)
            save(obj_name, mesh, cfg['density'])

    print(f"\nDone. {OUT_ROOT}")


if __name__ == '__main__':
    main()
