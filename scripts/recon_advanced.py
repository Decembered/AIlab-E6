#!/usr/bin/env python3.8
"""
Advanced 3D Object Reconstruction V2.

Per-object reconstruction strategies:
- drink_*: cylinder (circle-fit top mask → radius, side mask → height)
- bread:   organic extrusion (top mask contour + tapered height)
- pipette: variable-section (top mask contour + side mask width profile)
"""
import os, sys, json, argparse, math, shutil
from pathlib import Path
import numpy as np
import cv2
import trimesh

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get('HO_TRACKER_DATA', '/mnt/workspace/Hackthon/data/human_demo')
MASK_ROOT = os.path.join(SCRIPT_DIR, 'outputs', 'mask_pose')
OUT_ROOT = os.path.join(SCRIPT_DIR, 'runs', 'object_asset_v1')

sys.path.insert(0, os.path.join(SCRIPT_DIR, 'src'))
from object_recon.pose_tracking import load_intrinsics, load_extrinsics

OBJECTS = {
    'bread':      {'type': 'organic',          'seq': 'weigh_bread__2026_0701_0044_30',
                    'size': (0.12, 0.07, 0.04), 'density': 300,  'mass_ref': 0.4},
    'pipette':    {'type': 'variable_section',  'seq': 'grasp_pipette_stand__2026_0701_0019_19',
                    'size': (0.258, 0.02, 0.085), 'density': 1200, 'mass_ref': 0.15},
    'drink_ad':   {'type': 'cylinder',          'seq': 'weigh_drink_ad__2026_0701_0047_56',
                    'size': (0.07, 0.07, 0.20), 'density': 1000, 'mass_ref': 0.55},
    'drink_yykx': {'type': 'cylinder',          'seq': 'weigh_drink_yykx__2026_0701_0051_12',
                    'size': (0.07, 0.07, 0.20), 'density': 1000, 'mass_ref': 0.55},
}


def load_first_frame_mask(obj_name, seq_name, cam_name):
    p = os.path.join(MASK_ROOT, obj_name, seq_name, 'masks',
                     f'{cam_name}_frame_000000.png')
    if os.path.exists(p):
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            mask = (m > 128).astype(bool)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8)
            if num_labels > 1:
                largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                mask = (labels == largest)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)[:, 0] if contours else None
            return mask, contour
    return None, None


def load_object_center(obj_name, seq_name):
    p = os.path.join(MASK_ROOT, obj_name, seq_name, 'object_trajectory.json')
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        tf = d['trajectory'][0].get('transform_4x4')
        if tf:
            return np.array([tf[0][3], tf[1][3], tf[2][3]])
    return np.array([0.05, 0.35, 0.05])


def px_to_m(pixel_size, cam_name, seq_name, obj_pos):
    K = np.asarray(load_intrinsics(Path(DATA_ROOT) / seq_name / 'camera_calib', cam_name), dtype='float64')
    E = np.asarray(load_extrinsics(Path(DATA_ROOT) / seq_name / 'camera_calib', cam_name), dtype='float64')
    cam_center = -E[:3, :3].T @ E[:3, 3]
    depth = float(np.linalg.norm(obj_pos - cam_center))
    return float(pixel_size * depth / K[0, 0])


def build_cylinder(radius_m, height_m, n_slices=64):
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
    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    m.merge_vertices()
    return m


def build_organic(contour_2d, height_m, scale_xy=1.0, n_z=12, smooth=5):
    contour = contour_2d.copy().astype(np.float64)
    contour -= contour.mean(axis=0)
    contour *= scale_xy

    for _ in range(2):
        c2 = contour.copy()
        for i in range(len(contour)):
            c2[i] = (contour[(i - 1) % len(contour)] + contour[i] + contour[(i + 1) % len(contour)]) / 3.0
        contour = c2

    n_pts = len(contour)
    zs = np.linspace(-height_m / 2, height_m / 2, n_z + 1)
    verts = []
    for iz, z in enumerate(zs):
        scale = 1.0 - 0.5 * abs(z) / (height_m / 2) ** 1.5
        for pt in contour:
            verts.append([pt[0] * scale, z, pt[1] * scale])
    verts = np.array(verts)
    faces = []
    for iz in range(n_z):
        r0, r1 = iz * n_pts, (iz + 1) * n_pts
        for ip in range(n_pts):
            ip1 = (ip + 1) % n_pts
            faces.append([r0 + ip, r1 + ip, r1 + ip1])
            faces.append([r0 + ip, r1 + ip1, r0 + ip1])
    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    m.merge_vertices()
    for _ in range(smooth):
        m = trimesh.smoothing.filter_laplacian(m, iterations=1, lamb=0.5)
    return m


def build_variable_section(top_contour, side1_contour, expected_size):
    if top_contour is None or len(top_contour) < 5:
        top_contour = np.array([[-0.05, 0], [0.05, 0], [0.05, 0.01], [-0.05, 0.01]])

    contour = top_contour.copy().astype(np.float64)
    contour -= contour.mean(axis=0)
    contour[:, 0] *= expected_size[0] / max(contour[:, 0].ptp(), 0.001)
    contour[:, 1] *= expected_size[2] / max(contour[:, 1].ptp(), 0.001)

    rect = cv2.minAreaRect(contour.astype(np.float32))
    angle = -rect[2] * math.pi / 180
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotated = np.column_stack([
        contour[:, 0] * cos_a - contour[:, 1] * sin_a,
        contour[:, 0] * sin_a + contour[:, 1] * cos_a,
    ])

    n_sections = 64
    z_min, z_max = rotated[:, 0].min(), rotated[:, 0].max()
    zs = np.linspace(z_min, z_max, n_sections)
    height = expected_size[1]

    verts = []
    for iz, z in enumerate(zs):
        band = rotated[np.abs(rotated[:, 0] - z) < (zs[1] - zs[0]) * 1.2]
        if len(band) > 1:
            half_w = float(max(abs(band[:, 1].min()), abs(band[:, 1].max()), 0.001))
        else:
            half_w = expected_size[2] * 0.5
        y = (iz / max(1, n_sections - 1) - 0.5) * height
        verts.append([-half_w, y, z])
        verts.append([half_w, y, z])

    verts = np.array(verts)
    faces = []
    for i in range(n_sections - 1):
        a0, a1, b0, b1 = i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 3
        faces.append([a0, b0, b1])
        faces.append([a0, b1, a1])

    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    m.merge_vertices()
    return m


def build_collision(mesh, target=64):
    hull = mesh.convex_hull
    if len(hull.faces) > target * 2:
        try:
            from fast_simplification import simplify
            vs, fs = simplify(hull.vertices.astype(np.float32),
                              hull.faces.astype(np.int32), target_count=target)
            hull = trimesh.Trimesh(vertices=vs, faces=fs)
        except:
            pass
    return hull


def save(obj_name, mesh, density):
    out = os.path.join(OUT_ROOT, obj_name)
    os.makedirs(os.path.join(out, 'mesh'), exist_ok=True)
    os.makedirs(os.path.join(out, 'asset'), exist_ok=True)

    # Backup old
    old = os.path.join(out, 'mesh', 'visual_mesh.obj')
    bak = os.path.join(out, 'mesh', 'visual_mesh_backup.obj')
    if os.path.exists(old) and not os.path.exists(bak):
        shutil.copy(old, bak)

    mesh.export(old)
    col = build_collision(mesh)
    col.export(os.path.join(out, 'mesh', 'collision_mesh.obj'))

    vol = mesh.volume if mesh.is_watertight else np.prod(mesh.extents)
    mass = max(0.01, vol * density)
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
        obj_pos = load_object_center(obj_name, seq)

        print(f"\n{'='*60}")
        print(f"{obj_name} ({obj_type}) center=({obj_pos[0]:.3f},{obj_pos[1]:.3f},{obj_pos[2]:.3f})")
        print(f"{'='*60}")

        top_mask, top_contour = load_first_frame_mask(obj_name, seq, 'camera_top')
        s1_mask, s1_contour = load_first_frame_mask(obj_name, seq, 'camera_side_1')
        s2_mask, s2_contour = load_first_frame_mask(obj_name, seq, 'camera_side_2')

        for cam, info in [('top', top_mask), ('side_1', s1_mask), ('side_2', s2_mask)]:
            area = int(info.sum()) if info is not None else 0
            print(f"  mask_{cam}: {'OK' if area > 0 else 'MISSING'} area={area}")

        if obj_type == 'cylinder':
            if top_contour is None or len(top_contour) < 5:
                print("  SKIP: no top contour"); continue
            (cx, cy), radius_px = cv2.minEnclosingCircle(top_contour.astype(np.float32))
            radius_m = px_to_m(radius_px, 'camera_top', seq, obj_pos)
            radius_m = max(0.01, min(radius_m, expected[0] * 0.7))

            height_m = expected[1]
            if s1_contour is not None and len(s1_contour) > 4:
                h_px = s1_contour[:, 1].ptp()
                height_m = max(height_m, px_to_m(h_px, 'camera_side_1', seq, obj_pos))

            print(f"  cylinder: r={radius_m*1000:.0f}mm h={height_m*1000:.0f}mm")
            mesh = build_cylinder(radius_m, height_m)
            save(obj_name, mesh, cfg['density'])

        elif obj_type == 'organic':
            if top_contour is None or len(top_contour) < 5:
                print("  SKIP: no top contour"); continue
            w_px = max(top_contour[:, 0].ptp(), top_contour[:, 1].ptp())
            scale_xy = px_to_m(1.0, 'camera_top', seq, obj_pos)
            height_m = expected[1]
            if s1_contour is not None and len(s1_contour) > 4:
                h_px = s1_contour[:, 1].ptp()
                height_m = max(height_m, px_to_m(h_px, 'camera_side_1', seq, obj_pos) * 1.3)
            height_m = max(0.01, height_m)
            print(f"  organic: h={height_m*1000:.0f}mm scale_xy={scale_xy*1000:.2f}mm/px contour={len(top_contour)}pts")
            mesh = build_organic(top_contour, height_m, scale_xy)
            expected_scale = np.median(expected / np.maximum(mesh.extents, 1e-6))
            mesh.vertices *= expected_scale
            save(obj_name, mesh, cfg['density'])

        elif obj_type == 'variable_section':
            print(f"  variable-section: top_pts={len(top_contour) if top_contour is not None else 0}")
            mesh = build_variable_section(top_contour, s1_contour, expected)
            save(obj_name, mesh, cfg['density'])

    print(f"\nDone. {OUT_ROOT}")


if __name__ == '__main__':
    main()
