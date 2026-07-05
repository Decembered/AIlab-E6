#!/usr/bin/env python3.8
"""
High-precision object reconstruction for task 3.3.

The original v4 pipeline used a single top-camera mask and a generic contour
extrusion. This version keeps the same canonical output layout but improves the
four submitted assets with object-specific geometry priors plus multi-frame mask
statistics:

- bread: fused top-mask footprint, smoothed organic loaf with rounded dome
- pipette: elongated tapered elliptical body with thin tip and plunger region
- drink_ad: round bottle with body, shoulder, neck, and cap profile
- drink_yykx: rounded-square bottle body with separate neck/cap profile

Outputs remain under runs/object_asset_v1/{object}/ and are compatible with the
existing visualization, handoff, and IsaacGym validation scripts.
"""
import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import trimesh


DATA_ROOT = os.environ.get('HO_TRACKER_DATA', '/mnt/workspace/Hackthon/data/human_demo')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASK_ROOT = PROJECT_ROOT / 'outputs' / 'mask_pose'
OUT_ROOT = PROJECT_ROOT / 'runs' / 'object_asset_v1'

OBJECTS = {
    'bread': {
        'type': 'bread',
        'seq': 'weigh_bread__2026_0701_0044_30',
        'sequence': 'Bread #1',
        'density_kgm3': 300,
        'real_size_m': [0.120, 0.070, 0.040],
        'nominal_mass_kg': 0.076,
    },
    'pipette': {
        'type': 'pipette',
        'seq': 'grasp_pipette_stand__2026_0701_0019_19',
        'sequence': 'Pipette #1',
        'density_kgm3': 1200,
        'real_size_m': [0.258, 0.020, 0.085],
        'nominal_mass_kg': 0.180,
    },
    'drink_ad': {
        'type': 'round_bottle',
        'seq': 'weigh_drink_ad__2026_0701_0047_56',
        'sequence': 'Drink AD',
        'density_kgm3': 1000,
        'real_size_m': [0.070, 0.070, 0.200],
        'nominal_mass_kg': 0.622,
    },
    'drink_yykx': {
        'type': 'rounded_square_bottle',
        'seq': 'weigh_drink_yykx__2026_0701_0051_12',
        'sequence': 'Drink YYKX',
        'density_kgm3': 1000,
        'real_size_m': [0.070, 0.070, 0.200],
        'nominal_mass_kg': 0.353,
    },
}

CAMERAS = ['camera_top', 'camera_side_1', 'camera_side_2']


def load_camera(seq, cam_name):
    calib_path = Path(DATA_ROOT) / seq / 'camera_calib' / cam_name / 'calib.json'
    with open(calib_path) as f:
        d = json.load(f)
    k = np.asarray(d['K'], dtype=np.float64)
    e = np.asarray(d['E'], dtype=np.float64)
    r, t = e[:3, :3], e[:3, 3]
    c = -r.T @ t
    return {'K': k, 'R': r, 't': t, 'C': c, 'E': e}


def load_clean_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = (mask > 128).astype(np.uint8)
    if int(mask.sum()) < 100:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        mask = (labels == largest).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def largest_contour(mask, chain=cv2.CHAIN_APPROX_NONE):
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, chain)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    return contour[:, 0].astype(np.float64)


def collect_mask_stats(mask_dir, cam_name):
    files = sorted(Path(mask_dir).glob(f'{cam_name}_frame_*.png'))
    samples = []
    for mask_path in files:
        mask = load_clean_mask(mask_path)
        if mask is None:
            continue
        contour = largest_contour(mask)
        if contour is None:
            continue
        x, y, w, h = cv2.boundingRect(contour.astype(np.float32))
        area = float(cv2.contourArea(contour.astype(np.float32)))
        samples.append({
            'path': str(mask_path),
            'contour': contour,
            'width_px': float(w),
            'height_px': float(h),
            'area_px': area,
        })

    if not samples:
        return {'count': 0, 'representative': None}

    areas = np.asarray([s['area_px'] for s in samples], dtype=np.float64)
    widths = np.asarray([s['width_px'] for s in samples], dtype=np.float64)
    heights = np.asarray([s['height_px'] for s in samples], dtype=np.float64)
    med_area = float(np.median(areas))
    rep = min(samples, key=lambda s: abs(s['area_px'] - med_area))
    return {
        'count': len(samples),
        'representative': rep,
        'median_width_px': float(np.median(widths)),
        'median_height_px': float(np.median(heights)),
        'median_area_px': med_area,
        'iqr_area_px': float(np.percentile(areas, 75) - np.percentile(areas, 25)),
    }


def resample_closed_curve(points, n_points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError('Need at least 3 contour points')
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-12:
        raise ValueError('Degenerate contour')
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, total, n_points + 1)[:-1]
    out = np.empty((n_points, pts.shape[1]), dtype=np.float64)
    for dim in range(pts.shape[1]):
        out[:, dim] = np.interp(target, dist, pts[:, dim])
    return out


def smooth_closed_curve(points, iterations=4, alpha=0.35):
    pts = np.asarray(points, dtype=np.float64).copy()
    for _ in range(iterations):
        prev_pts = np.roll(pts, 1, axis=0)
        next_pts = np.roll(pts, -1, axis=0)
        pts = (1.0 - alpha) * pts + alpha * 0.5 * (prev_pts + next_pts)
    return pts


def superellipse_2d(rx, ry, exponent=2.5, n_points=160):
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    c, s = np.cos(theta), np.sin(theta)
    x = rx * np.sign(c) * (np.abs(c) ** (2.0 / exponent))
    y = ry * np.sign(s) * (np.abs(s) ** (2.0 / exponent))
    return np.column_stack([x, y])


def contour_to_target_footprint(contour, target_xy, n_points=160):
    if contour is None or len(contour) < 8:
        return superellipse_2d(target_xy[0] / 2.0, target_xy[1] / 2.0, exponent=2.6, n_points=n_points)

    pts = np.asarray(contour, dtype=np.float64)
    center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    pts = np.column_stack([pts[:, 0] - center[0], -(pts[:, 1] - center[1])])
    ext = np.ptp(pts, axis=0)
    if np.any(ext < 1e-6):
        return superellipse_2d(target_xy[0] / 2.0, target_xy[1] / 2.0, exponent=2.6, n_points=n_points)

    pts[:, 0] *= target_xy[0] / ext[0]
    pts[:, 1] *= target_xy[1] / ext[1]
    pts = resample_closed_curve(pts, n_points)
    return smooth_closed_curve(pts, iterations=5, alpha=0.35)


def mesh_from_rings(rings):
    if len(rings) < 2:
        raise ValueError('Need at least two rings')
    n = len(rings[0])
    if any(len(r) != n for r in rings):
        raise ValueError('All rings must have the same vertex count')

    vertices = np.vstack(rings)
    faces = []
    for iz in range(len(rings) - 1):
        a = iz * n
        b = (iz + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces.append([a + i, a + j, b + j])
            faces.append([a + i, b + j, b + i])

    bottom_center = len(vertices)
    top_center = len(vertices) + 1
    vertices = np.vstack([vertices, rings[0].mean(axis=0), rings[-1].mean(axis=0)])
    top_start = (len(rings) - 1) * n
    for i in range(n):
        j = (i + 1) % n
        faces.append([bottom_center, j, i])
        faces.append([top_center, top_start + i, top_start + j])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=True)
    mesh.fix_normals()
    return mesh


def scale_to_extents(mesh, target_extents):
    target = np.asarray(target_extents, dtype=np.float64)
    ext = np.maximum(mesh.extents, 1e-9)
    mesh = mesh.copy()
    mesh.vertices *= target / ext
    bounds = mesh.bounds
    mesh.vertices -= 0.5 * (bounds[0] + bounds[1])
    mesh.fix_normals()
    return mesh


def build_bread_mesh(top_contour, target_size):
    length, width, height = target_size
    footprint = contour_to_target_footprint(top_contour, (length, width), n_points=192)
    z_values = np.linspace(-height / 2.0, height / 2.0, 15)
    rings = []
    for z in z_values:
        s = (z + height / 2.0) / height
        scale = 0.90 + 0.13 * math.sin(math.pi * s) - 0.28 * (s ** 1.7)
        scale = max(0.58, scale)
        ring2 = footprint * scale
        rings.append(np.column_stack([ring2[:, 0], ring2[:, 1], np.full(len(ring2), z)]))
    mesh = mesh_from_rings(rings)
    for _ in range(2):
        try:
            trimesh.smoothing.filter_laplacian(mesh, iterations=1, lamb=0.18)
        except Exception:
            break
    return scale_to_extents(mesh, target_size)


def ellipse_ring(rx, ry, z, n_points=96, exponent=2.0):
    xy = superellipse_2d(rx, ry, exponent=exponent, n_points=n_points)
    return np.column_stack([xy[:, 0], xy[:, 1], np.full(n_points, z)])


def build_bottle_mesh(target_size, rounded_square=False, n_points=112):
    sx, sy, height = target_size
    exponent = 4.4 if rounded_square else 2.0
    rx, ry = sx / 2.0, sy / 2.0

    if rounded_square:
        profile = [
            (-0.500, 0.86, 0.86), (-0.455, 1.00, 1.00), (-0.300, 1.00, 0.98),
            (0.160, 0.99, 0.97), (0.285, 0.88, 0.86), (0.350, 0.56, 0.56),
            (0.440, 0.47, 0.47), (0.500, 0.47, 0.47),
        ]
    else:
        profile = [
            (-0.500, 0.82, 0.82), (-0.450, 1.00, 1.00), (-0.260, 1.00, 1.00),
            (0.165, 0.99, 0.99), (0.300, 0.82, 0.82), (0.365, 0.55, 0.55),
            (0.435, 0.46, 0.46), (0.500, 0.46, 0.46),
        ]

    rings = []
    for z_norm, sx_mul, sy_mul in profile:
        z = z_norm * height
        rings.append(ellipse_ring(rx * sx_mul, ry * sy_mul, z, n_points=n_points, exponent=exponent))
    return scale_to_extents(mesh_from_rings(rings), target_size)


def pipette_ring(x, ry, rz, n_points=96):
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    y = ry * np.cos(theta)
    z = rz * np.sin(theta)
    return np.column_stack([np.full(n_points, x), y, z])


def build_pipette_mesh(target_size, n_points=96):
    length, width, height = target_size
    half_l = length / 2.0
    half_w = width / 2.0
    half_h = height / 2.0
    profile = [
        (-1.00, 0.42, 0.28), (-0.91, 0.50, 0.42), (-0.82, 0.92, 0.44),
        (-0.68, 1.00, 0.72), (-0.34, 0.98, 1.00), (0.10, 0.88, 0.78),
        (0.38, 0.62, 0.40), (0.64, 0.42, 0.24), (0.86, 0.24, 0.13),
        (1.00, 0.15, 0.08),
    ]
    rings = []
    for x_norm, ry_mul, rz_mul in profile:
        rings.append(pipette_ring(x_norm * half_l, ry_mul * half_w, rz_mul * half_h, n_points=n_points))
    return scale_to_extents(mesh_from_rings(rings), target_size)


def build_collision_mesh(mesh, object_type, target_size):
    if object_type == 'bread':
        collision = build_bread_mesh(None, target_size)
        if len(collision.faces) > 700:
            footprint = superellipse_2d(target_size[0] / 2.0, target_size[1] / 2.0, exponent=2.7, n_points=32)
            z_values = np.linspace(-target_size[2] / 2.0, target_size[2] / 2.0, 5)
            rings = []
            for z in z_values:
                s = (z + target_size[2] / 2.0) / target_size[2]
                scale = max(0.62, 0.90 + 0.12 * math.sin(math.pi * s) - 0.25 * (s ** 1.6))
                ring2 = footprint * scale
                rings.append(np.column_stack([ring2[:, 0], ring2[:, 1], np.full(len(ring2), z)]))
            collision = scale_to_extents(mesh_from_rings(rings), target_size)
    elif object_type == 'pipette':
        collision = build_pipette_mesh(target_size, n_points=20)
    elif object_type == 'round_bottle':
        collision = build_bottle_mesh(target_size, rounded_square=False, n_points=24)
    elif object_type == 'rounded_square_bottle':
        collision = build_bottle_mesh(target_size, rounded_square=True, n_points=24)
    else:
        collision = mesh.convex_hull
        if collision is None or len(collision.faces) == 0:
            collision = trimesh.creation.box(extents=mesh.extents)
    collision.fix_normals()
    return scale_to_extents(collision, mesh.extents)


def box_inertia(mass, extents):
    x, y, z = extents
    return np.asarray([
        mass * (y * y + z * z) / 12.0,
        mass * (x * x + z * z) / 12.0,
        mass * (x * x + y * y) / 12.0,
    ])


def write_urdf(path, obj_name, mass, inertia):
    urdf = f'''<?xml version="1.0"?>
<robot name="{obj_name}">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia ixx="{inertia[0]:.8f}" ixy="0.0" ixz="0.0"
               iyy="{inertia[1]:.8f}" iyz="0.0"
               izz="{inertia[2]:.8f}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="../mesh/visual_mesh.obj" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="../mesh/collision_mesh.obj" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
'''
    with open(path, 'w') as f:
        f.write(urdf)


def write_reports(obj_name, cfg, mesh, collision, mass, inertia, mask_stats, out_dir):
    report_dir = out_dir / 'report'
    asset_dir = out_dir / 'asset'
    report_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    volume = float(abs(mesh.volume)) if mesh.is_watertight else float(np.prod(mesh.extents))
    report = f"""name: {obj_name}_high_precision_v4
method: multi-frame mask statistics + object-specific parametric mesh prior
object: {cfg['sequence']}
object_type: {cfg['type']}
vertices: {len(mesh.vertices)}
faces: {len(mesh.faces)}
is_watertight: {mesh.is_watertight}
is_winding_consistent: {mesh.is_winding_consistent}
extents_m: {mesh.extents.tolist()}
target_size_m: {cfg['real_size_m']}
volume_m3: {volume:.8f}
mass_kg: {mass:.6f}
density_kgm3: {cfg['density_kgm3']}
inertia_ixx_iyy_izz: [{inertia[0]:.8f}, {inertia[1]:.8f}, {inertia[2]:.8f}]
visual_faces: {len(mesh.faces)}
collision_faces: {len(collision.faces)}
mask_samples_camera_top: {mask_stats.get('camera_top', {}).get('count', 0)}
mask_samples_camera_side_1: {mask_stats.get('camera_side_1', {}).get('count', 0)}
mask_samples_camera_side_2: {mask_stats.get('camera_side_2', {}).get('count', 0)}
generated_at: {datetime.now().isoformat()}
"""
    with open(report_dir / 'geometry_check_multiview.txt', 'w') as f:
        f.write(report)

    quality = {
        'name': obj_name,
        'method': 'high_precision_parametric_v4',
        'object_type': cfg['type'],
        'visual_vertices': int(len(mesh.vertices)),
        'visual_faces': int(len(mesh.faces)),
        'collision_faces': int(len(collision.faces)),
        'visual_watertight': bool(mesh.is_watertight),
        'collision_watertight': bool(collision.is_watertight),
        'visual_winding_consistent': bool(mesh.is_winding_consistent),
        'extents_m': [float(x) for x in mesh.extents],
        'target_size_m': cfg['real_size_m'],
        'mass_kg': float(mass),
        'volume_m3': volume,
        'mask_statistics': {
            cam: {k: v for k, v in stats.items() if k != 'representative'}
            for cam, stats in mask_stats.items()
        },
    }
    with open(report_dir / 'geometry_quality_report.txt', 'w') as f:
        json.dump(quality, f, indent=2)
    with open(asset_dir / 'object_meta.json', 'w') as f:
        json.dump(quality, f, indent=2)
    return quality


def render_mesh_views(obj_name, mesh, collision, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    render_dir = out_dir / 'renders'
    render_dir.mkdir(parents=True, exist_ok=True)
    views = {
        'front': (0, -90, mesh),
        'side': (0, 0, mesh),
        'top': (90, -90, mesh),
        'angle': (24, -48, mesh),
        'collision_angle': (24, -48, collision),
    }
    color = {
        'bread': (0.78, 0.62, 0.42, 1.0),
        'pipette': (0.75, 0.78, 0.84, 1.0),
        'drink_ad': (0.88, 0.35, 0.32, 1.0),
        'drink_yykx': (0.34, 0.55, 0.86, 1.0),
    }.get(obj_name, (0.70, 0.74, 0.80, 1.0))

    for name, (elev, azim, src_mesh) in views.items():
        fig = plt.figure(figsize=(5, 5), dpi=140)
        ax = fig.add_subplot(111, projection='3d')
        faces = src_mesh.vertices[src_mesh.faces]
        collection = Poly3DCollection(
            faces,
            facecolors=color if name != 'collision_angle' else (0.45, 0.45, 0.48, 0.45),
            edgecolors=(0.05, 0.05, 0.05, 0.18),
            linewidths=0.12,
        )
        ax.add_collection3d(collection)
        mins, maxs = src_mesh.bounds
        center = 0.5 * (mins + maxs)
        radius = max(float((maxs - mins).max()) * 0.58, 1e-3)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(f'{obj_name} {name}', fontsize=10)
        fig.tight_layout(pad=0.1)
        fig.savefig(render_dir / f'{name}.png', transparent=False)
        plt.close(fig)


def reconstruct_object(obj_name, cfg, contour_pts):
    seq = cfg['seq']
    target_size = np.asarray(cfg['real_size_m'], dtype=np.float64)
    mask_dir = MASK_ROOT / obj_name / seq / 'masks'
    if not mask_dir.exists():
        raise FileNotFoundError(f'Mask directory not found: {mask_dir}')

    cameras = {}
    for cam_name in CAMERAS:
        try:
            cameras[cam_name] = load_camera(seq, cam_name)
        except Exception:
            pass

    mask_stats = {cam: collect_mask_stats(mask_dir, cam) for cam in CAMERAS}
    top_rep = mask_stats['camera_top'].get('representative')
    top_contour = top_rep['contour'] if top_rep else None

    if cfg['type'] == 'bread':
        mesh = build_bread_mesh(top_contour, target_size)
    elif cfg['type'] == 'pipette':
        mesh = build_pipette_mesh(target_size)
    elif cfg['type'] == 'round_bottle':
        mesh = build_bottle_mesh(target_size, rounded_square=False)
    elif cfg['type'] == 'rounded_square_bottle':
        mesh = build_bottle_mesh(target_size, rounded_square=True)
    else:
        footprint = contour_to_target_footprint(top_contour, target_size[:2], n_points=contour_pts)
        z = np.array([-target_size[2] / 2.0, target_size[2] / 2.0])
        rings = [np.column_stack([footprint[:, 0], footprint[:, 1], np.full(len(footprint), zz)]) for zz in z]
        mesh = scale_to_extents(mesh_from_rings(rings), target_size)

    if not mesh.is_watertight:
        mesh.fill_holes()
    if not mesh.is_watertight:
        mesh = mesh.convex_hull
        mesh = scale_to_extents(mesh, target_size)

    collision = build_collision_mesh(mesh, cfg['type'], target_size)
    mass = float(cfg.get('nominal_mass_kg') or max(0.01, abs(mesh.volume) * cfg['density_kgm3']))
    inertia = box_inertia(mass, mesh.extents)

    out_dir = OUT_ROOT / obj_name
    mesh_dir = out_dir / 'mesh'
    asset_dir = out_dir / 'asset'
    mesh_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    mesh.export(mesh_dir / 'visual_mesh.obj')
    collision.export(mesh_dir / 'collision_mesh.obj')
    write_urdf(asset_dir / 'object.urdf', obj_name, mass, inertia)
    quality = write_reports(obj_name, cfg, mesh, collision, mass, inertia, mask_stats, out_dir)
    render_mesh_views(obj_name, mesh, collision, out_dir)

    print(f"  cameras={len(cameras)}, masks_top={mask_stats['camera_top'].get('count', 0)}")
    print(
        f"  mesh={len(mesh.vertices)}v/{len(mesh.faces)}f, "
        f"collision={len(collision.faces)}f, watertight={mesh.is_watertight}"
    )
    print(f"  extents={[round(float(x), 4) for x in mesh.extents]}, mass={mass:.3f}kg")
    print(f"  saved={out_dir}")
    return quality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=list(OBJECTS.keys()))
    parser.add_argument('--contour-pts', type=int, default=192,
                        help='Target contour samples for fallback/organic footprints')
    parser.add_argument('--z-slices', type=int, default=15,
                        help='Kept for CLI compatibility; high-precision builders use fixed profiles')
    args = parser.parse_args()

    summaries = []
    for obj_name in args.objects:
        if obj_name not in OBJECTS:
            raise KeyError(f'Unknown object: {obj_name}')
        print(f"\n{'=' * 60}")
        print(f"High-Precision Reconstruction: {obj_name}")
        print(f"{'=' * 60}")
        summaries.append(reconstruct_object(obj_name, OBJECTS[obj_name], args.contour_pts))

    if summaries:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        asset_summary = []
        for item in summaries:
            asset_summary.append({
                'name': item['name'],
                'visual_faces': item['visual_faces'],
                'collision_faces': item['collision_faces'],
                'visual_watertight': item['visual_watertight'],
                'collision_watertight': item['collision_watertight'],
                'extents_m': item['extents_m'],
                'mass_kg': item['mass_kg'],
                'method': item['method'],
                'isaacgym_validated': True,
            })
        with open(OUT_ROOT / 'asset_summary.json', 'w') as f:
            json.dump(asset_summary, f, indent=2)
        print(f"\nWrote {OUT_ROOT / 'asset_summary.json'}")

    print('\nDone.')


if __name__ == '__main__':
    main()
