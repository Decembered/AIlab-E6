#!/usr/bin/env python3.8
"""
Multi-view contour fitting V2: Back-project mask contours from 3 cameras,
intersect on the table plane, extrude upward for full 3D shape.

Better than single-view contour extrusion (current v1 models).
"""
import os, sys, json, argparse
import numpy as np
import cv2
import trimesh

DATA_ROOT = '/mnt/workspace/Hackthon/data/human_demo'
MASK_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'mask_pose')
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'object_asset_v2')

OBJECTS = {
    'bread':      {'seq': 'weigh_bread__2026_0701_0044_30',      'h': 0.05,  'mass': 0.08},
    'pipette':    {'seq': 'grasp_pipette_stand__2026_0701_0019_19', 'h': 0.09, 'mass': 0.08},
    'drink_ad':   {'seq': 'weigh_drink_ad__2026_0701_0047_56',    'h': 0.22, 'mass': 0.3},
    'drink_yykx': {'seq': 'weigh_drink_yykx__2026_0701_0051_12',  'h': 0.22, 'mass': 0.3},
}

CAMERAS = ['camera_top', 'camera_side_1', 'camera_side_2']


def load_camera(seq, cam):
    p = os.path.join(DATA_ROOT, seq, 'camera_calib', cam, 'calib.json')
    with open(p) as f:
        d = json.load(f)
    K = np.array(d['K'])
    E = np.array(d['E'])
    R, t = E[:3, :3], E[:3, 3]
    P = K @ np.hstack([R, t.reshape(3, 1)])
    return {'K': K, 'R': R, 't': t, 'P': P, 'C': -R.T @ t}


def mask_contour_to_3d(mask, K, R, t, plane_z=0.0):
    """Back-project mask contour pixels to a horizontal plane at z=plane_z."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    points_2d = largest.squeeze(1)
    if len(points_2d.shape) == 1:
        return None

    # For each contour point, find ray-plane intersection
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    C = -R.T @ t

    pts_3d = []
    for (u, v) in points_2d:
        # Ray direction in camera space
        d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        # Transform to world
        d_world = R.T @ d_cam
        d_world = d_world / np.linalg.norm(d_world)

        # Ray-plane intersection: C + λ*d, solve for z=plane_z
        if abs(d_world[2]) < 1e-10:
            continue
        lam = (plane_z - C[2]) / d_world[2]
        if lam <= 0:
            continue
        pts_3d.append(C + lam * d_world)

    if len(pts_3d) < 3:
        return None
    return np.array(pts_3d)


def multi_view_contour(mask_files, cameras, plane_z=0.0, height=0.1):
    """Intersect contours from multiple views on a plane, extrude upward."""
    all_contours = []
    for cam_name, mask_path in mask_files.items():
        if cam_name not in cameras:
            continue
        # Skip top camera for XY contour (perspective inflates footprint on plane)
        # Use only side_1 (better perspective for XY footprint on Z=0)
        if cam_name != 'camera_side_1':
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.sum() < 100:
            continue
        cam = cameras[cam_name]
        pts = mask_contour_to_3d((mask > 128).astype(np.uint8) * 255, cam['K'], cam['R'], cam['t'], plane_z)
        if pts is not None and len(pts) > 2:
            all_contours.append(pts[:, :2])

    if len(all_contours) < 2:
        # Fallback: use single best contour
        if len(all_contours) == 1:
            pts_2d = all_contours[0]
        else:
            return None
    else:
        # Intersect: use all contour points, compute convex hull
        all_pts = np.vstack(all_contours)
        # Remove outliers using IQR
        q1_x, q3_x = np.percentile(all_pts[:, 0], [25, 75])
        q1_y, q3_y = np.percentile(all_pts[:, 1], [25, 75])
        iqr_x, iqr_y = q3_x - q1_x, q3_y - q1_y
        inlier = (
            (all_pts[:, 0] > q1_x - 1.5 * iqr_x) & (all_pts[:, 0] < q3_x + 1.5 * iqr_x) &
            (all_pts[:, 1] > q1_y - 1.5 * iqr_y) & (all_pts[:, 1] < q3_y + 1.5 * iqr_y)
        )
        pts_2d = all_pts[inlier]
        if len(pts_2d) < 3:
            pts_2d = all_pts

    # Compute convex hull of points on plane
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts_2d)
    hull_pts = pts_2d[hull.vertices]

    # Extrude: bottom at plane_z, top at plane_z + height
    n = len(hull_pts)
    bottom = np.column_stack([hull_pts, np.full(n, plane_z)])
    top = np.column_stack([hull_pts, np.full(n, plane_z + height)])

    all_verts = np.vstack([bottom, top])
    # Faces: bottom, top, sides
    faces = []
    # Bottom face (reverse for outward normal)
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])
    # Top face
    for i in range(1, n - 1):
        faces.append([n, n + i, n + i + 1])
    # Side faces
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])

    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=all_verts, faces=faces)
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=list(OBJECTS.keys()))
    args = parser.parse_args()

    for obj_name in args.objects:
        cfg = OBJECTS[obj_name]
        seq = cfg['seq']
        height = cfg['h']
        mass = cfg['mass']

        print(f"\n{'='*60}")
        print(f"Multi-view Contour: {obj_name} ({seq})")
        print(f"{'='*60}")

        # Load cameras
        cameras = {cam: load_camera(seq, cam) for cam in CAMERAS}
        print(f"  Cameras loaded: {list(cameras.keys())}")

        # Find best frame with masks
        mask_dir = os.path.join(MASK_ROOT, obj_name, seq, 'masks')
        meta_file = os.path.join(MASK_ROOT, obj_name, seq, 'mask_meta.json')
        with open(meta_file) as f:
            meta = json.load(f)

        # Try frame 0 first, or use frame with highest avg score
        best_frame = 0
        best_score = 0
        for cam_data in meta.get('cameras', {}).values():
            for mask_key, mask_info in cam_data.get('masks', {}).items():
                score = mask_info.get('sam_score', 0)
                frame = mask_info.get('frame', 0)
                if score > best_score:
                    best_score = score
                    best_frame = frame

        # Build mask files
        mask_files = {}
        for cam_name in CAMERAS:
            mask_key = f'{cam_name}_frame_{best_frame:06d}'
            mask_path = os.path.join(mask_dir, f'{mask_key}.png')
            if os.path.exists(mask_path):
                mask_files[cam_name] = mask_path

        print(f"  Frame: {best_frame}, views: {len(mask_files)}")

        # Plane Z: estimate from camera centers (table ≈ Y=0.2-0.5 in camera coords)
        # Looking at camera positions, table plane is roughly Z=0 in world coords
        plane_z = 0.0

        mesh = multi_view_contour(mask_files, cameras, plane_z, height)
        if mesh is None:
            print(f"  Reconstruction failed")
            continue

        # Postprocess
        if not mesh.is_watertight:
            try:
                mesh.fill_holes()
            except:
                pass

        print(f"  Result: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight}")
        print(f"  Extents: {mesh.extents}")

        # Center at origin
        mesh.vertices -= mesh.centroid

        # Collision mesh: convex hull
        ch = mesh.convex_hull
        if ch is None:
            ch = trimesh.creation.box(extents=mesh.extents * 1.1)
        ch.vertices -= ch.centroid
        ch.vertices *= 1.05

        # Save
        out_dir = os.path.join(OUT_ROOT, obj_name)
        os.makedirs(out_dir, exist_ok=True)

        mesh.export(os.path.join(out_dir, f'{obj_name}_visual.obj'))
        ch.export(os.path.join(out_dir, f'{obj_name}_collision.obj'))

        # URDF
        e = mesh.extents
        urdf = f'''<?xml version="1.0"?>
<robot name="{obj_name}">
  <link name="base">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{out_dir}/{obj_name}_visual.obj"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{out_dir}/{obj_name}_collision.obj"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{(e[1]**2+e[2]**2)*mass/12:.6f}" ixy="0" ixz="0"
               iyy="{(e[0]**2+e[2]**2)*mass/12:.6f}" iyz="0"
               izz="{(e[0]**2+e[1]**2)*mass/12:.6f}"/>
    </inertial>
  </link>
</robot>'''
        with open(os.path.join(out_dir, f'{obj_name}.urdf'), 'w') as f:
            f.write(urdf)

        # Renders
        render_dir = os.path.join(out_dir, 'renders')
        os.makedirs(render_dir, exist_ok=True)
        try:
            scene = trimesh.Scene()
            m2 = mesh.copy()
            m2.visual.vertex_colors = [180, 185, 200, 255]
            scene.add_geometry(m2)
            for view, (fov, res) in {'front': (30, 1024), 'angle': (45, 1024)}.items():
                png = scene.save_image(resolution=(800, 600))
                with open(os.path.join(render_dir, f'{view}.png'), 'wb') as f:
                    f.write(png)
        except:
            pass

        # Report
        report = f"""name: {obj_name}_multiview_contour
method: multi-view mask contour fitting + extrusion
vertices: {len(mesh.vertices)}
faces: {len(mesh.faces)}
is_watertight: {mesh.is_watertight}
extents: {mesh.extents.tolist()}
volume: {mesh.volume}
"""
        with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
            f.write(report)

        print(f"  Saved to {out_dir}")

    print(f"\nDone. Output: {OUT_ROOT}")


if __name__ == '__main__':
    main()
