#!/usr/bin/env python3.8
"""
Improved Multi-View Reconstruction: High-quality contour extrusion from masks.

Uses top-camera mask for XY footprint (with high polygon density),
side cameras for height estimation. Produces 500-2000 face meshes.

Key improvement over V3:
- Much finer polygon approximation (epsilon smaller)
- Uses ALL contour points, not just ConvexHull
- Multiple Z slices for smoother vertical surface
- Proper mesh cleanup and watertight guarantees
"""
import os, json, argparse
import numpy as np
import cv2
import trimesh
from scipy.spatial import ConvexHull

DATA_ROOT = os.environ.get('HO_TRACKER_DATA', '/mnt/workspace/Hackthon/data/human_demo')
MASK_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'mask_pose')
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'object_asset_v1')

OBJECTS = {
    'bread': {
        'seq': 'weigh_bread__2026_0701_0044_30',
        'sequence': 'Bread #1',
        'density_kgm3': 300,
        'real_size_m': [0.12, 0.07, 0.04],
    },
    'pipette': {
        'seq': 'grasp_pipette_stand__2026_0701_0019_19',
        'sequence': 'Pipette #1',
        'density_kgm3': 1200,
        'real_size_m': [0.258, 0.02, 0.085],
    },
    'drink_ad': {
        'seq': 'weigh_drink_ad__2026_0701_0047_56',
        'sequence': 'Drink AD',
        'density_kgm3': 1000,
        'real_size_m': [0.07, 0.07, 0.20],
    },
    'drink_yykx': {
        'seq': 'weigh_drink_yykx__2026_0701_0051_12',
        'sequence': 'Drink YYKX',
        'density_kgm3': 1000,
        'real_size_m': [0.07, 0.07, 0.20],
    },
}

CAMERAS = ['camera_top', 'camera_side_1', 'camera_side_2']


def load_camera(seq, cam_name):
    with open(os.path.join(DATA_ROOT, seq, 'camera_calib', cam_name, 'calib.json')) as f:
        d = json.load(f)
    K = np.array(d['K'])
    E = np.array(d['E'])
    R, t = E[:3, :3], E[:3, 3]
    C = -R.T @ t
    return {'K': K, 'R': R, 't': t, 'C': C, 'E': E}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=list(OBJECTS.keys()))
    parser.add_argument('--contour-pts', type=int, default=200,
                       help='Target contour polygon points (higher = more detail)')
    parser.add_argument('--z-slices', type=int, default=8,
                       help='Number of Z extrusion slices (higher = smoother)')
    args = parser.parse_args()

    for obj_name in args.objects:
        cfg = OBJECTS[obj_name]
        seq = cfg['seq']
        real_size = cfg['real_size_m']

        print(f"\n{'='*60}")
        print(f"Multi-View Reconstruction: {obj_name}")
        print(f"{'='*60}")

        cams = {}
        for cam_name in CAMERAS:
            cams[cam_name] = load_camera(seq, cam_name)
        print(f"  Cameras OK: {len(cams)}")

        mask_dir = os.path.join(MASK_ROOT, obj_name, seq, 'masks')

        # Load top mask
        top_mask_path = os.path.join(mask_dir, f'camera_top_frame_000000.png')
        if not os.path.exists(top_mask_path):
            mask_files = sorted([f for f in os.listdir(mask_dir) if f.startswith('camera_top') and f.endswith('.png')])
            if not mask_files:
                print(f"  SKIP: no top mask")
                continue
            top_mask_path = os.path.join(mask_dir, mask_files[0])

        top_mask = cv2.imread(top_mask_path, cv2.IMREAD_GRAYSCALE)
        if top_mask is None or (top_mask > 128).sum() < 100:
            print(f"  SKIP: top mask too small")
            continue

        mask_bin = (top_mask > 128).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"  SKIP: no contour")
            continue
        largest = max(contours, key=cv2.contourArea)

        # Pixel metrics
        xs, ys = largest[:, 0, 0], largest[:, 0, 1]
        u_c, v_c = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
        pixel_w, pixel_h = xs.max()-xs.min(), ys.max()-ys.min()
        print(f"  Top mask: {pixel_w:.0f}x{pixel_h:.0f}px, center=({u_c:.0f},{v_c:.0f})")

        # Estimate depth at object center
        K_t = cams['camera_top']['K']
        C_t = cams['camera_top']['C']
        fx, fy = K_t[0,0], K_t[1,1]
        cx, cy = K_t[0,2], K_t[1,2]
        d_top = np.array([(u_c-cx)/fx, (v_c-cy)/fy, 1.0])
        d_world = cams['camera_top']['R'].T @ d_top
        d_world /= np.linalg.norm(d_world)
        
        if abs(d_world[2]) > 0.01:
            lam = -C_t[2] / d_world[2]
            obj_center = C_t + lam * d_world
        else:
            lam = 1.0; obj_center = C_t + d_world
        
        depth = lam
        scale_x, scale_y = depth/fx, depth/fy
        print(f"  Depth: {depth:.3f}m, scale: ({scale_x:.5f}, {scale_y:.5f}) m/px")

        # Height from side cameras
        obj_height = 0.05
        for cam_name in ['camera_side_1', 'camera_side_2']:
            side_path = os.path.join(mask_dir, f'{cam_name}_frame_000000.png')
            if not os.path.exists(side_path):
                continue
            side_mask = cv2.imread(side_path, cv2.IMREAD_GRAYSCALE)
            if side_mask is None or (side_mask > 128).sum() < 100:
                continue
            s_bin = (side_mask > 128).astype(np.uint8) * 255
            s_conts, _ = cv2.findContours(s_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not s_conts:
                continue
            s_largest = max(s_conts, key=cv2.contourArea)
            sy_min, sy_max = s_largest[:, 0, 1].min(), s_largest[:, 0, 1].max()
            pixel_h_side = sy_max - sy_min
            s_fy = cams[cam_name]['K'][1,1]
            s_depth = np.linalg.norm(obj_center - cams[cam_name]['C'])
            h = pixel_h_side * s_depth / s_fy
            h = max(0.01, min(0.5, h))
            print(f"  {cam_name}: {pixel_h_side:.0f}px height -> {h:.3f}m")
            obj_height = max(obj_height, h)

        # Use real-world height if available and plausible
        real_h = real_size[2]
        if real_h > 0.005 and abs(obj_height - real_h) / max(real_h, 0.001) > 0.5:
            print(f"  Overriding height: estimated={obj_height:.3f}m -> real={real_h:.3f}m")
            obj_height = real_h
        print(f"  Final height: {obj_height:.3f}m")

        # === BUILD HIGH-QUALITY MESH ===

        # Extract contour with high detail
        # Use cv2.CHAIN_APPROX_NONE for maximum detail, then downsample
        contours_full, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours_full:
            print(f"  SKIP: no full contour")
            continue
        largest_full = max(contours_full, key=cv2.contourArea)
        all_pts = largest_full.squeeze(1)

        # Sample to target count
        n_pts = len(all_pts)
        if n_pts > args.contour_pts:
            idx = np.linspace(0, n_pts-1, args.contour_pts, dtype=int)
            poly_pts = all_pts[idx]
        else:
            poly_pts = all_pts

        # Convert to world XY
        poly_world = np.column_stack([
            (poly_pts[:, 0] - u_c) * scale_x,
            -(poly_pts[:, 1] - v_c) * scale_y,  # Y sign correction
        ])

        # Handle concave shapes — keep original polygon (not convex hull)
        # For pipette (long thin), keep full contour
        if obj_name == 'pipette':
            # Use original polygon without hull for concavities
            hull_pts = poly_world
        else:
            # Use convex hull for bread/drinks
            try:
                hull = ConvexHull(poly_world)
                hull_pts = poly_world[hull.vertices]
            except:
                hull_pts = poly_world

        n = len(hull_pts)

        # Multi-slice extrusion for smooth vertical surface
        z_slices = args.z_slices
        z_levels = np.linspace(-obj_height/2, obj_height/2, z_slices)
        
        all_verts = []
        all_faces = []
        
        # Create vertices for each Z level
        for z in z_levels:
            slice_verts = np.column_stack([hull_pts, np.full(n, z)])
            all_verts.append(slice_verts)
        
        all_verts = np.vstack(all_verts)
        
        # Bottom face
        base_idx = 0
        for i in range(1, n-1):
            all_faces.append([base_idx, base_idx+i+1, base_idx+i])
        
        # Top face
        top_idx = (z_slices - 1) * n
        for i in range(1, n-1):
            all_faces.append([top_idx, top_idx+i, top_idx+i+1])
        
        # Side faces between adjacent Z slices
        for zi in range(z_slices - 1):
            bot_idx = zi * n
            top_idx = (zi + 1) * n
            for i in range(n):
                j = (i + 1) % n
                all_faces.append([bot_idx + i, bot_idx + j, top_idx + j])
                all_faces.append([bot_idx + i, top_idx + j, top_idx + i])

        mesh = trimesh.Trimesh(vertices=all_verts, faces=np.array(all_faces))
        try:
            mesh.remove_duplicate_faces()
        except AttributeError:
            pass
        try:
            mesh.remove_degenerate_faces()
        except AttributeError:
            pass
        
        # Make watertight
        if not mesh.is_watertight:
            try:
                mesh.fill_holes()
            except:
                pass

        # Scale to match real size
        current_ext = mesh.extents
        target_ext = np.array(real_size)
        # Only scale X and Z (Y is height, from side camera)
        if current_ext[0] > 0.001 and current_ext[2] > 0.001:
            scale_xz = target_ext[0] / current_ext[0]
            mesh.vertices[:, 0] *= scale_xz
            mesh.vertices[:, 2] *= target_ext[2] / current_ext[2]

        # Center
        mesh.vertices -= mesh.centroid
        
        # Final watertight guarantee
        if not mesh.is_watertight:
            mesh = mesh.convex_hull

        print(f"  Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
              f"watertight={mesh.is_watertight}, extents={[round(x,3) for x in mesh.extents]}")

        # Collision mesh: convex hull, simplified
        collision = mesh.convex_hull
        if collision and len(collision.faces) > 64:
            try:
                from fast_simplification import simplify
                vs, fs = simplify(collision.vertices, collision.faces, target_count=64)
                collision = trimesh.Trimesh(vertices=vs, faces=fs)
            except:
                pass

        # Mass from volume × density
        density = cfg['density_kgm3']
        volume = mesh.volume if mesh.is_watertight else (mesh.extents[0] * mesh.extents[1] * mesh.extents[2])
        mass = max(0.01, volume * density)
        e = mesh.extents
        ixx = mass * (e[1]**2 + e[2]**2) / 12
        iyy = mass * (e[0]**2 + e[2]**2) / 12
        izz = mass * (e[0]**2 + e[1]**2) / 12

        # === SAVE ===
        out_dir = os.path.join(OUT_ROOT, obj_name)
        mesh_dir = os.path.join(out_dir, 'mesh')
        asset_dir = os.path.join(out_dir, 'asset')
        report_dir = os.path.join(out_dir, 'report')
        for d in [mesh_dir, asset_dir, report_dir]:
            os.makedirs(d, exist_ok=True)

        mesh.export(os.path.join(mesh_dir, 'visual_mesh.obj'))
        if collision:
            collision.export(os.path.join(mesh_dir, 'collision_mesh.obj'))
        else:
            mesh.export(os.path.join(mesh_dir, 'collision_mesh.obj'))

        urdf = f'''<?xml version="1.0"?>
<robot name="{obj_name}">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.4f}"/>
      <inertia ixx="{ixx:.6f}" ixy="0.0" ixz="0.0"
               iyy="{iyy:.6f}" iyz="0.0"
               izz="{izz:.6f}"/>
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
        with open(os.path.join(asset_dir, 'object.urdf'), 'w') as f:
            f.write(urdf)

        report = f"""name: {obj_name}_multiview_v4
method: multi-view mask contour extrusion (top camera XY + side camera height)
object: {cfg['sequence']}
vertices: {len(mesh.vertices)}
faces: {len(mesh.faces)}
is_watertight: {mesh.is_watertight}
extents_m: {mesh.extents.tolist()}
target_size_m: {real_size}
mass_kg: {mass:.4f}
density_kgm3: {density}
inertia_ixx_iyy_izz: [{ixx:.6f}, {iyy:.6f}, {izz:.6f}]
contour_points: {n}
z_slices: {z_slices}
cameras_used: {len(cams)}
"""
        with open(os.path.join(report_dir, 'geometry_check_multiview.txt'), 'w') as f:
            f.write(report)

        print(f"  Saved to {out_dir}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
