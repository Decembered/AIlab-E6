#!/usr/bin/env python3.8
"""
Multi-View Visual Hull Reconstruction (Voxel Carving)

Uses all 3 camera views' masks + camera calibration to reconstruct
proper 3D object models via voxel carving and marching cubes.

Input:  masks from mask_extraction_v2.py
Output: .obj files with 500+ faces, watertight, proper scale
"""
import os, sys, json, argparse
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
        'mass': 0.40, 'density': 300,
        'real_size_m': [0.12, 0.07, 0.04],
    },
    'pipette': {
        'seq': 'grasp_pipette_stand__2026_0701_0019_19',
        'mass': 0.15, 'density': 1200,
        'real_size_m': [0.258, 0.02, 0.085],
    },
    'drink_ad': {
        'seq': 'weigh_drink_ad__2026_0701_0047_56',
        'mass': 0.55, 'density': 1000,
        'real_size_m': [0.07, 0.07, 0.20],
    },
    'drink_yykx': {
        'seq': 'weigh_drink_yykx__2026_0701_0051_12',
        'mass': 0.55, 'density': 1000,
        'real_size_m': [0.07, 0.07, 0.20],
    },
}

VOXEL_RESOLUTION = 128  # 128^3 = 2M voxels, good for ~500-5000 face output
CAMERAS = ['camera_top', 'camera_side_1', 'camera_side_2']


def load_camera(seq, cam_name):
    with open(os.path.join(DATA_ROOT, seq, 'camera_calib', cam_name, 'calib.json')) as f:
        d = json.load(f)
    K = np.array(d['K'])
    E = np.array(d['E'])
    R, t = E[:3, :3], E[:3, 3]
    C = -R.T @ t
    return {'K': K, 'R': R, 't': t, 'C': C, 'P': K @ E[:3, :4]}


def project_to_pixel(P, point_3d):
    p = P @ np.append(point_3d, 1.0)
    if p[2] <= 1e-8:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]])


def build_visual_hull(cam_list, voxel_res, bounds):
    """
    Voxel carving: for each voxel in bounds, test if it projects inside
    the object mask in ALL camera views.
    
    cam_list: list of (cam_name, mask_image, projection_matrix_P)
    Returns voxel occupancy grid.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    # Add 10% margin to ensure we capture full object
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    z_pad = max(0.01, (z_max - z_min) * 0.1)
    x_min -= x_pad; x_max += x_pad
    y_min -= y_pad; y_max += y_pad
    z_min -= z_pad; z_max += z_pad
    
    xs = np.linspace(x_min, x_max, voxel_res)
    ys = np.linspace(y_min, y_max, voxel_res)
    zs = np.linspace(z_min, z_max, voxel_res)
    dx, dy, dz = xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]
    
    occupancy = np.ones((voxel_res, voxel_res, voxel_res), dtype=bool)
    
    for cam_name, mask_img, P in cam_list:
        h, w = mask_img.shape
        cam_occupancy = np.zeros((voxel_res, voxel_res, voxel_res), dtype=bool)
        
        # Process in XY slices for efficiency
        for iz, z in enumerate(zs):
            y_coords = np.broadcast_to(ys[:, None, None], (voxel_res, voxel_res, 1))
            x_coords = np.broadcast_to(xs[None, :, None], (voxel_res, voxel_res, 1))
            z_coords = np.full((voxel_res, voxel_res, 1), z)
            points = np.concatenate([x_coords, y_coords, z_coords, np.ones_like(x_coords)], axis=2)
            # points shape: (voxel_res, voxel_res, 1, 4)
            points_flat = points.reshape(-1, 4).T  # 4 x N
            
            # Project all points
            p = P @ points_flat  # 3 x N
            p_norm = p[:2] / np.maximum(p[2], 1e-8)
            
            # Check which pixels are within image and inside mask
            u = p_norm[0].reshape(voxel_res, voxel_res)
            v = p_norm[1].reshape(voxel_res, voxel_res)
            valid = (p[2] > 1e-8).reshape(voxel_res, voxel_res)
            
            # Quantize to pixel indices
            ui = np.clip(np.round(u).astype(int), 0, w-1)
            vi = np.clip(np.round(v).astype(int), 0, h-1)
            
            # Check mask
            in_mask = mask_img[vi, ui] > 128
            cam_occupancy[:, :, iz] = valid & in_mask
        
        occupancy &= cam_occupancy
    
    return occupancy, (xs, ys, zs), (dx, dy, dz)


def marching_cubes_mesh(occupancy, grid_xs, grid_ys, grid_zs, level=0.5):
    """Extract surface mesh from occupancy grid using simple marching."""
    from skimage import measure
    
    # skimage marching cubes expects (z, y, x) ordering
    occ_float = occupancy.astype(np.float32)
    spacing = (grid_zs[1]-grid_zs[0], grid_ys[1]-grid_ys[0], grid_xs[1]-grid_xs[0])
    
    try:
        verts, faces, _, _ = measure.marching_cubes(occ_float, level=level, spacing=spacing)
        # Shift vertices to world coordinates
        verts[:, 2] += grid_xs[0]
        verts[:, 1] += grid_ys[0]
        verts[:, 0] += grid_zs[0]
        # Reorder: (z,y,x) -> (x,y,z)
        verts = verts[:, [2, 1, 0]]
        return verts, faces
    except Exception as e:
        print(f"  marching_cubes failed: {e}")
        return None, None


def fallback_convex_hull(occupancy, grid_xs, grid_ys, grid_zs):
    """Fallback: extract occupied voxel centers and compute convex hull."""
    indices = np.argwhere(occupancy)
    if len(indices) < 4:
        return None, None
    
    points = np.column_stack([
        grid_xs[indices[:, 2]],
        grid_ys[indices[:, 1]],
        grid_zs[indices[:, 0]],
    ])
    
    hull = ConvexHull(points)
    return points, hull.simplices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=list(OBJECTS.keys()))
    parser.add_argument('--resolution', type=int, default=VOXEL_RESOLUTION,
                       help=f'Voxel grid resolution (default {VOXEL_RESOLUTION})')
    parser.add_argument('--target-faces', type=int, default=2000,
                       help='Target face count for final mesh')
    args = parser.parse_args()
    
    for obj_name in args.objects:
        cfg = OBJECTS[obj_name]
        seq = cfg['seq']
        real_size = cfg['real_size_m']
        
        print(f"\n{'='*60}")
        print(f"Visual Hull Reconstruction: {obj_name}")
        print(f"{'='*60}")
        
        # Load cameras
        cams = {}
        for cam_name in CAMERAS:
            cams[cam_name] = load_camera(seq, cam_name)
        print(f"  Cameras loaded: {list(cams.keys())}")
        
        # Load masks from best frame (frame 0)
        mask_dir = os.path.join(MASK_ROOT, obj_name, seq, 'masks')
        masks = {}
        for cam_name in CAMERAS:
            mask_path = os.path.join(mask_dir, f'{cam_name}_frame_000000.png')
            if not os.path.exists(mask_path):
                mask_files = sorted([f for f in os.listdir(mask_dir) if f.startswith(cam_name) and f.endswith('.png')])
                if mask_files:
                    mask_path = os.path.join(mask_dir, mask_files[0])
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None and (mask > 128).sum() > 50:
                    masks[cam_name] = mask
                    print(f"  {cam_name}: mask OK, {mask.shape}, {(mask>128).sum()} pixels")
                else:
                    print(f"  {cam_name}: mask too small")
            else:
                print(f"  {cam_name}: no mask found")
        
        if len(masks) < 2:
            print(f"  SKIP: need >=2 masks, got {len(masks)}")
            continue
        
        # Estimate workspace bounds from masks + cameras
        # Collect all rays from mask contours
        positions = []
        for cam_name, mask in masks.items():
            cam = cams[cam_name]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            for pt in contour[:, 0]:
                u, v = pt
                # Back-project to some depth
                d_cam = np.array([(u-cam['K'][0,2])/cam['K'][0,0], 
                                  (v-cam['K'][1,2])/cam['K'][1,1], 1.0])
                d_world = cam['R'].T @ d_cam
                d_world /= np.linalg.norm(d_world)
                # Intersect with Z=0 plane
                if abs(d_world[2]) > 0.01:
                    lam = -cam['C'][2] / d_world[2]
                    if lam > 0 and lam < 10:
                        pos = cam['C'] + lam * d_world
                        positions.append(pos)
        
        if len(positions) < 10:
            print(f"  SKIP: too few position estimates")
            continue
        
        positions = np.array(positions)
        
        # Bounds from positions + expand
        x_min, y_min, z_min = positions.min(axis=0) - 0.02
        x_max, y_max, z_max = positions.max(axis=0) + 0.02
        
        # Clamp height to be above table
        z_min = max(z_min, 0.0)
        
        print(f"  Bounds: X[{x_min:.3f},{x_max:.3f}] Y[{y_min:.3f},{y_max:.3f}] Z[{z_min:.3f},{z_max:.3f}]")
        
        # Build visual hull
        cam_list = [(name, masks[name], cams[name]['P']) for name in CAMERAS if name in masks]
        print(f"  Carving with {len(cam_list)} views, resolution={args.resolution}...")
        
        occupancy, grids, spacing = build_visual_hull(
            cam_list, args.resolution,
            (x_min, x_max, y_min, y_max, z_min, z_max))
        
        n_occupied = occupancy.sum()
        print(f"  Occupied voxels: {n_occupied} / {args.resolution**3} ({100*n_occupied/args.resolution**3:.1f}%)")
        
        if n_occupied < 10:
            print(f"  SKIP: too few occupied voxels")
            continue
        
        # Extract mesh
        verts, faces = marching_cubes_mesh(occupancy, grids[0], grids[1], grids[2])
        
        if verts is None or len(verts) < 4:
            print(f"  Falling back to convex hull...")
            verts, faces = fallback_convex_hull(occupancy, grids[0], grids[1], grids[2])
        
        if verts is None or len(verts) < 4:
            print(f"  SKIP: cannot extract surface")
            continue
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.fill_holes()
        mesh.merge_vertices()
        
        print(f"  Raw mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight}")
        
        # Decimate to target face count
        if len(mesh.faces) > args.target_faces * 2:
            try:
                from fast_simplification import simplify
                verts_s, faces_s = simplify(mesh.vertices, mesh.faces, target_count=args.target_faces)
                mesh = trimesh.Trimesh(vertices=verts_s, faces=faces_s)
            except ImportError:
                # Fallback: use trimesh's built-in
                pass
        
        # Scale to match real object size
        current_extents = mesh.extents
        target_extents = np.array(real_size)
        scale = target_extents / np.maximum(current_extents, 0.001)
        # Use median scale to avoid stretching
        median_scale = np.median(scale)
        mesh.vertices *= median_scale
        
        # Center
        mesh.vertices -= mesh.centroid
        
        # Ensure watertight
        if not mesh.is_watertight:
            try:
                mesh = mesh.convex_hull
            except:
                pass
        
        print(f"  Final mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
              f"watertight={mesh.is_watertight}, extents={[round(x,3) for x in mesh.extents]}")
        
        # Collision mesh (convex hull, simplified)
        collision = mesh.convex_hull
        if collision and len(collision.faces) > 100:
            try:
                from fast_simplification import simplify
                vs, fs = simplify(collision.vertices, collision.faces, target_count=64)
                collision = trimesh.Trimesh(vertices=vs, faces=fs)
            except:
                pass
        
        # Save
        out_dir = os.path.join(OUT_ROOT, obj_name)
        mesh_dir = os.path.join(out_dir, 'mesh')
        asset_dir = os.path.join(out_dir, 'asset')
        report_dir = os.path.join(out_dir, 'report')
        os.makedirs(mesh_dir, exist_ok=True)
        os.makedirs(asset_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        mesh.export(os.path.join(mesh_dir, 'visual_mesh.obj'))
        if collision:
            collision.export(os.path.join(mesh_dir, 'collision_mesh.obj'))
        else:
            mesh.export(os.path.join(mesh_dir, 'collision_mesh.obj'))
        
        # URDF with proper mass/inertia
        mass = cfg['mass']
        density = cfg['density']
        volume = mesh.volume if mesh.is_watertight else (mesh.extents[0] * mesh.extents[1] * mesh.extents[2])
        mass = max(0.01, volume * density)  # compute from volume
        
        e = mesh.extents
        ixx = mass * (e[1]**2 + e[2]**2) / 12
        iyy = mass * (e[0]**2 + e[2]**2) / 12
        izz = mass * (e[0]**2 + e[1]**2) / 12
        
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
        
        # Report
        report = f"""name: {obj_name}_visual_hull
method: multi-view voxel carving (visual hull) + marching cubes
cameras_used: {len(cam_list)}
voxel_resolution: {args.resolution}
raw_verts: {len(mesh.vertices)}
raw_faces: {len(mesh.faces)}
is_watertight: {mesh.is_watertight}
extents: {mesh.extents.tolist()}
target_size: {real_size}
mass_kg: {mass:.4f}
inertia: [{ixx:.6f}, {iyy:.6f}, {izz:.6f}]
approach: 3-view mask visual hull with marching cubes surface extraction
"""
        with open(os.path.join(report_dir, 'geometry_check_visual_hull.txt'), 'w') as f:
            f.write(report)
        
        print(f"  Saved to {out_dir}")
    
    print(f"\nDone. Output: {OUT_ROOT}")


if __name__ == '__main__':
    main()
