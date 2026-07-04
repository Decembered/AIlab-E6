#!/usr/bin/env python3.8
"""
Reconstruction V3: Use top-camera mask for XY footprint (pixel-to-meter scale),
side camera for height estimation. Simpler and more reliable than visual hull.

Key insight: top camera is nearly orthographic → pixel extent maps to world extent.
Side cameras provide height constraints via vertical pixel extent.
"""
import os, json, argparse
import numpy as np
import cv2
import trimesh

DATA_ROOT = '/mnt/workspace/Hackthon/data/human_demo'
MASK_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'mask_pose')
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'object_asset_v3')

OBJECTS = {
    'bread':      {'seq': 'weigh_bread__2026_0701_0044_30',      'mass': 0.08},
    'pipette':    {'seq': 'grasp_pipette_stand__2026_0701_0019_19', 'mass': 0.08},
    'drink_ad':   {'seq': 'weigh_drink_ad__2026_0701_0047_56',    'mass': 0.3},
    'drink_yykx': {'seq': 'weigh_drink_yykx__2026_0701_0051_12',  'mass': 0.3},
}


def load_camera(seq, cam):
    with open(os.path.join(DATA_ROOT, seq, 'camera_calib', cam, 'calib.json')) as f:
        d = json.load(f)
    K = np.array(d['K']); E = np.array(d['E'])
    R, t = E[:3, :3], E[:3, 3]
    C = -R.T @ t
    # Camera direction in world (Z axis of camera)
    cam_z = R[2, :]
    return {'K': K, 'R': R, 't': t, 'C': C, 'cam_z': cam_z}


def pixel_to_world_at_depth(u, v, K, R, t, depth):
    """Back-project pixel to world point at given depth from camera along -Z axis."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Ray in camera space (toward -Z)
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    d_cam = d_cam / np.linalg.norm(d_cam)
    # In camera space, Z axis points forward. The scene is at -Z from camera.
    # depth = distance along camera's view direction (-Z)
    d_world = R.T @ d_cam
    C = -R.T @ t
    # The ray goes from camera center in +d_cam direction in camera space
    # In world: C + λ * d_world, where λ is distance along ray
    # We want the point at distance=depth from camera along -Z_cam
    # For a pinhole camera pointing along +Z_cam, objects are in +Z direction
    # X_cam = R*X_world + t → X_world = R.T*(X_cam - t)
    # Camera center in world: C = -R.T * t
    # Point in camera coords at depth D: (X_cam, Y_cam, D) where (X_cam, Y_cam) come from pixel
    u_norm = (u - cx) / fx
    v_norm = (v - cy) / fy
    point_cam = np.array([u_norm * depth, v_norm * depth, depth])
    point_world = R.T @ (point_cam - t)
    return point_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=list(OBJECTS.keys()))
    args = parser.parse_args()

    for obj_name in args.objects:
        cfg = OBJECTS[obj_name]
        seq = cfg['seq']
        mass = cfg['mass']

        print(f"\n{'='*60}")
        print(f"Scaled Reconstruction: {obj_name} ({seq})")
        print(f"{'='*60}")

        # Load cameras
        cam_top = load_camera(seq, 'camera_top')
        cam_side = load_camera(seq, 'camera_side_1')

        # Find best frame
        mask_dir = os.path.join(MASK_ROOT, obj_name, seq, 'masks')
        meta_file = os.path.join(MASK_ROOT, obj_name, seq, 'mask_meta.json')
        with open(meta_file) as f:
            meta = json.load(f)

        best_frame = 0
        for cam_data in meta.get('cameras', {}).values():
            for mask_info in cam_data.get('masks', {}).values():
                frame = mask_info.get('frame', 0)
                if frame > best_frame:
                    best_frame = frame

        # Use frame 0 or whichever is clean
        frame = 0
        print(f"  Frame: {frame}")

        # Load top mask to get XY footprint
        top_mask_path = os.path.join(mask_dir, f'camera_top_frame_{frame:06d}.png')
        top_mask = cv2.imread(top_mask_path, cv2.IMREAD_GRAYSCALE)
        if top_mask is None or (top_mask > 128).sum() < 100:
            print(f"  Top mask too small or missing")
            continue

        mask_bin = (top_mask > 128).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)

        # Get pixel bounds and centroid
        xs = largest[:, 0, 0]
        ys = largest[:, 0, 1]
        u_min, u_max = xs.min(), xs.max()
        v_min, v_max = ys.min(), ys.max()
        u_center, v_center = (u_min + u_max) / 2, (v_min + v_max) / 2
        pixel_width = u_max - u_min
        pixel_height = v_max - v_min

        print(f"  Top mask pixels: ({u_min:.0f},{v_min:.0f}) to ({u_max:.0f},{v_max:.0f}), {pixel_width:.0f}×{pixel_height:.0f}")

        # Estimate depth: find where the ray from camera_top center intersects Y≈0 (ground)
        # Top camera looks approximately at table surface
        # For camera_top at C, looking toward table, the centroid ray intersection with Y=0 or some plane
        # We need the distance from camera to object
        # Compute 3D point at centroid using triangulation with side camera
        K_top = cam_top['K']
        R_top, t_top = cam_top['R'], cam_top['t']
        C_top = cam_top['C']

        # Back-project centroid pixel from top camera
        fx, fy = K_top[0, 0], K_top[1, 1]
        cx, cy = K_top[0, 2], K_top[1, 2]
        d_top_cam = np.array([(u_center - cx) / fx, (v_center - cy) / fy, 1.0])
        d_top_world = R_top.T @ d_top_cam
        d_top_world = d_top_world / np.linalg.norm(d_top_world)

        # Find intersection of top camera ray with Z≈0 (table plane roughly at Z=0)
        # The top camera is at Z=-0.348, looking upward (Z comp of ray direction ≈ 0.4)
        # Intersection with Z=0: λ * d_world[2] = 0 - C[2] → λ = -C[2]/d_world[2]
        if abs(d_top_world[2]) > 0.01:
            lam = -C_top[2] / d_top_world[2]
            obj_pos = C_top + lam * d_top_world
        else:
            lam = 1.0
            obj_pos = C_top + d_top_world

        depth = lam  # distance from camera to object center
        print(f"  Object depth from top cam: {depth:.3f}m, pos: ({obj_pos[0]:.3f},{obj_pos[1]:.3f},{obj_pos[2]:.3f})")

        # Pixel-to-meter scale at object depth
        scale_x = depth / fx
        scale_y = depth / fy

        # World XY extent
        obj_width_x = pixel_width * scale_x
        obj_width_y = pixel_height * scale_y
        print(f"  World XY from top: {obj_width_x:.3f}m × {obj_width_y:.3f}m")

        # Height estimation from side camera
        side_mask_path = os.path.join(mask_dir, f'camera_side_1_frame_{frame:06d}.png')
        side_mask = cv2.imread(side_mask_path, cv2.IMREAD_GRAYSCALE)

        obj_height = 0.05  # default
        if side_mask is not None and (side_mask > 128).sum() > 100:
            side_bin = (side_mask > 128).astype(np.uint8) * 255
            s_contours, _ = cv2.findContours(side_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if s_contours:
                s_largest = max(s_contours, key=cv2.contourArea)
                sy_min = s_largest[:, 0, 1].min()
                sy_max = s_largest[:, 0, 1].max()
                pixel_h = sy_max - sy_min

                # Scale: at depth from side camera, each pixel = depth/fy meters
                K_s = cam_side['K']
                s_fy = K_s[1, 1]
                C_s = cam_side['C']
                # Distance from side camera to object
                s_depth = np.linalg.norm(obj_pos - C_s)
                obj_height = pixel_h * s_depth / s_fy
                # Clamp to reasonable range
                obj_height = max(0.01, min(0.5, obj_height))
                print(f"  Side mask height: {pixel_h:.0f}px, depth={s_depth:.3f}m, height={obj_height:.3f}m")
        else:
            # Fallback: use scale from top camera
            print(f"  Side mask missing, using default height")

        # Build extruded mesh from contour shape
        # Simplify contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        poly_pts = approx.squeeze(1)

        # Convert polygon points to world XY (centered at origin)
        poly_world = np.column_stack([
            (poly_pts[:, 0] - u_center) * scale_x,
            (poly_pts[:, 1] - v_center) * scale_y,
        ])

        # Y is forward in world? Let's use the convention from camera extrinsic
        # The top camera's X axis in world roughly aligns with actual X (horizontal)
        # The top camera's Y axis points roughly in world -Y direction
        # So pixel X ≈ world X, pixel Y ≈ -world Y
        # Let's correct: swap sign of Y
        poly_world[:, 1] *= -1

        # Build hull of polygon
        from scipy.spatial import ConvexHull
        # Remove duplicates and ensure good hull
        if len(poly_world) > 30:
            # Sample every Nth point
            step = max(1, len(poly_world) // 20)
            poly_world = poly_world[::step]
        if len(poly_world) < 3:
            # Fallback: axis-aligned box
            corners = np.array([
                [-obj_width_x/2, -obj_width_y/2],
                [obj_width_x/2, -obj_width_y/2],
                [obj_width_x/2, obj_width_y/2],
                [-obj_width_x/2, obj_width_y/2],
            ])
            hull_pts = corners
        else:
            try:
                hull = ConvexHull(poly_world)
                hull_pts = poly_world[hull.vertices]
            except:
                hull_pts = poly_world

        # Extrude
        n = len(hull_pts)
        bottom_z = -obj_height / 2
        top_z = obj_height / 2

        bottom = np.column_stack([hull_pts, np.full(n, bottom_z)])
        top = np.column_stack([hull_pts, np.full(n, top_z)])

        all_verts = np.vstack([bottom, top])
        faces = []
        for i in range(1, n - 1):
            faces.append([0, i + 1, i])
        for i in range(1, n - 1):
            faces.append([n, n + i, n + i + 1])
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, n + j])
            faces.append([i, n + j, n + i])

        mesh = trimesh.Trimesh(vertices=all_verts, faces=np.array(faces))
        if not mesh.is_watertight:
            try:
                mesh.fill_holes()
            except:
                pass

        print(f"  Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight}")
        print(f"  Extents: {mesh.extents}")

        # Center
        mesh.vertices -= mesh.centroid

        # Collision mesh
        ch = mesh.convex_hull
        if ch is None:
            ch = trimesh.creation.box(extents=mesh.extents * 1.1)
        ch.vertices -= ch.centroid

        # Save
        out_dir = os.path.join(OUT_ROOT, obj_name)
        os.makedirs(out_dir, exist_ok=True)

        mesh.export(os.path.join(out_dir, f'{obj_name}_visual.obj'))
        ch.export(os.path.join(out_dir, f'{obj_name}_collision.obj'))

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

        # Render
        render_dir = os.path.join(out_dir, 'renders')
        os.makedirs(render_dir, exist_ok=True)
        try:
            scene = trimesh.Scene()
            m2 = mesh.copy()
            m2.visual.vertex_colors = [180, 185, 200, 255]
            scene.add_geometry(m2)
            png = scene.save_image(resolution=(800, 600))
            with open(os.path.join(render_dir, 'default.png'), 'wb') as f:
                f.write(png)
        except:
            pass

        # Report
        report = f"""name: {obj_name}_scaledv3
method: top-camera pixel-to-meter + contour extrusion
vertices: {len(mesh.vertices)}
faces: {len(mesh.faces)}
is_watertight: {mesh.is_watertight}
extents: {mesh.extents.tolist()}
pixel_width: {pixel_width}
pixel_height: {pixel_height}
object_depth: {depth}
estimated_height: {obj_height}
"""
        with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
            f.write(report)

        print(f"  Saved to {out_dir}")

    print(f"\nDone. Output: {OUT_ROOT}")


if __name__ == '__main__':
    main()
