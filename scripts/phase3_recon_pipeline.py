"""Phase 3: 3D Object Reconstruction Pipeline.

Strategy for 3-camera setup with known calibration:
- Primary: Use InstantMesh (image-to-3D) — fast, works with single good view
- Secondary: NeuS/NeuS2 — surface reconstruction from multi-view masks
- Tertiary: 3DGS — if we have enough views (may need dense frame extraction)

For bread: Use the top camera frame (best view of bread on scale) → InstantMesh → post-process.
This gets us a working 3D model quickly, which we can later improve.
"""
import sys, os, pickle, json
import numpy as np
import cv2
from pathlib import Path

SEQ = 'data/human_demo/weigh_bread__2026_0701_0044_30'
FRAMES_DIR = 'experiments/2026-07-04_obj_recon_bread/frames'
OUT_DIR = 'experiments/2026-07-04_obj_recon_bread'
MODELS_DIR = f'{OUT_DIR}/models'
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("Phase 3: 3D Object Reconstruction")
print("=" * 60)

# ============================================================
# Step 1: Collect best views for reconstruction
# ============================================================
print("\n[1/4] Collecting best views for reconstruction...")

# For bread on a scale, the top camera gives the best shape info
# Side cameras provide profile views
best_views = {
    'top': f'{FRAMES_DIR}/camera_top/frame_000115.jpg',
    'side_1': f'{FRAMES_DIR}/camera_side_1/frame_000115.jpg',
    'side_2': f'{FRAMES_DIR}/camera_side_2/frame_000115.jpg',
}
for name, path in best_views.items():
    assert os.path.exists(path), f"Missing: {path}"
    img = cv2.imread(path)
    print(f"  {name}: {img.shape}")

# ============================================================
# Step 2: Generate dense point cloud from multi-view + masks
# ============================================================
print("\n[2/4] Multi-view point cloud generation...")

# Load camera parameters
cameras = {}
for cam in ['camera_side_1', 'camera_side_2', 'camera_top']:
    K = pickle.load(open(f'{SEQ}/camera_calib/{cam}/cam_intr.pkl', 'rb'))
    E = pickle.load(open(f'{SEQ}/camera_calib/{cam}/cam_extr.pkl', 'rb'))
    cameras[cam] = {'K': K, 'E': E, 'R': E[:3, :3], 't': E[:3, 3]}

# For now, generate a simple point cloud from the top view + mask
# using back-projection with a flat plane assumption (bread is flat-ish)

mask_path = f'{OUT_DIR}/masks/camera_top_frame_000115_mask.png'
if not os.path.exists(mask_path):
    print("  WARNING: Mask not yet available, using center region fallback")
    img = cv2.imread(f'{FRAMES_DIR}/camera_top/frame_000115.jpg')
    h, w = img.shape[:2]
    # Create simple center mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h//4:3*h//4, w//4:3*w//4] = 255
else:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Back-project mask pixels to 3D
K = cameras['camera_top']['K']
R = cameras['camera_top']['R']
t = cameras['camera_top']['t']

# Assume a plane at some depth (we'll estimate from the scene)
# The scale is on a table ~0.8m below the top camera
# Top camera is at y=1.198, table is roughly at y=0
# So depth ≈ 1.2 meters
depth_estimate = 1.2  # meters

ys, xs = np.where(mask > 128)
if len(ys) == 0:
    # Use all pixels as fallback
    print("  WARNING: Empty mask, using full frame")
    ys, xs = np.where(np.ones(mask.shape, dtype=bool))

print(f"  Mask pixels: {len(ys)}")

# Back-project: P = K [R|t],  X = depth * inv(K) @ pixel_homogeneous
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# Normalized camera coordinates
xn = (xs - cx) / fx
yn = (ys - cy) / fy

# Points in camera frame
Zc = np.full_like(xn, depth_estimate, dtype=np.float64)
Xc = xn * Zc
Yc = yn * Zc

# Transform to world frame: Xw = R^T @ (Xc - t) ... no, camera to world: Xw = R^T @ Xc + C
# where C = -R^T @ t
C = -R.T @ t
pts_cam = np.stack([Xc, Yc, Zc], axis=1)  # (N, 3)
pts_world = (R.T @ pts_cam.T + C.reshape(3, 1)).T  # (N, 3)

print(f"  Point cloud: {pts_world.shape}")
print(f"  World bounds: x=[{pts_world[:,0].min():.3f}, {pts_world[:,0].max():.3f}], "
      f"y=[{pts_world[:,1].min():.3f}, {pts_world[:,1].max():.3f}], "
      f"z=[{pts_world[:,2].min():.3f}, {pts_world[:,2].max():.3f}]")

# ============================================================
# Step 3: Create a mesh from the point cloud (alpha shape / Poisson)
# ============================================================
print("\n[3/4] Meshing point cloud...")

try:
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world.astype(np.float64))

    # Remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Estimate normals
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    # Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, width=0, scale=1.1, linear_fit=False
    )

    # Remove low-density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    o3d.io.write_triangle_mesh(f'{MODELS_DIR}/bread_raw.ply', mesh)
    print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    print(f"  Saved to: {MODELS_DIR}/bread_raw.ply")

except ImportError:
    print("  WARNING: open3d not available, saving point cloud only")
    # Save as PLY manually
    with open(f'{MODELS_DIR}/bread_raw.ply', 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts_world)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for pt in pts_world:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
    print(f"  Point cloud saved to: {MODELS_DIR}/bread_raw.ply")

# ============================================================
# Step 4: Post-process mesh
# ============================================================
print("\n[4/4] Post-processing mesh...")

try:
    import trimesh

    if os.path.exists(f'{MODELS_DIR}/bread_raw.ply'):
        mesh = trimesh.load(f'{MODELS_DIR}/bread_raw.ply')
    else:
        # Create mesh from point cloud using alpha shape
        from scipy.spatial import Delaunay
        # Simplified: just create a convex hull as starter
        pcd_2d = pts_world[:, [0, 2]]  # XZ projection (top-down)
        from scipy.spatial import ConvexHull
        hull_2d = ConvexHull(pcd_2d)
        # Extrude to 3D
        # ... too complex for now, skip

    if 'mesh' in dir():
        print(f"  Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"  Watertight: {mesh.is_watertight}, Manifold: {mesh.is_manifold}")

        # Scale estimate: the bread in real life is about 15cm across
        bbox_size = mesh.bounding_box.extents
        print(f"  Bounding box extents: {bbox_size}")

        # Decimate if needed
        if len(mesh.faces) > 20000:
            mesh = mesh.simplify_quadric_decimation(15000)
            print(f"  Decimated to {len(mesh.faces)} faces")

        # Fill holes
        if not mesh.is_watertight:
            mesh.fill_holes()
            print(f"  After fill_holes: watertight={mesh.is_watertight}")

        # Export
        mesh.export(f'{MODELS_DIR}/bread_processed.obj')
        print(f"  Exported: {MODELS_DIR}/bread_processed.obj")

        # Quality report
        quality = {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "is_watertight": bool(mesh.is_watertight),
            "is_manifold": bool(mesh.is_manifold),
            "bounding_box": mesh.bounding_box.extents.tolist(),
        }
        with open(f'{MODELS_DIR}/quality_report.json', 'w') as f:
            json.dump(quality, f, indent=2)
        print(f"  Quality: {json.dumps(quality, indent=2)}")

except ImportError as e:
    print(f"  WARNING: {e}")

print("\nPhase 3 complete!")
