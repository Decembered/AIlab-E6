# 3d-scene-pipeline

## Purpose

Build small 3D spatial intelligence demos using Open3D, Nerfstudio, gsplat, PyTorch3D, point clouds, meshes, RGB-D frames, or 3D Gaussian Splatting assets.

The target is a compact visualization or reasoning result, not large-scene reconstruction training.

## When To Use

Use this skill when:

- A demo needs 3D visualization, target localization, occupancy, or path-planning illustration.
- A scene asset needs conversion between point cloud, mesh, image, or 3DGS representation.
- A VLA / VLN route needs spatial context or obstacle reasoning.
- The user wants a hackathon-ready visual result.

## Inputs

- Small image set, RGB-D frames, point cloud, mesh, COLMAP output, or 3DGS asset
- Camera poses or intrinsics when available
- Target object or navigation goal
- Desired output: visualization, localization, top-down map, path, rendered view

## Outputs

- Rendered images or screenshots
- Point cloud / mesh / 3DGS visualization
- Optional target coordinate or bounding primitive
- Optional path-planning overlay
- Metrics when available, such as number of points, map size, path length, collision count
- Demo assets saved under `experiments/` or `demos/`

## Steps

1. Start with the smallest local asset.
2. Use Open3D for quick point cloud / mesh loading and visualization.
3. Use Nerfstudio or gsplat only when existing assets or tiny examples are available.
4. Normalize coordinate frames and units.
5. Generate one or more views that clearly show the scene and target.
6. For navigation, create a simple occupancy or top-down projection.
7. For planning, draw a heuristic path or planner output and label it as a planning illustration unless it is executed.
8. Save figures and machine-readable outputs.

## Constraints

- Do not launch large 3DGS or NeRF training by default.
- Do not download large scenes without approval.
- Do not treat pretty renders as evidence of metric success.
- Always record coordinate frame assumptions.
- Prefer saved visual outputs that can be used in slides.

## Failure Debugging

- If Open3D import fails, run `sim-env-debugger`.
- If visualization cannot open a window, use offscreen rendering or save `.ply`, `.obj`, `.png`, or `.html`.
- If point clouds are misaligned, inspect units, pose convention, and coordinate handedness.
- If 3DGS render fails, check CUDA extension build, PyTorch version, and checkpoint format.
- If path overlays look wrong, verify map resolution and coordinate transforms.

## Minimum Runnable Demo

The minimum demo is one of:

- Load a small point cloud or mesh and save a rendered view.
- Convert RGB-D or sample points into a point cloud and save `.ply` plus screenshot.
- Create a top-down obstacle map and draw a start-goal path.
- Render an existing tiny 3DGS or Nerfstudio scene without training.

