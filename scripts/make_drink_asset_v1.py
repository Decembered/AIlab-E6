from pathlib import Path
import json
import trimesh

out = Path("runs/object_asset_v1/drink_yykx")
mesh_dir = out / "mesh"
asset_dir = out / "asset"
report_dir = out / "report"

mesh_dir.mkdir(parents=True, exist_ok=True)
asset_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

# 单位：meter。先做一个可 IsaacGym 加载的瓶子几何保底版。
height = 0.20
radius_body = 0.035
radius_neck = 0.018
neck_height = 0.045
mass = 0.30

body = trimesh.creation.cylinder(
    radius=radius_body,
    height=height - neck_height,
    sections=64
)
body.apply_translation([0, 0, (height - neck_height) / 2])

neck = trimesh.creation.cylinder(
    radius=radius_neck,
    height=neck_height,
    sections=64
)
neck.apply_translation([0, 0, height - neck_height / 2])

visual = trimesh.util.concatenate([body, neck])
visual.apply_translation([0, 0, -height / 2])
visual.process(validate=True)

# collision mesh 要简单，仿真更稳定
collision = trimesh.creation.cylinder(
    radius=radius_body,
    height=height,
    sections=16
)
collision.process(validate=True)

visual_path = mesh_dir / "visual_mesh.obj"
collision_path = mesh_dir / "collision_mesh.obj"
visual.export(visual_path)
collision.export(collision_path)

urdf = f'''<?xml version="1.0"?>
<robot name="drink_yykx">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../mesh/visual_mesh.obj" scale="1 1 1"/>
      </geometry>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../mesh/collision_mesh.obj" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
'''
(asset_dir / "object.urdf").write_text(urdf, encoding="utf-8")

meta = {
    "object_name": "Drink YYKX",
    "unit": "meter",
    "mesh_file": "../mesh/visual_mesh.obj",
    "collision_file": "../mesh/collision_mesh.obj",
    "urdf_file": "object.urdf",
    "mass_kg": mass,
    "scale": [1.0, 1.0, 1.0],
    "coordinate": "mesh centered at origin; z-axis is bottle vertical axis",
    "method": "parametric bottle asset v1; to be refined by video masks / visual hull",
    "visual_faces": int(len(visual.faces)),
    "collision_faces": int(len(collision.faces)),
    "visual_watertight": bool(visual.is_watertight),
    "collision_watertight": bool(collision.is_watertight)
}
(asset_dir / "object_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def report(mesh, name):
    return f"""name: {name}
vertices: {len(mesh.vertices)}
faces: {len(mesh.faces)}
is_watertight: {mesh.is_watertight}
is_winding_consistent: {mesh.is_winding_consistent}
bounds:
{mesh.bounds}
extents: {mesh.extents}
center_mass: {mesh.center_mass}
volume: {mesh.volume}
"""

(report_dir / "geometry_check_visual.txt").write_text(report(visual, "visual_mesh"), encoding="utf-8")
(report_dir / "geometry_check_collision.txt").write_text(report(collision, "collision_mesh"), encoding="utf-8")

print("Generated asset:")
print("  ", visual_path)
print("  ", collision_path)
print("  ", asset_dir / "object.urdf")
print("visual faces:", len(visual.faces))
print("collision faces:", len(collision.faces))
