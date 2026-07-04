"""IsaacGym asset generation from reconstructed object mesh.

Generates:
- URDF file with proper mass, inertia, collision mesh
- Simplified collision mesh (convex decomposition or decimated)
- Visual mesh link
- Optional articulated joint configuration (Object Bonus)
"""

from pathlib import Path
from typing import Optional
import json
import numpy as np


def generate_urdf(
    mesh_path: Path,
    output_dir: Path,
    object_name: str = "object",
    mass: Optional[float] = None,
    density: float = 1000.0,
    scale: tuple = (1.0, 1.0, 1.0),
    collision_mesh_path: Optional[Path] = None,
    joints: Optional[list] = None,
) -> Path:
    """Generate URDF for IsaacGym from a reconstructed mesh.

    Args:
        mesh_path: Path to visual mesh (.obj/.stl).
        output_dir: Output directory for URDF and collision mesh.
        object_name: Name for the URDF object.
        mass: Explicit mass in kg. If None, computed from mesh volume * density.
        density: Density in kg/m^3 for mass estimation.
        scale: Scale factors (x, y, z).
        collision_mesh_path: Optional pre-computed collision mesh path.
        joints: Optional list of joint configurations (Object Bonus).

    Returns:
        Path to generated URDF file.
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required for asset generation")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    mesh = trimesh.load(mesh_path)

    # Compute mass from volume if not given
    if mass is None:
        volume = mesh.volume if mesh.is_watertight else mesh.bounding_box.volume
        mass = volume * density

    # Generate collision mesh if not provided
    if collision_mesh_path is None:
        collision_mesh_path = output_dir / f"{object_name}_collision.obj"
        collision_mesh = mesh.simplify_quadric_decimation(
            max(4, len(mesh.faces) // 5)
        )
        collision_mesh.export(collision_mesh_path)

    # Compute inertia (approximate with bounding box)
    extents = mesh.bounding_box.extents
    inertia = compute_box_inertia(mass, extents)

    # Build URDF
    urdf_path = output_dir / f"{object_name}.urdf"
    urdf_content = _build_urdf_xml(
        object_name=object_name,
        visual_mesh=mesh_path,
        collision_mesh=collision_mesh_path,
        mass=mass,
        inertia=inertia,
        scale=scale,
        joints=joints,
    )

    with open(urdf_path, "w") as f:
        f.write(urdf_content)

    return urdf_path


def compute_box_inertia(
    mass: float,
    extents: np.ndarray,
) -> np.ndarray:
    """Compute moment of inertia for a solid rectangular box.

    Args:
        mass: Mass in kg.
        extents: (3,) array of box dimensions (x, y, z) in meters.

    Returns:
        (3,) diagonal inertia (Ixx, Iyy, Izz).
    """
    x, y, z = extents
    ixx = (1.0 / 12.0) * mass * (y**2 + z**2)
    iyy = (1.0 / 12.0) * mass * (x**2 + z**2)
    izz = (1.0 / 12.0) * mass * (x**2 + y**2)
    return np.array([ixx, iyy, izz])


def generate_collision_mesh(
    mesh_path: Path,
    output_path: Path,
    method: str = "decimation",
    max_faces: int = 200,
) -> Path:
    """Generate a simplified collision mesh for IsaacGym.

    Args:
        mesh_path: Path to high-resolution mesh.
        output_path: Output path for collision mesh.
        method: 'decimation' or 'convex_decomposition'.
        max_faces: Maximum face count for collision mesh.

    Returns:
        Path to collision mesh.
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required")

    mesh = trimesh.load(mesh_path)

    if method == "decimation":
        collision = mesh.simplify_quadric_decimation(max_faces)
    elif method == "convex_decomposition":
        # Uses v-hacd or similar
        try:
            import vhacd
            decomposed = vhacd.compute_vhacd(mesh.vertices, mesh.faces)
            collision = decomposed[0]  # Use first convex part
        except ImportError:
            # Fall back to convex hull
            collision = mesh.convex_hull
    else:
        raise ValueError(f"Unknown method: {method}")

    collision.export(output_path)
    return output_path


def _build_urdf_xml(
    object_name: str,
    visual_mesh: Path,
    collision_mesh: Path,
    mass: float,
    inertia: np.ndarray,
    scale: tuple,
    joints: Optional[list] = None,
) -> str:
    """Build URDF XML string.

    Args:
        object_name: Name for the URDF object.
        visual_mesh: Path to visual mesh.
        collision_mesh: Path to collision mesh.
        mass: Mass in kg.
        inertia: (Ixx, Iyy, Izz) diagonal inertia.
        scale: (sx, sy, sz) visual scale.
        joints: Optional list of joint configs.

    Returns:
        URDF XML string.
    """
    # Base URDF with a single link
    urdf = f'''<?xml version="1.0"?>
<robot name="{object_name}">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia
        ixx="{inertia[0]:.8f}" ixy="0" ixz="0"
        iyy="{inertia[1]:.8f}" iyz="0"
        izz="{inertia[2]:.8f}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{visual_mesh}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{collision_mesh}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
      </geometry>
    </collision>
  </link>
'''

    # Add joints for articulated objects (Object Bonus)
    if joints:
        for i, joint_cfg in enumerate(joints):
            child_name = joint_cfg.get("name", f"link_{i+1}")
            urdf += f'''
  <link name="{child_name}">
    <inertial>
      <origin xyz="{joint_cfg.get('origin_xyz', '0 0 0')}" rpy="{joint_cfg.get('origin_rpy', '0 0 0')}"/>
      <mass value="{joint_cfg.get('mass', mass * 0.1):.6f}"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="{joint_cfg.get('name', f'joint_{i+1}')}" type="{joint_cfg.get('type', 'revolute')}">
    <parent link="base_link"/>
    <child link="{child_name}"/>
    <origin xyz="{joint_cfg.get('origin_xyz', '0 0 0')}" rpy="{joint_cfg.get('origin_rpy', '0 0 0')}"/>
    <axis xyz="{joint_cfg.get('axis_xyz', '0 0 1')}"/>
    <limit lower="{joint_cfg.get('lower_limit', -1.57)}" upper="{joint_cfg.get('upper_limit', 1.57)}"
           effort="{joint_cfg.get('effort', 10)}" velocity="{joint_cfg.get('velocity', 1.0)}"/>
    <dynamics damping="{joint_cfg.get('damping', 0.1)}" friction="{joint_cfg.get('friction', 0.1)}"/>
  </joint>
'''

    urdf += '\n</robot>\n'
    return urdf


def validate_asset(
    urdf_path: Path,
) -> dict:
    """Validate that the URDF asset can be loaded in IsaacGym.

    Args:
        urdf_path: Path to URDF file.

    Returns:
        dict with keys: loadable, error, asset_options
    """
    try:
        from isaacgym import gymapi, gymutil

        gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False

        asset = gym.load_asset(sim, str(Path(urdf_path).parent), str(Path(urdf_path).name), asset_options)

        return {
            "loadable": True,
            "error": None,
            "num_bodies": gym.get_asset_rigid_body_count(asset),
            "num_shapes": gym.get_asset_rigid_shape_count(asset),
        }
    except Exception as e:
        return {
            "loadable": False,
            "error": str(e),
        }
