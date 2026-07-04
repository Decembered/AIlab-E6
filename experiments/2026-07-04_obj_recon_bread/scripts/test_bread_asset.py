from pathlib import Path
from isaacgym import gymapi

ASSET_ROOT = "/mnt/workspace/Hackthon/experiments/2026-07-04_obj_recon_bread/models"
ASSET_FILE = "bread.urdf"

asset_root = Path(ASSET_ROOT)
print("asset_root:", asset_root)
print("urdf exists:", (asset_root / ASSET_FILE).exists())
print("mesh exists:", (asset_root / "bread_extruded.obj").exists())

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.physx.use_gpu = True

sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)
assert sim is not None, "create_sim failed"

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.disable_gravity = False
asset_options.vhacd_enabled = True
asset_options.vhacd_params.resolution = 100000
asset_options.override_com = True
asset_options.override_inertia = True

asset = gym.load_asset(sim, str(asset_root), ASSET_FILE, asset_options)
assert asset is not None, "gym.load_asset failed"

env = gym.create_env(
    sim,
    gymapi.Vec3(-1.0, -1.0, 0.0),
    gymapi.Vec3(1.0, 1.0, 1.0),
    1,
)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.25)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

actor = gym.create_actor(env, asset, pose, "bread", 0, 1)
assert actor >= 0, "create_actor failed"

for _ in range(120):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

gym.destroy_sim(sim)
print("BREAD_ASSET_LOAD_OK")
