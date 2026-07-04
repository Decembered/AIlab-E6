from pathlib import Path
from isaacgym import gymapi

ASSET_ROOT = "/mnt/workspace/Hackthon/experiments/2026-07-04_obj_recon_bread/models"
ASSET_FILE = "bread.urdf"

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.physx.use_gpu = True

sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)

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

asset = gym.load_asset(sim, ASSET_ROOT, ASSET_FILE, asset_options)
assert asset is not None

env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.5)
pose.r = gymapi.Quat(0, 0, 0, 1)
actor = gym.create_actor(env, asset, pose, "bread", 0, 1)
assert actor >= 0

for _ in range(240):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

state = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
pose_state = state["pose"]["p"][0]
vel_state = state["vel"]["linear"][0]

print("final_position_xyz:", float(pose_state[0]), float(pose_state[1]), float(pose_state[2]))
print("final_linear_velocity_xyz:", float(vel_state[0]), float(vel_state[1]), float(vel_state[2]))
print("BREAD_DROP_TEST_OK")

gym.destroy_sim(sim)
