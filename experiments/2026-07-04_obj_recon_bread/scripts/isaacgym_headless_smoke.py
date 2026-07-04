from isaacgym import gymapi

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

for _ in range(120):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

gym.destroy_sim(sim)
print("IsaacGym headless smoke test OK")
