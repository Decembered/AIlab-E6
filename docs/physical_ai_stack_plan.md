# Physical AI Mini-Stack Plan

## Goal

Prepare a small, reusable stack for a Physical AI hackathon demo that can run quickly, produce figures, report metrics, and support a clear presentation.

Best personal fit:

- VLA / VLN / Physical AI with a safety wrapper
- Aerial or embodied navigation with local planner and controller integration
- 3D spatial reasoning
- Small RL or imitation baselines when they help the story

## Recommended Agent Tools To Configure First

### Context7 MCP

Use for current API and library documentation lookup. It is useful when working with fast-moving libraries such as LeRobot, OpenVLA-style repos, ManiSkill, Genesis, Open3D, Nerfstudio, and gsplat.

Best for:

- Finding current install commands
- Checking API examples
- Avoiding stale snippets

### GitHub MCP

Use for repository search, issue inspection, PR summaries, and release notes.

Best for:

- Scouting Physical AI repos
- Checking whether a bug is known
- Finding examples and minimal demos
- Understanding repo maintenance health

### Playwright MCP

Use for web demo verification and screenshots.

Best for:

- Testing Gradio / Streamlit / HTML demos
- Capturing screenshots for slides
- Checking that UI assets render correctly

### Filesystem / Local Repo Access

Use as the default coding and inspection surface.

Best for:

- Reading cloned repos
- Running smoke tests
- Creating experiment logs
- Packaging demos

## Recommended Open-Source Stack

### LeRobot

Useful for robot learning datasets, policy training / evaluation examples, and hackathon-friendly embodied AI tooling.

Good for:

- VLA / imitation-learning baselines
- Dataset format inspection
- Robot policy evaluation
- Language-conditioned manipulation demos

### OpenVLA

Useful for understanding image plus language to robot action pipelines.

Good for:

- VLA action format analysis
- Safety wrapper demos
- Showing why a planner / controller layer is required

### openpi

Useful as a modern robot policy reference stack, especially for action generation and policy deployment patterns.

Good for:

- Policy input-output analysis
- Comparing action formats
- Understanding deployment assumptions

### ManiSkill

Useful for fast manipulation simulation and RL / imitation baselines.

Good for:

- PPO / SAC / BC smoke tests
- Language-conditioned manipulation tasks
- Success-rate and rollout-video demos

### Genesis

Useful for modern simulation and fast visual demos.

Good for:

- Lightweight simulation experiments
- Quick scene and control demos
- RL baseline exploration

### Open3D

Useful for point clouds, meshes, RGB-D processing, visualization, and quick spatial reasoning.

Good for:

- 3D spatial intelligence demos
- Point cloud visualization
- Obstacle maps
- Target localization illustrations

### gsplat

Useful for 3D Gaussian Splatting rendering and experiments when assets already exist.

Good for:

- 3DGS visualization
- Scene rendering demos
- Spatial reasoning over compact 3D assets

### Nerfstudio

Useful for NeRF / 3DGS workflows and scene visualization when using existing small examples.

Good for:

- Rendered scene demos
- Camera trajectory visualization
- 3D reconstruction storytelling

## What Not To Install First

Avoid these at the beginning unless the demo route requires them:

- Full Isaac Lab: powerful but heavy, version-sensitive, and time-consuming.
- Full large VLA checkpoints: useful later, but first confirm inference path and action semantics.
- Full large datasets: download time and storage cost are high; start with sample data.
- Full RoboTwin large-scale tasks: valuable benchmark, but too heavy for first smoke tests.
- Full 3DGS large-scale training: start from existing tiny assets or Open3D visualizations.
- Distributed RL systems: only after a single-machine baseline is stable.

## Route 1: VLA + Safety Wrapper for Embodied / Aerial Navigation

This is the most recommended route.

Core idea:

Use a VLA / VLN-style model or simulated policy to propose a high-level action, waypoint, semantic target, or local motion intent. Do not directly execute the raw action. Pass it through a safety wrapper and local planner before generating controller commands.

Why it fits:

- Strong match with prior UAV, SLAM, planning, control, PX4/MAVROS, FAST-LIO2, EGO-Planner, and OpenFly-Agent experience
- Avoids training huge models
- Produces clear demo visuals: image, language instruction, proposed action, safety filter, planned path, final command

Minimal demo:

1. Input: one scene image or simulated camera frame plus language instruction.
2. VLA / mock VLA output: waypoint, velocity proposal, or action vector.
3. Safety wrapper: geofence, max velocity, collision check, altitude bounds, yaw-rate limit.
4. Local planner: simple 2D / 3D waypoint projection or obstacle-avoidance path.
5. Output: accepted / rejected action, planned trajectory, metrics, and visualization.

Metrics:

- Safety rejection count
- Distance to goal
- Path length
- Collision check result
- Command clipping magnitude

## Route 2: ManiSkill / Genesis RL Baseline + Language-Conditioned Manipulation

Core idea:

Run a small simulation task and produce a baseline result using PPO, SAC, BC, random, or scripted policy. Add language-conditioned task labels or a simple instruction interface if full VLA integration is too heavy.

Why it fits:

- Clear RL and Physical AI relevance
- Easy to produce metrics and videos
- Can be kept small for hackathon time

Minimal demo:

1. Pick one manipulation task.
2. Run reset plus random/scripted rollout.
3. Run tiny PPO / SAC / BC baseline only if setup is stable.
4. Save reward curve, success rate, rollout video, and failure cases.
5. Package as Gradio / Streamlit or static HTML.

Metrics:

- Success rate
- Episode return
- Episode length
- Runtime
- Failure mode categories

## Route 3: 3DGS / Open3D Spatial Intelligence Demo

Core idea:

Use Open3D, gsplat, or Nerfstudio to create a small 3D visualization and show target localization or path-planning reasoning.

Why it fits:

- Strong visual impact
- Does not require training a large model
- Connects naturally to spatial intelligence, navigation, and robotics safety

Minimal demo:

1. Load a small point cloud, mesh, RGB-D sample, or tiny 3DGS scene.
2. Show 3D scene rendering.
3. Mark a target or obstacle.
4. Produce a top-down map or projected path.
5. Save screenshots and metrics.

Metrics:

- Number of points / objects
- Target coordinate
- Path length
- Obstacle clearance estimate
- Render time

## First Practical Sequence

1. Run `python scripts/check_physical_ai_env.py`.
2. Configure GitHub MCP and Context7 MCP.
3. Scout LeRobot, OpenVLA, openpi, ManiSkill, Genesis, Open3D, gsplat, and Nerfstudio with `physical-ai-repo-scout`.
4. Build the VLA safety-wrapper toy demo first, using mock actions if a real VLA checkpoint is too heavy.
5. Add a small Open3D visualization for spatial context.
6. Package outputs with `hackathon-demo-builder`.

