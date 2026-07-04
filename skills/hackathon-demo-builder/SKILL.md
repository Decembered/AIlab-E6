# hackathon-demo-builder

## Purpose

Package experiment results into a small, clear demo using Gradio, Streamlit, or static HTML.

The demo should show inputs, algorithm or model outputs, metrics, visualizations, and a concise conclusion.

## When To Use

Use this skill when:

- An experiment has enough output to present.
- A VLA, RL, 3D, safety, or planner result needs an interactive demo.
- The user needs a hackathon booth or final presentation artifact.
- We need to turn logs and figures into a simple story.

## Inputs

- Experiment folder
- Input examples, such as image, language instruction, scene, or task
- Output files, such as actions, trajectories, metrics, screenshots, videos
- Preferred demo framework: Gradio, Streamlit, or static HTML
- Runtime constraints

## Outputs

- Demo app under `demos/demo_name/`
- `README.md` with run command and expected behavior
- `requirements.txt` or minimal dependency note when needed
- Visual assets copied or linked from experiments
- A conclusion panel or section with limitations

## Steps

1. Inspect the experiment folder and identify the clearest input-output story.
2. Choose the smallest demo framework:

   - Static HTML for saved outputs and no runtime inference
   - Gradio for simple interactive ML-style demos
   - Streamlit for dashboards and experiment comparisons

3. Create a demo folder under `demos/`.
4. Include sample inputs and saved outputs, or reference experiment paths.
5. Show metrics in a small table or JSON panel.
6. Show visualizations directly.
7. Include a limitations or safety note when actions could affect robots or drones.
8. Provide one run command.

## Constraints

- Do not make a marketing-only landing page. Build the usable demo first.
- Do not require large model downloads at demo startup unless approved.
- Do not hide failed cases; use them as diagnostics when useful.
- Keep UI simple enough to run during a hackathon.
- For robot or drone outputs, label generated actions as proposals unless a safety wrapper and controller are included.

## Failure Debugging

- If Gradio or Streamlit is missing, provide install commands instead of failing silently.
- If media files do not load, check relative paths and file sizes.
- If model inference is too slow, switch to saved-output replay mode.
- If the demo is visually unclear, add side-by-side input, output, metric, and conclusion panels.

## Minimum Runnable Demo

A minimum demo must:

- Launch locally or open as static HTML
- Show at least one input
- Show at least one model / algorithm output
- Show at least one metric or diagnostic
- Show one visualization
- State the main conclusion and limitation

