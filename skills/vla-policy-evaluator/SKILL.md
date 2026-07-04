# vla-policy-evaluator

## Purpose

Evaluate VLA or robot policy models such as OpenVLA, LeRobot policies, openpi, Octo-style models, or similar systems by analyzing the path from image plus language input to action output.

The focus is action semantics, deployability, safety, and integration with local planners or controllers.

## When To Use

Use this skill when:

- A VLA or policy model needs input / output analysis.
- The user wants to know whether a model can command a robot, manipulator, or drone.
- A demo needs to show language-conditioned action generation.
- We need a safety wrapper design around model actions.

## Inputs

- Model or repo name / path
- Image, video frame, robot observation, or simulator observation
- Language instruction
- Robot embodiment, for example arm, mobile robot, drone, dexterous hand
- Action space documentation, if available
- Control stack target, for example MAVROS, PX4, local planner, joint controller, end-effector controller

## Outputs

Create an evaluation report with:

- Model summary
- Required inputs and preprocessing
- Output action format
- Action units, frame, frequency, and horizon if known
- Whether actions are joint, end-effector, delta pose, velocity, waypoint, gripper, or tokenized
- Safety risks
- Required safety filter / planner / controller
- Minimal inference command
- Example predicted action saved as JSON
- Recommendation for hackathon use

## Steps

1. Inspect model docs and code to identify the exact observation and action interfaces.
2. Find preprocessing: image size, normalization, camera convention, language tokenizer, proprioception fields.
3. Find postprocessing: action unnormalization, discretization, chunking, control frequency, coordinate frame.
4. Run the smallest inference path when possible, using a tiny sample image or synthetic observation.
5. Save predicted action to `outputs/predicted_action.json`.
6. Explain action deployability:

   - Can it directly command the target embodiment?
   - What state estimator is required?
   - What planner or controller must sit downstream?
   - What safety checks must reject or modify actions?

7. For aerial navigation, map VLA outputs to safe high-level commands only, such as waypoint proposals or semantic goals.

## Constraints

- Default conclusion: VLA actions are not directly safe for real robots or drones.
- Always require a safety filter, local planner, controller, and emergency stop path for real execution.
- Do not run real hardware commands.
- Do not download large checkpoints without approval.
- Do not hide uncertainty about action units or coordinate frames.

## Failure Debugging

- If model loading fails, check checkpoint path, `transformers` version, tokenizer files, and GPU memory.
- If image preprocessing fails, verify RGB/BGR order, PIL vs OpenCV, dtype, shape, and normalization.
- If action dimensions mismatch, inspect dataset statistics and action unnormalization code.
- If output is tokenized, find the decoder or policy head that converts tokens to continuous actions.
- If action frame is unknown, label it unknown and do not propose direct execution.

## Minimum Runnable Demo

A valid minimum demo is:

- One image plus one instruction producing one action vector, action chunk, waypoint, or policy output
- A JSON explanation of each action dimension when known
- A safety wrapper diagram or pseudocode showing how unsafe actions are clipped, rejected, or converted to planner goals

