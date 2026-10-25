# Language-Guided Robot Arm

A simulation-only learning project using ROS 2 Jazzy, Gazebo Harmonic, Python, and OpenCV. The robot currently detects a red block with an overhead camera, estimates its position, solves inverse kinematics, and moves its gripper above it after an explicit execution command.

The longer-term goal is to select objects by color or shape using a simple natural-language instruction, then approach them. Language integration and camera-guided pick-and-place are future work.

## Motivation

I am a Robotics, Cognition, Intelligence master’s student at the Technical University of Munich. This project is practical preparation for working with camera-equipped robotic arms and natural-language instructions.

It builds on my previous ROS 2 mobile robot simulation project, where I explored sensors, SLAM, localization, and autonomous navigation. The focus is understanding each component and integrating it progressively with a simple custom robot.

## Current Status

- Custom arm with base rotation, shoulder, elbow, wrist pitch, and a parallel gripper.
- Primitive visual and collision geometry, link inertias, and joint limits.
- Gazebo joint-position controllers and ROS–Gazebo bridges.
- Wrist and fixed overhead RGB cameras, with camera information and TF frames.
- OpenCV red detection with an annotated image and pixel-center output.
- Camera-ray intersection with a known horizontal surface to estimate the red block’s position in `base_link`.
- Analytical inverse kinematics for a downward-facing `tool0`, checked against forward kinematics and joint limits.
- RViz previews of the IK candidate and the proposed motion.
- Explicit prepare, execute, and cancel services for a gradual approach above the detected block.

The end-to-end approach has been tested in simulation, including moving the red block and preparing another approach. This milestone moves above the object; it does not automatically grasp or place it.

## Repository Structure

| Package | Responsibility |
|---|---|
| `arm_description` | Robot Xacro, geometry, joints, inertias, and standalone RViz display |
| `arm_gazebo` | World, simulated sensors, joint controllers, bridges, simulation launch, and RViz configuration |
| `arm_perception` | Red detection and camera-based position estimation |
| `arm_control` | Forward/inverse kinematics, visualization, and explicit motion execution |

The current motion implementation uses Gazebo joint-position controllers directly. MoveIt 2 and an LLM are not part of the current execution pipeline.

## Build

The commands below assume Ubuntu 24.04 / WSL2, ROS 2 Jazzy, Gazebo Harmonic, and this repository under `~/ros2_arm_ws/src/`.

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_arm_ws
rosdep install --from-paths src --ignore-src --rosdistro jazzy -y
colcon build --packages-select arm_description arm_gazebo arm_perception arm_control
source install/setup.bash
```

## Run the Simulation and Perception

Use a separate terminal for each long-running command. Source the workspace in every terminal.

**Terminal 1 — Gazebo and RViz:**

```bash
source ~/ros2_arm_ws/install/setup.bash
ros2 launch arm_gazebo gazebo.launch.py
```

Gazebo starts paused. Press **Play** to advance physics, publish joint states, and produce camera images. RViz opens with the robot, cameras, detection, and motion-preview displays. To omit RViz, add `rviz:=false`.

**Terminal 2 — overhead-camera red detection and localization:**

```bash
source ~/ros2_arm_ws/install/setup.bash
ros2 launch arm_perception red_perception.launch.py
```

The overhead camera supplies the image used for localization. The wrist camera remains available for inspection. Perception is software processing these camera images, rather than another camera.

To inspect the estimated surface position in metres:

```bash
ros2 topic echo /arm/perception/red/position
```

## Preview and Move Above the Red Block

**Terminal 3 — motion node:**

```bash
source ~/ros2_arm_ws/install/setup.bash
ros2 run arm_control move_to_red --ros-args -p use_sim_time:=true
```

Starting this node does not move the arm.

**Terminal 4 — prepare an approach:**

```bash
source ~/ros2_arm_ws/install/setup.bash
ros2 service call /arm/prepare_red_move std_srvs/srv/Trigger '{}'
```

Inspect the animated ghost and tool path in RViz’s **Planned Motion** display. Preparation captures the current target and measured joint positions. By default, `tool0` is commanded 0.10 m above the estimated red surface, with the gripper pointing downward. `tool0` is the reference frame halfway along the fingers, not their tips.

If the preview is suitable, execute within 60 simulation seconds:

```bash
ros2 service call /arm/execute_red_move std_srvs/srv/Trigger '{}'
```

Watch Terminal 3 for completion or rejection details. To cancel and request a hold:

```bash
ros2 service call /arm/cancel_red_move std_srvs/srv/Trigger '{}'
```

After moving the block, prepare a new approach before executing again. The motion uses the captured target; it does not continuously chase the block. A significant observed target change during motion causes an abort.

For an IK-only candidate preview instead of execution, run:

```bash
ros2 run arm_control ik_preview --ros-args -p use_sim_time:=true
```

Stop that preview node before using `move_to_red` to keep the RViz overlays clear. During simulation, do not also run the standalone `display.launch.py`: its joint-state publisher would compete with Gazebo’s measured joint states.

## How the Pipeline Works

1. The overhead RGB image is thresholded in HSV to find the largest red region.
2. Its bounding-box center provides a pixel coordinate.
3. Camera intrinsics turn that pixel into a viewing ray; TF places the ray in the world.
4. The ray intersects the assumed red top-surface plane at world `z = 0.04 m`.
5. The intersection is expressed in `base_link` and published as the target surface point.
6. The control node adds clearance and solves for base, shoulder, elbow, and wrist angles.
7. After preparation and explicit execution, interpolated joint targets drive the Gazebo controllers while measured joint states provide feedback.

## Current Limits

- Localization uses an RGB camera and a known surface height, not measured depth. Different object heights need appropriate localization assumptions or a depth-based method.
- Only the red target is detected. Other colored shapes are present for later exercises.
- The four-joint arm supports the selected downward-facing approach, not arbitrary six-dimensional tool poses. Unreachable targets or joint-limit violations are rejected.
- The IK geometry constants match the current robot. Geometry changes require updating and checking those constants.
- Motion preparation checks joint limits and sampled floor clearance. It does not plan around other objects or check the full path for self-collisions.
- Execution checks target freshness, joint feedback, and tracking error. These checks are intended for this simulation exercise and are not a hardware safety system.
- Grasping, placing, general obstacle avoidance, and natural-language commands remain future work.

## Next Learning Steps

- Extend detection to selectable colors and then simple shapes.
- Handle their different geometry and heights when estimating positions.
- Reuse the approach pipeline for a selected detected object.
- Add a simple text-command interface before considering an LLM integration.
- Explore grasping and more complete motion planning as separate milestones.

## Learning Resources

- [Automatic Addison: Simulate a Robotic Arm in Gazebo with ROS 2 Jazzy](https://automaticaddison.com/how-to-simulate-a-robotic-arm-in-gazebo-ros-2-jazzy/)
- [MOGI-ROS Simple Arm tutorial](https://github.com/MOGI-ROS/Week-9-10-Simple-arm)
- [ROS 2 Jazzy documentation](https://docs.ros.org/en/jazzy/)
- [Gazebo Harmonic documentation](https://gazebosim.org/docs/harmonic/)
- [MoveIt 2 tutorials](https://moveit.picknik.ai/main/doc/tutorials/tutorials.html)

Any adapted tutorial code should retain its original license and attribution.
