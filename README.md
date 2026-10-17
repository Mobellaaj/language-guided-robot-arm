# Language-Guided Robot Arm

A simulation-only learning project exploring how computer vision, large language models (LLMs), and ROS 2 can work together to control a robotic arm.

The intended demonstration is simple: give the robot a command such as **“Pick the red block and place it in the left tray”**, and have it identify the object, plan its movements, and execute the task in simulation.

## Motivation

I am a Robotics, Cognition, Intelligence master’s student at the Technical University of Munich. This project is practical preparation for working with camera-equipped robotic arms and natural-language instructions.

It builds on my previous ROS 2 mobile robot simulation project, where I explored sensors, SLAM, localization, and autonomous navigation.

The focus is understanding each component and integrating them progressively.

## Planned Demonstration

The initial environment will contain:

- A simulated robotic arm with a gripper.
- A fixed camera overlooking a table.
- Red, green, and blue blocks.
- Two named placement zones.
- A text interface for commands.

The project will integrate an existing LLM rather than train a language model from scratch.

## Intended Architecture

| Component | Responsibility |
|---|---|
| Gazebo | Simulate the robot, camera, objects, and physical interactions |
| Computer vision | Detect colored blocks and estimate their positions |
| LLM | Convert instructions into structured task requests |
| ROS 2 task node | Validate requests and coordinate manipulation skills |
| MoveIt 2 | Plan arm movements and check for collisions |
| Robot controllers | Execute arm and gripper commands |

Object positions will come from perception. The LLM will select detected objects and supported actions rather than generate joint commands directly.

## Learning Roadmap

- [ ] Launch an existing arm model and inspect its links and joints.
- [ ] Control individual joints and understand their limits.
- [ ] Explore forward and inverse kinematics.
- [ ] Plan movements to gripper pose targets.
- [ ] Pick and place a block at a known position.
- [ ] Add a simulated camera and detect colored blocks.
- [ ] Transform detected positions into the robot’s coordinate frame.
- [ ] Perform pick-and-place using camera observations.
- [ ] Integrate an LLM for typed instructions.
- [ ] Handle ambiguous commands and report execution failures.
- [ ] Document and record the complete demonstration.

## Planned Environment

- Ubuntu 24.04 through WSL2
- ROS 2 Jazzy
- Gazebo Harmonic
- MoveIt 2
- Python and OpenCV
- Existing LLM through a local runtime or API

The arm model, controller configuration, and LLM integration will be selected during development.

## Current Status

**Project planning and repository setup.**

No arm simulation, perception pipeline, or language integration has been implemented yet. Setup instructions and verified results will be added as milestones are completed.

## Learning Resources

The initial learning reference is the [MOGI-ROS Simple Arm tutorial](https://github.com/MOGI-ROS/Week-9-10-Simple-arm).

Additional references:

- [MoveIt 2 tutorials](https://moveit.picknik.ai/main/doc/tutorials/tutorials.html)
- [ROS 2 Jazzy documentation](https://docs.ros.org/en/jazzy/)
- [Gazebo Harmonic documentation](https://gazebosim.org/docs/harmonic/)

Any adapted tutorial code will retain its original license and attribution.
