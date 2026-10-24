# Gripper exercise

The current gripper uses two Gazebo joint-position controllers driven by the same
`/arm/gripper/target` command. The finger axes are opposite, so equal positive
joint positions open symmetrically. RViz uses a mimic joint instead; the Gazebo
Xacro omits it for the current physics engine.

## Geometry and commands

- Each finger is 20 x 12 x 60 mm, mass 30 g, with matching box collision geometry.
- One finger travels from 0 to 35 mm; the total free opening is twice this travel.
- Fully open: command 0.035 m (70 mm gap).
- Empty closed: command 0.0 m.
- Grip the 40 mm block: command 0.015 m. Contact should stop each finger near
  0.020 m instead of allowing it to reach the requested 0.015 m.
- Each finger controller is limited to +/-5 N, with provisional PID gains.
- Identical commands are not a mechanical linkage: asymmetric contact can still
  produce unequal positions. This exercise validates only the current setup.

Run only the Gazebo launch when testing physics; do not also run display.launch.py.
Press Play because the normal Gazebo launch starts paused. Example opening command:

```bash
ros2 topic pub --once /arm/gripper/target std_msgs/msg/Float64 '{data: 0.035}'
```

Inspect actual joint positions with `ros2 topic echo /joint_states`.

## Block location and arm poses

The 40 mm, 50 g block spawns at (0.326506350946, 0, 0.021) m and settles with
its center around z=0.020 m. The shoulder pivot is (0.11, 0, 0.17) m in world
coordinates when base rotation is zero.

Nominal grasp x = 0.11 + 0.25*sin(pi/3) = 0.326506350946 m.
With shoulder + elbow = pi, the fingers point down. At the grasp pose, their
nominal tips are 15 mm above the ground. The fingertip and tool-center positions
are different; tool0 is halfway along the fingers, not at the tips.

| Stage | Base (rad) | Shoulder (rad) | Elbow (rad) | Finger command (m) |
|---|---:|---:|---:|---:|
| Open at home | 0 | 0 | 0 | 0.035 |
| Approach | 0 | 0.88 | 2.261592654 | 0.035 |
| Lower to grasp | 0 | 1.047197551 | 2.094395102 | 0.035 |
| Grip | 0 | 1.047197551 | 2.094395102 | 0.015 |
| Lift | 0 | 0.9 | 2.241592654 | 0.015 |
| Lower | 0 | 1.047197551 | 2.094395102 | 0.015 |
| Release | 0 | 1.047197551 | 2.094395102 | 0.035 |
| Retreat | 0 | 0.88 | 2.261592654 | 0.035 |

These are reference poses, not an instruction to jump between joint targets.
The headless verification interpolates shoulder and elbow commands together over
4-6 seconds and pauses between stages. This is a fixed demonstration, not a
collision-aware motion planner or a general grasping system.

## Verification

The isolated headless test uses the source robot geometry, Gazebo controllers,
gravity, self-collisions, ground, block collisions, and contact friction. A temporary
block-pose publisher measures object movement; it is not required by the normal
world. Joint feedback checks opening, empty closure, and symmetry. Block-position
feedback checks actual lifting, holding, lowering, and release. Runtime results are summarized below.

### Observed result (2026-10-24)

- Passed the complete open, approach, grasp, lift, 4-second hold, lower, release,
  and retreat cycle in an isolated headless Gazebo instance.
- Block center before grasp: 0.02000 m above ground.
- Block center after lifting and holding: 0.04859 m; rise 28.6 mm.
- Block center after release and retreat: 0.02000 m.
- Contact finger positions: approximately 0.020 m each, matching the 40 mm cube.
- Finger position difference stayed below 0.5 mm at every measured stage.
- Grasp and lift also succeeded in the preceding measurement run.
- Test processes were stopped after completion. These results apply to this block,
  pose, approach, and controller setup; they do not establish general grasping.
