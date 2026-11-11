"""A slow joint-space interpolation for the learning arm; not a motion planner."""
import math
import numpy as np
from arm_control.ik_solver import JOINT_LIMITS
from arm_control.ik_visualization import origin_matrix

JOINTS = ('base_rotation_joint', 'shoulder_joint', 'elbow_joint', 'wrist_pitch_joint')
TOPICS = ('/arm/base/target', '/arm/shoulder/target', '/arm/elbow/target', '/arm/wrist/target')
# rad/s, below the current URDF limits.
SPEEDS = (0.30, 0.25, 0.35, 0.20)
# Unloaded travel may be faster; grasping/carrying retain the original limits.
UNLOADED_SPEEDS = tuple(1.25*speed for speed in SPEEDS)
UNLOADED_MIN_DURATION = 6.0


def interpolate(start, goal, fraction):
    u = max(0.0, min(1.0, fraction))
    blend = u*u*(3-2*u)
    return tuple(a + blend*(b-a) for a, b in zip(start, goal))


def duration_for(start, goal, minimum=8.0):
    # Cubic smoothstep has a peak derivative of 1.5.
    return max(minimum, *(1.5*abs(b-a)/speed for a,b,speed in zip(start, goal, SPEEDS)))


def check_limits(angles):
    if len(angles) != 4 or not all(math.isfinite(a) for a in angles):
        raise ValueError('Four finite arm angles are required')
    if not all(low-1e-3 <= a <= high+1e-3 for a,(low,high) in zip(angles, JOINT_LIMITS)):
        raise ValueError('A sampled joint angle exceeds the model limits')


def check_ground(model, angles, gripper):
    """Reject sampled floor penetration using the actual URDF collision shapes."""
    poses = model.link_poses(angles, gripper)
    for link in model.robot.findall('link'):
        for collision in link.findall('collision'):
            pose = poses[link.get('name')] @ origin_matrix(collision.find('origin'))
            geometry = collision.find('geometry')
            row = pose[2, :3]
            if geometry.find('box') is not None:
                half = np.array([float(v) for v in geometry.find('box').get('size').split()])/2
                extent = np.abs(row) @ half
            elif geometry.find('cylinder') is not None:
                shape = geometry.find('cylinder')
                extent = (float(shape.get('radius'))*math.hypot(row[0],row[1])
                          + float(shape.get('length'))/2*abs(row[2]))
            elif geometry.find('sphere') is not None:
                extent = float(geometry.find('sphere').get('radius'))
            else:
                raise ValueError('Unsupported collision shape in the floor check')
            if pose[2,3]-extent < -0.002:
                raise ValueError(f'Path crosses the floor at {link.get("name")}')


def check_finger_ground(model, angles, gripper):
    """Keep at least 1 mm under the measured finger collision geometry."""
    poses = model.link_poses(angles, gripper)
    for name in ('left_finger_link', 'right_finger_link'):
        link = model.robot.find(f"link[@name='{name}']")
        for collision in link.findall('collision'):
            pose = poses[name] @ origin_matrix(collision.find('origin'))
            half = np.array([float(v) for v in collision.find('geometry/box').get('size').split()])/2
            clearance = pose[2, 3]-np.abs(pose[2, :3]) @ half
            if clearance < 0.001:
                raise ValueError(f'{name} is less than 1 mm above the floor; holding before contact.')
