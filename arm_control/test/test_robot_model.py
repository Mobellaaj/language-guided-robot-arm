"""Check the real URDF, IK geometry, finger symmetry and sampled motion speeds."""
from types import SimpleNamespace
import math

import numpy as np
import pytest
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose

from arm_control.ik_solver import JOINT_LIMITS, forward_kinematics, solve_downward_ik
from arm_control.ik_visualization import PreviewModel, origin_matrix
from arm_control.motion_math import JOINTS, SPEEDS, UNLOADED_SPEEDS, UNLOADED_MIN_DURATION, duration_for, interpolate, check_finger_ground
from arm_control.pick_sequence import vertical_path, path_duration, sample_path
from arm_control.place_sequence import PlaceSequence
from arm_control.scene_planner import ScenePlanner
from arm_control.gripper_control import FINGER_SPEED, FINGER_JOINTS
from arm_perception.platform_geometry import PlatformLayout


@pytest.fixture(scope='module')
def model():
    return PreviewModel()


def test_solver_joint_limits_match_the_robot(model):
    for name, expected in zip(JOINTS, JOINT_LIMITS):
        limit = model.robot.find(f"joint[@name='{name}']/limit")
        assert (float(limit.get('lower')), float(limit.get('upper'))) == pytest.approx(expected)


def test_forward_kinematics_matches_urdf_over_joint_ranges(model):
    rng = np.random.default_rng(2026)
    for _ in range(300):
        q = tuple(rng.uniform(a, b) for a, b in JOINT_LIMITS)
        assert np.allclose(model.link_poses(q)['tool0'][:3, 3], forward_kinematics(q), atol=1e-10)


@pytest.mark.parametrize('travel', [0., .02, .035])
def test_fingers_have_symmetric_inner_faces_and_expected_gap(model, travel):
    q = (.4, .6, 2.1, .4)
    poses = model.link_poses(q, travel)
    palm_inverse = np.linalg.inv(poses['gripper_palm_link'])
    left = palm_inverse @ poses['left_finger_link']
    right = palm_inverse @ poses['right_finger_link']
    thickness = float(model.robot.find("link[@name='left_finger_link']/collision/geometry/box").get('size').split()[1])
    assert left[:3, 3] == pytest.approx([0., thickness/2+travel, .02])
    assert right[:3, 3] == pytest.approx([0., -thickness/2-travel, .02])
    assert left[1, 3]-right[1, 3]-thickness == pytest.approx(2*travel)


def test_masses_and_inertia_tensors_are_physical(model):
    for link in model.robot.findall('link'):
        inertial = link.find('inertial')
        if inertial is None:
            continue
        assert float(inertial.find('mass').get('value')) > 0
        t = inertial.find('inertia').attrib
        tensor = np.array([[float(t['ixx']), float(t['ixy']), float(t['ixz'])],
                           [float(t['ixy']), float(t['iyy']), float(t['iyz'])],
                           [float(t['ixz']), float(t['iyz']), float(t['izz'])]])
        eigenvalues = np.linalg.eigvalsh(tensor)
        assert min(eigenvalues) > 0
        assert 2*max(eigenvalues) <= sum(eigenvalues)+1e-12


def test_pick_tip_clearance_uses_the_actual_finger_geometry(model):
    q = solve_downward_ik(.327, 0., .035)[0]
    poses = model.link_poses(q, .035)
    for name in ('left_finger_link', 'right_finger_link'):
        tip = poses[name] @ np.array([0., 0., .06, 1.])
        assert tip[2] == pytest.approx(.005)
    assert .07 > math.sqrt(2)*.04  # Enough opening for a 4 cm cube at any yaw.


def test_sampled_joint_and_vertical_speeds_stay_below_limits():
    start = (0., 0., 0., 0.)
    goal = solve_downward_ik(.327, 0., .14)[0]
    duration = duration_for(start, goal)
    samples = np.array([interpolate(start, goal, i/500) for i in range(501)])
    assert np.all(np.max(abs(np.diff(samples, axis=0)), axis=0)/(duration/500) <= np.array(SPEEDS)*1.001)
    path = vertical_path(goal, .045)
    duration = path_duration(path)
    samples = np.array([sample_path(path, i/500) for i in range(501)])
    assert np.all(np.max(abs(np.diff(samples, axis=0)), axis=0)/(duration/500) <= np.array(SPEEDS)*1.001)
    heights = np.array([forward_kinematics(q)[2] for q in samples])
    assert np.max(abs(np.diff(heights)))/(duration/500) <= .0201


def test_placement_rechecks_a_block_moved_into_the_prepared_route(model):
    n = SimpleNamespace(model=model, layout=PlatformLayout(), objects=Detection3DArray())
    n.objects.header.frame_id = 'base_link'
    n.planner = ScenePlanner(model, n.layout)
    sequence = PlaceSequence(n)
    q = solve_downward_ik(.24, -.23, .071)[0]
    sequence.program = dict(held=dict(offset=.025, source=(.327,0.,.04)), center=(.24,-.23,.02),
                            stages=[dict(name='lower', path=[q], finger=.02)])
    sequence.check_current_scene()
    result = ObjectHypothesisWithPose()
    result.pose.pose.position.x = .24
    result.pose.pose.position.y = -.23
    result.pose.pose.position.z = .06
    detection = Detection3D()
    detection.results.append(result)
    n.objects.detections.append(detection)
    with pytest.raises(ValueError, match='collision|collide'):
        sequence.check_current_scene()


def test_finger_contacts_support_both_sides_of_cube_center(model):
    # In the downward grasp, patches must bracket the 4 cm cube's COM in
    # height and across its face, rather than supporting only a single line.
    poses = model.link_poses(solve_downward_ik(.327, 0., .035)[0], .02)
    for name in ('left_finger_link', 'right_finger_link'):
        collisions = model.robot.find(f"link[@name='{name}']").findall('collision')
        centers = [poses[name] @ origin_matrix(c.find('origin')) for c in collisions]
        assert min(p[2, 3] for p in centers) < .02 < max(p[2, 3] for p in centers)
        xs = [float(c.find('origin').get('xyz').split()[0]) for c in collisions]
        assert min(xs) < 0 < max(xs)


def test_measured_finger_floor_clearance_blocks_contact(model):
    check_finger_ground(model, solve_downward_ik(.327, 0., .035)[0], .035)
    with pytest.raises(ValueError, match='less than 1 mm'):
        check_finger_ground(model, solve_downward_ik(.327, 0., .030)[0], .035)


def test_unloaded_profile_is_faster_and_stays_within_robot_joint_limits(model):
    start = (0.,0.,0.,0.)
    goal = solve_downward_ik(.327,0.,.14)[0]
    path = [start,goal]
    slow = path_duration(path,minimum=8.,tool_speed=None)
    fast = path_duration(path,minimum=UNLOADED_MIN_DURATION,tool_speed=None,joint_speeds=UNLOADED_SPEEDS)
    assert fast < slow
    samples = np.array([sample_path(path,i/500) for i in range(501)])
    assert np.all(np.max(abs(np.diff(samples,axis=0)),axis=0)/(fast/500) <= np.array(UNLOADED_SPEEDS)*1.001)
    limits = [float(model.robot.find(f"joint[@name='{name}']/limit").get('velocity')) for name in JOINTS]
    assert np.all(np.array(UNLOADED_SPEEDS) <= limits)
    for name in FINGER_JOINTS:
        assert FINGER_SPEED <= float(model.robot.find(f"joint[@name='{name}']/limit").get('velocity'))
