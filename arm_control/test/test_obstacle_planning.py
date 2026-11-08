"""Offline checks of route search, collision rejection and waypoint execution."""
from pathlib import Path
from types import SimpleNamespace
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from rclpy.clock import Clock
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from arm_control.ik_solver import solve_downward_ik, forward_kinematics
from arm_control.ik_visualization import PreviewModel
from arm_control.scene_planner import ScenePlanner
from arm_control.pick_sequence import PickSequence
from arm_control.place_sequence import PlaceSequence
from arm_control.move_to_color import MoveToColor
from arm_control.task_runner import TaskRunner
from arm_perception.platform_geometry import PlatformLayout


def observations(entries):
    message = Detection3DArray()
    message.header.frame_id = 'base_link'
    for color, x, y, z in entries:
        detection = Detection3D()
        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id = color
        result.pose.pose.position.x, result.pose.pose.position.y, result.pose.pose.position.z = x,y,z
        detection.results.append(result)
        message.detections.append(detection)
    return message


@pytest.fixture(scope='module')
def world():
    path = Path(__file__).resolve().parents[2]/'arm_gazebo/worlds/arm_world.sdf'
    scene = ET.parse(path)
    barrier = scene.find("world/model[@name='planning_obstacle']")
    box = tuple(map(float, barrier.findtext('pose').split()[:3]
                          + barrier.findtext('link/collision/geometry/box/size').split()))
    entries = []
    for model in scene.findall('world/model'):
        if 'block' in model.get('name',''):
            x,y = map(float, model.findtext('pose').split()[:2])
            entries.append((model.get('name').split('_')[0],x,y,.04))
    return PreviewModel(), PlatformLayout(), box, entries


def planner_for(world):
    model, layout, box, entries = world
    planner = ScenePlanner(model, layout, [box])
    message = observations(entries)
    planner.capture(message)
    return planner, message


def test_blocked_straight_transfer_gets_a_downward_collision_free_detour(world):
    planner, objects = planner_for(world)
    start = solve_downward_ik(.327,0.,.15)[0]
    goal = solve_downward_ik(.24,-.23,.15)[0]
    held = dict(source=(.327,0.,.04), offset=.015)
    with pytest.raises(ValueError, match='box_1'):
        planner.validate([start,goal],.02,objects,held=held)
    path = planner.plan(start,goal,.02,objects,held=held,keep_down=True)
    assert path[0] == pytest.approx(start)
    assert path[-1] == pytest.approx(goal)
    assert all(sum(q[1:]) == pytest.approx(math.pi) for q in path)
    planner.validate(path,.02,objects,held=held)
    # An obstacle newly inserted into the inspected route must invalidate it.
    boxes, cubes, offset = planner.scene(objects, held=held)
    point = forward_kinematics(path[len(path)//2])
    with pytest.raises(ValueError, match='collision|collide'):
        planner.engine.validate(path,.02,boxes+[(*point,.03,.03,.03)],cubes,.04,offset)


def test_goal_inside_barrier_and_self_collision_are_rejected(world):
    planner, objects = planner_for(world)
    start = solve_downward_ik(.327,0.,.15)[0]
    blocked = solve_downward_ik(.29,-.158,.12)[0]
    with pytest.raises(ValueError, match='Goal pose is invalid'):
        planner.plan(start,blocked,.02,objects,held=dict(source=(.327,0.,.04),offset=.015),keep_down=True)
    with pytest.raises(ValueError, match='base_link'):
        planner.validate([(0.,-1.0739,-2.1560,-.7086)],.035,objects)


def test_hidden_cubes_remain_obstacles_and_verified_placement_updates_the_scene(world):
    planner, _ = planner_for(world)
    count = len(world[3])
    _, centres, _ = planner.scene(observations([]))
    assert len(centres) == count
    placed = (.24,-.23,.04)
    planner.remember_placement((world[3][0][1],world[3][0][2],.04),placed)
    _, centres, _ = planner.scene(None)
    assert len(centres) == count
    assert placed in centres


@pytest.mark.parametrize('index', range(6))
def test_all_pickups_and_both_slots_with_the_barrier(world, index):
    model, layout, _, entries = world
    planner, objects = planner_for(world)
    color,x,y,z = entries[index]
    n = SimpleNamespace(model=model,layout=layout,planner=planner,objects=objects,
                        clearance=.1,observation_joints=(0.,0.,0.,0.),get_clock=lambda:Clock(),
                        gripper=SimpleNamespace(opened=.035,positions=(.02,.02),hold_problem=lambda *args:''))
    pick = PickSequence(n)
    program = pick.prepare((x,y,z),(0.,0.,0.,0.),0.)
    pick.check_current_scene()
    for slot in layout.slots:
        n.held = dict(color=color,source=(x,y,z),offset=.015,gap=.04,fingers=(.02,.02))
        extra = [('red',.24,-.23,.06)] if slot == 'slot_2' else []
        n.objects = observations(entries+extra)
        n.platform = SimpleNamespace(available=lambda slot=slot:[slot],reserve=lambda name:None)
        place = PlaceSequence(n)
        place.prepare(program['stages'][-1]['path'][-1],.02)
        place.check_current_scene()


def test_execution_follows_the_detour_instead_of_interpolating_endpoints():
    sent = []
    start, detour, goal = (0.,0.,0.,0.), (.3,.4,.5,.2), (.1,.1,.1,.1)
    n = SimpleNamespace(now_seconds=lambda:1.,last_clock=.95,elapsed=3.95,active=True,
                        gripper=SimpleNamespace(active=False,tick=lambda dt:None),
                        pick=SimpleNamespace(active=False),place=SimpleNamespace(active=False),
                        feedback_problem=lambda:'',measured=lambda:start,last_command=start,
                        plan=dict(kind='observation',duration=8.,start=start,goal=goal,
                                  path=[start,detour,goal]),send=sent.append,last_preview_wall=float('inf'))
    n.held, n.grasp_uncertain = None, False
    n.task_status_pub = SimpleNamespace(publish=lambda message:None)
    n.task = TaskRunner(n)
    MoveToColor.tick(n)
    assert np.allclose(sent[0],detour)


@pytest.mark.parametrize('source_slot', ['slot_1','slot_2'])
@pytest.mark.parametrize('destination_slot', ['slot_1','slot_2'])
def test_platform_pickup_and_floor_return_with_the_known_obstacle(world, source_slot, destination_slot):
    model, layout, _, entries = world
    planner, _ = planner_for(world)
    x,y,support = layout.slots[source_slot]
    surface = (x,y,support+layout.cube)
    # The first floor cube has already been placed on the platform.
    current = entries[1:]+[('red',*surface)]
    objects = observations(current)
    planner.capture(objects)
    n = SimpleNamespace(model=model,layout=layout,planner=planner,objects=objects,
                        clearance=.1,observation_joints=(0.,0.,0.,0.),get_clock=lambda:Clock(),
                        gripper=SimpleNamespace(opened=.035,positions=(.02,.02),hold_problem=lambda *args:''))
    pick = PickSequence(n)
    program = pick.prepare(surface,(0.,0.,0.,0.),.035,source='platform')
    pick.check_current_scene()
    assert forward_kinematics(program['stages'][2]['path'][-1])[2] == pytest.approx(.055)
    n.held = dict(color='red',source=surface,offset=.015,gap=.04,fingers=(.02,.02))
    n.floor = SimpleNamespace(available=lambda:[destination_slot],reserve=lambda name:None)
    if destination_slot == 'slot_2':
        fx,fy,_ = layout.placement_slots('floor')['slot_1']
        n.objects = observations(current+[('blue',fx,fy,.04)])
    place = PlaceSequence(n)
    place.prepare(program['stages'][-1]['path'][-1],.02,destination='floor')
    place.check_current_scene()
    assert place.program['destination'] == 'floor'
    assert place.program['center'][2] == 0


@pytest.mark.parametrize('slot', ['slot_1','slot_2'])
@pytest.mark.parametrize('dx', [-.003,.003])
def test_platform_grasp_has_clearance_for_small_camera_position_bias(world, slot, dx):
    from arm_control.pick_sequence import vertical_path
    planner, objects = planner_for(world)
    x,y,support = world[1].slots[slot]
    # The overhead bbox centre can shift a few millimetres toward the camera.
    x,y = x+dx,y+.003
    above = solve_downward_ik(x,y,support+.04+.1)[0]
    planner.validate(vertical_path(above,support+.035),.035,objects)
