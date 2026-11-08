"""Task semantics and orchestration, without Gazebo or joint-command publishers."""
import json
from types import SimpleNamespace

import pytest
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from arm_control.arm_command import parse_command, parse_task
from arm_perception.platform_geometry import PlatformLayout
from arm_control.target_selection import AmbiguousTarget, choose_object
from arm_control.task_request import ClarificationRequired, resolve_action, validate_task
from arm_control.task_runner import TaskRunner


def objects(*entries):
    message = Detection3DArray()
    message.header.frame_id = 'base_link'
    for color,x,y in entries:
        d = Detection3D()
        h = ObjectHypothesisWithPose()
        h.hypothesis.class_id = color
        h.pose.pose.position.x, h.pose.pose.position.y, h.pose.pose.position.z = x,y,.04
        d.results.append(h)
        message.detections.append(d)
    return message


COMBINED = json.dumps(dict(action='pick_and_place',object=dict(color='red',selector='rightmost'),
                           destination='platform'))


def response():
    return SimpleNamespace(success=False,needs_clarification=False,message='',normalized_json='')


class Controller:
    """Low-level sequences are already tested separately; finish them explicitly."""
    def __init__(self):
        self.active = False
        self.pick, self.place = SimpleNamespace(active=False), SimpleNamespace(active=False)
        self.held, self.grasp_uncertain, self.plan = None, False, None
        self.now = 10.
        self.target_color, self.target_side = 'red', 'unique'
        self.objects = objects(('red',.33,0.),('red',.39,.04),('blue',.33,-.085))
        self.scene_problem, self.grip_problem, self.place_problem = '', '', ''
        self.slots_ready = True
        self.calls, self.reports, self.status = [], [], []
        self.task_status_pub = SimpleNamespace(publish=lambda m:self.status.append(json.loads(m.data)))
        self.gripper = SimpleNamespace(active=False,positions=(.02,.02),
                                       hold_problem=lambda *args:self.grip_problem)
        self.layout = PlatformLayout()
        self.platform = SimpleNamespace(available=self.available)
        self.floor = SimpleNamespace(available=self.available)
        self.pick_source, self.place_destination = None, None
        self.task = TaskRunner(self)

    def available(self):
        if not self.slots_ready:
            raise ValueError('No confirmed empty platform slot.')
        return ['slot_1']

    def now_seconds(self):
        return self.now

    def set_status(self, message):
        self.reports.append(message)

    def objects_problem(self):
        return self.scene_problem

    def feedback_problem(self):
        return ''

    def grasp_recovery_message(self):
        return 'Grasp uncertain. Inspect the cube before another action.'

    def clear_plan(self):
        self.task.discard_prepared()
        self.plan = None
        self.pick.active = self.place.active = False

    def clear_selection(self):
        pass

    def set_parameters_atomically(self, parameters):
        self.clear_plan()
        for parameter in parameters:
            setattr(self,parameter.name,parameter.value)
        return SetParametersResult(successful=True)

    def prepare_pick(self, request, reply, selected=None, source='floor'):
        self.pick_source = source
        self.calls.append(('prepare_pick',selected))
        self.plan = dict(kind='pick',surface=selected)
        reply.success, reply.message = True, 'Pickup prepared.'
        return reply

    def prepare_place(self, request, reply, destination='platform'):
        self.place_destination = destination
        self.calls.append(('prepare_place',self.held['source']))
        self.clear_plan()
        if self.place_problem:
            reply.message = self.place_problem
            return reply
        self.plan = dict(kind='place')
        reply.success, reply.message = True, 'Placement prepared.'
        return reply

    def execute(self, request, reply):
        if self.plan is None or self.task.busy:
            reply.message = 'No executable plan, or another task is active.'
            return reply
        self.calls.append(('execute',self.plan['kind']))
        (self.pick if self.plan['kind']=='pick' else self.place).active = True
        self.task.started()
        reply.success = True
        return reply

    def stop_motion(self, reason, task_state='failed'):
        self.task.interrupted(task_state)
        self.clear_plan()
        self.task.publish(task_state,reason)
        return reason

    def finish_pick(self):
        self.held = dict(color=self.target_color,source=self.plan['surface'],
                         offset=.015,gap=.04,fingers=(.02,.02))
        self.clear_plan()


def test_schema_defaults_and_strict_fields():
    task = validate_task('{"action":"pick"}')
    assert task['object'] == dict(color=None,selector='unique',source='floor')
    for text in ('{"action":"execute"}', '{"action":"pick","joint_positions":[1,2,3,4]}',
                 '{"action":"pick","object":{"selector":"right"}}',
                 '{"action":"place"}', '{"action":"pick","object":{"source":"table"}}',
                 '{"action":"pick","action":"place"}', 'NaN', '{"action":true}'):
        with pytest.raises(ValueError):
            validate_task(text)


def test_place_semantics_use_held_state_and_do_not_release_a_different_color():
    task = validate_task('{"action":"place","object":{"color":"red"},"destination":"platform"}')
    assert resolve_action(task,None)['action'] == 'pick_and_place'
    held = dict(color='red')
    assert resolve_action(validate_task('{"action":"place","destination":"platform"}'),held)['action']=='place'
    for text, context in [('{"action":"place","destination":"platform"}',None),
                          ('{"action":"pick"}',held),
                          ('{"action":"place","object":{"color":"blue"},"destination":"platform"}',held),
                          ('{"action":"place","object":{"selector":"rightmost"},"destination":"platform"}',held)]:
        with pytest.raises(ClarificationRequired):
            resolve_action(validate_task(text),context)


def test_region_and_ranking_are_different_and_color_is_optional():
    seen = objects(('red',.4,-.02),('blue',.3,-.1),('green',.2,.08))
    assert choose_object(seen,None,'rightmost') == ('blue',(.3,-.1,.04))
    assert choose_object(seen,None,'nearest') == ('green',(.2,.08,.04))
    assert choose_object(seen,None,'farthest')[0] == 'red'
    assert choose_object(seen,None,'left_of_robot')[0] == 'green'
    with pytest.raises(AmbiguousTarget):
        choose_object(seen,None,'right_of_robot')
    with pytest.raises(AmbiguousTarget):
        choose_object(objects(('red',.3,0.),('blue',.3,0.)),None,'nearest')


def test_combined_task_prepares_only_then_automatically_places_the_same_object_once():
    n = Controller()
    prepared = n.task.prepare(COMBINED,response())
    assert prepared.success and n.task.prepared and not n.task.active
    assert n.calls == [('prepare_pick',(.33,0.,.04))]
    assert n.execute(None,Trigger.Response()).success
    n.finish_pick()
    # Another red becomes rightmost; the task must keep the object already held.
    n.objects = objects(('red',.39,-.2))
    n.task.tick()
    assert n.task.state == 'waiting_for_scene'
    n.task.tick()
    assert n.place.active and n.task.phase == 'place'
    for _ in range(3):
        n.task.tick()
    assert n.calls.count(('execute','place')) == 1
    assert ('prepare_place',(.33,0.,.04)) in n.calls
    n.held = None
    n.clear_plan()
    n.task.tick()
    assert n.task.state == 'succeeded' and not n.task.active
    assert n.status[-1]['held_color'] is None


def test_ambiguous_request_discards_the_previous_preview_and_executes_nothing():
    n = Controller()
    assert n.task.prepare(COMBINED,response()).success
    answer = n.task.prepare('{"action":"pick","object":{"color":"red"}}',response())
    assert not answer.success and answer.needs_clarification
    assert n.plan is None and not n.task.prepared
    assert not n.execute(None,Trigger.Response()).success


def test_full_platform_is_rejected_before_pickup():
    n = Controller()
    n.slots_ready = False
    answer = n.task.prepare(COMBINED,response())
    assert not answer.success and 'platform' in answer.message
    assert n.calls == []


def begin_handoff():
    n = Controller()
    assert n.task.prepare(COMBINED,response()).success
    n.execute(None,Trigger.Response())
    n.finish_pick()
    n.task.tick()
    return n


def test_cancel_during_handoff_retains_the_cube_and_never_restarts():
    n = begin_handoff()
    held = n.held
    n.stop_motion('Cancelled.',task_state='cancelled')
    for _ in range(5):
        n.task.tick()
    assert n.task.state == 'cancelled' and n.held is held
    assert n.calls.count(('execute','place')) == 0


def test_paused_clock_does_not_expire_the_scene_wait_but_timeout_stops_safely():
    n = begin_handoff()
    n.scene_problem = 'Camera messages are stale.'
    for _ in range(100):
        n.task.tick()
    assert n.task.active and n.task.state == 'waiting_for_scene'
    n.now += 3.1
    n.task.tick()
    assert n.task.state == 'failed' and n.held is not None
    assert ('execute','place') not in n.calls


@pytest.mark.parametrize('failure', ['planning','grasp'])
def test_handoff_failure_never_starts_placement_or_opens_fingers(failure):
    n = begin_handoff()
    if failure == 'planning':
        n.place_problem = 'No collision-free placement route.'
    else:
        n.grip_problem = 'Finger gap changed; possible slip.'
    positions = n.gripper.positions
    n.task.tick()
    assert n.task.state == 'failed' and n.held is not None
    assert n.gripper.positions == positions and ('execute','place') not in n.calls
    assert n.grasp_uncertain == (failure == 'grasp')


def test_handoff_blocks_a_second_task_and_parameter_changes():
    n = begin_handoff()
    original = n.task.request
    answer = n.task.prepare('{"action":"pick"}',response())
    assert not answer.success and n.task.request is original
    from arm_control.move_to_color import MoveToColor
    from rclpy.parameter import Parameter
    result = MoveToColor.validate_color_change(n,[Parameter('target_color',value='blue')])
    assert not result.successful
    assert not MoveToColor.prepare_observation(n,None,Trigger.Response()).success


def test_plain_text_templates_share_the_task_semantics():
    assert parse_command('pick red and place it on the platform') == ('pick_and_place','red','unique')
    assert parse_command('put the rightmost red block on the platform') == ('place','red','rightmost')
    assert parse_command('pick and place the nearest block on platform') == ('pick_and_place',None,'nearest')
    assert parse_command('pick block on your right') == ('pick',None,'right_of_robot')
    assert parse_command('put it on the platform') == ('place',None,None)
    assert parse_command('stop') == ('cancel',None,None)


def test_platform_to_floor_task_keeps_the_surfaces_through_the_handoff():
    n = Controller()
    n.objects = objects(('red',.24,-.23),('red',.39,.04))
    n.objects.detections[0].results[0].pose.pose.position.z = .06
    task = dict(action='pick_and_place',object=dict(color='red',source='platform'),destination='floor')
    answer = n.task.prepare(json.dumps(task),response())
    assert answer.success and n.pick_source == 'platform'
    assert n.calls == [('prepare_pick',(.24,-.23,.06))]
    n.execute(None,Trigger.Response())
    n.finish_pick()
    n.task.tick()
    n.task.tick()
    assert n.place.active and n.place_destination == 'floor'
    assert n.task.request['destination'] == 'floor'


def test_reverse_text_command_and_held_placement_preserve_surfaces():
    task = parse_task('pick the rightmost red block from the platform and place it on the floor')
    assert task == dict(action='pick_and_place',object=dict(color='red',selector='rightmost',
                                                         source='platform'),destination='floor')
    assert parse_task('place on the floor') == dict(action='place',destination='floor')
    assert parse_task('pick red from platform')['object']['source'] == 'platform'
    assert parse_command('inspect floor') == ('inspect_floor',None,None)


def test_unknown_destination_is_rejected_and_full_floor_prevents_pickup():
    with pytest.raises(ValueError):
        validate_task('{"action":"place","destination":"table"}')
    n = Controller()
    def full():
        raise ValueError('No confirmed empty floor slot.')
    n.floor.available = full
    task = json.loads(COMBINED)
    task['destination'] = 'floor'
    answer = n.task.prepare(json.dumps(task),response())
    assert not answer.success and 'floor slot' in answer.message and not n.calls
