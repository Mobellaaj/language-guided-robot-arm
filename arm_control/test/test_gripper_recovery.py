"""Grasp interruption must not release a payload or silently become an empty-hand move."""
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
from std_srvs.srv import Trigger

from arm_control.arm_command import parse_command
from arm_control.gripper_control import GripperControl
from arm_control.task_runner import TaskRunner
from arm_control.move_to_color import MoveToColor
from arm_control.pick_sequence import PickSequence
from arm_control.ik_visualization import PreviewModel


class Publisher:
    def __init__(self):
        self.sent = []

    def get_subscription_count(self):
        return 1

    def publish(self, message):
        self.sent.append(message.data)


def gripper():
    publisher = Publisher()
    reports = []
    node = SimpleNamespace(create_publisher=lambda *args: publisher, age=lambda stamp: 0.)
    robot = ET.fromstring('''<robot>
        <joint name="gripper_opening_joint"><limit lower="0" upper="0.035"/></joint>
        <joint name="right_finger_joint"><limit lower="0" upper="0.035"/></joint>
        </robot>''')
    control = GripperControl(node, robot, reports.append)
    control.positions = (.02, .02)
    control.stamp = object()
    return control, publisher, reports


def finish_at(control, positions):
    control.positions = positions
    for _ in range(200):
        control.tick(.05)
        if not control.active:
            return
    pytest.fail('The finger action did not finish within the test interval')


def test_open_timeout_rejects_short_gap_and_reports_both_fingers():
    control, _, reports = gripper()
    control.start('open')
    finish_at(control, (.03115, .03115))  # The reported 62.3 mm gap.
    assert control.last_result['kind'] == 'failed'
    assert 'requested gap 70.0 mm, measured 62.3 mm' in control.last_result['reason']
    assert 'left 31.2, right 31.2' in reports[-1]
    assert 'no arm motion was started' not in reports[-1]


def test_only_settled_full_opening_succeeds():
    control, _, _ = gripper()
    control.start('open')
    finish_at(control, (.035, .035))
    assert control.last_result['kind'] == 'reached'


def test_cancel_can_preserve_squeezing_target_instead_of_zero_contact_force():
    control, publisher, _ = gripper()
    control.start('close')
    control.tick(.1)
    sent = list(publisher.sent)
    assert sent[-1] < .02
    control.cancel(preserve_target=True)
    assert publisher.sent == sent
    assert not control.active
    assert control.last_result['kind'] == 'cancelled'


class Sequence:
    def __init__(self, stage=None):
        self.active = stage is not None
        self.program = dict(stages=[dict(name=stage)]) if self.active else None
        self.index = 0

    def clear(self):
        self.active = False
        self.program = None


def motion_node(stage=None, held=None):
    control, publisher, reports = gripper()
    n = SimpleNamespace(active=False, pick=Sequence(stage), place=Sequence(),
                        gripper=control, held=held, grasp_uncertain=False, plan={'kind': 'pick'},
                        now=0., last_clock=0., last_command=(0., .5, 2., .6))
    n.task_status_pub = SimpleNamespace(publish=lambda message:None)
    n.task = TaskRunner(n)
    n.model = PreviewModel()
    n.measured = lambda: (0., .51, 2., .6)
    n.fresh_feedback = lambda: True
    n.send = lambda q: setattr(n, 'last_command', q)
    n.set_status = reports.append
    n.clear_selection = lambda: None
    n.grasp_recovery_message = lambda: MoveToColor.grasp_recovery_message(n)
    n.now_seconds = lambda: n.now

    def clear_plan():
        n.plan = None
        n.pick.clear()
        n.place.clear()

    n.clear_plan = clear_plan
    n.stop_motion = lambda reason, **kwargs: MoveToColor.stop_motion(n, reason, **kwargs)
    return n, publisher, reports


@pytest.mark.parametrize('stage', ['close', 'lift'])
def test_interrupted_grasp_blocks_observe_new_pick_and_place(stage):
    n, _, _ = motion_node(stage)
    reply = MoveToColor.cancel(n, None, Trigger.Response())
    assert reply.success and n.grasp_uncertain
    assert n.plan is None and not n.pick.active
    assert n.last_command == n.measured()
    for prepare in (MoveToColor.prepare_observation, MoveToColor.prepare_pick, MoveToColor.prepare_place):
        rejected = prepare(n, None, Trigger.Response())
        assert not rejected.success and 'uncertain' in rejected.message


def test_cancel_before_grasp_does_not_invent_a_payload():
    n, _, _ = motion_node('approach')
    MoveToColor.cancel(n, None, Trigger.Response())
    assert not n.grasp_uncertain and n.held is None and n.plan is None


def test_cancel_transport_keeps_completed_pick_context():
    payload = dict(color='red', gap=.04, offset=.025, source=(.33, 0., .04))
    n, _, _ = motion_node(held=payload)
    n.place = Sequence('transfer')
    reply = MoveToColor.cancel(n, None, Trigger.Response())
    assert reply.success and n.held is payload
    assert 'Held-cube context and closure target retained' in reply.message
    assert n.plan is None and not n.place.active


@pytest.mark.parametrize('positions, expected_uncertain', [((.03115, .03115), True),
                                                         ((.035, .035), False)])
def test_payload_state_cleared_only_after_confirmed_manual_release(positions, expected_uncertain):
    n, _, _ = motion_node(held=dict(color='red', gap=.04))
    reply = MoveToColor.command_gripper(n, 'open', Trigger.Response())
    assert reply.success and n.held is not None and n.grasp_uncertain
    n.gripper.positions = positions
    for _ in range(200):
        n.now += .05
        MoveToColor.tick(n)
        if not n.gripper.active:
            break
    assert n.grasp_uncertain == expected_uncertain
    assert (n.held is not None) == expected_uncertain


def test_sequence_reports_actual_gripper_failure():
    n, _, _ = motion_node()
    n.feedback_problem = lambda: ''
    n.fresh_target = lambda: False
    sequence = PickSequence(n)
    sequence.active = True
    sequence.program = dict(stages=[dict(name='open', pose=n.measured())])
    n.gripper.last_result = dict(kind='failed', reason='Opening timed out at 62.3 mm.')
    failures = []
    sequence.fail = failures.append
    sequence.tick(.05)
    assert failures == ['Opening timed out at 62.3 mm.']


def test_stop_is_an_immediate_cancel_command():
    assert parse_command('stop') == parse_command('cancel') == ('cancel', None, None)


def test_clock_reset_during_grasp_does_not_keep_old_payload_uncertainty():
    n, _, reports = motion_node('close')
    n.platform = SimpleNamespace(message=object(), stability={'old': 1}, reserved={'slot_1': 'reserved'})
    n.floor = SimpleNamespace(message=object(), stability={'old': 1}, reserved={'slot_2': 'reserved'})
    n.planner = SimpleNamespace(snapshot=[(.3,0.,.02)])
    n.planner.capture = lambda message: n.planner.snapshot.clear()
    n.now = -1.
    MoveToColor.tick(n)
    assert not n.grasp_uncertain and n.held is None
    assert not n.pick.active and n.plan is None
    assert n.platform.message is None and not n.platform.reserved
    assert n.floor.message is None and not n.floor.reserved and not n.floor.stability
    assert not n.planner.snapshot
    assert reports[-1] == 'Simulation clock reset; prepare again.'



def test_closing_timeout_requires_settled_contact():
    control, _, _ = gripper()
    control.start('close')
    finish_at(control, (.02, .02))
    assert control.last_result['kind'] == 'blocked'
    control.start('close')
    for i in range(200):
        p = .019 if i % 2 else .021
        control.positions = (p, p)
        control.tick(.05)
        if not control.active:
            break
    assert control.last_result['kind'] == 'failed'
    assert 'still moving' in control.last_result['reason']


def test_constant_total_gap_does_not_hide_asymmetric_slip():
    control, _, _ = gripper()
    assert not control.hold_problem((.02, .02), .04)
    control.positions = (.03, .01)
    assert abs(sum(control.positions)-.04) < 1e-9
    assert 'asymmetry' in control.hold_problem((.02, .02), .04)


def test_individual_finger_shift_is_checked_against_grasp_sample():
    control, _, _ = gripper()
    control.positions = (.022, .018)
    assert 'individual finger' in control.hold_problem((.017, .023), .04)


def test_arm_drift_during_gripper_action_stops_the_sequence():
    n, _, _ = motion_node()
    n.feedback_problem = lambda: ''
    n.fresh_target = lambda: False
    sequence = PickSequence(n)
    sequence.active = True
    sequence.program = dict(stages=[dict(name='open', pose=(0., .2, 2., .6))])
    n.gripper.start('open')
    failures = []
    sequence.fail = failures.append
    sequence.tick(.05)
    assert failures and 'Arm drifted' in failures[0]


def test_empty_closure_does_not_start_a_lift():
    n, _, _ = motion_node()
    n.feedback_problem = lambda: ''
    n.fresh_target = lambda: False
    n.gripper.positions = (0., 0.)
    n.gripper.last_result = dict(kind='reached', action='close', gap=0.)
    sequence = PickSequence(n)
    sequence.active = True
    sequence.program = dict(stages=[dict(name='close', pose=n.measured())])
    failures = []
    sequence.fail = failures.append
    sequence.tick(.05)
    assert failures and 'No lift requested' in failures[0]


@pytest.mark.parametrize('kind', ['approach', 'observation', 'pick', 'place'])
def test_finger_changes_invalidate_every_kind_of_arm_preview(kind):
    n, _, _ = motion_node()
    n.input_problem = lambda **kwargs: ''
    n.surface = lambda: (.33, 0., .04)
    n.plan = dict(kind=kind, prepared=0., start=n.measured(), fingers=(.02, .02),
                  surface=n.surface())
    n.gripper.positions = (.023, .017)  # Same total gap; geometry has changed.
    reply = MoveToColor.execute(n, None, Trigger.Response())
    assert not reply.success and 'Finger positions changed' in reply.message
    assert n.plan is None and not n.active


def test_pausing_simulation_does_not_advance_finger_targets():
    control, publisher, _ = gripper()
    control.start('open')
    for _ in range(10):
        control.tick(0.)
    assert control.active and control.motion['elapsed'] == 0.
    assert not publisher.sent


def test_stale_finger_feedback_stops_the_action():
    control, _, _ = gripper()
    control.start('open')
    control.node.age = lambda stamp: .6
    control.tick(.05)
    assert not control.active and control.last_result['kind'] == 'failed'
    assert 'not current' in control.last_result['reason']


def test_execute_starts_the_prepared_combined_task_in_the_real_motion_handler():
    n, publisher, _ = motion_node()
    n.input_problem = lambda **kwargs: ''
    n.surface = lambda: (.33,0.,.04)
    n.plan = dict(kind='pick',prepared=0.,start=n.measured(),fingers=(.02,.02),surface=n.surface())
    n.publishers_by_joint = [Publisher() for _ in range(4)]
    n.platform = SimpleNamespace(available=lambda:['slot_1'])
    n.pick.check_current_scene = lambda:None
    n.pick.start = lambda:setattr(n.pick,'active',True)
    n.task.request = dict(action='pick_and_place',object=dict(color='red'),destination='platform')
    n.task.state, n.task.phase = 'prepared', 'pick'
    reply = MoveToColor.execute(n,None,Trigger.Response())
    assert reply.success and 'Combined task started' in reply.message
    assert n.pick.active and n.task.active and n.task.phase == 'pick'
    assert not publisher.sent


def test_stop_during_task_handoff_holds_the_arm_and_retains_the_grasp():
    payload = dict(color='red',gap=.04,source=(.33,0.,.04),offset=.015,fingers=(.02,.02))
    n, publisher, _ = motion_node(held=payload)
    n.task.state, n.task.phase = 'waiting_for_scene', 'handoff'
    reply = MoveToColor.cancel(n,None,Trigger.Response())
    assert reply.success and n.task.state == 'cancelled' and not n.task.active
    assert n.last_command == n.measured() and n.held is payload
    assert not publisher.sent
