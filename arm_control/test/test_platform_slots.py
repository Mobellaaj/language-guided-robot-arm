"""Unknown/full slots and unverified releases must not be reused."""
from types import SimpleNamespace
import pytest
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from arm_interfaces.msg import PlatformSlot, PlatformState
from arm_control.platform_manager import PlatformManager
from arm_perception.platform_geometry import PlatformLayout


def observed(manager, states):
    for t in (9.0, 9.3, 9.6, 9.9):
        message = PlatformState()
        message.header.frame_id = 'base_link'
        message.header.stamp = Time(sec=int(t), nanosec=int((t-int(t))*1e9))
        for (name, center), state in zip(manager.slots.items(), states):
            message.slots.append(PlatformSlot(name=name, state=state, center=Point(
                x=center[0], y=center[1], z=center[2])))
        manager.update(message)


def manager():
    node = SimpleNamespace(age=lambda stamp: 10-(stamp.sec+stamp.nanosec/1e9))
    return PlatformManager(node, PlatformLayout())


def test_choose_another_slot_when_first_is_occupied():
    state = manager()
    observed(state, (PlatformSlot.OCCUPIED, PlatformSlot.EMPTY))
    assert state.available() == ['slot_2']


@pytest.mark.parametrize('slots', [(PlatformSlot.UNKNOWN, PlatformSlot.UNKNOWN),
                                   (PlatformSlot.OCCUPIED, PlatformSlot.OCCUPIED)])
def test_full_or_hidden_platform_has_no_available_slot(slots):
    state = manager()
    observed(state, slots)
    with pytest.raises(ValueError, match='No confirmed empty'):
        state.available()


def test_release_is_not_confirmed_by_command_completion():
    state = manager()
    observed(state, (PlatformSlot.EMPTY, PlatformSlot.EMPTY))
    state.reserve('slot_1')
    state.released('slot_1')
    state.release_reservation('slot_1')
    assert state.reserved == {'slot_1': 'unverified'}
    assert state.available() == ['slot_2']


def test_canceling_unused_preview_releases_its_reservation():
    state = manager()
    observed(state, (PlatformSlot.EMPTY, PlatformSlot.EMPTY))
    state.reserve('slot_1')
    state.release_reservation('slot_1')
    assert state.available() == ['slot_1', 'slot_2']


def test_old_platform_observations_cannot_allocate_slots():
    state = manager()
    observed(state, (PlatformSlot.EMPTY, PlatformSlot.EMPTY))
    state.node.age = lambda stamp: 1.0
    with pytest.raises(ValueError, match='stale'):
        state.available()


def test_floor_and_platform_reservations_are_independent():
    platform = manager()
    floor = PlatformManager(platform.node, platform.layout, 'floor')
    observed(platform, (PlatformSlot.EMPTY, PlatformSlot.EMPTY))
    observed(floor, (PlatformSlot.EMPTY, PlatformSlot.EMPTY))
    platform.reserve('slot_1')
    assert floor.available() == ['slot_1','slot_2']
    floor.reserve('slot_2')
    floor.released('slot_2')
    assert floor.available() == ['slot_1']
    assert platform.available() == ['slot_2']


def test_floor_verification_requires_a_floor_level_cube_and_matching_color():
    from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
    floor = PlatformManager(manager().node, PlatformLayout(), 'floor')
    objects = Detection3DArray()
    objects.header.frame_id = 'base_link'
    objects.header.stamp = Time(sec=9,nanosec=900000000)
    obj, result = Detection3D(), ObjectHypothesisWithPose()
    result.hypothesis.class_id = 'red'
    x,y,_ = floor.slots['slot_1']
    result.pose.pose.position = Point(x=x,y=y,z=.04)
    obj.results.append(result)
    objects.detections.append(obj)
    floor.node.objects = objects
    observed(floor, (PlatformSlot.OCCUPIED, PlatformSlot.EMPTY))
    floor.message.slots[0].color = 'red'
    floor.stability['slot_1'] = ((PlatformSlot.OCCUPIED,'red'),9.,9.9,4)
    assert floor.verified('slot_1','red')
    result.pose.pose.position.z = .06
    # ROS messages copy their child fields; modify the stored observation too.
    objects.detections[0].results[0].pose.pose.position.z = .06
    assert not floor.verified('slot_1','red')
    objects.detections[0].results[0].pose.pose.position.z = .04
    assert not floor.verified('slot_1','blue')
