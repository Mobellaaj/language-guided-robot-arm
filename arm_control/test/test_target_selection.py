"""Selection uses robot-frame geometry, never per-image labels or list order."""
import pytest
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from arm_control.arm_command import parse_command
from arm_control.target_selection import choose_target, match_selected, choose_object
from arm_perception.platform_geometry import PlatformLayout


def objects(*entries):
    message = Detection3DArray()
    message.header.frame_id = 'base_link'
    for color, x, y in entries:
        detection = Detection3D()
        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id = color
        result.pose.pose.position.x = float(x)
        result.pose.pose.position.y = float(y)
        result.pose.pose.position.z = 0.04
        detection.results.append(result)
        message.detections.append(detection)
    return message


def test_left_and_right_use_y_not_image_order_or_forward_distance():
    message = objects(('blue', .4, -.1), ('red', .3, .3), ('blue', .3, .1))
    assert choose_target(message, 'blue', 'leftmost') == (.3, .1, .04)
    assert choose_target(message, 'blue', 'rightmost') == (.4, -.1, .04)
    message.detections.reverse()
    assert choose_target(message, 'blue', 'leftmost') == (.3, .1, .04)


def test_color_only_requires_one_candidate():
    assert choose_target(objects(('red', .3, 0)), 'red') == (.3, 0., .04)
    with pytest.raises(ValueError, match='Ambiguous'):
        choose_target(objects(('red', .3, 0), ('red', .4, .1)), 'red')
    with pytest.raises(ValueError, match='No visible'):
        choose_target(objects(('red', .3, 0)), 'blue')


def test_near_tied_y_is_ambiguous():
    with pytest.raises(ValueError, match='too close'):
        choose_target(objects(('red', .3, .1), ('red', .4, .102)), 'red', 'leftmost')


def test_invalid_coordinates_and_frame_are_rejected():
    message = objects(('blue', .3, .1))
    message.header.frame_id = 'camera_optical_frame'
    with pytest.raises(ValueError, match='base_link'):
        choose_target(message, 'blue')
    with pytest.raises(ValueError, match='finite'):
        choose_target(objects(('blue', float('nan'), 0)), 'blue')


def test_occlusion_does_not_switch_to_other_same_color_cube():
    anchor = (.39, -.045, .04)
    assert match_selected(objects(('blue', .33, -.085)), 'blue', anchor) is None
    assert match_selected(objects(), 'blue', anchor) is None


def test_existing_selection_is_matched_by_proximity_not_new_rank():
    anchor = (.3, .1, .04)
    message = objects(('blue', .3, .101), ('blue', .4, .2))
    assert match_selected(message, 'blue', anchor) == (.3, .101, .04)
    assert choose_target(message, 'blue', 'leftmost') == (.4, .2, .04)


def test_uncertain_proximity_match_is_rejected():
    assert match_selected(objects(('blue', .3, .09), ('blue', .3, .11)),
                          'blue', (.3, .1, .04)) is None


def test_parser_distinguishes_select_prepare_and_execute():
    assert parse_command('select the leftmost blue block') == ('select', 'blue', 'leftmost')
    assert parse_command('Move above the rightmost RED cube') == ('approach', 'red', 'rightmost')
    assert parse_command('move above blue') == ('approach', 'blue', 'unique')
    assert parse_command('execute') == ('execute', None, None)
    with pytest.raises(ValueError):
        parse_command('move above blue and execute')


def test_platform_cube_is_excluded_from_floor_pick():
    message = objects(('red', .24, -.23), ('red', .39, .04))
    message.detections[0].results[0].pose.pose.position.z = .06
    assert choose_target(message, 'red', 'rightmost') == (.39, .04, .04)


def test_platform_selection_and_matching_do_not_switch_to_floor_cube():
    layout = PlatformLayout()
    message = objects(('red',.24,-.23),('red',.39,.04))
    message.detections[0].results[0].pose.pose.position.z = .06
    assert choose_object(message,'red','unique','platform',layout) == ('red',(.24,-.23,.06))
    assert match_selected(message,'red',(.24,-.23,.06),'platform',layout) == (.24,-.23,.06)
    message.detections.pop(0)
    assert match_selected(message,'red',(.24,-.23,.06),'platform',layout) is None
