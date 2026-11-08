"""Known-support height selection and visibility evidence for empty slots."""
import cv2
import numpy as np
from arm_perception.platform_geometry import PlatformLayout, brown_visible, support_visible


def test_floor_and_platform_have_different_top_planes():
    layout = PlatformLayout()
    down = np.array([0., 0., -1.])
    raised = layout.surface(np.array([.24, -.23, .879]), down, .04)
    floor = layout.surface(np.array([.33, 0., .879]), down, .04)
    assert np.allclose(raised, [.24, -.23, .06])
    assert np.allclose(floor, [.33, 0., .04])


def test_empty_requires_visible_brown_surface_not_absence_of_detections():
    polygon = np.array([[10, 10], [80, 10], [80, 80], [10, 80]])
    brown = np.full((100, 100, 3), (20, 51, 89), dtype=np.uint8)
    assert brown_visible(brown, polygon)
    covered = brown.copy()
    cv2.rectangle(covered, (20, 20), (70, 70), (0, 0, 0), -1)
    assert not brown_visible(covered, polygon)
    assert not brown_visible(brown, polygon+200)


def test_floor_area_is_flat_and_separate_from_platform():
    layout = PlatformLayout()
    for center in layout.placement_slots('floor').values():
        assert center[2] == 0 and center[1] > 0 and not layout.contains(center)
    assert all(p[2] == .02 for p in layout.slots.values())


def test_floor_empty_requires_visible_grey_ground():
    polygon = np.array([[10,10],[80,10],[80,80],[10,80]])
    frame = np.full((100,100,3),180,dtype=np.uint8)
    assert support_visible(frame,polygon,'floor')
    cv2.rectangle(frame,(20,20),(70,70),(0,0,0),-1)
    assert not support_visible(frame,polygon,'floor')
    cv2.rectangle(frame,(20,20),(70,70),(0,0,200),-1)
    assert not support_visible(frame,polygon,'floor')
    assert not support_visible(frame,polygon+200,'floor')
