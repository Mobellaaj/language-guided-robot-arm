"""A partly hidden color patch must not masquerade as a displaced cube."""
import cv2
import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import BoundingBox2D

from arm_perception.object_detector import detect_regions, color_mask
from arm_perception.object_localizer import cube_sized_region
from arm_perception.red_localizer import camera_ray, intersect_plane

INFO = CameraInfo(width=640, height=480,
                  k=[554.2547, 0., 320., 0., 554.2547, 240., 0., 0., 1.])
ORIGIN = np.array([.33, 0., .879])
ROTATION = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])


def red_box(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    regions = detect_regions(image, [0, 0, 0, 0], kernel, 100.)
    x, y, width, height = next(box for color, box in regions if color == 'red')
    box = BoundingBox2D(size_x=float(width), size_y=float(height))
    box.center.position.x, box.center.position.y = x+width/2, y+height/2
    return box


def position(box):
    return intersect_plane(ORIGIN, ROTATION @ camera_ray(
        box.center.position.x, box.center.position.y, INFO), .04)


def test_occlusion_reproduces_false_16_mm_shift_and_is_rejected():
    image = np.full((480, 640, 3), 128, dtype=np.uint8)
    image[226:253, 307:333] = (0, 0, 200)
    full = red_box(image)
    image[231:253, 307:333] = 0  # Arm covers the bottom 22 rows; cube never moves.
    partial = red_box(image)
    assert np.linalg.norm(position(partial)-position(full)) == pytest.approx(.01665, abs=.0001)
    assert cube_sized_region(full, INFO, ORIGIN, ROTATION, .04, .04)
    assert not cube_sized_region(partial, INFO, ORIGIN, ROTATION, .04, .04)


def test_a_genuinely_moved_full_cube_still_produces_a_position_update():
    image = np.full((480, 640, 3), 128, dtype=np.uint8)
    image[226:253, 307:333] = (0, 0, 200)
    before = red_box(image)
    image[:] = 128
    image[246:273, 307:333] = (0, 0, 200)
    moved = red_box(image)
    assert np.linalg.norm(position(moved)-position(before)) > .015
    assert cube_sized_region(moved, INFO, ORIGIN, ROTATION, .04, .04)


@pytest.mark.parametrize('height', [.04, .06])
def test_visible_cubes_on_floor_and_platform_are_accepted(height):
    pixels = .04*INFO.k[0]/(ORIGIN[2]-height)
    box = BoundingBox2D(size_x=pixels, size_y=pixels)
    box.center.position.x, box.center.position.y = 320., 240.
    assert cube_sized_region(box, INFO, ORIGIN, ROTATION, height, .04)


def test_rotated_fragment_with_a_large_box_is_rejected_by_visible_area():
    image = np.full((480,640,3),128,dtype=np.uint8)
    diamond = np.array([[320,220],[340,240],[320,260],[300,240]])
    cv2.fillConvexPoly(image,diamond,(0,0,200))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    full = red_box(image)
    mask = color_mask(cv2.cvtColor(image,cv2.COLOR_BGR2HSV),'red',kernel)
    assert cube_sized_region(full,INFO,ORIGIN,ROTATION,.06,.04,mask)
    # A diagonal occlusion retains broad X/Y extents but removes most of the top.
    cv2.fillConvexPoly(image,np.array([[280,260],[360,210],[360,280],[280,280]]),(0,0,0))
    partial = red_box(image)
    assert cube_sized_region(partial,INFO,ORIGIN,ROTATION,.06,.04)
    mask = color_mask(cv2.cvtColor(image,cv2.COLOR_BGR2HSV),'red',kernel)
    assert not cube_sized_region(partial,INFO,ORIGIN,ROTATION,.06,.04,mask)
