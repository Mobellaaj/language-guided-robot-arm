"""Known support surfaces for the single-layer, 4 cm cube exercise."""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from arm_perception.red_localizer import intersect_plane

DEFAULTS = dict(platform_x=0.28, platform_y=-0.23, platform_size_x=0.16,
                platform_size_y=0.14, platform_top=0.02, cube_size=0.04,
                floor_x=0.28, floor_y=0.23, floor_size_x=0.16, floor_size_y=0.14)


@dataclass
class PlatformLayout:
    x: float = 0.28
    y: float = -0.23
    sx: float = 0.16
    sy: float = 0.14
    top: float = 0.02
    cube: float = 0.04
    floor_x: float = 0.28
    floor_y: float = 0.23
    floor_sx: float = 0.16
    floor_sy: float = 0.14

    @classmethod
    def from_node(cls, node):
        values = []
        for name, default in DEFAULTS.items():
            node.declare_parameter(name, default)
            values.append(float(node.get_parameter(name).value))
        layout = cls(*values)
        if not all(math.isfinite(v) for v in values) or min(
                layout.sx, layout.sy, layout.top, layout.cube, layout.floor_sx, layout.floor_sy) <= 0:
            raise ValueError('Platform dimensions/heights and cube size must be positive and finite')
        for sx, sy in ((layout.sx, layout.sy), (layout.floor_sx, layout.floor_sy)):
            if sx < 0.08+math.sqrt(2)*layout.cube+0.01 or sy < math.sqrt(2)*layout.cube+0.01:
                raise ValueError('Placement areas must fit two gripper-accessible slots')
        return layout

    @property
    def slots(self):
        return self.placement_slots('platform')

    def support_height(self, surface):
        if surface not in ('floor', 'platform'):
            raise ValueError('Choose floor or platform.')
        return self.top if surface == 'platform' else 0.0

    def placement_slots(self, surface):
        z = self.support_height(surface)
        x, y = (self.x, self.y) if surface == 'platform' else (self.floor_x, self.floor_y)
        return {'slot_1': (x-0.04, y, z), 'slot_2': (x+0.04, y, z)}

    def contains(self, point):
        return abs(point[0]-self.x) <= self.sx/2 and abs(point[1]-self.y) <= self.sy/2

    def surface(self, origin, direction, floor_top):
        floor = intersect_plane(origin, direction, floor_top)
        raised = intersect_plane(origin, direction, self.top+self.cube)
        if self.contains(raised):
            return raised
        if self.contains(floor):
            raise ValueError('Pixel is at an ambiguous floor/platform boundary')
        return floor

    def slot_polygon(self, center, half=0.028):
        x, y, z = center
        return np.array([[x-half,y-half,z], [x+half,y-half,z],
                         [x+half,y+half,z], [x-half,y+half,z]], dtype=float)


def project_polygon(world_points, camera_origin, camera_rotation, info):
    optical = (camera_rotation.T @ (world_points-camera_origin).T).T
    if np.any(optical[:,2] <= 0):
        return None
    distortion = np.asarray(info.d, dtype=float) if len(info.d) else None
    uv, _ = cv2.projectPoints(optical, np.zeros(3), np.zeros(3),
                             np.asarray(info.k).reshape(3,3), distortion)
    return uv.reshape(-1,2)


def support_visible(frame, polygon, surface):
    """Require visible support, rather than treating missing detections as empty."""
    if polygon is None or not np.isfinite(polygon).all():
        return False
    h, w = frame.shape[:2]
    if np.any(polygon[:,0] < 0) or np.any(polygon[:,0] >= w) or np.any(polygon[:,1] < 0) or np.any(polygon[:,1] >= h):
        return False
    region = np.zeros((h,w), dtype=np.uint8)
    cv2.fillConvexPoly(region, np.rint(polygon).astype(np.int32), 255)
    total = cv2.countNonZero(region)
    if total < 25:
        return False
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if surface == 'platform':
        support = cv2.inRange(hsv, (10,70,35), (30,255,255))
    elif surface == 'floor':
        # The marked floor is light grey; black arm parts and coloured cubes fail.
        support = cv2.inRange(hsv, (0,0,110), (179,45,255))
    else:
        raise ValueError('Choose floor or platform.')
    return cv2.countNonZero(cv2.bitwise_and(support, region))/total >= 0.93


def brown_visible(frame, polygon):
    return support_visible(frame, polygon, 'platform')


def overlap_slot(point, center, cube_size):
    # Conservative XY envelope for a cube at any yaw, plus the placement footprint.
    limit = cube_size/math.sqrt(2)+cube_size/2+0.005
    return abs(point[0]-center[0]) < limit and abs(point[1]-center[1]) < limit
