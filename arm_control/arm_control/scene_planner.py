"""Turn our known world and camera observations into MoveIt collision geometry."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import Marker, MarkerArray
from arm_planning import Planner

APPROACH_CLEARANCE = 0.008
TRANSFER_CLEARANCE = 0.020


class ScenePlanner:
    def __init__(self, model, layout, obstacle_boxes=()):
        self.layout = layout
        self.obstacle_boxes = [tuple(map(float, box)) for box in obstacle_boxes]
        if any(len(box) != 6 or not all(math.isfinite(v) for v in box)
               or min(box[3:]) <= 0 for box in self.obstacle_boxes):
            raise ValueError('Each obstacle needs finite centre XYZ and positive size XYZ.')
        srdf = Path(get_package_share_directory('arm_planning'))/'config/arm.srdf'
        self.engine = Planner(ET.tostring(model.robot, encoding='unicode'), srdf.read_text())
        self.snapshot = []

    def capture(self, objects):
        # Keep other cubes in the scene when the arm later hides them.
        # A new pick starts a new snapshot; this is not moving-object tracking.
        self.snapshot = self.observed_centres(objects)

    def remember_placement(self, source, centre):
        self.snapshot = self.scene(None, removed=source, extra=[centre])[1]

    def observed_centres(self, objects):
        if objects is None:
            return []
        if objects.header.frame_id != 'base_link':
            raise ValueError('Planning observations must use base_link.')
        centres = []
        for detection in objects.detections:
            if detection.results:
                p = detection.results[0].pose.pose.position
                centre = (p.x, p.y, p.z-self.layout.cube/2)
                if not all(math.isfinite(v) for v in centre):
                    raise ValueError('An observed cube position is not finite.')
                centres.append(centre)
        return centres

    def scene(self, objects, held=None, removed=None, extra=(), transfer=False):
        # Include the platform and the fixed obstacle read from the actual world.
        layout = self.layout
        boxes = [(layout.x, layout.y, layout.top/2, layout.sx, layout.sy, layout.top)]
        # Carrying needs room for tracking lag and movement inside the grasp.
        # Keep the smaller approach margin for close pickup/release poses.
        clearance = TRANSFER_CLEARANCE if transfer else APPROACH_CLEARANCE
        boxes += [(*b[:3], *(v+2*clearance for v in b[3:])) for b in self.obstacle_boxes]
        centres = list(self.snapshot)
        for point in self.observed_centres(objects):
            near = next((i for i, old in enumerate(centres) if math.dist(old, point) < .02), None)
            if near is None:
                centres.append(point)
            else:
                centres[near] = point
        source = held['source'] if held is not None else removed
        if source is not None:
            source_centre = (source[0], source[1], source[2]-layout.cube/2)
            centres = [p for p in centres if math.dist(p, source_centre) >= layout.cube]
        centres.extend(extra)
        return boxes, centres, held['offset'] if held is not None else None

    def plan(self, start, goal, finger, objects, held=None, keep_down=False,
             removed=None, extra=()):
        if held is not None and not keep_down:
            raise ValueError('Carrying paths must keep the gripper pointing down.')
        boxes, centres, offset = self.scene(objects, held, removed, extra, transfer=held is not None)
        return self.engine.plan(start, goal, finger, boxes, centres,
                                self.layout.cube, offset, keep_down)

    def validate(self, path, finger, objects, held=None, removed=None, extra=(), transfer=False):
        boxes, centres, offset = self.scene(objects, held, removed, extra, transfer=transfer)
        self.engine.validate(path, finger, boxes, centres, self.layout.cube, offset)

    def validate_fingers(self, q, before, after, objects, held=None, removed=None, extra=()):
        steps = max(1, math.ceil(abs(after-before)/.003))
        for i in range(steps+1):
            self.validate([q], before+(after-before)*i/steps, objects, held=held,
                          removed=removed, extra=extra)

    def markers(self):
        result = MarkerArray()
        for i, box in enumerate(self.obstacle_boxes):
            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.ns = 'known_obstacles'
            marker.id = i
            marker.type = Marker.CUBE
            marker.pose.orientation.w = 1.
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = box[:3]
            marker.scale.x, marker.scale.y, marker.scale.z = box[3:]
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = .65, .65, .65, .65
            result.markers.append(marker)
        return result
