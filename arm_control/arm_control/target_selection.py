"""Robot-relative selection rules; frame-local detections are not tracked IDs."""
import math
from arm_control.errors import PerceptionNotReady

COLORS = ('red', 'green', 'blue')
SIDES = ('unique', 'leftmost', 'rightmost', 'left_of_robot', 'right_of_robot', 'nearest', 'farthest')


class AmbiguousTarget(ValueError):
    """More than one object fits the request, including near-tied rankings."""


def surface_objects(objects, color=None, source='floor', layout=None):
    if source not in ('floor', 'platform') or (source == 'platform' and layout is None):
        raise ValueError('Platform selection needs the configured surface geometry.')
    if objects.header.frame_id != 'base_link':
        raise ValueError('Object positions must be expressed in base_link.')
    candidates = []
    for detection in objects.detections:
        if not detection.results:
            continue
        detected = detection.results[0].hypothesis.class_id
        if detected not in COLORS or (color is not None and detected != color):
            continue
        p = detection.results[0].pose.pose.position
        point = (p.x, p.y, p.z)
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f'A {detected} position is not finite.')
        height = layout.support_height(source)+layout.cube if layout is not None else .04
        on_platform = layout.contains(point) if layout is not None else False
        if abs(point[2]-height) <= .005 and (on_platform == (source == 'platform')):
            candidates.append((detected, point))
    return candidates


def floor_objects(objects, color=None):
    return surface_objects(objects, color)


def positions_for_color(objects, color):
    return [point for _, point in floor_objects(objects, color)]


def choose_object(objects, color=None, selector='unique', source='floor', layout=None):
    """Choose once in base_link: left is +Y, right is -Y, distance uses XY."""
    if (color is not None and color not in COLORS) or selector not in SIDES:
        raise ValueError('Unsupported color or selector.')
    candidates = surface_objects(objects, color, source, layout)
    if selector in ('left_of_robot', 'right_of_robot'):
        sign = 1 if selector == 'left_of_robot' else -1
        # Keep a 5 mm centre band out of left/right region decisions.
        candidates = [(c,p) for c,p in candidates if sign*p[1] > .005]
    description = f'{color} ' if color else ''
    if not candidates:
        raise PerceptionNotReady(f'No visible {description}{source} cube matches {selector}. Return to observation.')
    if selector in ('unique', 'left_of_robot', 'right_of_robot'):
        if len(candidates) != 1:
            raise AmbiguousTarget(f'Ambiguous: {len(candidates)} {description}{source} cubes match. '
                                  'Specify a color or choose leftmost, rightmost, nearest, or farthest.')
        return candidates[0]
    def score(item):
        point = item[1]
        if selector in ('nearest', 'farthest'):
            return math.hypot(point[0], point[1])
        return point[1]

    candidates.sort(key=score, reverse=selector in ('leftmost', 'farthest'))
    if len(candidates) > 1 and abs(score(candidates[0])-score(candidates[1])) < .005:
        raise AmbiguousTarget(f'The {description}cubes are too close to distinguish {selector} reliably.')
    return candidates[0]


def choose_target(objects, color, side='unique'):
    return choose_object(objects, color, side)[1]


def match_selected(objects, color, anchor, source='floor', layout=None):
    """Keep a fixed selected location; never re-rank when a cube disappears."""
    points = sorted((p for _, p in surface_objects(objects, color, source, layout)),
                    key=lambda p: math.dist(p, anchor))
    if not points or math.dist(points[0], anchor) > .05:
        return None
    if len(points) > 1 and math.dist(points[1], anchor)-math.dist(points[0], anchor) < .01:
        return None
    return points[0]
