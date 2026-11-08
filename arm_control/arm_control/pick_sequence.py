"""Preview and run one pick attempt for the 4 cm cubes on a known support.

No object attachment is created. Finger contact and friction must lift the cube.
MoveIt checks the scene; OMPL plans the approach. Descent and lift stay vertical.
"""
import math
import time

from geometry_msgs.msg import Point
from rclpy.duration import Duration
from visualization_msgs.msg import Marker

from arm_control.ik_solver import forward_kinematics, solve_downward_ik
from arm_control.motion_math import SPEEDS, check_ground, check_limits, interpolate, check_finger_ground


def sample_path(path, fraction):
    u = max(0.0, min(1.0, fraction))
    index = u*u*(3-2*u)*(len(path)-1)
    lo = min(int(index), len(path)-2)
    blend = index-lo
    return tuple(a+blend*(b-a) for a, b in zip(path[lo], path[lo+1]))


def vertical_path(start, end_z):
    """Solve successive IK waypoints at constant X/Y and downward orientation."""
    x, y, start_z = forward_kinematics(start)
    path = [start]
    for i in range(1, 41):
        candidates = solve_downward_ik(x, y, start_z+(end_z-start_z)*i/40)
        path.append(min(candidates, key=lambda q: sum((a-b)**2 for a, b in zip(q, path[-1]))))
    return path


def path_duration(path, minimum=4.0, tool_speed=0.02):
    # Waypoints are traversed with smoothstep: bound peak joint speeds along the path.
    joint_time = max(1.5*(len(path)-1)*abs(b-a)/speed
                     for before, after in zip(path, path[1:])
                     for a, b, speed in zip(before, after, SPEEDS))
    tool_time = 0.0
    if tool_speed is not None:
        points = [forward_kinematics(q) for q in path]
        tool_time = 1.5*(len(path)-1)*max(math.dist(a,b) for a,b in zip(points,points[1:]))/tool_speed
    return max(minimum, joint_time, tool_time)


class PickSequence:
    operation = 'Pick'

    def __init__(self, node):
        self.node = node
        self.program = None
        self.active = False
        self.index = 0
        self.elapsed = 0.0
        self.stable_since = None
        self.last_command = None
        self.last_preview_wall = 0.0
        self.closed_gap = None
        self.closed_positions = None

    def clear(self):
        self.active = False
        self.program = None

    def prepare(self, surface, start, finger, source='floor'):
        n = self.node
        expected_top = n.layout.support_height(source)+n.layout.cube
        if abs(surface[2]-expected_top) > 0.005:
            raise ValueError(f'The selected cube does not match the {source} support height.')
        opened = n.gripper.opened
        approach_z = surface[2]+n.clearance
        grasp_z = surface[2]-0.005  # Tips remain 5 mm above the selected support.
        lift_z = grasp_z+0.08
        candidates = solve_downward_ik(surface[0], surface[1], approach_z)
        candidates.sort(key=lambda q: sum((a-b)**2 for a, b in zip(q, start)))
        error = None
        for above in candidates:
            try:
                n.model.markers(above, (surface[0], surface[1], approach_z), n.get_clock().now().to_msg())
                approach = n.planner.plan(start, above, finger, n.objects)
                descend = vertical_path(above, grasp_z)
                lift = vertical_path(descend[-1], lift_z)
                for path, gap in ((approach, finger), (descend, opened), (lift, 0.02)):
                    for q in path:
                        check_limits(q)
                        check_ground(n.model, q, gap)
                n.planner.validate_fingers(above, finger, opened, n.objects)
                n.planner.validate(descend, opened, n.objects)
                payload = dict(source=surface, offset=grasp_z-(surface[2]-n.layout.cube/2))
                n.planner.validate(lift, .02, n.objects, held=payload)
                break
            except ValueError as exc:
                error = exc
        else:
            raise ValueError(f'No pick path passed the sampled checks: {error}')
        stages = [
            dict(name='approach', path=approach, duration=path_duration(approach, minimum=8.0, tool_speed=None), finger=finger),
            dict(name='open', pose=above, duration=max(1.0, 1.5*abs(opened-finger)/0.01),
                 finger=finger, end_finger=opened),
            dict(name='descend', path=descend, duration=path_duration(descend), finger=opened),
            dict(name='close', pose=descend[-1], duration=1.5*opened/0.01+3.0,
                 finger=opened, end_finger=0.02),
            dict(name='lift', path=lift, duration=path_duration(lift), finger=0.02),
        ]
        self.program = dict(stages=stages, surface=surface, finger=finger,
                            duration=sum(stage['duration'] for stage in stages))
        return self.program

    def check_current_scene(self):
        n = self.node
        stages = self.program['stages']
        surface = self.program['surface']
        payload = dict(source=surface, offset=forward_kinematics(stages[2]['path'][-1])[2]
                       -(surface[2]-n.layout.cube/2))
        n.planner.validate(stages[0]['path'], stages[0]['finger'], n.objects)
        n.planner.validate_fingers(stages[1]['pose'], stages[1]['finger'],
                                   stages[1]['end_finger'], n.objects)
        n.planner.validate(stages[2]['path'], stages[2]['finger'], n.objects)
        n.planner.validate(stages[4]['path'], stages[4]['finger'], n.objects, held=payload)

    def start(self):
        self.active = True
        self.index = 0
        self.closed_gap = None
        self.closed_positions = None
        self.begin_stage()

    def begin_stage(self):
        n = self.node
        stage = self.program['stages'][self.index]
        self.elapsed = 0.0
        self.stable_since = None
        n.set_status(f'{self.operation} stage {self.index+1}/{len(self.program["stages"])}: {stage["name"]}.')
        if 'path' in stage:
            self.last_command = stage['path'][0]
        else:
            n.gripper.start(stage['name'])
            if stage['name'] == 'close':
                # Until lift completes, cancellation must retain possible-grasp state.
                n.grasp_uncertain = True
                # RGB localization assumes a support plane, not a cube being lifted.
                n.clear_selection()

    def fail(self, reason):
        self.node.stop_motion(f'{self.operation} aborted: {reason} Fingers are not automatically opened.')

    def next_stage(self):
        self.index += 1
        if self.index == len(self.program['stages']):
            self.complete()
            return
        try:
            self.begin_stage()
        except ValueError as error:
            self.fail(str(error))

    def complete(self):
        n = self.node
        gap = sum(n.gripper.positions)
        n.grasp_uncertain = False
        n.held = dict(color=n.target_color, gap=gap, fingers=tuple(n.gripper.positions),
                      offset=self.program['grasp_offset'], source=self.program['surface'])
        self.active = False
        n.clear_plan()
        n.set_status(f'Pick motion completed; holding the closure target. Finger gap {gap*1000:.1f} mm. '
                     'Held-cube context recorded from the grasp pose and finger feedback.')

    def decorate_preview(self, markers, q, stage):
        pass

    def tick(self, dt):
        if not self.active or dt <= 0:
            return
        n = self.node
        problem = n.feedback_problem() or n.gripper.feedback_problem()
        if problem:
            self.fail(problem)
            return
        stage = self.program['stages'][self.index]
        if stage['name'] in ('descend', 'close'):
            try:
                check_finger_ground(n.model, n.measured(), sum(n.gripper.positions)/2)
            except ValueError as error:
                self.fail(str(error))
                return
        if stage['name'] in ('approach', 'open', 'descend') and n.fresh_target():
            if math.dist(n.surface(), self.program['surface']) > 0.015:
                self.fail(f'The observed target shifted {math.dist(n.surface(), self.program["surface"])*1000:.1f} mm '
                          'before closing (limit 15 mm). It may have moved or become partly occluded; '
                          'return to observation and select again.')
                return
        if stage['name'] in ('open', 'close'):
            if (max(abs(a-b) for a,b in zip(n.measured(), stage['pose'])) > 0.08
                    or math.dist(forward_kinematics(n.measured()),
                                 forward_kinematics(stage['pose'])) > 0.015):
                self.fail('Arm drifted while operating the fingers; the sequence will not advance.')
                return
            if n.gripper.active:
                return
            result = n.gripper.last_result
            if result is None or result['kind'] not in ('reached', 'blocked'):
                self.fail((result or {}).get('reason', 'The gripper operation did not finish normally.'))
                return
            if stage['name'] == 'open' and result['kind'] != 'reached':
                self.fail('The fingers did not fully open.')
                return
            if stage['name'] == 'open':
                n.grasp_uncertain = False
                n.held = None
            if stage['name'] == 'close':
                left, right = n.gripper.positions
                gap = left+right
                if not 0.025 <= gap <= 0.060 or abs(left-right) > 0.008:
                    self.fail(f'Finger gap ({gap*1000:.1f} mm) or asymmetry is not consistent '
                              'with this cube. No lift requested.')
                    return
                self.closed_gap = gap
                self.closed_positions = tuple(n.gripper.positions)
                self.program['grasp_offset'] = (forward_kinematics(n.measured())[2]
                                                -(self.program['surface'][2]-n.layout.cube/2))
                n.set_status('Finger gap is consistent with possible cube contact; attempting lift. '
                             'This is not yet a verified grasp.')
            self.next_stage()
            return
        if max(abs(a-b) for a, b in zip(n.measured(), self.last_command)) > 0.25:
            self.fail('Arm tracking error exceeded 0.25 rad.')
            return
        if stage['name'] == 'lift':
            problem = n.gripper.hold_problem(self.closed_positions, self.closed_gap)
            if problem:
                self.fail(problem)
                return
        self.elapsed += min(dt, 0.1)
        fraction = min(1.0, self.elapsed/stage['duration'])
        command = sample_path(stage['path'], fraction)
        n.send(command)
        self.last_command = command
        if fraction == 1:
            goal = stage['path'][-1]
            joint_error = max(abs(a-b) for a, b in zip(n.measured(), goal))
            position_error = math.dist(forward_kinematics(n.measured()), forward_kinematics(goal))
            # Descent needs tighter settling than a distant approach.
            tolerance = stage.get('tolerance', 0.005 if stage['name'] == 'descend' else 0.01)
            if joint_error <= 0.035 and position_error <= tolerance:
                if self.stable_since is None:
                    self.stable_since = self.elapsed
                if self.elapsed-self.stable_since >= 0.4:
                    self.next_stage()
                    return
            else:
                self.stable_since = None
            if self.elapsed > stage['duration']+6.0:
                self.fail(f'{stage["name"]} did not settle at its target.')

    def draw_preview(self, fraction=None):
        if self.program is None or time.monotonic()-self.last_preview_wall < 0.1:
            return
        self.last_preview_wall = time.monotonic()
        if self.active:
            stage = self.program['stages'][self.index]
            fraction = min(1.0, self.elapsed/stage['duration']) if 'path' in stage else 1.0
        else:
            elapsed = max(0.0, min(1.0, fraction))*self.program['duration']
            stage = self.program['stages'][-1]
            for candidate in self.program['stages']:
                stage = candidate
                if elapsed <= stage['duration']:
                    break
                elapsed -= stage['duration']
            fraction = min(1.0, elapsed/stage['duration'])
        if 'path' in stage:
            q, finger = sample_path(stage['path'], fraction), stage['finger']
        else:
            q = stage['pose']
            finger = interpolate((stage['finger'],), (stage['end_finger'],), fraction)[0]
        if self.active and self.node.gripper.positions is not None:
            finger = sum(self.node.gripper.positions)/2
        markers = self.node.model.pose_markers(q, self.node.get_clock().now().to_msg(), finger)
        for marker in markers.markers:
            marker.ns = 'planned_motion'
        line = Marker()
        line.header.frame_id = 'base_link'
        line.header.stamp = self.node.get_clock().now().to_msg()
        line.ns = 'planned_tool_path'
        line.id = 1000
        line.type = Marker.LINE_STRIP
        line.pose.orientation.w = 1.0
        line.scale.x = 0.003
        line.color.r = line.color.b = line.color.a = 1.0
        line.lifetime = Duration(seconds=0.8).to_msg()
        for path_stage in self.program['stages']:
            for path_q in path_stage.get('path', []):
                x, y, z = forward_kinematics(path_q)
                line.points.append(Point(x=float(x), y=float(y), z=float(z)))
        markers.markers.append(line)
        self.decorate_preview(markers, q, stage)
        self.node.marker_pub.publish(markers)
