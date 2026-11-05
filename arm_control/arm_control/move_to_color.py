"""Prepare, inspect, then explicitly execute a color approach or return to an observation pose."""
import math
import time

import rclpy
from geometry_msgs.msg import Point, PointStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from arm_control.ik_solver import solve_downward_ik, forward_kinematics
from arm_control.ik_visualization import PreviewModel
from arm_control.motion_math import JOINTS, TOPICS, interpolate, duration_for, check_limits, check_ground


class MoveToColor(Node):
    def __init__(self):
        super().__init__('move_to_color')
        self.declare_parameter('target_color', 'red')
        self.target_color = self.get_parameter('target_color').value

        if self.target_color not in ('red', 'green', 'blue'):
            raise ValueError('target_color must be red, green, or blue')

        self.declare_parameter('clearance', 0.10)
        self.clearance = float(self.get_parameter('clearance').value)
        if not math.isfinite(self.clearance) or self.clearance < 0.05:
            raise ValueError('This exercise requires clearance >= 0.05 m')
        # Upright arm: base, shoulder, elbow, wrist (radians).
        self.declare_parameter('observation_joints', [0.0, 0.0, 0.0, 0.0])
        self.observation_joints = tuple(self.get_parameter('observation_joints').value)
        check_limits(self.observation_joints)
        self.model = PreviewModel()
        self.position = None
        self.position_valid = False
        self.joints = {}
        self.joint_stamp = None
        self.plan = None
        self.active = False
        self.elapsed = 0.0
        self.stable_since = None
        self.last_command = None
        self.last_clock = self.now_seconds()
        self.publishers_by_joint = [self.create_publisher(Float64, t, 10) for t in TOPICS]
        self.marker_pub = self.create_publisher(MarkerArray, '/arm/control/move_preview', 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/arm/control/motion_status', qos)
        self.subs = self.target_subscriptions() + [
            self.create_subscription(JointState, '/joint_states', self.on_joints, qos_profile_sensor_data),
        ]
        self.add_on_set_parameters_callback(self.validate_color_change)
        self.add_post_set_parameters_callback(self.apply_color_change)
        self.service_handles = [
            self.create_service(Trigger, '/arm/prepare_move', self.prepare),
            self.create_service(Trigger, '/arm/prepare_observation', self.prepare_observation),
            self.create_service(Trigger, '/arm/execute_move', self.execute),
            self.create_service(Trigger, '/arm/cancel_move', self.cancel),
        ]
        # A steady timer lets cancellation and clock-reset checks work while paused.
        self.timer = self.create_timer(0.05, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.last_preview_wall = 0.0
        self.set_status(f'Idle. Selected color: {self.target_color}. Prepare a target move or observation pose, then execute.')

    def target_subscriptions(self):
        color = self.target_color
        prefix = f'/arm/perception/{color}'
        # Ignore an old-color callback if a subscription was replaced meanwhile.
        return [
            self.create_subscription(
                PointStamped, f'{prefix}/position',
                lambda msg: self.on_position(msg) if self.target_color == color else None, 10),
            self.create_subscription(
                Bool, f'{prefix}/position_valid',
                lambda msg: self.on_valid(msg) if self.target_color == color else None, 10),
        ]

    def validate_color_change(self, parameters):
        for parameter in parameters:
            if parameter.name != 'target_color':
                continue
            if parameter.value not in ('red', 'green', 'blue'):
                return SetParametersResult(successful=False, reason='Choose red, green, or blue.')
            if self.active and parameter.value != self.target_color:
                return SetParametersResult(
                    successful=False, reason='Movement is active. Wait for completion or cancel first.')
        return SetParametersResult(successful=True)

    def apply_color_change(self, parameters):
        for parameter in parameters:
            if parameter.name != 'target_color' or parameter.value == self.target_color:
                continue
            self.clear_plan()
            self.position = None
            self.position_valid = False
            for subscription in self.subs[:2]:
                self.destroy_subscription(subscription)
            self.target_color = parameter.value
            self.subs[:2] = self.target_subscriptions()
            self.set_status(f'Selected {self.target_color}. Old plan cleared; '
                            'waiting for a fresh target. Prepare a new move before execution.')

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def age(self, stamp):
        t = Time.from_msg(stamp, clock_type=self.get_clock().clock_type)
        return (self.get_clock().now()-t).nanoseconds / 1e9

    def on_position(self, msg):
        self.position = msg

    def on_valid(self, msg):
        self.position_valid = msg.data

    def on_joints(self, msg):
        sample = dict(zip(msg.name, msg.position))
        # Keep the four arm angles and their stamp from one complete sample.
        # Partial/unrelated messages must not refresh or mix the arm feedback.
        if all(j in sample for j in JOINTS):
            self.joints.update(sample)
            self.joint_stamp = msg.header.stamp

    def feedback_problem(self):
        if self.joint_stamp is None:
            return 'No complete arm joint feedback received on /joint_states.'
        if not all(j in self.joints and math.isfinite(self.joints[j]) for j in JOINTS):
            return 'Joint feedback contains missing or non-finite arm angles.'
        age = self.age(self.joint_stamp)
        # /clock and /joint_states travel independently. A joint sample can
        # arrive just before its corresponding clock update; tolerate 50 ms.
        if age < -0.05:
            return f'Joint feedback is {-age:.3f} s ahead of /clock (limit 0.050 s).'
        if age > 0.5:
            return f'Joint feedback is stale: {age:.3f} simulation seconds old (limit 0.500 s).'
        return ''

    def fresh_feedback(self):
        return not self.feedback_problem()

    def measured(self):
        return tuple(self.joints[j] for j in JOINTS)

    def fresh_target(self):
        return (self.position is not None and self.position_valid
                and self.position.header.frame_id == 'base_link'
                and 0 <= self.age(self.position.header.stamp) <= 0.5
                and all(math.isfinite(v) for v in self.surface()))

    def surface(self):
        p = self.position.point
        return (p.x, p.y, p.z)

    def set_status(self, text):
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def clear_plan(self):
        self.plan = None
        self.marker_pub.publish(self.model.clear())

    def input_problem(self, require_target=True):
        problems = []
        feedback_problem = self.feedback_problem()
        if feedback_problem:
            problems.append(feedback_problem + ' Check the joint-state/clock bridges and Gazebo playback.')
        if require_target and not self.fresh_target():
            problems.append(f'No fresh valid {self.target_color} position. The cube may be hidden '
                            'or perception may be stopped. Use /arm/prepare_observation to regain a view.')
        return ' '.join(problems)

    def checked_path(self, start, goal, gripper):
        path = [interpolate(start, goal, i/40) for i in range(41)]
        for sample in path:
            check_limits(sample)
            check_ground(self.model, sample, gripper)
        return path

    def store_plan(self, start, goal, gripper, path, response, kind, surface=None):
        duration = duration_for(start, goal)
        self.plan = dict(start=start, goal=goal, surface=surface,
                         target=forward_kinematics(goal), kind=kind,
                         gripper=gripper, duration=duration, path=path, prepared=self.now_seconds())
        response.success = True
        response.message = (f'Prepared {kind} movement ({duration:.1f}s). Inspect Planned Motion in RViz. '
                            'Only joint limits and sampled floor clearance checked; no obstacle/self-collision planner. '
                            'Execute within 60 simulation seconds using /arm/execute_move.')
        self.set_status(response.message)
        return response

    def prepare_observation(self, request, response):
        if self.active:
            response.message = 'A movement is active; cancel it first.'
            return response
        self.clear_plan()
        problem = self.input_problem(require_target=False)
        if problem:
            response.message = problem
            return response
        start = self.measured()
        gripper = self.joints.get('gripper_opening_joint', 0.0)
        try:
            path = self.checked_path(start, self.observation_joints, gripper)
        except ValueError as exc:
            response.message = str(exc)
            return response
        return self.store_plan(start, self.observation_joints, gripper, path,
                               response, kind='observation')

    def prepare(self, request, response):
        if self.active:
            response.message = 'A movement is active; cancel it first.'
            return response
        self.clear_plan()
        problem = self.input_problem()
        if problem:
            response.message = problem
            return response
        start = self.measured()
        surface = self.surface()
        target = (surface[0], surface[1], surface[2]+self.clearance)
        gripper = self.joints.get('gripper_opening_joint', 0.0)
        try:
            candidates = solve_downward_ik(*target)
            candidates.sort(key=lambda q: sum((a-b)**2 for a,b in zip(q,start)))
            error = None
            for goal in candidates:
                try:
                    self.model.markers(goal, target, self.get_clock().now().to_msg())
                    path = self.checked_path(start, goal, gripper)
                    break
                except ValueError as exc:
                    error = exc
            else:
                raise ValueError(f'No candidate passed sampled checks: {error}')
        except ValueError as exc:
            response.message = str(exc)
            return response
        return self.store_plan(start, goal, gripper, path, response,
                               kind='approach', surface=surface)

    def execute(self, request, response):
        if self.active:
            response.message = 'Already moving.'
            return response
        if self.plan is None:
            response.message = 'Prepare and inspect a move first.'
            return response
        problem = self.input_problem(require_target=self.plan['kind'] == 'approach')
        if problem:
            response.message = problem
            return response
        age = self.now_seconds()-self.plan['prepared']
        target_moved = (self.plan['kind'] == 'approach'
                        and math.dist(self.surface(), self.plan['surface']) > 0.015)
        if not 0 <= age <= 60 or target_moved:
            self.clear_plan()
            response.message = 'Plan expired or target moved. Prepare again.'
            return response
        if max(abs(a-b) for a,b in zip(self.measured(),self.plan['start'])) > 0.03:
            self.clear_plan()
            response.message = 'Arm moved since preparation. Prepare again.'
            return response
        if not all(p.get_subscription_count() for p in self.publishers_by_joint):
            response.message = 'One or more command bridges are unavailable.'
            return response
        self.active = True
        self.elapsed = 0.0
        self.stable_since = None
        self.last_command = self.plan['start']
        self.last_clock = self.now_seconds()
        response.success = True
        response.message = 'Movement started. Completion is reported on /arm/control/motion_status.'
        self.set_status(response.message)
        return response

    def send(self, angles):
        for pub, angle in zip(self.publishers_by_joint, angles):
            pub.publish(Float64(data=float(angle)))
        self.last_command = angles

    def stop_motion(self, reason):
        if self.active:
            # Hold observed pose when feedback is fresh; otherwise retain last target.
            self.send(self.measured() if self.fresh_feedback() else self.last_command)
        self.active = False
        self.clear_plan()
        self.set_status(reason)

    def cancel(self, request, response):
        self.stop_motion('Cancelled; controllers retain the hold target.')
        response.success = True
        response.message = 'Cancelled.'
        return response

    def draw_plan(self, fraction):
        q = interpolate(self.plan['start'], self.plan['goal'], fraction)
        markers = self.model.pose_markers(q,self.get_clock().now().to_msg(),self.plan['gripper'])
        for marker in markers.markers:
            marker.ns = 'planned_motion'
        line = Marker()
        line.header.frame_id = 'base_link'
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns = 'planned_tool_path';line.id = 1000;line.type = Marker.LINE_STRIP
        line.pose.orientation.w = 1.0;line.scale.x = 0.004
        line.color.r = line.color.b = line.color.a = 1.0
        from rclpy.duration import Duration
        line.lifetime = Duration(seconds=0.8).to_msg()
        for sample in self.plan['path']:
            x,y,z = forward_kinematics(sample)
            line.points.append(Point(x=float(x),y=float(y),z=float(z)))
        markers.markers.append(line)
        self.marker_pub.publish(markers)

    def tick(self):
        now = self.now_seconds()
        dt = now-self.last_clock
        self.last_clock = now
        if dt < -1e-6:
            self.stop_motion('Simulation clock reset; prepare again.')
            return
        if self.plan is None:
            return
        if self.active and dt > 0:
            feedback_problem = self.feedback_problem()
            if feedback_problem:
                self.stop_motion(f'Aborted: {feedback_problem}')
                return
            if max(abs(a-b) for a,b in zip(self.measured(),self.last_command)) > 0.25:
                self.stop_motion('Aborted: tracking error exceeded 0.25 rad.')
                return
            if (self.plan['kind'] == 'approach' and self.fresh_target()
                    and math.dist(self.surface(),self.plan['surface']) > 0.03):
                self.stop_motion('Aborted: observed block moved more than 3 cm.')
                return
            # Pauses do not advance the path; cap time steps after scheduling delays.
            self.elapsed += min(dt,0.1)
            fraction = min(1.0,self.elapsed/self.plan['duration'])
            self.send(interpolate(self.plan['start'],self.plan['goal'],fraction))
            if fraction == 1.0:
                q = self.measured()
                joint_error = max(abs(a-b) for a,b in zip(q,self.plan['goal']))
                position_error = math.dist(forward_kinematics(q),self.plan['target'])
                if joint_error <= 0.04 and position_error <= 0.02:
                    if self.stable_since is None:self.stable_since = self.elapsed
                    if self.elapsed-self.stable_since >= 0.5:
                        self.active = False
                        self.clear_plan()
                        self.set_status(f'Completed. Tool position error {position_error:.4f} m; holding final targets.')
                        return
                else:self.stable_since = None
                if self.elapsed > self.plan['duration']+6:
                    self.stop_motion('Aborted: final pose did not settle within tolerance.')
                    return
        if time.monotonic()-self.last_preview_wall >= 0.1:
            self.last_preview_wall = time.monotonic()
            fraction = (min(1.0,self.elapsed/self.plan['duration']) if self.active
                        else ((now-self.plan['prepared']) % (self.plan['duration']+2))/self.plan['duration'])
            self.draw_plan(fraction)


def main(args=None):
    rclpy.init(args=args)
    node = MoveToColor()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if rclpy.ok():
            node.stop_motion('Movement node stopped.')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
