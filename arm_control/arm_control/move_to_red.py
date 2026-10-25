"""Prepare, inspect, then explicitly execute one simulated approach to red."""
import math
import time

import rclpy
from geometry_msgs.msg import Point, PointStamped
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


class MoveToRed(Node):
    def __init__(self):
        super().__init__('move_to_red')
        self.declare_parameter('clearance', 0.10)
        self.clearance = float(self.get_parameter('clearance').value)
        if not math.isfinite(self.clearance) or self.clearance < 0.05:
            raise ValueError('This exercise requires clearance >= 0.05 m')
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
        self.subs = [
            self.create_subscription(PointStamped, '/arm/perception/red/position', self.on_position, 10),
            self.create_subscription(Bool, '/arm/perception/red/position_valid', self.on_valid, 10),
            self.create_subscription(JointState, '/joint_states', self.on_joints, qos_profile_sensor_data),
        ]
        self.service_handles = [
            self.create_service(Trigger, '/arm/prepare_red_move', self.prepare),
            self.create_service(Trigger, '/arm/execute_red_move', self.execute),
            self.create_service(Trigger, '/arm/cancel_red_move', self.cancel),
        ]
        # A steady timer lets cancellation and clock-reset checks work while paused.
        self.timer = self.create_timer(0.05, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.last_preview_wall = 0.0
        self.set_status('Idle. Prepare a move to preview it; execute is a separate command.')

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
        self.joints.update(dict(zip(msg.name, msg.position)))
        # A complete arm sample is needed; unrelated publishers cannot refresh it.
        if all(j in msg.name for j in JOINTS):
            self.joint_stamp = msg.header.stamp

    def fresh_feedback(self):
        return (self.joint_stamp is not None and 0 <= self.age(self.joint_stamp) <= 0.5
                and all(j in self.joints and math.isfinite(self.joints[j]) for j in JOINTS))

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

    def prepare(self, request, response):
        if self.active:
            response.message = 'A movement is active; cancel it first.'
            return response
        self.clear_plan()
        if not self.fresh_target() or not self.fresh_feedback():
            response.message = 'Need fresh valid perception and joint feedback. Press Play.'
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
                    path = [interpolate(start,goal,i/40) for i in range(41)]
                    for sample in path:
                        check_limits(sample)
                        check_ground(self.model, sample, gripper)
                    break
                except ValueError as exc:
                    error = exc
            else:
                raise ValueError(f'No candidate passed sampled checks: {error}')
        except ValueError as exc:
            response.message = str(exc)
            return response
        duration = duration_for(start,goal)
        self.plan = dict(start=start, goal=goal, surface=surface, target=target,
                         gripper=gripper, duration=duration, path=path, prepared=self.now_seconds())
        response.success = True
        response.message = (f'Prepared {duration:.1f}s approach. Inspect Planned Motion in RViz. '
                            'Only joint limits and sampled floor clearance checked; no obstacle/self-collision planner. '
                            'Execute within 60 simulation seconds or prepare again.')
        self.set_status(response.message)
        return response

    def execute(self, request, response):
        if self.active:
            response.message = 'Already moving.'
            return response
        if self.plan is None:
            response.message = 'Prepare and inspect a move first.'
            return response
        if not self.fresh_target() or not self.fresh_feedback():
            response.message = 'Execution needs fresh target and joint feedback. Press Play.'
            return response
        age = self.now_seconds()-self.plan['prepared']
        if not 0 <= age <= 60 or math.dist(self.surface(),self.plan['surface']) > 0.015:
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
            if not self.fresh_feedback():
                self.stop_motion('Aborted: joint feedback is stale.')
                return
            if max(abs(a-b) for a,b in zip(self.measured(),self.last_command)) > 0.25:
                self.stop_motion('Aborted: tracking error exceeded 0.25 rad.')
                return
            if self.fresh_target() and math.dist(self.surface(),self.plan['surface']) > 0.03:
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
    node = MoveToRed()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if rclpy.ok():
            node.stop_motion('Movement node stopped.')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
