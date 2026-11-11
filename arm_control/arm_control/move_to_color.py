"""Prepare, inspect, then explicitly execute a color approach or return to an observation pose."""
import math
import time

import rclpy
from geometry_msgs.msg import Point, PointStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from vision_msgs.msg import Detection3DArray
from arm_interfaces.msg import PlatformState
from arm_interfaces.srv import PrepareTask
from arm_perception.platform_geometry import PlatformLayout
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from arm_control.pick_sequence import PickSequence, sample_path, path_duration
from arm_control.scene_planner import ScenePlanner
from arm_control.task_runner import TaskRunner
from arm_control.place_sequence import PlaceSequence
from arm_control.platform_manager import PlatformManager
from arm_control.gripper_control import GripperControl
from arm_control.target_selection import COLORS, SIDES, choose_target, match_selected
from arm_control.ik_solver import solve_downward_ik, forward_kinematics
from arm_control.ik_visualization import PreviewModel
from arm_control.motion_math import JOINTS, TOPICS, check_limits, UNLOADED_SPEEDS, UNLOADED_MIN_DURATION


class MoveToColor(Node):
    def __init__(self):
        super().__init__('move_to_color')
        self.declare_parameter('target_color', 'red')
        self.target_color = self.get_parameter('target_color').value

        if self.target_color not in ('red', 'green', 'blue'):
            raise ValueError('target_color must be red, green, or blue')

        self.declare_parameter('target_side', 'unique')
        self.target_side = self.get_parameter('target_side').value
        if self.target_side not in SIDES:
            raise ValueError('target_side must be one of: '+', '.join(SIDES))

        self.declare_parameter('clearance', 0.10)
        self.clearance = float(self.get_parameter('clearance').value)
        if not math.isfinite(self.clearance) or self.clearance < 0.05:
            raise ValueError('This exercise requires clearance >= 0.05 m')
        # Upright arm: base, shoulder, elbow, wrist (radians).
        self.declare_parameter('observation_joints', [0.0, 0.0, 0.0, 0.0])
        self.observation_joints = tuple(self.get_parameter('observation_joints').value)
        check_limits(self.observation_joints)
        self.layout = PlatformLayout.from_node(self)
        self.platform = PlatformManager(self, self.layout)
        self.floor = PlatformManager(self, self.layout, 'floor')
        self.held = None
        self.grasp_uncertain = False
        self.model = PreviewModel()
        self.position = None
        self.position_valid = False
        self.objects = None
        self.declare_parameter('obstacle_boxes', Parameter.Type.DOUBLE_ARRAY)
        flat_boxes = list(self.get_parameter_or('obstacle_boxes').value or [])
        if len(flat_boxes) % 6:
            raise ValueError('obstacle_boxes must contain centre XYZ and size XYZ for each box.')
        self.planner = ScenePlanner(self.model, self.layout,
                                    [flat_boxes[i:i+6] for i in range(0, len(flat_boxes), 6)])
        self.selection_anchor = None
        self.selection_source = 'floor'
        self.target_error = 'Select a target first.'
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
        self.selection_pub = self.create_publisher(Marker, '/arm/control/selected_target', 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/arm/control/motion_status', qos)
        self.task_status_pub = self.create_publisher(String, '/arm/control/task_status', qos)
        self.scene_pub = self.create_publisher(MarkerArray, '/arm/control/planning_scene', qos)
        self.scene_pub.publish(self.planner.markers())
        self.gripper = GripperControl(self, self.model.robot, self.set_status)
        self.pick = PickSequence(self)
        self.place = PlaceSequence(self)
        self.task = TaskRunner(self)
        self.subs = [
            self.create_subscription(PlatformState, '/arm/perception/platform', self.on_platform, 10),
            self.create_subscription(PlatformState, '/arm/perception/floor',
                                     lambda message: self.on_surface(message, 'floor'), 10),
            self.create_subscription(Detection3DArray, '/arm/perception/objects', self.on_objects, 10),
            self.create_subscription(JointState, '/joint_states', self.on_joints, qos_profile_sensor_data),
        ]
        self.add_on_set_parameters_callback(self.validate_color_change)
        self.add_post_set_parameters_callback(self.apply_color_change)
        self.service_handles = [
            self.create_service(PrepareTask, '/arm/prepare_task',
                                lambda request, response: self.task.prepare(request.task_json, response)),
            self.create_service(Trigger, '/arm/select_target', self.select),
            self.create_service(Trigger, '/arm/prepare_move', self.prepare),
            self.create_service(Trigger, '/arm/prepare_pick', self.prepare_pick),
            self.create_service(Trigger, '/arm/prepare_place', self.prepare_place),
            self.create_service(Trigger, '/arm/inspect_platform', self.inspect_platform),
            self.create_service(Trigger, '/arm/inspect_floor',
                                lambda request, response: self.inspect_platform(request, response, 'floor')),
            self.create_service(Trigger, '/arm/prepare_observation', self.prepare_observation),
            self.create_service(Trigger, '/arm/execute_move', self.execute),
            self.create_service(Trigger, '/arm/cancel_move', self.cancel),
            self.create_service(Trigger, '/arm/open_gripper',
                                lambda request, response: self.command_gripper('open', response)),
            self.create_service(Trigger, '/arm/close_gripper',
                                lambda request, response: self.command_gripper('close', response)),
        ]
        # A steady timer lets cancellation and clock-reset checks work while paused.
        self.timer = self.create_timer(0.05, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.last_preview_wall = 0.0
        self.set_status('Idle. Prepare a text/JSON task or an observation return. Execution is separate.')
        self.task.publish()

    def validate_color_change(self, parameters):
        for parameter in parameters:
            choices = {'target_color': COLORS, 'target_side': SIDES}
            if parameter.name not in choices:
                continue
            if parameter.value not in choices[parameter.name]:
                return SetParametersResult(successful=False,
                                           reason=f'{parameter.name}: choose {choices[parameter.name]}.')
            if (self.active or self.gripper.active or self.pick.active or self.place.active or self.task.busy) and parameter.value != getattr(self, parameter.name):
                return SetParametersResult(
                    successful=False, reason='Movement is active. Wait for completion or cancel first.')
        return SetParametersResult(successful=True)

    def apply_color_change(self, parameters):
        changes = {p.name: p.value for p in parameters
                   if p.name in ('target_color', 'target_side') and p.value != getattr(self, p.name)}
        if not changes:
            return
        self.clear_plan()
        self.clear_selection()
        for name, value in changes.items():
            setattr(self, name, value)
        self.set_status(f'Target request: {self.target_side} {self.target_color}. '
                        'Old plan cleared; select or prepare a new target.')

    def clear_selection(self):
        self.selection_anchor = None
        self.selection_source = 'floor'
        self.position = None
        self.position_valid = False
        self.target_error = 'No target selected.'
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.ns = 'selected_target'
        marker.id = 0
        marker.action = Marker.DELETE
        self.selection_pub.publish(marker)

    def objects_problem(self):
        if self.objects is None:
            return 'No object list received. Start multiple-object perception.'
        if self.objects.header.frame_id != 'base_link':
            return 'Object positions must be in base_link.'
        age = self.age(self.objects.header.stamp)
        if not -0.05 <= age <= 0.5:
            return f'Object list is not current (age {age:.3f} s). Check perception and Gazebo playback.'
        return ''

    def set_selected_position(self, point, header):
        self.position = PointStamped()
        self.position.header = header
        self.position.point.x, self.position.point.y, self.position.point.z = map(float, point)
        self.position_valid = True
        self.target_error = ''
        marker = Marker()
        marker.header = header
        marker.ns = 'selected_target'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.003
        marker.color.r = marker.color.g = marker.color.a = 1.0
        from rclpy.duration import Duration
        marker.lifetime = Duration(seconds=0.5).to_msg()
        for i in range(33):
            angle = 2 * math.pi * i / 32
            marker.points.append(Point(x=point[0]+0.028*math.cos(angle),
                                       y=point[1]+0.028*math.sin(angle), z=point[2]+0.004))
        self.selection_pub.publish(marker)

    def on_objects(self, message):
        self.objects = message
        if self.selection_anchor is None:
            return
        try:
            problem = self.objects_problem()
            if problem:
                raise ValueError(problem)
            point = match_selected(message, self.target_color, self.selection_anchor,
                                   self.selection_source, self.layout)
            if point is None:
                raise ValueError('Selected cube is hidden, moved too far, or cannot be matched uniquely. '
                                 'Return to observation and select again.')
            self.set_selected_position(point, message.header)
        except ValueError as error:
            self.position_valid = False
            self.target_error = str(error)
            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.ns = 'selected_target'
            marker.action = Marker.DELETE
            self.selection_pub.publish(marker)

    def select_surface(self):
        self.clear_selection()
        problem = self.objects_problem()
        if problem:
            raise ValueError(problem)
        point = choose_target(self.objects, self.target_color, self.target_side)
        self.selection_anchor = point
        self.set_selected_position(point, self.objects.header)
        return point

    def select(self, request, response):
        if self.active or self.gripper.active or self.pick.active or self.place.active or self.task.busy:
            response.message = 'A movement is active; cancel it first.'
            return response
        self.clear_plan()
        try:
            point = self.select_surface()
        except ValueError as error:
            response.message = str(error)
            return response
        response.success = True
        response.message = (f'Selected {self.target_side} {self.target_color} at '
                            f'({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) m in base_link. '
                            'Yellow ring only; no motion prepared or executed.')
        self.set_status(response.message)
        return response

    def on_platform(self, message):
        self.on_surface(message, 'platform')

    def on_surface(self, message, surface):
        getattr(self, surface).update(message)
        if self.place.program is None or self.place.program['destination'] != surface:
            return
        center = self.place.program['center']
        marker = Marker()
        marker.header = message.header
        marker.ns = 'selected_target'
        marker.type = Marker.LINE_STRIP
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.003
        marker.color.r = marker.color.g = marker.color.a = 1.0
        from rclpy.duration import Duration
        marker.lifetime = Duration(seconds=0.5).to_msg()
        for dx,dy in ((-.028,-.028),(.028,-.028),(.028,.028),(-.028,.028),(-.028,-.028)):
            marker.points.append(Point(x=center[0]+dx,y=center[1]+dy,z=center[2]+0.002))
        self.selection_pub.publish(marker)

    def inspect_platform(self, request, response, surface='platform'):
        if self.active or self.gripper.active or self.pick.active or self.place.active or self.task.busy:
            response.message = 'Wait for the movement to finish before rescanning reservations.'
            return response
        try:
            response.message = getattr(self, surface).inspect()
            response.success = True
        except ValueError as error:
            response.message = str(error)
        return response

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def age(self, stamp):
        t = Time.from_msg(stamp, clock_type=self.get_clock().clock_type)
        return (self.get_clock().now()-t).nanoseconds / 1e9

    def on_joints(self, msg):
        self.gripper.on_joints(msg)
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
        try:
            check_limits(self.measured())
        except ValueError as error:
            return f'Arm feedback is outside the configured joint limits: {error}'
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
                and -0.05 <= self.age(self.position.header.stamp) <= 0.5
                and all(math.isfinite(v) for v in self.surface()))

    def surface(self):
        p = self.position.point
        return (p.x, p.y, p.z)

    def set_status(self, text):
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def clear_plan(self):
        self.task.discard_prepared()
        self.plan = None
        self.pick.clear()
        self.place.clear()
        self.marker_pub.publish(self.model.clear())

    def input_problem(self, require_target=True):
        problems = []
        feedback_problem = self.feedback_problem()
        if feedback_problem:
            problems.append(feedback_problem + ' Check the joint-state/clock bridges and Gazebo playback.')
        finger_problem = self.gripper.feedback_problem()
        if finger_problem:
            problems.append(finger_problem)
        if require_target and not self.fresh_target():
            problems.append(self.target_error or 'Selected target is stale. Return to observation or select again.')
        return ' '.join(problems)

    def checked_path(self, start, goal, gripper):
        return self.planner.plan(start, goal, gripper, self.objects)

    def store_plan(self, start, goal, gripper, path, response, kind, surface=None):
        duration = path_duration(path, minimum=UNLOADED_MIN_DURATION, tool_speed=None, joint_speeds=UNLOADED_SPEEDS)
        self.plan = dict(start=start, goal=goal, surface=surface,
                         target=forward_kinematics(goal), kind=kind,
                         gripper=gripper, fingers=tuple(self.gripper.positions),
                         duration=duration, path=path, prepared=self.now_seconds())
        response.success = True
        response.message = (f'Prepared {kind} movement ({duration:.1f}s). Inspect Planned Motion in RViz. '
                            'MoveIt collision checks and OMPL route search; static scene. '
                            'Execute within 60 simulation seconds using /arm/execute_move.')
        self.set_status(response.message)
        return response

    def grasp_recovery_message(self):
        if self.grasp_uncertain:
            return ('A grasp or release was interrupted; the cube state is uncertain. '
                    'Inspect Gazebo before explicitly releasing with open gripper at the current pose, '
                    'or restart the simulation. Wait for opening to complete before preparing another movement.')
        return 'A cube may be held. Place it on the platform or explicitly release it first.'

    def prepare_observation(self, request, response):
        if self.active or self.gripper.active or self.pick.active or self.place.active or self.task.busy:
            response.message = 'A movement is active; cancel it first.'
            return response
        if self.grasp_uncertain or self.held is not None:
            response.message = self.grasp_recovery_message()
            return response
        self.clear_plan()
        self.clear_selection()
        problem = self.input_problem(require_target=False)
        if problem:
            response.message = problem
            return response
        start = self.measured()
        gripper = sum(self.gripper.positions)/2
        try:
            path = self.checked_path(start, self.observation_joints, gripper)
        except ValueError as exc:
            response.message = str(exc)
            return response
        return self.store_plan(start, self.observation_joints, gripper, path,
                               response, kind='observation')

    def prepare(self, request, response):
        if self.active or self.gripper.active or self.pick.active or self.place.active or self.task.busy:
            response.message = 'A movement is active; cancel it first.'
            return response
        if self.grasp_uncertain or self.held is not None:
            response.message = self.grasp_recovery_message()
            return response
        self.clear_plan()
        try:
            self.select_surface()
        except ValueError as error:
            response.message = str(error)
            return response
        problem = self.input_problem()
        if problem:
            response.message = problem
            return response
        start = self.measured()
        surface = self.surface()
        target = (surface[0], surface[1], surface[2]+self.clearance)
        gripper = sum(self.gripper.positions)/2
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

    def prepare_pick(self, request, response, selected=None, source='floor'):
        if self.active or self.gripper.active or self.pick.active or self.place.active or self.task.busy:
            response.message = 'A movement is active; wait for completion or cancel first.'
            return response
        if self.grasp_uncertain or self.held is not None:
            response.message = self.grasp_recovery_message()
            return response
        self.clear_plan()
        try:
            if selected is None:
                surface = self.select_surface()
            else:
                self.selection_source = source
                self.selection_anchor = selected
                self.set_selected_position(selected, self.objects.header)
                surface = selected
            self.planner.capture(self.objects)
            problem = self.input_problem() or self.gripper.feedback_problem()
            if problem:
                raise ValueError(problem)
            start = self.measured()
            finger = sum(self.gripper.positions)/2
            program = self.pick.prepare(surface, start, finger, source)
        except ValueError as error:
            self.pick.clear()
            response.message = str(error)
            return response
        self.plan = dict(kind='pick', start=start, surface=surface, gripper=finger,
                         fingers=tuple(self.gripper.positions),
                         prepared=self.now_seconds(), duration=program['duration'])
        response.success = True
        response.message = (f'Prepared pick attempt ({program["duration"]:.1f}s plus settling): '
                            'approach, open, descend, close, lift. Inspect the full Planned Motion preview. '
                            'MoveIt checks the arm, other cubes and known obstacles. '
                            'Execute within 60 simulation seconds. This is not a verified grasp.')
        self.set_status(response.message)
        return response

    def prepare_place(self, request, response, destination='platform'):
        if self.active or self.gripper.active or self.pick.active or self.place.active or self.task.busy:
            response.message = 'A movement is active; wait or cancel first.'
            return response
        if self.grasp_uncertain:
            response.message = self.grasp_recovery_message()
            return response
        self.clear_plan()
        self.clear_selection()
        try:
            problem = self.input_problem(require_target=False) or self.gripper.feedback_problem() or self.objects_problem()
            if problem:
                raise ValueError(problem)
            start = self.measured()
            finger = sum(self.gripper.positions)/2
            program = self.place.prepare(start, finger, destination)
        except ValueError as error:
            self.place.clear()
            response.message = str(error)
            return response
        self.plan = dict(kind='place',start=start,gripper=finger,prepared=self.now_seconds(),
                         fingers=tuple(self.gripper.positions),
                         duration=program['duration'])
        response.success = True
        response.message = (f'Prepared place into {destination} {program["slot"]}: raise, transfer, lower, open, retreat, '
                            'observe, verify. Inspect Planned Motion before execute. '
                            'MoveIt collision checks; OMPL searches transfer and return routes. '
                            'Execute within 60 simulation seconds.')
        if not self.task.internal:
            self.set_status(response.message)
        return response

    def execute(self, request, response):
        if self.active or self.gripper.active or self.pick.active or self.place.active or self.task.busy:
            response.message = 'Already moving.'
            return response
        if self.plan is None:
            response.message = 'Prepare and inspect a move first.'
            return response
        problem = self.input_problem(require_target=self.plan['kind'] in ('approach', 'pick'))
        if problem:
            response.message = problem
            return response
        age = self.now_seconds()-self.plan['prepared']
        target_moved = (self.plan['kind'] in ('approach', 'pick')
                        and math.dist(self.surface(), self.plan['surface']) > 0.015)
        if not 0 <= age <= 60 or target_moved:
            self.clear_plan()
            response.message = 'Plan expired or target moved. Prepare again.'
            return response
        if max(abs(a-b) for a,b in zip(self.measured(),self.plan['start'])) > 0.03:
            self.clear_plan()
            response.message = 'Arm moved since preparation. Prepare again.'
            return response
        if max(abs(a-b) for a,b in zip(self.gripper.positions,self.plan['fingers'])) > 0.002:
            self.clear_plan()
            response.message = 'Finger positions changed since preview. Prepare the movement again.'
            return response
        if not all(p.get_subscription_count() for p in self.publishers_by_joint):
            response.message = 'One or more command bridges are unavailable.'
            return response
        if self.plan['kind'] == 'place':
            try:
                problem = self.gripper.feedback_problem()
                if problem:
                    raise ValueError(problem)
                if self.held is None:
                    raise ValueError('Held-cube context changed. Check the cube and prepare again.')
                problem = self.gripper.hold_problem(self.held.get('fingers'),self.held['gap']) or self.objects_problem()
                if problem:
                    raise ValueError(problem)
                self.place.check_current_scene()
                self.place.manager.assert_empty(self.place.program['slot'])
                if not self.gripper.publisher.get_subscription_count():
                    raise ValueError('Gripper command bridge is unavailable.')
            except ValueError as error:
                self.clear_plan()
                response.message = str(error)
                return response
            self.last_clock = self.now_seconds()
            self.place.start()
            self.task.started()
            response.success = True
            response.message = 'Placement started. Completion requires camera confirmation after retreat.'
            return response
        if self.plan['kind'] == 'pick':
            try:
                self.task.check_start()
                self.pick.check_current_scene()
            except ValueError as error:
                self.clear_plan()
                response.message = str(error)
                return response
            problem = self.gripper.feedback_problem()
            if problem:
                response.message = problem
                return response
            if not self.gripper.publisher.get_subscription_count():
                response.message = 'Gripper command bridge is unavailable.'
                return response
            self.last_clock = self.now_seconds()
            self.pick.start()
            self.task.started()
            response.success = True
            if self.task.active and self.task.request['action'] == 'pick_and_place':
                response.message = 'Combined task started. Placement will follow pickup automatically.'
            else:
                response.message = 'Pick attempt started. Watch stage reports in the launch terminal.'
            return response
        try:
            self.planner.validate(self.plan['path'], self.plan['gripper'], self.objects)
        except ValueError as error:
            self.clear_plan()
            response.message = str(error)
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

    def command_gripper(self, action, response):
        if self.active or self.gripper.active or self.pick.active or self.place.active or self.task.busy:
            response.message = 'A movement is active; wait for completion or cancel first.'
            return response
        try:
            response.message = self.gripper.start(action)
        except ValueError as error:
            response.message = str(error)
            return response
        if action == 'open' and self.held is not None:
            # Clear payload state only after confirmed opening, not request acceptance.
            self.grasp_uncertain = True
        # Finger travel changes invalidate the geometry used for an old arm preview.
        self.clear_plan()
        response.success = True
        self.set_status(response.message)
        return response

    def send(self, angles):
        for pub, angle in zip(self.publishers_by_joint, angles):
            pub.publish(Float64(data=float(angle)))
        self.last_command = angles

    def stop_motion(self, reason, reset_grasp=False, task_state='failed'):
        task_was_active = self.task.active
        self.task.interrupted(task_state)
        # Keep the existing finger target when contact may already have occurred.
        # Cancelling the arm must not silently release a cube or forget an interrupted grasp.
        if self.pick.active and self.pick.program is not None:
            if self.pick.program['stages'][self.pick.index]['name'] in ('close', 'lift'):
                self.grasp_uncertain = True
        if self.gripper.active and self.gripper.motion['action'] == 'close':
            self.grasp_uncertain = True
        if self.active or self.pick.active or self.place.active or task_was_active:
            # Hold the arm for either a standalone movement or a pick phase.
            hold = self.measured() if self.fresh_feedback() else self.last_command
            if hold is not None:
                self.send(hold)
        self.active = False
        self.pick.clear()
        self.gripper.cancel(preserve_target=self.grasp_uncertain or self.held is not None)
        self.clear_plan()
        self.clear_selection()
        if reset_grasp:
            self.held = None
            self.grasp_uncertain = False
        if self.grasp_uncertain:
            reason += ' Existing finger target retained. ' + self.grasp_recovery_message()
        elif self.held is not None:
            reason += ' Held-cube context and closure target retained; prepare a new placement when ready.'
        self.set_status(reason)
        self.task.publish(task_state, reason)
        return reason

    def cancel(self, request, response):
        response.message = self.stop_motion('Cancelled. Arm hold requested; old preview discarded. '
                                            'A new movement must be prepared before execute.', task_state='cancelled')
        response.success = True
        return response

    def draw_plan(self, fraction):
        if self.plan['kind'] == 'place':
            self.place.draw_preview(fraction)
            return
        if self.plan['kind'] == 'pick':
            self.pick.draw_preview(fraction)
            return
        q = sample_path(self.plan['path'], fraction)
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
            self.held = None
            self.grasp_uncertain = False
            self.planner.capture(None)
            for manager in (self.platform, self.floor):
                manager.message = None
                manager.stability.clear()
                manager.reserved.clear()
            self.stop_motion('Simulation clock reset; prepare again.', reset_grasp=True)
            return
        gripper_was_active = self.gripper.active
        self.gripper.tick(dt)
        if gripper_was_active and not self.gripper.active:
            result = self.gripper.last_result or {}
            if result.get('action') == 'open' and result.get('kind') == 'reached':
                self.held = None
                self.grasp_uncertain = False
        self.task.tick()
        self.task.publish()
        if self.place.active:
            self.place.tick(dt)
            self.place.draw_preview()
            return
        if self.pick.active:
            self.pick.tick(dt)
            self.pick.draw_preview()
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
            self.send(sample_path(self.plan['path'], fraction))
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
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node.context.ok():
                node.stop_motion('Movement node stopped.', task_state='cancelled')
        except RCLError:
            # SIGINT can close ROS between the context check and a publish.
            # Suppress only that shutdown race; keep genuine runtime errors visible.
            if node.context.ok():
                raise
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
