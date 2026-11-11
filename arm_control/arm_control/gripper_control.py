"""Gradual manual finger commands; contact and successful grasp are not inferred."""
import math

from std_msgs.msg import Float64

from arm_control.motion_math import interpolate

FINGER_JOINTS = ('gripper_opening_joint', 'right_finger_joint')
FINGER_SPEED = 0.015  # m/s per finger, below the URDF's 0.05 m/s limit.
POSITION_SETTLE_TIME = 0.3
CONTACT_SETTLE_TIME = 1.0


def finger_motion_duration(start, target):
    """Smoothstep peak speed is 1.5 * travel / duration."""
    return max(0.3, 1.5*abs(target-start)/FINGER_SPEED)


class GripperControl:
    def __init__(self, node, robot, report):
        self.node = node
        self.report = report
        limits = [robot.find(f"joint[@name='{name}']/limit") for name in FINGER_JOINTS]
        self.closed = max(float(limit.get('lower')) for limit in limits)
        self.opened = min(float(limit.get('upper')) for limit in limits)
        if not 0 <= self.closed < self.opened:
            raise ValueError('The finger joints need compatible nonnegative travel limits')
        self.publisher = node.create_publisher(Float64, '/arm/gripper/target', 10)
        self.positions = None
        self.stamp = None
        self.motion = None
        self.last_result = None

    @property
    def active(self):
        return self.motion is not None

    def on_joints(self, message):
        sample = dict(zip(message.name, message.position))
        if all(name in sample for name in FINGER_JOINTS):
            self.positions = tuple(sample[name] for name in FINGER_JOINTS)
            self.stamp = message.header.stamp

    def feedback_problem(self):
        if self.stamp is None:
            return 'Need joint feedback from both fingers. Check Gazebo playback and /joint_states.'
        if not all(math.isfinite(p) and self.closed-0.002 <= p <= self.opened+0.002
                   for p in self.positions):
            return 'Finger feedback is invalid or outside the travel limits.'
        age = self.node.age(self.stamp)
        if not -0.05 <= age <= 0.5:
            return f'Finger feedback is not current (age {age:.3f} s).'
        return ''

    def hold_problem(self, fingers, gap):
        problem = self.feedback_problem()
        if problem:
            return problem
        left, right = self.positions
        if abs(left-right) > 0.008:
            return 'Finger asymmetry exceeded 8 mm; possible off-center contact or slip.'
        if abs(left+right-gap) > 0.008:
            return 'Finger gap changed by more than 8 mm; possible loss of the cube.'
        if fingers is not None and max(abs(a-b) for a,b in zip(self.positions, fingers)) > 0.004:
            return 'An individual finger shifted more than 4 mm since contact; possible slip.'
        return ''

    def start(self, action):
        if action not in ('open', 'close'):
            raise ValueError('Gripper action must be open or close')
        problem = self.feedback_problem()
        if problem:
            raise ValueError(problem)
        if not self.publisher.get_subscription_count():
            raise ValueError('The gripper command bridge is unavailable.')
        # One shared target must not initially reverse either finger.
        edge = max(self.positions) if action == 'open' else min(self.positions)
        start = max(self.closed, min(self.opened, edge))
        target = self.opened if action == 'open' else self.closed
        duration = finger_motion_duration(start, target)
        self.last_result = None
        self.motion = dict(action=action, start=start, target=target, duration=duration,
                           elapsed=0.0, stable_since=None,
                           contact_anchor=None, contact_since=None)
        return (f'Gripper {action} started; target travel {target:.3f} m per finger. '
                'Completion is reported on /arm/control/motion_status. '
                'Closing is not proof of a grasp. Any previous arm preview is cleared.')

    def send(self, target):
        self.publisher.publish(Float64(data=float(target)))

    def cancel(self, preserve_target=False):
        if self.active:
            self.last_result = dict(kind='cancelled', action=self.motion['action'])
        # A measured-position target removes squeezing force. Retain the
        # existing target when interrupting a possible grasp instead.
        if self.active and not preserve_target and not self.feedback_problem():
            self.send(max(self.closed, min(self.opened, sum(self.positions)/2)))
        self.motion = None

    def tick(self, dt):
        if not self.active or dt <= 0:
            return
        problem = self.feedback_problem()
        if problem:
            self.cancel()
            self.last_result = dict(kind='failed', reason=problem)
            self.report(f'Gripper aborted: {problem} Last available finger target is retained.')
            return
        motion = self.motion
        motion['elapsed'] += min(dt, 0.1)
        fraction = min(1.0, motion['elapsed']/motion['duration'])
        target = interpolate((motion['start'],), (motion['target'],), fraction)[0]
        self.send(target)
        if fraction < 1:
            return
        gap = sum(self.positions)
        error = max(abs(p-motion['target']) for p in self.positions)
        if error <= 0.001:
            if motion['stable_since'] is None:
                motion['stable_since'] = motion['elapsed']
            if motion['elapsed']-motion['stable_since'] >= POSITION_SETTLE_TIME:
                self.motion = None
                self.last_result = dict(kind='reached', action=motion['action'], gap=gap)
                self.report(f'Gripper {motion["action"]} completed. Measured gap {gap*1000:.1f} mm. '
                            'No grasp verification has been performed.')
                return
        else:
            motion['stable_since'] = None
        # A timeout alone is not evidence of settled contact. Require both
        # fingers to stay within 0.5 mm of a fixed sample for one second.
        if motion['contact_anchor'] is None or max(
                abs(a-b) for a,b in zip(self.positions, motion['contact_anchor'])) > 0.0005:
            motion['contact_anchor'] = tuple(self.positions)
            motion['contact_since'] = motion['elapsed']
        contact_settled = motion['elapsed']-motion['contact_since'] >= CONTACT_SETTLE_TIME
        if motion['action'] == 'close' and contact_settled:
            # The full closing target has been sent and both fingers have
            # settled. Keep squeezing; a timeout is not needed to report contact.
            self.motion = None
            self.last_result = dict(kind='blocked', action='close', gap=gap)
            self.report(f'Gripper closure settled. Measured gap {gap*1000:.1f} mm; '
                        'possible contact or obstruction, not a verified grasp. '
                        'The closing target remains active; use open gripper to release.')
            return
        if motion['elapsed'] > motion['duration']+3.0:
            if motion['action'] == 'close':
                self.cancel(preserve_target=True)
                reason = 'Closing timed out while the fingers were still moving; no settled contact. No lift requested.'
                self.last_result = dict(kind='failed', action='close', gap=gap, reason=reason)
                self.report(reason)
            else:
                self.cancel()
                reason = (f'Opening timed out: requested gap {2*motion["target"]*1000:.1f} mm, '
                          f'measured {gap*1000:.1f} mm '
                          f'(left {self.positions[0]*1000:.1f}, right {self.positions[1]*1000:.1f} mm travel). '
                          'Each finger must be within 1 mm of its target. '
                          'The sequence will not advance; check the finger controller or obstruction.')
                self.last_result = dict(kind='failed', action='open', gap=gap, reason=reason)
                self.report(reason)
