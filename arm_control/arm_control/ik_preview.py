import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool
from visualization_msgs.msg import MarkerArray

from arm_control.ik_visualization import PreviewModel

from arm_control.ik_solver import (
    forward_kinematics,
    solve_downward_ik,
)


class IKPreview(Node):
    def __init__(self):
        super().__init__('ik_preview')

        self.declare_parameter('clearance', 0.10)
        self.clearance = float(self.get_parameter('clearance').value)

        if self.clearance <= 0:
            raise ValueError('clearance must be positive')

        self.preview_model = PreviewModel()
        self.marker_publisher = self.create_publisher(
            MarkerArray, '/arm/control/ik_preview', 10
        )
        self.preview_visible = False
        self.position = None
        self.position_valid = False

        self.position_subscription = self.create_subscription(
            PointStamped,
            '/arm/perception/red/position',
            self.on_position,
            10,
        )
        self.valid_subscription = self.create_subscription(
            Bool,
            '/arm/perception/red/position_valid',
            self.on_valid,
            10,
        )

        self.timer = self.create_timer(0.5, self.preview)
        self.get_logger().info(
            'IK preview only: waiting for a fresh red-object position'
        )

    def on_position(self, message):
        self.position = message

    def on_valid(self, message):
        self.position_valid = message.data
        if not message.data:
            self.clear_preview()

    def clear_preview(self):
        if self.preview_visible:
            self.marker_publisher.publish(self.preview_model.clear())
            self.preview_visible = False

    def preview(self):
        if self.position is None or not self.position_valid:
            self.clear_preview()
            return

        message = self.position

        if message.header.frame_id != 'base_link':
            self.clear_preview()
            self.get_logger().warning(
                'Expected position in base_link',
                throttle_duration_sec=2.0,
            )
            return

        stamp = Time.from_msg(
            message.header.stamp,
            clock_type=self.get_clock().clock_type,
        )
        age = (self.get_clock().now() - stamp).nanoseconds / 1e9

        if not 0 <= age <= 0.5:
            self.clear_preview()
            return

        # Perception supplies the surface point.
        # Our tool target is above it.
        target = (
            message.point.x,
            message.point.y,
            message.point.z + self.clearance,
        )

        try:
            solutions = solve_downward_ik(*target)
        except ValueError as error:
            self.clear_preview()
            self.get_logger().warning(
                str(error), throttle_duration_sec=2.0
            )
            return

        # Show the first joint-limit-valid candidate; no command topics are used.
        try:
            markers = self.preview_model.markers(
                solutions[0], target, self.get_clock().now().to_msg())
        except ValueError as error:
            self.clear_preview()
            self.get_logger().warning(str(error), throttle_duration_sec=2.0)
            return
        self.marker_publisher.publish(markers)
        self.preview_visible = True

        for index, angles in enumerate(solutions, start=1):
            reconstructed = forward_kinematics(angles)
            formatted = ', '.join(f'{angle:.3f}' for angle in angles)

            self.get_logger().info(
                f'Solution {index}: '
                f'[base, shoulder, elbow, wrist] = [{formatted}] rad; '
                f'tool0 = {tuple(round(v, 4) for v in reconstructed)}',
                throttle_duration_sec=1.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = IKPreview()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()