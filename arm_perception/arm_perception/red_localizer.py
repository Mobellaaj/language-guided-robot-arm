"""Estimate a visible red surface point using a known horizontal world plane."""
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker


def rotation_matrix(quaternion):
    """Convert a ROS quaternion to a rotation matrix; no translation here."""
    q = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid TF quaternion')
    x, y, z, w = q / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def camera_ray(u, v, info):
    """Unproject an image pixel to optical coordinates: X right, Y down, Z forward."""
    if not (0 <= u < info.width and 0 <= v < info.height):
        raise ValueError('Pixel outside the image')
    k = np.asarray(info.k, dtype=float).reshape(3, 3)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Camera intrinsics are not initialized')
    if info.distortion_model not in ('', 'plumb_bob', 'rational_polynomial'):
        raise ValueError('This example expects a pinhole camera')
    distortion = np.asarray(info.d, dtype=float) if len(info.d) else None
    pixel = np.array([[[u, v]]], dtype=float)
    x, y = cv2.undistortPoints(pixel, k, distortion)[0, 0]
    return np.array([x, y, 1.0])


def intersect_plane(origin, direction, height):
    """Ray p = origin + distance * direction; solve p.z = height."""
    if not np.isfinite(origin).all() or not np.isfinite(direction).all():
        raise ValueError('Invalid viewing ray')
    if not np.isfinite(height) or abs(direction[2]) < 1e-9:
        raise ValueError('Ray is parallel to the plane or plane height is invalid')
    distance = (height - origin[2]) / direction[2]
    if distance <= 0:
        raise ValueError('Plane lies behind the camera')
    return origin + distance * direction


def translation(transform):
    t = transform.translation
    return np.array([t.x, t.y, t.z])


class RedLocalizer(Node):
    def __init__(self):
        super().__init__('red_localizer')
        self.declare_parameter('camera_info_topic', '/overhead_camera/camera_info')
        self.declare_parameter('plane_frame', 'world')
        self.declare_parameter('plane_height', 0.04)
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('max_age', 0.5)
        self.plane_frame = self.get_parameter('plane_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.plane_height = float(self.get_parameter('plane_height').value)
        self.max_age = float(self.get_parameter('max_age').value)
        if self.max_age <= 0:
            raise ValueError('max_age must be positive')
        self.info = None
        self.pixel = None
        self.valid = False
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.position_pub = self.create_publisher(PointStamped, '/arm/perception/red/position', 10)
        self.marker_pub = self.create_publisher(Marker, '/arm/perception/red/marker', 10)
        self.valid_pub = self.create_publisher(Bool, '/arm/perception/red/position_valid', 10)
        self.subscriptions_to_keep = [
            self.create_subscription(CameraInfo, self.get_parameter('camera_info_topic').value,
                                     self.on_info, qos_profile_sensor_data),
            self.create_subscription(PointStamped, '/arm/perception/red/pixel_center',
                                     self.on_pixel, 10),
            self.create_subscription(Bool, '/arm/perception/red/detected', self.on_detection, 10),
        ]
        # Retry after TF has caught up with the image; never substitute latest TF.
        self.timer = self.create_timer(0.05, self.update_position)
        self.get_logger().info(
            f'Waiting for pixels, CameraInfo and TF; plane {self.plane_frame} Z={self.plane_height} m')

    def on_info(self, message):
        self.info = message

    def on_pixel(self, message):
        self.pixel = message

    def on_detection(self, message):
        if not message.data:
            self.pixel = None
            self.invalidate()

    def invalidate(self):
        self.valid_pub.publish(Bool(data=False))
        if self.valid:
            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.ns = 'red_surface'
            marker.id = 0
            marker.action = Marker.DELETE
            self.marker_pub.publish(marker)
        self.valid = False

    def update_position(self):
        if self.pixel is None:
            return
        pixel = self.pixel
        stamp = Time.from_msg(pixel.header.stamp, clock_type=self.get_clock().clock_type)
        age = (self.get_clock().now() - stamp).nanoseconds / 1e9
        if stamp.nanoseconds == 0 or age < 0 or age > self.max_age:
            self.invalidate()
            return
        if self.info is None or pixel.header.frame_id != self.info.header.frame_id:
            self.invalidate()
            return
        try:
            ray = camera_ray(pixel.point.x, pixel.point.y, self.info)
            camera_tf = self.buffer.lookup_transform(
                self.plane_frame, pixel.header.frame_id, stamp).transform
            # Translate the ray origin, but only rotate its direction.
            point_in_plane = intersect_plane(
                translation(camera_tf), rotation_matrix(camera_tf.rotation) @ ray,
                self.plane_height)
            target_tf = self.buffer.lookup_transform(
                self.target_frame, self.plane_frame, stamp).transform
            point = rotation_matrix(target_tf.rotation) @ point_in_plane + translation(target_tf)
        except (TransformException, ValueError, cv2.error) as error:
            self.invalidate()
            self.get_logger().warning(str(error), throttle_duration_sec=2.0)
            return

        output = PointStamped()
        output.header.stamp = pixel.header.stamp
        output.header.frame_id = self.target_frame
        output.point.x, output.point.y, output.point.z = map(float, point)
        self.position_pub.publish(output)
        self.valid_pub.publish(Bool(data=True))
        self.valid = True

        marker = Marker()
        marker.header = output.header
        marker.ns = 'red_surface'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = output.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.015
        marker.color.g = marker.color.b = marker.color.a = 1.0
        marker.lifetime = Duration(seconds=self.max_age).to_msg()
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = RedLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
