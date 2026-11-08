"""Estimate all detected surface points on a known plane; not depth sensing."""
import math
from collections import OrderedDict

import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from arm_interfaces.msg import PlatformSlot, PlatformState

import cv2
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray, Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from arm_perception.platform_geometry import PlatformLayout, project_polygon, support_visible, overlap_slot
from arm_perception.object_detector import color_mask

from arm_perception.red_localizer import camera_ray, intersect_plane, rotation_matrix, translation

COLORS = {'red': (1.0, 0.0, 0.0), 'green': (0.0, 1.0, 0.0), 'blue': (0.0, 0.3, 1.0)}


def cube_sized_region(box, info, origin, rotation, height, cube_size, mask=None):
    """Reject small visible fragments in this known-size, overhead-camera setup.

    A partial color region's centre is not the cube's centre. Allow 20 percent
    size/area loss for pixel rounding/shading, using the known support plane.
    A rotated fragment may have a large bounding box but too few colour pixels.
    """
    u, v = box.center.position.x, box.center.position.y
    spans = []
    for du, dv in ((box.size_x, 0.0), (0.0, box.size_y)):
        a = intersect_plane(origin, rotation @ camera_ray(u-du/2, v-dv/2, info), height)
        b = intersect_plane(origin, rotation @ camera_ray(u+du/2, v+dv/2, info), height)
        spans.append(math.dist(a, b))
        if spans[-1] < 0.8*cube_size:
            return False
    if mask is not None:
        x, y = int(round(u-box.size_x/2)), int(round(v-box.size_y/2))
        w, h = int(round(box.size_x)), int(round(box.size_y))
        if x < 0 or y < 0 or x+w > mask.shape[1] or y+h > mask.shape[0] or w*h == 0:
            return False
        # With this fixed downward camera, plane area per pixel is constant.
        pixel_area = spans[0]*spans[1]/(box.size_x*box.size_y)
        if cv2.countNonZero(mask[y:y+h,x:x+w])*pixel_area < 0.8*cube_size**2:
            return False
    return True


class ObjectLocalizer(Node):
    def __init__(self):
        super().__init__('object_localizer')
        for name, default in (
            ('camera_info_topic', '/overhead_camera/camera_info'),
            ('plane_frame', 'world'), ('target_frame', 'base_link'),
            ('plane_height', 0.04), ('max_age', 0.5),
        ):
            self.declare_parameter(name, default)
        self.plane_frame = self.get_parameter('plane_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.plane_height = float(self.get_parameter('plane_height').value)
        self.max_age = float(self.get_parameter('max_age').value)
        if not math.isfinite(self.plane_height) or not 0 < self.max_age < float('inf'):
            raise ValueError('plane_height must be finite and max_age finite and positive')
        self.layout = PlatformLayout.from_node(self)
        self.bridge = CvBridge()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.images = OrderedDict()
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.info = None
        self.pending = None
        self.last_processed = None
        self.has_output = False
        self.objects_pub = self.create_publisher(Detection3DArray, '/arm/perception/objects', 10)
        self.platform_pub = self.create_publisher(PlatformState, '/arm/perception/platform', 10)
        self.floor_pub = self.create_publisher(PlatformState, '/arm/perception/floor', 10)
        self.scene_pub = self.create_publisher(Image, '/arm/perception/scene/image', qos_profile_sensor_data)
        self.markers_pub = self.create_publisher(MarkerArray, '/arm/perception/objects/markers', 10)
        self.subs = [
            self.create_subscription(Image, '/overhead_camera/image_raw', self.on_image, qos_profile_sensor_data),
            self.create_subscription(CameraInfo, self.get_parameter('camera_info_topic').value,
                                     self.on_info, qos_profile_sensor_data),
            self.create_subscription(Detection2DArray, '/arm/perception/detections',
                                     self.on_detections, 10),
        ]
        self.timer = self.create_timer(0.05, self.update_objects)
        self.get_logger().info(f'Localizing surface centers on {self.plane_frame} '
                               f'Z={self.plane_height} m; waiting for camera data and TF.')

    def on_image(self, message):
        stamp = message.header.stamp
        self.images[stamp.sec*1000000000+stamp.nanosec] = message
        while len(self.images) > 8:
            self.images.popitem(last=False)

    def unknown_platform(self):
        for surface in ('platform', 'floor'):
            state = PlatformState()
            state.header.frame_id = self.target_frame
            state.header.stamp = self.get_clock().now().to_msg()
            for name, center in self.layout.placement_slots(surface).items():
                slot = PlatformSlot(name=name, state=PlatformSlot.UNKNOWN)
                slot.center.x, slot.center.y, slot.center.z = center
                state.slots.append(slot)
            getattr(self, surface+'_pub').publish(state)

    def placement_scene(self, frame, records, camera_origin, camera_rotation,
                        target_rotation, target_origin, header, surface):
        state = PlatformState()
        state.header = header
        markers = []
        raw_frame = frame.copy()
        for i, (name, center) in enumerate(self.layout.placement_slots(surface).items()):
            polygon = project_polygon(self.layout.slot_polygon(center), camera_origin,
                                      camera_rotation, self.info)
            matches = [color for color, point, _ in records
                       if abs(point[2]-(center[2]+self.layout.cube)) < 0.005
                       and overlap_slot(point, center, self.layout.cube)]
            slot = PlatformSlot(name=name)
            slot.center.x, slot.center.y, slot.center.z = map(float,
                target_rotation @ np.asarray(center)+target_origin)
            if matches:
                slot.state = PlatformSlot.OCCUPIED
                slot.color = matches[0] if len(set(matches)) == 1 else 'mixed'
            elif support_visible(raw_frame, polygon, surface):
                slot.state = PlatformSlot.EMPTY
            else:
                slot.state = PlatformSlot.UNKNOWN
            state.slots.append(slot)
            rgb = {PlatformSlot.EMPTY: (0.0, 1.0, 0.0),
                   PlatformSlot.OCCUPIED: (1.0, 0.2, 0.0),
                   PlatformSlot.UNKNOWN: (0.6, 0.6, 0.6)}[slot.state]
            marker = Marker()
            marker.header = header
            marker.ns = surface+'_slots'
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.002
            marker.color.r, marker.color.g, marker.color.b = rgb
            marker.color.a = 1.0
            marker.lifetime = Duration(seconds=self.max_age).to_msg()
            corners = self.layout.slot_polygon(center, self.layout.cube/2)
            for point in list(corners)+[corners[0]]:
                p = target_rotation @ point+target_origin
                marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])+0.001))
            markers.append(marker)
            if polygon is not None and np.isfinite(polygon).all():
                points = np.rint(polygon).astype(np.int32)
                bgr = tuple(int(v*255) for v in rgb[::-1])
                cv2.polylines(frame, [points], True, bgr, 1)
                u, v = np.mean(polygon, axis=0)
                label = f'{surface[0].upper()}{i+1} ' + (slot.color if matches else ('free' if slot.state == PlatformSlot.EMPTY else '?'))
                cv2.putText(frame, label, (int(u)-20, int(v)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (255,255,255), 1)
        getattr(self, surface+'_pub').publish(state)
        return markers

    def on_info(self, message):
        self.info = message

    def on_detections(self, message):
        self.pending = message

    def clear_outputs(self):
        output = Detection3DArray()
        output.header.frame_id = self.target_frame
        output.header.stamp = self.get_clock().now().to_msg()
        self.objects_pub.publish(output)
        self.unknown_platform()
        self.markers_pub.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        self.has_output = False
        self.last_processed = None

    def update_objects(self):
        if self.pending is None:
            return
        message = self.pending
        stamp = Time.from_msg(message.header.stamp, clock_type=self.get_clock().clock_type)
        age = (self.get_clock().now() - stamp).nanoseconds / 1e9
        if stamp.nanoseconds == 0 or age > self.max_age or age < -0.05:
            if self.has_output:
                self.clear_outputs()
            return
        # Allow the /clock callback to catch up before using this image.
        if age < 0 or stamp.nanoseconds == self.last_processed:
            return
        if self.info is None or self.info.header.frame_id != message.header.frame_id:
            if self.has_output:
                self.clear_outputs()
            return
        raw_image = self.images.get(stamp.nanoseconds)
        if raw_image is None or raw_image.header.frame_id != message.header.frame_id:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(raw_image, desired_encoding='bgr8').copy()
        except CvBridgeError:
            self.clear_outputs()
            return
        try:
            camera_tf = self.buffer.lookup_transform(
                self.plane_frame, message.header.frame_id, stamp).transform
            target_tf = self.buffer.lookup_transform(
                self.target_frame, self.plane_frame, stamp).transform
        except TransformException as error:
            if self.has_output:
                self.clear_outputs()
            self.get_logger().warning(str(error), throttle_duration_sec=2.0)
            return
        output = Detection3DArray()
        output.header.stamp = message.header.stamp
        output.header.frame_id = self.target_frame
        markers = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        records = []
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            masks = {color: color_mask(hsv, color, self.kernel) for color in COLORS}
            camera_rotation = rotation_matrix(camera_tf.rotation)
            target_rotation = rotation_matrix(target_tf.rotation)
            camera_origin = translation(camera_tf)
            target_origin = translation(target_tf)
            for detection in message.detections:
                if not detection.results:
                    continue
                color = detection.results[0].hypothesis.class_id
                if color not in COLORS:
                    continue
                center = detection.bbox.center.position
                ray = camera_ray(center.x, center.y, self.info)
                try:
                    surface = self.layout.surface(camera_origin, camera_rotation @ ray, self.plane_height)
                    if not cube_sized_region(detection.bbox, self.info, camera_origin,
                                             camera_rotation, surface[2], self.layout.cube, masks[color]):
                        continue
                except ValueError:
                    # Reject boundary/partial regions, without discarding other cubes.
                    continue
                records.append((color, surface, detection))
                point = target_rotation @ surface + target_origin
                obj = Detection3D()
                obj.header = output.header
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = color
                hypothesis.hypothesis.score = detection.results[0].hypothesis.score
                pose = hypothesis.pose.pose
                pose.position.x, pose.position.y, pose.position.z = map(float, point)
                # Only position is estimated: identity orientation is a placeholder.
                pose.orientation.w = 1.0
                obj.results.append(hypothesis)
                # The reference point is the visible top surface, not the cube COM.
                # Leave bbox dimensions and id unset: no 3D extent or tracking estimated.
                output.detections.append(obj)
                marker = Marker()
                marker.header = output.header
                marker.ns = 'object_surfaces'
                marker.id = len(output.detections)-1
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, point)
                marker.pose.orientation.w = 1.0
                marker.color.r, marker.color.g, marker.color.b = COLORS[color]
                marker.color.a = 1.0
                marker.lifetime = Duration(seconds=self.max_age).to_msg()
                marker.scale.x = marker.scale.y = marker.scale.z = 0.012
                markers.markers.append(marker)
        except (ValueError, cv2.error) as error:
            self.clear_outputs()
            self.get_logger().warning(str(error), throttle_duration_sec=2.0)
            return
        for surface in ('platform', 'floor'):
            markers.markers.extend(self.placement_scene(
                frame, records, camera_origin, camera_rotation, target_rotation, target_origin,
                output.header, surface))
        counts = dict.fromkeys(COLORS, 0)
        for color, point, detection in records:
            counts[color] += 1
            box = detection.bbox
            x = int(box.center.position.x-box.size_x/2)
            y = int(box.center.position.y-box.size_y/2)
            cv2.rectangle(frame, (x,y), (x+int(box.size_x),y+int(box.size_y)),
                          tuple(int(v*255) for v in COLORS[color][::-1]), 2)
            region = 'platform' if self.layout.contains(point) else 'floor'
            label = f'{color} {counts[color]} {"P" if region == "platform" else "F"}'
            cv2.putText(frame, label, (max(0,x-15),max(15,y-7)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (0,0,0), 3)
            cv2.putText(frame, label, (max(0,x-15),max(15,y-7)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (255,255,255), 1)
        cv2.putText(frame, 'F=floor  P=platform  F1/F2, P1/P2: free, color, ?', (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        scene = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        scene.header = message.header
        self.scene_pub.publish(scene)
        self.objects_pub.publish(output)
        self.markers_pub.publish(markers)
        self.has_output = True
        self.last_processed = stamp.nanoseconds


def main(args=None):
    rclpy.init(args=args)
    node = ObjectLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
