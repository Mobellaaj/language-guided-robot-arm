"""Detect every separated red, green and blue region in an overhead image."""
import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from arm_perception.red_detector import COLOR_RANGES

DRAW_COLORS = {'red': (0, 0, 255), 'green': (0, 220, 0), 'blue': (255, 100, 0)}


def color_mask(hsv, color, kernel):
    """Use the same colour pixels for detection and visibility checks."""
    mask = cv2.inRange(hsv, *COLOR_RANGES[color][0])
    for lower, upper in COLOR_RANGES[color][1:]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def find_color_boxes(hsv, color, kernel, min_area=100.0):
    """Return all large-enough regions, ordered by image X then Y, not size."""
    mask = color_mask(hsv, color, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted((cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area),
                  key=lambda box: (box[0], box[1]))


def detect_regions(frame, roi, kernel, min_area):
    """Return (color, full-image box) pairs, applying an optional image ROI."""
    x, y, width, height = roi
    if width == height == 0:
        x, y, width, height = 0, 0, frame.shape[1], frame.shape[0]
    if (x < 0 or y < 0 or width <= 0 or height <= 0
            or x + width > frame.shape[1] or y + height > frame.shape[0]):
        raise ValueError('image_roi is outside the image')
    hsv = cv2.cvtColor(frame[y:y+height, x:x+width], cv2.COLOR_BGR2HSV)
    return [(color, (bx+x, by+y, bw, bh))
            for color in COLOR_RANGES
            for bx, by, bw, bh in find_color_boxes(hsv, color, kernel, min_area)]


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.declare_parameter('image_topic', '/overhead_camera/image_raw')
        self.declare_parameter('image_roi', [0, 0, 0, 0])
        self.declare_parameter('min_area', 100.0)
        self.roi = list(self.get_parameter('image_roi').value)
        self.min_area = float(self.get_parameter('min_area').value)
        if (len(self.roi) != 4 or any(v < 0 for v in self.roi)
                or (self.roi[2] == 0) != (self.roi[3] == 0)):
            raise ValueError('image_roi must be [x, y, width, height]; zero size means full image')
        if not 0 < self.min_area < float('inf'):
            raise ValueError('min_area must be finite and positive')
        self.bridge = CvBridge()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.detections_pub = self.create_publisher(
            Detection2DArray, '/arm/perception/detections', 10)
        self.image_pub = self.create_publisher(
            Image, '/arm/perception/objects/image', qos_profile_sensor_data)
        self.subscription = self.create_subscription(
            Image, self.get_parameter('image_topic').value, self.on_image, qos_profile_sensor_data)
        self.get_logger().info('Detecting all red, green and blue regions; no object tracking yet.')

    def on_image(self, message):
        detections = Detection2DArray()
        detections.header = message.header
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            regions = detect_regions(frame, self.roi, self.kernel, self.min_area)
        except (CvBridgeError, ValueError, cv2.error) as error:
            self.get_logger().error(str(error), throttle_duration_sec=2.0)
            self.detections_pub.publish(detections)
            return
        image = frame.copy()
        counts = dict.fromkeys(COLOR_RANGES, 0)
        for color, (x, y, width, height) in regions:
            counts[color] += 1
            u, v = x + width / 2, y + height / 2
            detection = Detection2D()
            detection.header = message.header
            detection.bbox.center.position.x = u
            detection.bbox.center.position.y = v
            detection.bbox.size_x = float(width)
            detection.bbox.size_y = float(height)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = color
            # Binary acceptance by HSV rules, not a learned confidence estimate.
            hypothesis.hypothesis.score = 1.0
            detection.results.append(hypothesis)
            # Leave id empty: numbered labels are local to this image, not tracked IDs.
            detections.detections.append(detection)
            cv2.rectangle(image, (x, y), (x+width, y+height), DRAW_COLORS[color], 2)
            cv2.circle(image, (int(u), int(v)), 2, (255, 255, 255), -1)
            label = f'{color} {counts[color]}'
            text_at = (max(0, x-3), max(35, y-7))
            cv2.putText(image, label, text_at, cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 3)
            cv2.putText(image, label, text_at, cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1)
        cv2.putText(image, f'{len(regions)} regions: ' + ', '.join(
            f'{color}={count}' for color, count in counts.items()), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        output = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        output.header = message.header
        self.image_pub.publish(output)
        # Publish empty lists too, so vanished objects are not retained.
        self.detections_pub.publish(detections)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
