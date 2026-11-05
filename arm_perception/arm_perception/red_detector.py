import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool


# OpenCV hue is in [0, 179]. Red spans both ends of that range.
COLOR_RANGES = {
    'red': (((0, 120, 70), (10, 255, 255)),
            ((170, 120, 70), (179, 255, 255))),
    'green': (((35, 120, 70), (85, 255, 255)),),
    'blue': (((100, 120, 70), (130, 255, 255)),),
}


def find_color_box(hsv, color, kernel):
    """Return the largest sufficiently large color region, or None."""
    mask = cv2.inRange(hsv, *COLOR_RANGES[color][0])
    for lower, upper in COLOR_RANGES[color][1:]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest) if cv2.contourArea(largest) >= 100 else None


class RedDetector(Node):
    def __init__(self):
        super().__init__('red_detector')

        self.declare_parameter('color', 'red')
        self.color = self.get_parameter('color').value
        if self.color not in COLOR_RANGES:
            raise ValueError('color must be red, green, or blue')
        topic_prefix = f'/arm/perception/{self.color}'

        # Zero width/height means the whole image (legacy red workflow).
        self.declare_parameter('image_roi', [0, 0, 0, 0])
        self.roi = list(self.get_parameter('image_roi').value)
        if len(self.roi) != 4 or any(v < 0 for v in self.roi):
            raise ValueError('image_roi must be [x, y, width, height], all nonnegative')
        if (self.roi[2] == 0) != (self.roi[3] == 0):
            raise ValueError('ROI width and height must both be zero or both positive')

        self.declare_parameter('image_topic', '/arm/camera/image_raw')
        image_topic = self.get_parameter('image_topic').value

        self.bridge = CvBridge()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.on_image,
            qos_profile_sensor_data,
        )
        self.publisher = self.create_publisher(
            Image, f'{topic_prefix}/image', 10
        )
        self.center_publisher = self.create_publisher(
            PointStamped, f'{topic_prefix}/pixel_center', 10
        )
        self.detected_publisher = self.create_publisher(
            Bool, f'{topic_prefix}/detected', 10
        )
        self.get_logger().info(f'Waiting for camera images on {image_topic}')

    def on_image(self, message):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                message, desired_encoding='bgr8'
            )
        except CvBridgeError as error:
            self.get_logger().error(str(error))
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        rx, ry, rw, rh = self.roi
        if rw and rh:
            if rx + rw > frame.shape[1] or ry + rh > frame.shape[0]:
                self.get_logger().error('image_roi is outside the image')
                self.detected_publisher.publish(Bool(data=False))
                return
            box = find_color_box(hsv[ry:ry+rh, rx:rx+rw], self.color, self.kernel)
            if box is not None:
                x, y, width, height = box
                box = (x + rx, y + ry, width, height)
        else:
            box = find_color_box(hsv, self.color, self.kernel)

        result = frame.copy()
        status = f'No {self.color} region detected'

        detected = False

        if box is not None:
            x, y, width, height = box
            u = x + width // 2
            v = y + height // 2
            cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.circle(result, (u, v), 4, (0, 255, 255), -1)
            status = f'{self.color.capitalize()} region: pixel center ({u}, {v})'
            detected = True
            center = PointStamped()
            center.header = message.header
            center.point.x = float(u)
            center.point.y = float(v)
            center.point.z = 0.0
            self.center_publisher.publish(center)

        detection = Bool()
        detection.data = detected
        self.detected_publisher.publish(detection)

        cv2.putText(
            result, status, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        output = self.bridge.cv2_to_imgmsg(result, encoding='bgr8')
        output.header = message.header
        self.publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = RedDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
