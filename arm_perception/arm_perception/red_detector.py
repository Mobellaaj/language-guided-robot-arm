import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool


class RedDetector(Node):
    def __init__(self):
        super().__init__('red_detector')

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
            Image, '/arm/perception/red/image', 10
        )
        self.center_publisher = self.create_publisher(
            PointStamped, '/arm/perception/red/pixel_center', 10
        )
        self.detected_publisher = self.create_publisher(
            Bool, '/arm/perception/red/detected', 10
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

        # Red wraps around both ends of OpenCV's hue range.
        low_red = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        high_red = cv2.inRange(hsv, (170, 120, 70), (179, 255, 255))
        mask = cv2.bitwise_or(low_red, high_red)

        # Remove isolated pixels and small patches.
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, self.kernel
        )

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        result = frame.copy()
        status = 'No red region detected'

        detected = False

        if contours:
            largest = max(contours, key=cv2.contourArea)

            # Ignore regions smaller than 100 square pixels.
            if cv2.contourArea(largest) >= 100:
                x, y, width, height = cv2.boundingRect(largest)
                u = x + width // 2
                v = y + height // 2

                cv2.rectangle(
                    result, (x, y), (x + width, y + height),
                    (0, 255, 0), 2,
                )
                cv2.circle(result, (u, v), 4, (0, 255, 255), -1)
                status = f'Red region: pixel center ({u}, {v})'

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
