import math
import threading
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class _OdomSubscriber(Node):
    def __init__(self, topics):
        super().__init__('track_odom_subscriber')

        best_effort_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        reliable_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.lock = threading.Lock()
        self.pose = None
        self.topic = None
        self.qos_name = None
        self.receive_count = 0
        self.last_receive_time = None
        self.odom_subscriptions = []

        for topic, msg_type in topics:
            self.odom_subscriptions.append(
                self.create_subscription(msg_type, topic, self.MakeOdomCallback(topic, 'best_effort'), best_effort_qos)
            )
            self.odom_subscriptions.append(
                self.create_subscription(msg_type, topic, self.MakeOdomCallback(topic, 'reliable'), reliable_qos)
            )

    def MakeOdomCallback(self, topic, qos_name):
        def callback(msg):
            self.OdomCallback(topic, qos_name, msg)
        return callback

    def OdomCallback(self, topic, qos_name, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = self.QuaternionToYaw(orientation.x, orientation.y, orientation.z, orientation.w)

        with self.lock:
            self.pose = (position.x, position.y, yaw)
            self.topic = topic
            self.qos_name = qos_name
            self.receive_count += 1
            self.last_receive_time = time.time()

    def QuaternionToYaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


class OdomWrapper:
    def __init__(self, topics=None):
        if not rclpy.ok():
            rclpy.init()

        if topics is None:
            topics = [
                ('/leg_odom2', Odometry),
                ('/leg_odom', PoseWithCovarianceStamped),
            ]

        self.node = _OdomSubscriber(topics)
        self.break_flag = False
        self.spin_thread = threading.Thread(target=self.SpinThreadFunc)
        self.spin_thread.start()

        print('OdomWrapper topics: {}'.format(', '.join([topic for topic, _ in topics])))
        time.sleep(1.0)
        print('OdomWrapper visible odom topics: {}'.format(', '.join(self.GetVisibleOdomTopics())))

    def __del__(self):
        try:
            self.StopThread()
        except Exception:
            pass

    def StopThread(self):
        if self.break_flag:
            return

        self.break_flag = True
        if self.spin_thread.is_alive():
            self.spin_thread.join()
        self.node.destroy_node()

    def SpinThreadFunc(self):
        while self.break_flag != True:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            time.sleep(0.001)

    def GetPose(self):
        with self.node.lock:
            return self.node.pose

    def GetTopic(self):
        with self.node.lock:
            return self.node.topic

    def GetStatus(self):
        with self.node.lock:
            if self.node.last_receive_time is None:
                age = None
            else:
                age = time.time() - self.node.last_receive_time
            return self.node.topic, self.node.qos_name, self.node.receive_count, age

    def GetVisibleOdomTopics(self):
        visible_topics = []
        for topic, topic_types in self.node.get_topic_names_and_types():
            if 'odom' in topic.lower():
                visible_topics.append('{}:{}'.format(topic, ','.join(topic_types)))
        return visible_topics
