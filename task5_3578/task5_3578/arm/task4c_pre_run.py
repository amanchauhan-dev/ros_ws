#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import tf2_ros

class Task4cPreCheck(Node):
    def __init__(self):
        super().__init__('task4c_precheck')

        self.team_id = '3578'
        self.base_frame = 'base_link'
        self.fertilizer_frame = f'{self.team_id}_fertilizer_1'

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("=== TASK4C PRE-RUN CHECK STARTED ===")

        self.check_topics()
        self.check_services()
        self.check_tf()

        self.get_logger().info("✅ ALL CHECKS PASSED — SAFE TO RUN TASK4C")
        rclpy.shutdown()

    # ---------------- TOPIC CHECK ----------------
    def check_topics(self):
        self.get_logger().info("🔍 Checking Topics...")

        required_topics = {
            '/delta_twist_cmds': 'geometry_msgs/msg/TwistStamped',
            '/delta_joint_cmds': 'control_msgs/msg/JointJog',
            '/joint_states': 'sensor_msgs/msg/JointState',
            '/tcp_pose_raw': 'std_msgs/msg/Float64MultiArray',
            '/net_wrench': 'std_msgs/msg/Float32'
        }

        topic_list = self.get_topic_names_and_types()

        for topic, expected_type in required_topics.items():
            found = False
            for t_name, t_types in topic_list:
                if t_name == topic:
                    found = True
                    if expected_type not in t_types:
                        self.error(f"Topic {topic} has wrong type {t_types}")
                    else:
                        self.get_logger().info(f"✅ {topic} [{expected_type}]")
            if not found:
                self.error(f"Topic {topic} NOT FOUND")

    # ---------------- SERVICE CHECK ----------------
    def check_services(self):
        self.get_logger().info("🔍 Checking Services...")

        service_list = self.get_service_names_and_types()

        required_service = '/magnet'
        required_type = 'std_srvs/srv/SetBool'

        for name, types in service_list:
            if name == required_service:
                if required_type in types:
                    self.get_logger().info(f"✅ {name} [{required_type}]")
                    return
                else:
                    self.error(f"Service {name} has wrong type {types}")

        self.error(f"Service {required_service} NOT FOUND")

    # ---------------- TF CHECK ----------------
    def check_tf(self):
        self.get_logger().info("🔍 Checking TF Frames...")

        try:
            self.tf_buffer.lookup_transform(
                self.base_frame,
                self.fertilizer_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=2.0)
            )
            self.get_logger().info(f"✅ TF {self.base_frame} → {self.fertilizer_frame}")
        except Exception as e:
            self.error(f"TF lookup failed: {self.base_frame} → {self.fertilizer_frame}")

    # ---------------- ERROR HANDLER ----------------
    def error(self, msg):
        self.get_logger().error(f"❌ {msg}")
        self.get_logger().error("⛔ FIX THE ISSUE BEFORE RUNNING TASK4C")
        rclpy.shutdown()
        exit(1)


def main():
    rclpy.init()
    Task4cPreCheck()
    rclpy.spin_once(rclpy.create_node('dummy'), timeout_sec=0.1)

if __name__ == '__main__':
    main()
