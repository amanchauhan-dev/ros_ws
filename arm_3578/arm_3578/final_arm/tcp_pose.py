#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# ---------------------------------------------------------------------------------
# Team ID:          <Your Team ID>
# Theme:            Krishi coBot (KC)
# Author List:      <Your Name>
# Filename:         twist_delta_monitor.py
# Description:      Publishes Cartesian velocity commands and measures actual
#                   change in TCP position using TF.
# ---------------------------------------------------------------------------------
"""

import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import TwistStamped
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time


class TwistDeltaMonitor(Node):

    # ---------------------------------------------------------------------------------
    def __init__(self):
        """
        Initializes publisher and TF listener.
        """
        super().__init__('twist_delta_monitor')

        # Publisher
        self.twist_pub = self.create_publisher(
            TwistStamped, '/delta_twist_cmds', 10)

        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Store previous position
        self.prev_pos = None

        # Timer
        self.timer = self.create_timer(0.1, self.control_loop)

    # ---------------------------------------------------------------------------------
    def get_tcp_position(self):
        """
        Gets current TCP position from TF.
        """
        try:
            trans = self.tf_buffer.lookup_transform(
                'base_link',        # change if needed
                'tool0',            # TCP frame
                Time()
            )

            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z

            return np.array([x, y, z])

        except Exception as e:
            self.get_logger().warn(f"TF Error: {e}")
            return None

    # ---------------------------------------------------------------------------------
    def publish_twist(self):
        """
        Publishes a simple upward velocity.
        """
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()

        twist.twist.linear.x = 0.0
        twist.twist.linear.y = 0.0
        twist.twist.linear.z = 0.05  # upward velocity

        twist.twist.angular.x = 0.0
        twist.twist.angular.y = 0.0
        twist.twist.angular.z = 0.0

        self.twist_pub.publish(twist)

    # ---------------------------------------------------------------------------------
    def control_loop(self):
        """
        Main loop:
        - publish twist
        - compute change in position
        """
        current_pos = self.get_tcp_position()

        if current_pos is None:
            return

        # First iteration
        if self.prev_pos is None:
            self.prev_pos = current_pos
            return

        # Compute delta
        delta = current_pos - self.prev_pos

        self.get_logger().info(
            f"Δx: {delta[0]:.4f}, Δy: {delta[1]:.4f}, Δz: {delta[2]:.4f}"
        )

        # Update previous
        self.prev_pos = current_pos

        # Publish command
        self.publish_twist()


# ---------------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = TwistDeltaMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()