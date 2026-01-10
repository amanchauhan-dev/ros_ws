#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class LidarGlobalPlotter(Node):
    def __init__(self):
        super().__init__("lidar_global_plotter")

        # ---------------- Subscribers ----------------
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)

        # ---------------- Robot pose ----------------
        self.x = None
        self.y = None
        self.yaw = None

        # ---------------- Plot ----------------
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title("Global LiDAR Points")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)
        self.ax.set_aspect("equal")

        self.scatter = self.ax.scatter([], [], s=2)
        self.robot_plot, = self.ax.plot([], [], "bo")

    # =====================================================
    # Callbacks
    # =====================================================

    def odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

    def scan_cb(self, msg: LaserScan):
        if self.x is None or self.y is None or self.yaw is None:
            return

        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # remove invalid points
        valid = np.isfinite(ranges)
        ranges = ranges[valid]
        angles = angles[valid]

        # local coordinates
        x_local = ranges * np.cos(angles)
        y_local = ranges * np.sin(angles)

        # rotation
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)

        # global coordinates
        x_global = self.x + (x_local * cy - y_local * sy)
        y_global = self.y + (x_local * sy + y_local * cy)

        self.update_plot(x_global, y_global)

    # =====================================================
    # Plot update
    # =====================================================

    def update_plot(self, xs, ys):
        self.ax.cla()
        self.ax.set_title("Global LiDAR Points")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)
        self.ax.set_aspect("equal")

        self.ax.scatter(xs, ys, s=2, c="red", label="LiDAR")
        self.ax.plot(self.x, self.y, "bo", label="Robot")

        # robot heading
        self.ax.arrow(
            self.x,
            self.y,
            0.5 * math.cos(self.yaw),
            0.5 * math.sin(self.yaw),
            head_width=0.05,
            color="blue",
        )

        self.ax.legend()
        plt.pause(0.001)

    # =====================================================
    # Cleanup
    # =====================================================

    def destroy_node(self):
        plt.close(self.fig)
        super().destroy_node()


def main():
    rclpy.init()
    node = LidarGlobalPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
