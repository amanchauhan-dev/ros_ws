#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
import numpy as np
import matplotlib.pyplot as plt

from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


# ================= CONFIG =================
WAYPOINTS = [
    (-1.53, -5.6),
    (0.32, -5.6),
    (0.32, 1.38),
]

LINEAR_SPEED = 1.5
ANGULAR_SPEED = 1.5

POS_TOL = 0.10
ANG_TOL = 0.10

CONTROL_PERIOD = 0.1
# ==========================================


class WaypointNavigator(Node):
    def __init__(self):
        super().__init__("simple_waypoint_nav")

        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.timer = self.create_timer(CONTROL_PERIOD, self.control_loop)

        # robot state
        self.x = None
        self.y = None
        self.yaw = None

        # lidar
        self.latest_scan = None

        # waypoint state
        self.waypoints = WAYPOINTS
        self.wp_idx = 0

        # plotting
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

        self.get_logger().info("Waypoint navigator started")

    # ---------------- Callbacks ----------------

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

    def scan_cb(self, msg):
        self.latest_scan = msg

    # ---------------- Control ----------------

    def control_loop(self):
        if self.x is None or self.latest_scan is None:
            return

        # navigation
        if self.wp_idx < len(self.waypoints):
            self.navigate()
        else:
            self.stop()

        # plot lidar frame
        self.plot_lidar_frame()

    def navigate(self):
        gx, gy = self.waypoints[self.wp_idx]

        dx = gx - self.x
        dy = gy - self.y
        dist = math.hypot(dx, dy)

        target_yaw = math.atan2(dy, dx)
        yaw_err = self.normalize_angle(target_yaw - self.yaw)

        cmd = Twist()

        if dist > POS_TOL:

            # BIG error → rotate in place
            if abs(yaw_err) > 2.0 * ANG_TOL:
                cmd.angular.z = ANGULAR_SPEED * yaw_err
                cmd.linear.x = 0.0

            # SMALL error → move forward + correct slowly
            else:
                cmd = self.compute_motion(dist, yaw_err)

        else:
            self.get_logger().info(f"Reached waypoint {self.wp_idx}")
            self.wp_idx += 1
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)


    # ---------------- Plotting ----------------

    def plot_lidar_frame(self):
        """Plot current frame LiDAR points in global frame"""

        scan = self.latest_scan
        ranges = np.array(scan.ranges, dtype=float)

        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        # remove invalid
        mask = np.isfinite(ranges)
        ranges = ranges[mask]
        angles = angles[mask]

        # local → global
        x_local = ranges * np.cos(angles)
        y_local = ranges * np.sin(angles)

        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)

        xg = self.x + (x_local * cy - y_local * sy)
        yg = self.y + (x_local * sy + y_local * cy)

        # redraw
        self.ax.clear()
        self.ax.scatter(xg, yg, s=2, c="red", label="LiDAR")
        self.ax.plot(self.x, self.y, "bo", label="Robot")

        self.ax.arrow(
            self.x,
            self.y,
            0.4 * math.cos(self.yaw),
            0.4 * math.sin(self.yaw),
            head_width=0.05,
            color="blue",
        )

        self.ax.set_aspect("equal")
        self.ax.set_xlim(self.x - 5, self.x + 5)
        self.ax.set_ylim(self.y - 5, self.y + 5)
        self.ax.set_title("Global LiDAR – Current Frame")
        self.ax.legend()
        self.ax.grid(True)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ---------------- Utils ----------------
    def compute_motion(self, dist, yaw_err):
        """
        Compute forward and angular velocity while moving.
        - Linear speed reduces near waypoint
        - Angular correction scales with yaw error
        """

        cmd = Twist()

        # ---------- Linear speed ----------
        # Slow down near waypoint
        if dist > 1.0:
            cmd.linear.x = LINEAR_SPEED
        else:
            cmd.linear.x = LINEAR_SPEED * (dist / 1.0)

        # ---------- Angular correction ----------
        cmd.angular.z = ANGULAR_SPEED * yaw_err

        return cmd

    def stop(self):
        self.cmd_pub.publish(Twist())

    @staticmethod
    def normalize_angle(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a


def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
