#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
#                       IMPORTS
# ============================================================

import math
import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

# ============================================================
#                    GLOBAL CONSTANTS
# ============================================================

TOPIC_LIDAR_SCAN = "/scan"
TOPIC_ODOMETRY = "/odom"
TOPIC_IN_LANE = "/in_lane"
TOPIC_FERTILIZER_STATUS = "/fertilizer_placement_status"
TOPIC_DETECTION_STATUS = "/detection_status"

CONTROL_LOOP_PERIOD_SEC = 0.1

GLOBAL_BUFFER_LENGTH = 3
GLOBAL_BUFFER_MIN_DISTANCE = 0.2
SIDE_DISTANCE_TOLERANCE = 1.5

RANSAC_ITERATIONS = 200
RANSAC_DISTANCE_THRESHOLD = 0.01
RANSAC_MIN_INLIERS = 5
RANSAC_MAX_LINES = 6

EXPECTED_SIDE_LENGTH = 0.25
SIDE_LENGTH_TOLERANCE = 0.08
ANGLE_90_TOLERANCE_DEG = 15
ANGLE_60_TOLERANCE_DEG = 20
SHAPE_CONFIRMATION_COUNT = 5

PLOT_COLORS = ["red", "blue", "green", "orange", "purple", "cyan"]

PLANT_TRACKS = [
    [2.2, -2.0],
    [1.2, -0.8],
    [2.0, -0.8],
    [2.8, -0.8],
    [3.6, -0.8],
    [1.2,  0.8],
    [2.0,  0.8],
    [2.8,  0.8],
    [3.6,  0.8],
]



# ============================================================
#                         NODE
# ============================================================

class Ransac(Node):
    def __init__(self):
        super().__init__("ransac_detection_node")

        # ---------------- Subscribers ----------------
        self.create_subscription(LaserScan, TOPIC_LIDAR_SCAN, self.lidar_callback, 10)
        self.create_subscription(Odometry, TOPIC_ODOMETRY, self.odom_callback, 10)
        self.create_subscription(Bool, TOPIC_IN_LANE, self.in_lane_callback, 10)
        self.create_subscription(Bool, TOPIC_FERTILIZER_STATUS,
                                 self.fertilizer_placement_status_callback, 10)

        # ---------------- Publisher ----------------
        self.detection_pub = self.create_publisher(String, TOPIC_DETECTION_STATUS, 10)

        # ---------------- State ----------------
        self.in_lane = True
        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.last_point_buffered = None

        # ---------------- Buffers ----------------
        self.current_buffer = np.empty((0, 2))
        self.right_buffer = []
        self.left_buffer = []

        # ---------------- Detection memory ----------------
        self.right_votes = [0] * len(PLANT_TRACKS)
        self.left_votes = [0] * len(PLANT_TRACKS)

        # ---------------- Plot ----------------
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.colors = PLOT_COLORS

        # ---------------- Timer ----------------
        self.create_timer(CONTROL_LOOP_PERIOD_SEC, self.control_loop)

        self.get_logger().info("RANSAC perception node started")

    # ============================================================
    #                       CALLBACKS
    # ============================================================

    def fertilizer_placement_status_callback(self, msg: Bool):
        self.get_logger().info(f"Fertilizer status received: {msg.data}")

    def in_lane_callback(self, msg: Bool):
        self.in_lane = msg.data
        self.get_logger().info(f"In-lane flag: {self.in_lane}")

    def odom_callback(self, msg: Odometry):
        if not self.in_lane:
            return

        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny, cosy)

        self.get_logger().debug(
            f"Odom: x={self.current_x:.2f}, y={self.current_y:.2f}, yaw={self.current_yaw:.2f}"
        )

    def lidar_callback(self, msg: LaserScan):
        if not self.in_lane:
            self.right_buffer.clear()
            self.left_buffer.clear()
            self.last_point_buffered = None
            return

        if self.current_x is None:
            return

        ranges = np.nan_to_num(np.array(msg.ranges), nan=0.0)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        x_local = ranges * np.cos(angles)
        y_local = ranges * np.sin(angles)

        cy, sy = math.cos(self.current_yaw), math.sin(self.current_yaw)
        xg = self.current_x + (x_local * cy - y_local * sy)
        yg = self.current_y + (x_local * sy + y_local * cy)

        pts = np.vstack((xg, yg)).T
        self.current_buffer = pts

        if self.last_point_buffered is None:
            self._buffer_sides(pts)
            self.last_point_buffered = (self.current_x, self.current_y)
            self.get_logger().info("First LiDAR frame buffered")
            return

        dist = math.hypot(
            self.current_x - self.last_point_buffered[0],
            self.current_y - self.last_point_buffered[1],
        )

        if dist >= GLOBAL_BUFFER_MIN_DISTANCE:
            self._buffer_sides(pts)
            self.last_point_buffered = (self.current_x, self.current_y)
            self.get_logger().debug("Buffered new LiDAR frame")

    def _buffer_sides(self, pts):
        self.right_buffer.append(self._filter_side(pts[0:90]))
        self.left_buffer.append(self._filter_side(pts[270:360]))

        self.right_buffer = self.right_buffer[-GLOBAL_BUFFER_LENGTH:]
        self.left_buffer = self.left_buffer[-GLOBAL_BUFFER_LENGTH:]

    # ============================================================
    #                       HELPERS
    # ============================================================

    def _filter_side(self, pts):
        pts = np.array(pts)
        if len(pts) == 0:
            return pts

        robot = np.array([self.current_x, self.current_y])
        dists = np.linalg.norm(pts - robot, axis=1)

        mask = dists < SIDE_DISTANCE_TOLERANCE
        pts = pts[mask]
        dists = dists[mask]

        if len(pts) == 0:
            return pts

        med = np.median(dists)
        mad = np.median(np.abs(dists - med)) + 1e-6
        pts = pts[np.abs(dists - med) < 2.5 * mad]

        return pts

    # ============================================================
    #                       CONTROL LOOP
    # ============================================================

    def control_loop(self):
        if not self.in_lane:
            return

        self.ax.clear()
        self.ax.set_title("RANSAC Detection (No Stop Logic)")
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        if len(self.current_buffer) > 0:
            self.ax.scatter(
                self.current_buffer[:, 0],
                self.current_buffer[:, 1],
                s=3,
                c="gray",
                label="LiDAR"
            )

        if self.current_x is not None:
            self.ax.plot(self.current_x, self.current_y, "bo", label="Robot")

        self._process_side(self.right_buffer, True)
        self._process_side(self.left_buffer, False)

        self.ax.legend()
        plt.pause(0.001)

    def _process_side(self, buffers, is_right):
        lines = self.extract_multiple_lines(buffers)
        side = "RIGHT" if is_right else "LEFT"

        for model, pts in lines:
            p1, p2, _, _ = self.compute_line_endpoints(model, pts)
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=2)

        shape = self.detect_shape([
            {"model": m, "p1": p1, "p2": p2}
            for m, pts in lines
            for p1, p2, _, _ in [self.compute_line_endpoints(m, pts)]
        ])

        if shape:
            shape_type, group = shape
            cx, cy, _ = self.compute_shape_position(group)

            self.get_logger().info(
                f"[{side}] Detected {shape_type.upper()} at ({cx:.2f}, {cy:.2f})"
            )

            msg = String()
            msg.data = f"{side},{shape_type},{cx:.2f},{cy:.2f}"
            self.detection_pub.publish(msg)

    # ============================================================
    #                       RANSAC + SHAPE
    # ============================================================

    def ransac_line(self, points):
        pts = np.array(points)
        if len(pts) < RANSAC_MIN_INLIERS:
            return None, None

        best_model, best_inliers = None, None

        for _ in range(RANSAC_ITERATIONS):
            p1, p2 = pts[np.random.choice(len(pts), 2, replace=False)]
            a, b = p2[1] - p1[1], -(p2[0] - p1[0])
            c = p2[0]*p1[1] - p2[1]*p1[0]

            norm = math.hypot(a, b)
            if norm < 1e-6:
                continue

            dist = np.abs(a*pts[:, 0] + b*pts[:, 1] + c) / norm
            inliers = dist < RANSAC_DISTANCE_THRESHOLD

            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best_model, best_inliers = (a, b, c), inliers

        return best_model, best_inliers

    def extract_multiple_lines(self, buffers):
        if not buffers:
            return []

        pts = np.vstack(buffers)
        lines = []

        while len(pts) >= RANSAC_MIN_INLIERS:
            model, inliers = self.ransac_line(pts)
            if model is None:
                break

            lines.append((model, pts[inliers]))
            pts = pts[~inliers]

            if len(lines) >= RANSAC_MAX_LINES:
                break

        return lines

    def detect_shape(self, lines):
        if len(lines) < 3:
            return None

        from itertools import combinations

        for group in combinations(lines, 3):
            lengths = [np.linalg.norm(g["p2"] - g["p1"]) for g in group]
            near = sum(abs(l - EXPECTED_SIDE_LENGTH) < SIDE_LENGTH_TOLERANCE for l in lengths)

            angles = [
                self.angle_between_lines(group[i]["model"], group[j]["model"])
                for i, j in [(0,1),(1,2),(2,0)]
            ]

            if near >= 2 and sum(abs(a - 90) < ANGLE_90_TOLERANCE_DEG for a in angles) == 2:
                return "square", group

            if near >= 2 and sum(abs(a - 60) < ANGLE_60_TOLERANCE_DEG for a in angles) >= 2:
                return "triangle", group

        return None

    def angle_between_lines(self, m1, m2):
        d1 = np.array([m1[1], -m1[0]])
        d2 = np.array([m2[1], -m2[0]])
        d1 /= np.linalg.norm(d1)
        d2 /= np.linalg.norm(d2)
        ang = math.degrees(math.acos(np.clip(np.dot(d1, d2), -1, 1)))
        return min(ang, 180-ang)

    def compute_line_endpoints(self, model, pts):
        a, b, c = model
        d = np.array([b, -a]) / np.linalg.norm([b, -a])
        p0 = np.array([-a*c/(a*a+b*b), -b*c/(a*a+b*b)])
        proj = [(np.dot(p - p0, d), p) for p in pts]
        return min(proj)[1], max(proj)[1], None, None

    def compute_shape_position(self, group):
        pts = np.vstack([g["p1"] for g in group] + [g["p2"] for g in group])
        cx, cy = np.mean(pts, axis=0)
        dist = math.hypot(cx - self.current_x, cy - self.current_y)
        return cx, cy, dist

    # ============================================================
    #                       CLEANUP
    # ============================================================

    def destroy_node(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = Ransac()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
