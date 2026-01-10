#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
#                    GLOBAL CONSTANTS
# ============================================================

# ---------------- ROS topics ----------------
TOPIC_LIDAR_SCAN = "/scan"
TOPIC_ODOMETRY = "/odom"
TOPIC_IN_LANE = "/in_lane"
TOPIC_FERTILIZER_STATUS = "/fertilizer_placement_status"
TOPIC_DETECTION_STATUS = "/detection_status"
TOPIC_STOP_COMMAND = "/to_stop"
TOPIC_DOCK_STATUS = "/ebot_dock_status"

# ---------------- Timings ----------------
CONTROL_LOOP_PERIOD_SEC = 0.1
STOP_DURATION_SEC = 2.0

# ---------------- LiDAR / buffer tuning ----------------
GLOBAL_BUFFER_LENGTH = 3
GLOBAL_BUFFER_MIN_DISTANCE = 0.2
SIDE_DISTANCE_TOLERANCE = 1.5
NEIGHBOR_POINT_DISTANCE = 0.15

# ---------------- RANSAC parameters ----------------
RANSAC_ITERATIONS = 200
RANSAC_DISTANCE_THRESHOLD = 0.01
RANSAC_MIN_INLIERS = 5
RANSAC_MAX_LINES = 6

# ---------------- Shape detection ----------------
EXPECTED_SIDE_LENGTH = 0.25
SIDE_LENGTH_TOLERANCE = 0.08
ANGLE_90_TOLERANCE_DEG = 15
ANGLE_60_TOLERANCE_DEG = 20
SHAPE_CONFIRMATION_COUNT = 5

# ---------------- Plot ----------------
PLOT_COLORS = ["red", "blue", "green", "orange", "purple", "cyan"]

# ---------------- Plant map ----------------
PLANT_TRACKS = [
    [2.2, -2.0],
    [1.2, -0.8],
    [2,0, -0.8],
    [2.8, -0.8],
    [3.6, -0.8],
    [1.2, 0.8],
    [2,0, 0.8],
    [2.8, 0.8],
    [3.6, 0.8],
]

# ============================================================
#                       IMPORTS
# ============================================================

import math
import time
import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


# ============================================================
#                         NODE
# ============================================================

class Ransac(Node):
    def __init__(self):
        super().__init__("ransac_node")

        # ---------------- Subscribers ----------------
        self.create_subscription(LaserScan, TOPIC_LIDAR_SCAN, self.lidar_callback, 10)
        self.create_subscription(Odometry, TOPIC_ODOMETRY, self.odom_callback, 10)
        self.create_subscription(Bool, TOPIC_IN_LANE, self.in_lane_callback, 10)
        self.create_subscription(
            Bool, TOPIC_FERTILIZER_STATUS,
            self.fertilizer_placement_status_callback, 10
        )

        # ---------------- Publishers ----------------
        self.detection_pub = self.create_publisher(String, TOPIC_DETECTION_STATUS, 10)
        self.stop_pub = self.create_publisher(Bool, TOPIC_STOP_COMMAND, 10)
        self.at_dock_pub = self.create_publisher(Bool, TOPIC_DOCK_STATUS, 10)

        # ---------------- State ----------------
        self.in_lane = True
        self.at_dock = False

        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self.last_point_buffered = None

        # ---------------- Buffers ----------------
        self.current_buffer = np.empty((0, 2))
        self.right_buffer = []
        self.left_buffer = []

        # ---------------- Detection state ----------------
        self.stop_active = False
        self.stop_start_time = 0.0
        self.current_shape = None
        self.current_plant_idx = None

        self.right_side_next_shape = [0, 0]
        self.left_side_next_shape = [0, 0]

        self.right_side_next_plant_index = [0] * len(PLANT_TRACKS)
        self.left_side_next_plant_index = [0] * len(PLANT_TRACKS)

        self.right_last_broadcast_plant_id = None
        self.left_last_broadcast_plant_id = None

        # ---------------- Plot ----------------
        self.colors = PLOT_COLORS
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        # ---------------- Timer ----------------
        self.create_timer(CONTROL_LOOP_PERIOD_SEC, self.control_loop)

    # ============================================================
    #                       CALLBACKS
    # ============================================================

    def fertilizer_placement_status_callback(self, msg: Bool):
        if msg.data:
            self.at_dock = False

    def in_lane_callback(self, msg: Bool):
        self.in_lane = msg.data
        if not msg.data:
            self.right_last_broadcast_plant_id = None
            self.left_last_broadcast_plant_id = None

    def odom_callback(self, msg: Odometry):
        if not self.in_lane:
            return

        self.current_x = float(msg.pose.pose.position.x)
        self.current_y = float(msg.pose.pose.position.y)

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = float(math.atan2(siny, cosy))

    def lidar_callback(self, msg: LaserScan):
        if not self.in_lane:
            self.current_buffer = []
            self.right_buffer = []
            self.left_buffer = []
            self.last_point_buffered = None
            return

        if self.current_x is None or self.current_y is None or self.current_yaw is None:
            return

        distances = np.nan_to_num(np.array(msg.ranges), nan=0.0)
        angles = msg.angle_min + np.arange(len(distances)) * msg.angle_increment

        x_local = distances * np.cos(angles)
        y_local = distances * np.sin(angles)

        cy, sy = math.cos(self.current_yaw), math.sin(self.current_yaw)

        xg = self.current_x + (x_local * cy - y_local * sy)
        yg = self.current_y + (x_local * sy + y_local * cy)

        pts = np.vstack((xg, yg)).T
        self.current_buffer = pts

        if self.last_point_buffered is None:
            self.right_buffer.append(self.get_side(pts[0:90]))
            self.left_buffer.append(self.get_side(pts[270:360]))
            self.last_point_buffered = (self.current_x, self.current_y)
            return

        dist = math.hypot(
            self.current_x - self.last_point_buffered[0],
            self.current_y - self.last_point_buffered[1],
        )

        if dist >= GLOBAL_BUFFER_MIN_DISTANCE:
            self.right_buffer.append(self.get_side(pts[0:90]))
            self.left_buffer.append(self.get_side(pts[270:360]))
            self.last_point_buffered = (self.current_x, self.current_y)

        self.right_buffer = self.right_buffer[-GLOBAL_BUFFER_LENGTH:]
        self.left_buffer = self.left_buffer[-GLOBAL_BUFFER_LENGTH:]

    # ============================================================
    #                       HELPERS
    # ============================================================

    def get_side(self, pts):
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
        if self.stop_active:
            elapsed = time.time() - self.stop_start_time

            if elapsed >= STOP_DURATION_SEC:
                self.stop_active = False
                self.stop_pub.publish(Bool(data=False))
            else:
                self.stop_pub.publish(Bool(data=True))
            return

        if not self.in_lane:
            return

        # ===== RIGHT SIDE =====
        self.right_lines = self.extract_multiple_lines(self.right_buffer)
        self._process_side(self.right_lines, True)

        # ===== LEFT SIDE =====
        self.left_lines = self.extract_multiple_lines(self.left_buffer)
        self._process_side(self.left_lines, False)

    def _process_side(self, lines, is_right):
        side_shape = self.right_side_next_shape if is_right else self.left_side_next_shape
        side_votes = self.right_side_next_plant_index if is_right else self.left_side_next_plant_index
        last_id = self.right_last_broadcast_plant_id if is_right else self.left_last_broadcast_plant_id

        line_dicts = []
        for model, pts in lines:
            p1, p2, _, _ = self.compute_line_endpoints(model, pts)
            line_dicts.append({"model": model, "p1": p1, "p2": p2})

        shape = self.detect_shape(line_dicts)
        if not shape:
            return

        shape_type, group = shape
        cx, cy, _ = self.compute_shape_position(group)

        for idx, (px, py) in enumerate(PLANT_TRACKS):
            if abs(cx - px) <= 1.0 and abs(cy - py) <= 0.5 and idx != last_id:
                side_votes[idx] += 1
                break

        max_count = max(side_votes)
        if max_count >= SHAPE_CONFIRMATION_COUNT:
            plant_id = side_votes.index(max_count)

            if abs(self.current_y - PLANT_TRACKS[plant_id][1]) <= 0.2:
                self.current_plant_idx = plant_id
                self.current_shape = 0 if shape_type == "triangle" else 1
                self.stop_active = True
                self.stop_start_time = time.time()

                for i in range(len(side_votes)):
                    side_votes[i] = 0
                side_shape[0] = side_shape[1] = 0

    # ============================================================
    #                       RANSAC
    # ============================================================

    def ransac_line(self, points):
        pts = np.array(points)
        if len(pts) < RANSAC_MIN_INLIERS:
            return None, None

        best_model, best_inliers = None, None

        for _ in range(RANSAC_ITERATIONS):
            p1, p2 = pts[np.random.choice(len(pts), 2, replace=False)]
            a, b = p2[1] - p1[1], -(p2[0] - p1[0])
            c = p2[0] * p1[1] - p2[1] * p1[0]

            norm = math.hypot(a, b)
            if norm < 1e-6:
                continue

            dist = np.abs(a * pts[:, 0] + b * pts[:, 1] + c) / norm
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

    # ============================================================
    #                       SHAPE
    # ============================================================

    def detect_shape(self, lines):
        if len(lines) < 3:
            return None

        enriched = []
        for L in lines:
            enriched.append({**L, "length": self.segment_length(L)})

        from itertools import combinations

        for group in combinations(enriched, 3):
            lengths = [g["length"] for g in group]
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

    def line_direction(self, model):
        a, b, _ = model
        d = np.array([b, -a])
        return d / (np.linalg.norm(d) + 1e-9)

    def angle_between_lines(self, m1, m2):
        d1 = self.line_direction(m1)
        d2 = self.line_direction(m2)
        ang = math.degrees(math.acos(np.clip(np.dot(d1, d2), -1.0, 1.0)))
        return min(ang, 180 - ang)

    def compute_line_endpoints(self, model, pts):
        a, b, c = model
        d = np.array([b, -a]) / np.linalg.norm([b, -a])
        p0 = np.array([-a*c/(a*a+b*b), -b*c/(a*a+b*b)])

        proj = [(np.dot(p - p0, d), p) for p in pts]
        return min(proj)[1], max(proj)[1], None, None

    def segment_length(self, seg):
        return np.linalg.norm(seg["p2"] - seg["p1"])

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


def main(args=None):
    rclpy.init(args=args)
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
