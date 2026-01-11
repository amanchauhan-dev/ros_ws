#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


# ============================================================
# CONSTANTS
# ============================================================

# Lane boundaries
LANE_Y_MIN = -5.0
LANE_Y_MAX = 1.0

# Buffer parameters
BUFFER_LENGTH = 3
GLOBAL_BUFFER_DISTANCE_DIFFERENCE = 0.2

# Side filtering parameters
SIDE_DISTANCE_TOLERANCE = 1.5
SIDE_LENGTH = 0.25

# RANSAC parameters
RANSAC_ITERATIONS = 200
RANSAC_THRESHOLD = 0.01
RANSAC_MIN_INLIERS = 5
MAX_LINES = 6

# Shape detection parameters
SHAPE_SIDE_LENGTH = 0.25
LENGTH_TOLERANCE = 0.08
ANGLE_90_TOLERANCE = 15
ANGLE_60_TOLERANCE = 20

# Stop detection parameters
MIN_FREQUENCY_THRESHOLD = 5
PLANT_VALIDATION_WINDOW_Y = 0.20
PLANT_DETECTION_TOLERANCE_Y = 0.5
PLANT_DETECTION_TOLERANCE_X = 1.00

# Plant track locations
PLANTS_TRACKS = [
    [0.26, -1.95],
    [-0.50, -4.0941],
    [-0.50, -2.7702],
    [-0.50, -1.4044],
    [-0.50, -0.0461],
    [-2.40, -4.0941],
    [-2.40, -2.7702],
    [-2.40, -1.4044],
    [-2.40, -0.0461],
]

# LiDAR scan indices
FRONT_ANGLE = 0.0
LEFT_ANGLE  = math.pi / 2
RIGHT_ANGLE = -math.pi / 2
SECTOR_WIDTH = math.radians(15)

# MAD filtering parameters
MAD_MULTIPLIER = 2.5
MAD_EPSILON = 1e-6

# Noise filtering parameters
NEIGHBOR_DISTANCE_THRESHOLD = 0.15

# Plot parameters
PLOT_COLORS = ["red", "blue", "green", "orange", "purple", "cyan"]
PLOT_XLIM = (-6, 6)
PLOT_YLIM = (-6, 6)
ROBOT_ARROW_LENGTH = 0.5
ROBOT_ARROW_HEAD_WIDTH = 0.05

# Misc constants
EPSILON = 1e-9
NORM_EPSILON = 1e-6


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def check_in_lane(x, y):
    """Check if position (x, y) is within the lane boundaries."""
    return LANE_Y_MIN < y < LANE_Y_MAX


def quaternion_to_yaw(q):
    """Convert quaternion to yaw angle."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def transform_to_global(x_local, y_local, robot_x, robot_y, robot_yaw):
    """Transform local coordinates to global frame."""
    cy = math.cos(robot_yaw)
    sy = math.sin(robot_yaw)
    xg = robot_x + (x_local * cy - y_local * sy)
    yg = robot_y + (x_local * sy + y_local * cy)
    return xg, yg


def compute_distance(x1, y1, x2, y2):
    """Compute Euclidean distance between two points."""
    return math.hypot(x2 - x1, y2 - y1)


def line_direction_vector(model):
    """Compute normalized direction vector for a line model (a, b, c)."""
    a, b, c = model
    d = np.array([b, -a], float)
    d /= (np.linalg.norm(d) + EPSILON)
    return d


def angle_between_lines(model1, model2):
    """Compute the smallest angle between two line models."""
    d1 = line_direction_vector(model1)
    d2 = line_direction_vector(model2)
    dot = np.clip(np.dot(d1, d2), -1.0, 1.0)
    ang = math.degrees(math.acos(dot))
    if ang > 90:
        ang = 180 - ang
    return ang


def compute_line_slope_intercept(model):
    """Compute slope and intercept from line model (a, b, c)."""
    a, b, c = model
    if abs(b) > 1e-8:
        m = -a / b
        k = -c / b
    else:
        m = None
        k = -c / a
    return m, k


def compute_line_endpoints(model, inlier_pts):
    """Compute endpoints, slope, and intercept for a line from its inliers."""
    a, b, c = model
    m, k = compute_line_slope_intercept(model)
    
    d = line_direction_vector(model)
    
    denom = a * a + b * b
    point0 = np.array([-a * c / denom, -b * c / denom])
    
    projections = []
    for p in inlier_pts:
        v = p - point0
        t = np.dot(v, d)
        projections.append((t, p))
    
    t_vals = [tp[0] for tp in projections]
    p_min = projections[np.argmin(t_vals)][1]
    p_max = projections[np.argmax(t_vals)][1]
    
    return p_min, p_max, m, k


def segment_length(seg):
    """Compute length of a line segment."""
    return np.linalg.norm(seg["p2"] - seg["p1"])


def compute_shape_centroid_distance(group, robot_x, robot_y):
    """Compute centroid and distance from robot for a group of lines."""
    pts = []
    for L in group:
        pts.append(L["p1"])
        pts.append(L["p2"])
    
    pts = np.array(pts)
    cx, cy = np.mean(pts, axis=0)
    dist = math.sqrt((cx - robot_x) ** 2 + (cy - robot_y) ** 2)
    
    return cx, cy, dist


def filter_far_points(pts, robot_x, robot_y, tolerance):
    """Remove points that are too far from the robot."""
    dists = np.linalg.norm(pts - np.array([robot_x, robot_y]), axis=1)
    mask = dists < tolerance
    return pts[mask], dists[mask]


def filter_outliers_mad(pts, dists):
    """Remove statistical outliers using Median Absolute Deviation."""
    if len(pts) == 0:
        return pts, dists
    
    median_dist = np.median(dists)
    mad = np.median(np.abs(dists - median_dist)) + MAD_EPSILON
    stable_mask = np.abs(dists - median_dist) < MAD_MULTIPLIER * mad
    
    return pts[stable_mask], dists[stable_mask]


def filter_isolated_points(pts):
    """Remove isolated single-point noise based on neighborhood distance."""
    if len(pts) < 3:
        return pts
    
    clean_pts = [pts[0]]
    for i in range(1, len(pts) - 1):
        left_d = np.linalg.norm(pts[i] - pts[i - 1])
        right_d = np.linalg.norm(pts[i] - pts[i + 1])
        
        if left_d < NEIGHBOR_DISTANCE_THRESHOLD or right_d < NEIGHBOR_DISTANCE_THRESHOLD:
            clean_pts.append(pts[i])
    
    clean_pts.append(pts[-1])
    return np.array(clean_pts)


# ============================================================
# RANSAC NODE CLASS
# ============================================================

class Ransac(Node):
    def __init__(self):
        super().__init__("ransac_node")

        # Subscribers
        self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        # Publishers
        self.detection_pub = self.create_publisher(String, "/detection_status", 10)
        self.stop_pub = self.create_publisher(Bool, "/to_stop", 10)
        self.at_dock_pub = self.create_publisher(Bool, "/ebot_dock_status", 10)

        # State variables
        self.in_lane = False
        self.right_buffer = []
        self.current_buffer = np.empty((0, 2))
        self.left_buffer = []
        self.global_points = []

        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self.last_point_buffered = None

        self.right_lines = []
        self.left_lines = []

        self.stop_active = False
        self.stop_start_time = 0.0

        self.current_shape = None
        self.current_plant_idx = None
        self.at_dock = False

        # Right side tracking
        self.right_side_next_shape = [0, 0]
        self.right_side_next_plant_index = [0] * 9
        self.right_last_broadcast_plant_id = None

        # Left side tracking
        self.left_side_next_shape = [0, 0]
        self.left_side_next_plant_index = [0] * 9
        self.left_last_broadcast_plant_id = None

        # Plotting setup
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_title("RANSAC", fontsize=14, fontweight="bold")
        self.ax.set_xlabel("X (m)", fontsize=11)
        self.ax.set_ylabel("Y (m)", fontsize=11)
        self.ax.set_xlim(*PLOT_XLIM)
        self.ax.set_ylim(*PLOT_YLIM)
        self.ax.grid(True, alpha=0.3)

        # Timer
        self.create_timer(0.1, self.control_loop)

    def fertilizer_placement_status_callback(self, msg: Bool):
        """Handle fertilizer placement status updates."""
        if msg.data:
            self.at_dock = False

    # ============================================================
    # CALLBACKS
    # ============================================================

    def odom_callback(self, msg: Odometry):
        """Extract position and yaw from odometry."""
        self.current_x = float(msg.pose.pose.position.x)
        self.current_y = float(msg.pose.pose.position.y)
        self.current_yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.in_lane = check_in_lane(self.current_x, self.current_y)

    def lidar_callback(self, msg: LaserScan):
        """Process incoming LaserScan data."""
        if not self.in_lane:
            self.reset_buffers()
            return

        if self.current_x is None or self.current_y is None or self.current_yaw is None:
            return

        distances = np.nan_to_num(
            np.array(msg.ranges, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        
        if distances.size == 0 or np.all(distances == 0.0):
            return

        pts = self.convert_scan_to_global(msg, distances)
        self.current_buffer = pts

        if self.should_buffer_points():
            self.buffer_side_points(pts)

        self.trim_buffers()

    # ============================================================
    # LIDAR PROCESSING
    # ============================================================

    def convert_scan_to_global(self, msg, distances):
        """Convert LaserScan ranges to global coordinates."""
        idx = np.arange(len(distances))
        angles = msg.angle_min + idx * msg.angle_increment

        x_local = distances * np.cos(angles)
        y_local = distances * np.sin(angles)

        xg, yg = transform_to_global(
            x_local, y_local, self.current_x, self.current_y, self.current_yaw
        )

        return np.vstack((xg, yg)).T

    def should_buffer_points(self):
        """Determine if points should be added to buffers based on robot movement."""
        if self.last_point_buffered is None:
            self.last_point_buffered = (self.current_x, self.current_y)
            return True

        x1, y1 = self.last_point_buffered
        dist = compute_distance(x1, y1, self.current_x, self.current_y)

        if dist >= GLOBAL_BUFFER_DISTANCE_DIFFERENCE:
            self.last_point_buffered = (self.current_x, self.current_y)
            return True

        return False

    def buffer_side_points(self, pts):
        """Extract and buffer right and left side points."""
        right_pts , left_pts =[], []

        for gx, gy in pts:
            dx, dy = gx - self.current_x, gy - self.current_y
            ang = math.atan2(dy, dx) - self.current_yaw
            ang = math.atan2(math.sin(ang), math.cos(ang))

            if abs(ang - RIGHT_ANGLE) <= SECTOR_WIDTH:
                right_pts.append([gx, gy])
            elif abs(ang - LEFT_ANGLE) <= SECTOR_WIDTH:
                left_pts.append([gx, gy])

        self.right_buffer.append(self.get_side(right_pts))
        self.left_buffer.append(self.get_side(left_pts))

    def trim_buffers(self):
        """Trim buffers to maximum length."""
        self.right_buffer = self.right_buffer[-BUFFER_LENGTH:]
        self.left_buffer = self.left_buffer[-BUFFER_LENGTH:]

    def reset_buffers(self):
        """Reset all buffers and state."""
        self.current_buffer = []
        self.right_buffer = []
        self.left_buffer = []
        self.global_points = []
        self.left_lines = []
        self.right_lines = []
        self.last_point_buffered = None

    def get_side(self, pts):
        """Filter raw LiDAR points for a specific side."""
        pts = np.array(pts)
        if len(pts) == 0:
            return pts

        pts, dists = filter_far_points(
            pts, self.current_x, self.current_y, SIDE_DISTANCE_TOLERANCE
        )
        if len(pts) == 0:
            return pts

        pts, dists = filter_outliers_mad(pts, dists)
        if len(pts) == 0:
            return pts

        pts = filter_isolated_points(pts)
        return pts

    # ============================================================
    # RANSAC LINE DETECTION
    # ============================================================

    def ransac_line(self, points):
        """Perform RANSAC to find the best fitting line."""
        pts = np.array(points)
        best_inliers = None
        best_model = None
        n = len(pts)

        if n < RANSAC_MIN_INLIERS:
            return None, None

        for _ in range(RANSAC_ITERATIONS):
            idx = np.random.choice(n, 2, replace=False)
            p1, p2 = pts[idx]

            x1, y1 = p1
            x2, y2 = p2

            a = y2 - y1
            b = -(x2 - x1)
            c = x2 * y1 - y2 * x1

            norm = np.hypot(a, b)
            if norm < NORM_EPSILON:
                continue

            dist = np.abs(a * pts[:, 0] + b * pts[:, 1] + c) / norm
            mask = dist < RANSAC_THRESHOLD
            inliers = np.sum(mask)

            if inliers > RANSAC_MIN_INLIERS and (
                best_inliers is None or inliers > np.sum(best_inliers)
            ):
                best_inliers = mask
                best_model = (a, b, c)

        return best_model, best_inliers

    def extract_multiple_lines(self, pts):
        """Iteratively extract multiple lines using RANSAC."""
        if not pts or len(pts) == 0:
            return []
        
        pts = np.array(np.vstack(pts))
        lines = []

        while len(pts) >= RANSAC_MIN_INLIERS:
            model, inliers = self.ransac_line(pts)
            if model is None:
                break

            line_pts = pts[inliers]
            lines.append((model, line_pts))

            pts = pts[~inliers]
            if len(lines) >= MAX_LINES:
                break

        return lines

    # ============================================================
    # SHAPE DETECTION
    # ============================================================

    def detect_shape(self, lines):
        """Detect geometric shapes (Triangle or Square) from lines."""
        if len(lines) < 3:
            return None

        enriched = self.enrich_lines_with_length(lines)
        best_square = self.find_best_square(enriched)
        best_triangle = self.find_best_triangle(enriched)

        if best_triangle:
            return ("triangle", best_triangle)
        if best_square:
            return ("square", best_square)

        return None

    def enrich_lines_with_length(self, lines):
        """Add length property to each line."""
        enriched = []
        for L in lines:
            Llen = segment_length(L)
            enriched.append({**L, "length": Llen})
        return enriched

    def find_best_square(self, enriched):
        """Find best square candidate from enriched lines."""
        for group in combinations(enriched, 3):
            if self.is_square(group):
                return group
        return None

    def find_best_triangle(self, enriched):
        """Find best triangle candidate from enriched lines."""
        for group in combinations(enriched, 3):
            if self.is_triangle(group):
                return group
        return None

    def is_square(self, group):
        """Check if group of lines forms a square."""
        L1, L2, L3 = group
        lengths = [L1["length"], L2["length"], L3["length"]]
        near_count = sum(abs(L - SHAPE_SIDE_LENGTH) < LENGTH_TOLERANCE for L in lengths)

        ang12 = angle_between_lines(L1["model"], L2["model"])
        ang23 = angle_between_lines(L2["model"], L3["model"])
        ang31 = angle_between_lines(L3["model"], L1["model"])

        angles = [ang12, ang23, ang31]
        right_angles = sum(abs(a - 90) < ANGLE_90_TOLERANCE for a in angles)

        return right_angles == 2 and near_count >= 2

    def is_triangle(self, group):
        """Check if group of lines forms a triangle."""
        L1, L2, L3 = group
        lengths = [L1["length"], L2["length"], L3["length"]]
        near_count = sum(abs(L - SHAPE_SIDE_LENGTH) < LENGTH_TOLERANCE for L in lengths)

        ang12 = angle_between_lines(L1["model"], L2["model"])
        ang23 = angle_between_lines(L2["model"], L3["model"])
        ang31 = angle_between_lines(L3["model"], L1["model"])

        angles = [ang12, ang23, ang31]
        sixty_angles = sum(abs(a - 60) < ANGLE_60_TOLERANCE for a in angles)

        return near_count >= 2 and sixty_angles >= 2

    # ============================================================
    # DETECTION AND TRACKING
    # ============================================================

    def process_side_detection(self, buffer, side_shape, side_plant_index, last_broadcast_id, side_name):
        """Process shape detection for one side (right or left)."""
        lines = self.extract_multiple_lines(buffer)
        line_dicts = self.convert_lines_to_dicts(lines)

        shape = self.detect_shape(line_dicts)
        if shape:
            shape_type, group = shape
            cx, cy, dist = compute_shape_centroid_distance(
                group, self.current_x, self.current_y
            )
            
            is_valid = False
            self.get_logger().info(
                f"current status: {side_shape} {side_plant_index}"
            )
            
            for idx, (px, py) in enumerate(PLANTS_TRACKS):
                if (
                    abs(cy - py) <= PLANT_DETECTION_TOLERANCE_Y
                    and abs(cx - px) <= PLANT_DETECTION_TOLERANCE_X
                    and idx != last_broadcast_id
                ):
                    is_valid = True
                    side_plant_index[idx] += 1
                    break

            if is_valid:
                if shape_type == "triangle":
                    side_shape[0] += 1
                elif shape_type == "square":
                    side_shape[1] += 1

        return self.check_stop_condition(
            side_plant_index, side_shape, last_broadcast_id
        )

    def convert_lines_to_dicts(self, lines):
        """Convert line tuples to dictionary format with endpoints."""
        line_dicts = []
        for model, pts_line in lines:
            p1, p2, m, k = compute_line_endpoints(model, pts_line)
            line_dicts.append({
                "model": model,
                "p1": p1,
                "p2": p2,
                "slope": m,
                "intercept": k,
            })
        return line_dicts

    def check_stop_condition(self, plant_index, shape_tracker, last_broadcast_id):
        """Check if stop condition is met for detected shapes."""
        max_count = max(plant_index)
        if max_count >= MIN_FREQUENCY_THRESHOLD:
            plant_id = plant_index.index(max_count)

            if last_broadcast_id != plant_id:
                plant_x, plant_y = PLANTS_TRACKS[plant_id]
                if abs(self.current_y - plant_y) <= PLANT_VALIDATION_WINDOW_Y:
                    self.current_plant_idx = plant_id
                    self.current_shape = shape_tracker.index(max(shape_tracker))

                    self.stop_active = True
                    self.stop_start_time = time.time()

                    return plant_id, [0] * 9, [0, 0]

        return last_broadcast_id, plant_index, shape_tracker

    # ============================================================
    # MAIN CONTROL LOOP
    # ============================================================

    def control_loop(self):
        """Main control loop for processing detections."""
        if not self.in_lane:
            return

        # Right side detection
        (
            self.right_last_broadcast_plant_id,
            self.right_side_next_plant_index,
            self.right_side_next_shape,
        ) = self.process_side_detection(
            self.right_buffer,
            self.right_side_next_shape,
            self.right_side_next_plant_index,
            self.right_last_broadcast_plant_id,
            "RIGHT",
        )

        # Left side detection
        (
            self.left_last_broadcast_plant_id,
            self.left_side_next_plant_index,
            self.left_side_next_shape,
        ) = self.process_side_detection(
            self.left_buffer,
            self.left_side_next_shape,
            self.left_side_next_plant_index,
            self.left_last_broadcast_plant_id,
            "LEFT",
        )

        # self.graph()

    # ============================================================
    # VISUALIZATION
    # ============================================================

    def graph(self):
        """Plot current state including buffers, lines, and robot position."""
        self.ax.clear()
        self.ax.set_title("RANSAC", fontsize=14, fontweight="bold")
        self.ax.set_xlabel("X (m)", fontsize=11)
        self.ax.set_ylabel("Y (m)", fontsize=11)
        self.ax.set_xlim(*PLOT_XLIM)
        self.ax.set_ylim(*PLOT_YLIM)
        self.ax.grid(True, alpha=0.3)

        if len(self.current_buffer) > 0:
            pts = np.vstack(self.current_buffer)
            self.ax.scatter(pts[:, 0], pts[:, 1], s=8)

        if len(self.right_buffer) > 0:
            rpts = np.vstack(self.right_buffer)
            self.ax.scatter(rpts[:, 0], rpts[:, 1], s=8)

        if len(self.left_buffer) > 0:
            lpts = np.vstack(self.left_buffer)
            self.ax.scatter(lpts[:, 0], lpts[:, 1], s=8)

        if (
            self.current_x is not None
            and self.current_y is not None
            and self.current_yaw is not None
        ):
            self.ax.plot(self.current_x, self.current_y, "bo", markersize=6)
            self.ax.arrow(
                self.current_x,
                self.current_y,
                ROBOT_ARROW_LENGTH * math.cos(self.current_yaw),
                ROBOT_ARROW_LENGTH * math.sin(self.current_yaw),
                head_width=ROBOT_ARROW_HEAD_WIDTH,
                color="blue",
            )

        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ============================================================
    # CLEANUP
    # ============================================================

    def destroy_node(self):
        """Clean up resources on node destruction."""
        try:
            plt.close(self.fig)
        except Exception:
            pass
        super().destroy_node()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = Ransac()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()