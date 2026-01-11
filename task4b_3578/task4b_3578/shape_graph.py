#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
# Team ID:          3578
# Theme:            Krishi coBot
# Author List:      Raghav Jibachha Mandal, Ashishkumar Rajeshkumar Jha,
#                   Aman Ratanlal Chauhan, Harshil Rahulbhai Mehta
# Filename:         shape_detector_task3b.py
# Functions:        __init__, fertilizer_placement_status_callback, in_lane_callback,
#                   odom_callback, lidar_callback, get_side, control_loop, graph,
#                   ransac_line, extract_multiple_lines, detect_shape,
#                   line_direction, angle_between_lines, compute_line_endpoints,
#                   segment_length, compute_shape_position, destroy_node, main
# Global variables: None
'''

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import Bool, String
import time


# ============================================================================
# TUNABLE CONSTANTS
# ============================================================================

# Lane boundaries
LANE_MIN = 0.912
LANE_MAX = 4.0

# LiDAR angle constants
FRONT_ANGLE = 0.0
LEFT_ANGLE = math.pi / 2
RIGHT_ANGLE = -math.pi / 2
SECTOR_WIDTH = math.radians(45)

# Buffer parameters
BUFFER_LENGTH = 3
GLOBAL_BUFFER_DISTANCE_DIFFERENCE = 0.2

# Side filtering parameters
SIDE_DISTANCE_TOLERANCE = 1.5
SIDE_MAD_MULTIPLIER = 2.5
SIDE_NEIGHBOR_THRESHOLD = 0.15

# RANSAC parameters
RANSAC_ITERATIONS = 200
RANSAC_THRESHOLD = 0.01
RANSAC_MIN_INLIERS = 5
RANSAC_MAX_LINES = 6

# Shape detection parameters
SHAPE_SIDE_LENGTH = 0.25
SHAPE_LENGTH_TOLERANCE = 1.0
SHAPE_ANGLE_90_TOLERANCE = 15
SHAPE_ANGLE_60_TOLERANCE = 20

# Stop parameters
STOP_DURATION = 2.0
STOP_MIN_FREQUENCY = 5
STOP_VALIDATION_WINDOW_Y = 0.20

# Shape validation parameters
SHAPE_VALIDATION_Y_TOLERANCE = 0.5
SHAPE_VALIDATION_X_TOLERANCE = 1.00

# Plot parameters
PLOT_UPDATE_INTERVAL = 0.1
PLOT_X_LIMIT = 6
PLOT_Y_LIMIT = 6
PLOT_POINT_SIZE = 8
PLOT_ROBOT_SIZE = 6
PLOT_ARROW_LENGTH = 0.5
PLOT_ARROW_HEAD_WIDTH = 0.05

# Plant track positions
PLANTS_TRACKS = [
    [2.333, -1.877],
    [1.301, -0.8475],
    [2.0115, -0.8475],
    [2.748, -0.8475],
    [3.573, -0.8475],
    [1.301, 0.856],
    [2.0115, 0.856],
    [2.748, 0.856],
    [3.573, 0.856],
]

# ============================================================================


def check_in_lane(x, y):
    """Check if position (x, y) is within the lane boundaries."""
    return LANE_MIN < x < LANE_MAX


class Ransac(Node):
    def __init__(self):
        super().__init__("ransac_node")

        # subscribers
        self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(Bool, "/fertilizer_placement_status", self.fertilizer_placement_status_callback, 10)

        self.detection_pub = self.create_publisher(String, "/detection_status", 10)
        self.stop_pub = self.create_publisher(Bool, "/to_stop", 10)
        self.at_dock_pub = self.create_publisher(Bool, "/ebot_dock_status", 10)

        # memory / buffers
        self.in_lane = False
        self.right_buffer = []
        self.current_buffer = np.empty((0, 2))
        self.left_buffer = []
        self.global_points = []

        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self.last_point_buffered = None

        # ransac
        self.right_lines = []
        self.left_lines = []

        self.stop_active = False
        self.stop_start_time = 0.0

        # shape detection
        self.current_shape = None
        self.current_plant_idx = None
        self.at_dock = False

        # Right side shape
        self.right_side_next_shape = [0, 0]
        self.right_side_next_plant_index = [0] * 9
        self.right_last_broadcast_plant_id = None

        # left side shape
        self.left_side_next_shape = [0, 0]
        self.left_side_next_plant_index = [0] * 9
        self.left_last_broadcast_plant_id = None

        # plotting
        plt.ion()  # FIXED: Enable interactive mode
        self.colors = ["red", "blue", "green", "orange", "purple", "cyan"]
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_title("RANSAC", fontsize=14, fontweight="bold")
        self.ax.set_xlabel("X (m)", fontsize=11)
        self.ax.set_ylabel("Y (m)", fontsize=11)
        self.ax.set_xlim(-PLOT_X_LIMIT, PLOT_X_LIMIT)
        self.ax.set_ylim(-PLOT_Y_LIMIT, PLOT_Y_LIMIT)
        self.ax.grid(True, alpha=0.3)

        # timer
        self.create_timer(PLOT_UPDATE_INTERVAL, self.control_loop)

    def fertilizer_placement_status_callback(self, msg: Bool):
        if msg.data == True:
            self.at_dock = False

    # ---------------- callbacks ---------------- #
    def odom_callback(self, msg: Odometry):
        """Extract position and yaw from odometry (store as floats)."""
        
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        self.in_lane = check_in_lane(x, y)

        if not self.in_lane:
            self.right_side_next_shape = [0, 0]
            self.right_side_next_plant_index = [0] * 9
            self.left_side_next_shape = [0, 0]
            self.left_side_next_plant_index = [0] * 9
            self.right_buffer = []
            self.left_buffer = []
            self.right_last_broadcast_plant_id = None
            self.left_last_broadcast_plant_id = None
            return
        
        self.current_x = x
        self.current_y = y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = float(math.atan2(siny_cosp, cosy_cosp))
    
    def buffer_side_points(self, pts):
        """Extract and buffer right and left side points."""
        right_pts, left_pts = [], []

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

    def lidar_callback(self, msg: LaserScan):
        """
        Convert incoming LaserScan to global points (uses current pose).
        Store to current_buffer. Also append to side buffers when robot moved
        at least global_buffer_distance_difference since last stored frame.
        """
        if not self.in_lane:
            self.current_buffer = []
            self.right_buffer = []
            self.left_buffer = []
            self.global_points = []
            self.left_lines = []
            self.right_lines = []
            self.last_point_buffered = None
            return
        
        if self.current_x is None or self.current_y is None or self.current_yaw is None:
            return

        distances = np.nan_to_num(np.array(msg.ranges, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if distances.size == 0 or np.all(distances == 0.0):
            return

        idx = np.arange(len(distances))
        angles = msg.angle_min + idx * msg.angle_increment

        x_local = distances * np.cos(angles)
        y_local = distances * np.sin(angles)

        cy = math.cos(self.current_yaw)
        sy = math.sin(self.current_yaw)

        xg = self.current_x + (x_local * cy - y_local * sy)
        yg = self.current_y + (x_local * sy + y_local * cy)

        pts = np.vstack((xg, yg)).T
        self.current_buffer = pts

        if self.last_point_buffered is None:
            self.buffer_side_points(pts)
            self.last_point_buffered = (self.current_x, self.current_y)
            return

        x1, y1 = self.last_point_buffered
        dist = math.hypot(self.current_x - x1, self.current_y - y1)

        if dist >= GLOBAL_BUFFER_DISTANCE_DIFFERENCE:
            self.buffer_side_points(pts)
            self.last_point_buffered = (self.current_x, self.current_y)
                
        self.right_buffer = self.right_buffer[-BUFFER_LENGTH:]
        self.left_buffer = self.left_buffer[-BUFFER_LENGTH:]

    # ---------------- helpers ---------------- #

    def get_side(self, pts):
        '''Filters raw LiDAR points for a specific side.'''
        robot_x, robot_y = self.current_x, self.current_y
        pts = np.array(pts)
        if len(pts) == 0:
            return pts
    
        dists = np.linalg.norm(pts - np.array([robot_x, robot_y]), axis=1)
    
        mask = dists < SIDE_DISTANCE_TOLERANCE
        pts = pts[mask]
        dists = dists[mask]
    
        if len(pts) == 0:
            return pts
    
        median_dist = np.median(dists)
        mad = np.median(np.abs(dists - median_dist)) + 1e-6
    
        stable_mask = np.abs(dists - median_dist) < SIDE_MAD_MULTIPLIER * mad
        pts = pts[stable_mask]
        dists = dists[stable_mask]
    
        if len(pts) == 0:
            return pts
    
        clean_pts = [pts[0]]
        for i in range(1, len(pts) - 1):
            left_d = np.linalg.norm(pts[i] - pts[i - 1])
            right_d = np.linalg.norm(pts[i] - pts[i + 1])
    
            if left_d < SIDE_NEIGHBOR_THRESHOLD or right_d < SIDE_NEIGHBOR_THRESHOLD:
                clean_pts.append(pts[i])
    
        clean_pts.append(pts[-1])
    
        return np.array(clean_pts)

    # ---------------- main loop / plotting ---------------- #
    def control_loop(self):
        if self.stop_active:
            elapsed = time.time() - self.stop_start_time

            if elapsed >= STOP_DURATION:
                self.stop_active = False
                self.stop_pub.publish(Bool(data=False))
                self.get_logger().info("==== STOP FALSE (timeout) ====")
            else:
                detection_msg = String()
                if self.current_plant_idx == 0:
                    detection_msg.data = f"DOCK_STATION,{self.current_x:.2f},{self.current_y:.2f},{self.current_plant_idx}"
                    self.at_dock = True
                else:
                    s = "FERTILIZER_REQUIRED" if self.current_shape == 0 else "BAD_HEALTH"
                    detection_msg.data = f"{s},{self.current_x:.2f},{self.current_y:.2f},{self.current_plant_idx}"

                self.detection_pub.publish(detection_msg)
                self.get_logger().info("==== publishing ====")
                self.stop_pub.publish(Bool(data=True))
            return 
        else:
            self.stop_pub.publish(Bool(data=False))
        
        if not self.in_lane:
            return
            
        # ============> RIGHT SIDE DETECTION <===========
        self.right_lines = self.extract_multiple_lines(self.right_buffer)

        line_dicts = []
        for i, (model, pts_line) in enumerate(self.right_lines):
            p1, p2, m, k = self.compute_line_endpoints(model, pts_line)
            line_dicts.append({
                "model": model,
                "p1": p1,
                "p2": p2,
                "slope": m,
                "intercept": k
            })

        shape = self.detect_shape(line_dicts)
        if shape:
            shape_type, group = shape
            cx, cy, dist = self.compute_shape_position(group)
            is_valid = False
            self.get_logger().info(f"current status: {self.right_side_next_shape} {self.right_side_next_plant_index}")
            for idx, (px, py) in enumerate(PLANTS_TRACKS):
                if abs(cy - py) <= SHAPE_VALIDATION_Y_TOLERANCE and abs(cx - px) <= SHAPE_VALIDATION_X_TOLERANCE and idx != self.right_last_broadcast_plant_id:
                    is_valid = True
                    self.right_side_next_plant_index[idx] += 1
                    break 
            
            if is_valid:
                if shape_type == "triangle":
                    self.right_side_next_shape[0] += 1
                elif shape_type == "square":
                    self.right_side_next_shape[1] += 1
       
        max_count = max(self.right_side_next_plant_index)
        if max_count >= STOP_MIN_FREQUENCY:
            plant_id = self.right_side_next_plant_index.index(max_count)

            if self.right_last_broadcast_plant_id != plant_id:
                plant_x, plant_y = PLANTS_TRACKS[plant_id]
                if abs(self.current_y - plant_y) <= STOP_VALIDATION_WINDOW_Y:
                    self.right_last_broadcast_plant_id = plant_id
                    self.current_plant_idx = plant_id
                    self.current_shape = self.right_side_next_shape.index(max(self.right_side_next_shape))
                    self.stop_active = True
                    self.stop_start_time = time.time()
                    self.right_side_next_plant_index = [0] * 9
                    self.right_side_next_shape = [0, 0]

        # ============> LEFT SIDE DETECTION <===========
        self.left_lines = self.extract_multiple_lines(self.left_buffer)

        line_dicts = []
        for i, (model, pts_line) in enumerate(self.left_lines):
            p1, p2, m, k = self.compute_line_endpoints(model, pts_line)
            line_dicts.append({
                "model": model,
                "p1": p1,
                "p2": p2,
                "slope": m,
                "intercept": k
            })

        shape = self.detect_shape(line_dicts)
        if shape:
            shape_type, group = shape
            cx, cy, dist = self.compute_shape_position(group)
            is_valid = False
            self.get_logger().info(f"current status: {self.left_side_next_shape} {self.left_side_next_plant_index}")
            for idx, (px, py) in enumerate(PLANTS_TRACKS):
                if abs(cy - py) <= SHAPE_VALIDATION_Y_TOLERANCE and abs(cx - px) <= SHAPE_VALIDATION_X_TOLERANCE and idx != self.left_last_broadcast_plant_id:
                    is_valid = True
                    self.left_side_next_plant_index[idx] += 1
                    break 
            
            if is_valid:
                if shape_type == "triangle":
                    self.left_side_next_shape[0] += 1
                elif shape_type == "square":
                    self.left_side_next_shape[1] += 1
       
        max_count = max(self.left_side_next_plant_index)
        if max_count >= STOP_MIN_FREQUENCY:
            plant_id = self.left_side_next_plant_index.index(max_count)

            if self.left_last_broadcast_plant_id != plant_id:
                plant_x, plant_y = PLANTS_TRACKS[plant_id]
                if abs(self.current_y - plant_y) <= STOP_VALIDATION_WINDOW_Y:
                    self.left_last_broadcast_plant_id = plant_id
                    self.current_plant_idx = plant_id
                    self.current_shape = self.left_side_next_shape.index(max(self.left_side_next_shape))
                    self.stop_active = True
                    self.stop_start_time = time.time()
                    self.left_side_next_plant_index = [0] * 9
                    self.left_side_next_shape = [0, 0]

        self.graph()

    def graph(self):
        """FIXED: Proper matplotlib plotting with interactive updates"""
        self.ax.clear()
        self.ax.set_title("RANSAC", fontsize=14, fontweight="bold")
        self.ax.set_xlabel("X (m)", fontsize=11)
        self.ax.set_ylabel("Y (m)", fontsize=11)
        self.ax.set_xlim(-PLOT_X_LIMIT, PLOT_X_LIMIT)
        self.ax.set_ylim(-PLOT_Y_LIMIT, PLOT_Y_LIMIT)
        self.ax.grid(True, alpha=0.3)

        # current buffer 
        if len(self.current_buffer) > 0:
            pts = np.array(self.current_buffer)
            self.ax.scatter(pts[:, 0], pts[:, 1], s=PLOT_POINT_SIZE, c='gray', alpha=0.5, label='Current Scan')

        # right buffer 
        if len(self.right_buffer) > 0:
            rpts = np.vstack(self.right_buffer)
            self.ax.scatter(rpts[:, 0], rpts[:, 1], s=PLOT_POINT_SIZE, c='red', alpha=0.6, label='Right Buffer')

        # left buffer 
        if len(self.left_buffer) > 0:
            lpts = np.vstack(self.left_buffer)
            self.ax.scatter(lpts[:, 0], lpts[:, 1], s=PLOT_POINT_SIZE, c='green', alpha=0.6, label='Left Buffer')

        # Plot right lines
        for i, (model, pts_line) in enumerate(self.right_lines):
            color = self.colors[i % len(self.colors)]
            p1, p2, m, k = self.compute_line_endpoints(model, pts_line)
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2, label=f'R{i+1}')

        # Plot left lines
        for i, (model, pts_line) in enumerate(self.left_lines):
            color = self.colors[i % len(self.colors)]
            p1, p2, m, k = self.compute_line_endpoints(model, pts_line)
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2, linestyle='--', label=f'L{i+1}')

        # plot robot if known
        if self.current_x is not None and self.current_y is not None and self.current_yaw is not None:
            self.ax.plot(self.current_x, self.current_y, "bo", markersize=PLOT_ROBOT_SIZE, label='Robot')
            self.ax.arrow(
                self.current_x,
                self.current_y,
                PLOT_ARROW_LENGTH * math.cos(self.current_yaw),
                PLOT_ARROW_LENGTH * math.sin(self.current_yaw),
                head_width=PLOT_ARROW_HEAD_WIDTH,
                color="blue",
            )

        self.ax.legend(loc='upper right', fontsize=8)
        
        # FIXED: Proper drawing and pausing
        plt.draw()
        plt.pause(0.001)

    # Ransac
    def ransac_line(self, points):
        '''Performs a single RANSAC iteration to find the best fitting line.'''
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
            if norm < 1e-6:
                continue

            dist = np.abs(a * pts[:, 0] + b * pts[:, 1] + c) / norm
            mask = dist < RANSAC_THRESHOLD
            inliers = np.sum(mask)

            if inliers > RANSAC_MIN_INLIERS and (best_inliers is None or inliers > np.sum(best_inliers)):
                best_inliers = mask
                best_model = (a, b, c)

        return best_model, best_inliers

    def extract_multiple_lines(self, pts):
        '''Iteratively extracts multiple lines from a point cloud using RANSAC.'''
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
            if len(lines) >= RANSAC_MAX_LINES:
                break
            
        return lines

    # Shape detect
    def detect_shape(self, lines):
        '''Analyzes a set of lines to detect geometric shapes (Triangle or Square).'''
        if len(lines) < 3:
            return None

        enriched = []
        for L in lines:
            Llen = self.segment_length(L)
            enriched.append({
                **L,
                "length": Llen
            })

        best_square = None
        best_triangle = None

        from itertools import combinations

        for group in combinations(enriched, 3):
            L1, L2, L3 = group

            len1, len2, len3 = L1["length"], L2["length"], L3["length"]
            lengths = [len1, len2, len3]

            near_len = [abs(L - SHAPE_SIDE_LENGTH) < SHAPE_LENGTH_TOLERANCE for L in lengths]
            near_count = sum(near_len)

            ang12 = self.angle_between_lines(L1["model"], L2["model"])
            ang23 = self.angle_between_lines(L2["model"], L3["model"])
            ang31 = self.angle_between_lines(L3["model"], L1["model"])

            angles = [ang12, ang23, ang31]

            # SQUARE LOGIC
            right_angles = sum(abs(a - 90) < SHAPE_ANGLE_90_TOLERANCE for a in angles)

            if right_angles == 2 and near_count >= 2:
                best_square = group

            # TRIANGLE LOGIC
            sixty_angles = sum(abs(a - 60) < SHAPE_ANGLE_60_TOLERANCE for a in angles)

            if near_count >= 2 and sixty_angles >= 2:
                best_triangle = group

        if best_triangle:
            return ("triangle", best_triangle)
        
        if best_square:
            return ("square", best_square)

        return None

    def line_direction(self, model):
        a, b, c = model
        d = np.array([b, -a], float)
        d /= (np.linalg.norm(d) + 1e-9)
        return d

    def angle_between_lines(self, model1, model2):
        d1 = self.line_direction(model1)
        d2 = self.line_direction(model2)

        dot = np.clip(np.dot(d1, d2), -1.0, 1.0)
        ang = math.degrees(math.acos(dot))

        if ang > 90:
            ang = 180 - ang
        return ang
    
    def compute_line_endpoints(self, model, inlier_pts):
        a, b, c = model
    
        if abs(b) > 1e-8:  
            m = -a / b
            k = -c / b
        else:
            m = None
            k = -c / a
    
        d = np.array([b, -a], dtype=float)
        d = d / np.linalg.norm(d)
    
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
       
    def segment_length(self, seg):
        return np.linalg.norm(seg["p2"] - seg["p1"])

    def compute_shape_position(self, group):
        """Returns (cx, cy, dist)"""
        pts = []
        for L in group:
            pts.append(L["p1"])
            pts.append(L["p2"])

        pts = np.array(pts)
        cx, cy = np.mean(pts, axis=0)

        dist = math.sqrt((cx - self.current_x)**2 + (cy - self.current_y)**2)

        return cx, cy, dist
    
    # ---------------- cleanup ---------------- #
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
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        rclpy.shutdown()

if __name__ == "__main__":
    main()