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


class Ransac(Node):
    def __init__(self):
        super().__init__("ransac_node")

        # subscribers
        self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(Bool, "/in_lane", self.in_lane_callback, 10)
        self.create_subscription(Bool, "/fertilizer_placement_status",self.fertilizer_placement_status_callback, 10)   # arm will publish true when done

        self.detection_pub = self.create_publisher(String, "/detection_status", 10)
        self.stop_pub = self.create_publisher(Bool, "/to_stop", 10)
        self.at_dock_pub = self.create_publisher(Bool, "/ebot_dock_status", 10)   # it will publish true when reached at dock

        # memory / buffers
        self.in_lane = False
        self.right_buffer = []
        self.current_buffer = np.empty((0, 2))
        self.left_buffer = []
        self.global_points = []  # store recent global scans (list of Nx2 arrays)

        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self.last_point_buffered = None  # last position (x,y) where we stored a frame

        self.right_lines = []  # RIGHT lines
        self.left_lines = []  # LEFT lines

        # parameters (kept same as you set)
        self.buffer_length = 3
        self.global_buffer_distance_difference = 0.2 
        self.side_distance_tolerance = 1.5  
        self.side_length = 0.25  

        # ransac
        self.right_lines = []
        self.left_lines = []
        self.iterations=200
        self.threshold=0.01
        self.min_inliers=5
        self.max_lines = 6

        self.stop_active = False
        self.stop_start_time = 0.0

        #  shape detection
        self.side_length=0.25, 
        self.len_tol=0.08 
        self.ang90_tol=15
        self.ang60_tol=20

        self.plants_tracks  = [
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

        # shape
        self.current_shape = None
        self.current_plant_idx = None
        self.at_dock = False

        # Right side shape
        self.right_side_next_shape = [0, 0]
        self.right_side_next_plant_index = [0]*9
        self.right_last_broadcast_plant_id = None

        # left side shape
        self.left_side_next_shape = [0, 0]
        self.left_side_next_plant_index = [0]*9
        self.left_last_broadcast_plant_id = None

        # plotting
        # plt.ion()
        self.colors = ["red", "blue", "green", "orange", "purple", "cyan"]
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_title("RANSAC", fontsize=14, fontweight="bold")
        self.ax.set_xlabel("X (m)", fontsize=11)
        self.ax.set_ylabel("Y (m)", fontsize=11)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.grid(True, alpha=0.3)

        # timer
        self.create_timer(0.1, self.control_loop)

    def fertilizer_placement_status_callback(self, msg:Bool):
        if msg.data == True:
            self.at_dock = False

    def in_lane_callback(self, msg:Bool):
        self.in_lane = msg.data
        if not msg.data:
            self.right_last_broadcast_plant_id = None
            self.left_last_broadcast_plant_id = None
        # self.get_logger().info(f"Received flag = {self.in_lane}")

    # ---------------- callbacks ---------------- #
    def odom_callback(self, msg: Odometry):
        """Extract position and yaw from odometry (store as floats)."""
        if not self.in_lane:
            return
        
        self.current_x = float(msg.pose.pose.position.x)
        self.current_y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = float(math.atan2(siny_cosp, cosy_cosp))

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
        # require pose to be known (use is None checks so zeros are allowed)
        if self.current_x is None or self.current_y is None or self.current_yaw is None:
            return

        distances = np.nan_to_num(np.array(msg.ranges, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if distances.size == 0 or np.all(distances == 0.0):
            # nothing useful in scan
            return

        # build local coordinates using scan angle info
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

        # decide if we should append to global/side buffers based on movement
        if self.last_point_buffered is None:
            # first frame -> store
            right_pts = pts[0:90]
            left_pts  = pts[270:360]

            self.right_buffer.append(self.get_side(right_pts))
            self.left_buffer.append(self.get_side(left_pts))
            self.last_point_buffered = (self.current_x, self.current_y)
            return

        # distance moved since last buffer store
        x1, y1 = self.last_point_buffered
        dist = math.hypot(self.current_x - x1, self.current_y - y1)

        if dist >= self.global_buffer_distance_difference:
            right_pts = pts[0:90]
            left_pts  = pts[270:360]

            self.right_buffer.append(self.get_side(right_pts))
            self.left_buffer.append(self.get_side(left_pts))

            self.last_point_buffered = (self.current_x, self.current_y)
                
        # self.global_buffer = self.global_buffer[-self.buffer_length:]
        self.right_buffer  = self.right_buffer[-self.buffer_length:]
        self.left_buffer   = self.left_buffer[-self.buffer_length:]

    # ---------------- helpers ---------------- #

    def get_side(self, pts):
        '''
        Purpose:
        ---
        Filters raw LiDAR points for a specific side. Removes points that are 
        too far, statistical outliers (MAD), and isolated noise.

        Input Arguments:
        ---
        `pts` :  [ numpy array ]
            Array of (x, y) coordinates from LiDAR.

        Returns:
        ---
        `clean_pts` :  [ numpy array ]
            Filtered array of (x, y) coordinates.

        Example call:
        ---
        cleaned_right = self.get_side(right_pts)
        '''
        robot_x, robot_y = self.current_x, self.current_y
        pts = np.array(pts)
        if len(pts) == 0:
            return pts
    
        # -------------------------------
        # 1. Compute distance of each LiDAR point from robot
        # -------------------------------
        dists = np.linalg.norm(pts - np.array([robot_x, robot_y]), axis=1)
    
        # -------------------------------
        # 2. Remove far points (> tolerance)
        # -------------------------------
        mask = dists < self.side_distance_tolerance
        pts = pts[mask]
        dists = dists[mask]
    
        if len(pts) == 0:
            return pts
    
        # -------------------------------
        # 3. Remove noisy outliers using median absolute deviation (MAD)
        # -------------------------------
        median_dist = np.median(dists)
        mad = np.median(np.abs(dists - median_dist)) + 1e-6  # prevent divide by zero
    
        # keep points close to trend
        stable_mask = np.abs(dists - median_dist) < 2.5 * mad
        pts = pts[stable_mask]
        dists = dists[stable_mask]
    
        if len(pts) == 0:
            return pts
    
        # -------------------------------
        # 4. Remove isolated single-point noise
        #    (if neighborhood difference is too large)
        # -------------------------------
        clean_pts = [pts[0]]
        for i in range(1, len(pts)-1):
            left_d  = np.linalg.norm(pts[i] - pts[i-1])
            right_d = np.linalg.norm(pts[i] - pts[i+1])
    
            # keep if close to neighbors
            if left_d < 0.15 or right_d < 0.15:
                clean_pts.append(pts[i])
    
        clean_pts.append(pts[-1])
    
        return np.array(clean_pts)

    # ---------------- main loop / plotting ---------------- #
    def control_loop(self):
        # stop at dock station ( <=== If want to keep stop at dock until arm sends signal uncomment this code ===>)
        if self.at_dock:
            self.at_dock_pub.publish(Bool(data=True))
            return
        else:
            self.at_dock_pub.publish(Bool(data=False))
        # (<=== end dock keep stop code ==> )

         # If STOP mode active → check if 2 seconds passed
        if self.stop_active:
            elapsed = time.time() - self.stop_start_time

            if elapsed >= 2.0:
                self.stop_active = False
                self.stop_pub.publish(Bool(data=False))
                self.get_logger().info("==== STOP FALSE (timeout) ====")
            else:
                # Keep STOP = TRUE
                detection_msg = String()
                if self.current_plant_idx ==0:
                    # docstation
                    detection_msg.data = f"DOCK_STATION,{self.current_x:.2f},{self.current_y:.2f},{self.current_plant_idx}"
                    # self.at_dock = True   # <========== Note: uncomment if no need to keep stop at dock 
                else:
                    # plants
                    s = "FERTILIZER_REQUIRED" if self.current_shape == 0 else "BAD_HEALTH"
                    detection_msg.data = f"{s},{self.current_x:.2f},{self.current_y:.2f},{self.current_plant_idx}"

                self.detection_pub.publish(detection_msg)
                self.get_logger().info("==== publishing ====")
                self.stop_pub.publish(Bool(data=True))
            return 
        else:
            self.stop_pub.publish(Bool(data=False))
            self.at_dock = False
        
        if not self.in_lane:
            return
            
        #  ============> RIGHT SIDE DETECTION <===========
        self.right_lines = self.extract_multiple_lines(self.right_buffer)

        line_dicts = []
        for i, (model, pts_line) in enumerate(self.right_lines):
            color = self.colors[i % len(self.colors)]
            p1, p2, m, k = self.compute_line_endpoints(model, pts_line)
            eq = f"{m:.2f}x + {k:.2f}" if m else f"x = {k:.2f}"
            line_dicts.append({
                "model": model,
                "p1": p1,
                "p2": p2,
                "slope": m,
                "intercept": k
            })
            self.ax.plot([p1[0],p2[0]], [p1[1], p2[1]],
                 color=color, linewidth=2, label=f"R{i+1}: {eq}")

        shape = self.detect_shape(line_dicts)
        if shape:
            shape_type, group = shape
            cx, cy, dist = self.compute_shape_position(group)
            is_valid = False
            self.get_logger().info(f"current status: {self.right_side_next_shape} {self.right_side_next_plant_index}")
            for idx, (px, py) in enumerate(self.plants_tracks):
                if abs(cy - py) <= 0.5 and abs(cx - px) <= 1.00 and idx != self.right_last_broadcast_plant_id:
                    is_valid = True
                    self.right_side_next_plant_index[idx] +=1
                    break 
            
            if is_valid:
                if shape_type == "triangle":
                    self.right_side_next_shape[0] += 1
                elif shape_type == "square":
                    self.right_side_next_shape[1] += 1
       
        # stop logic    
        max_count = max(self.right_side_next_plant_index)
        if max_count >= 5:  # minimum frequency threshold
                    plant_id = self.right_side_next_plant_index.index(max_count)

                    # Check if already used
                    if self.right_last_broadcast_plant_id != plant_id:
                    
                        # Check robot is near the plant (validate by Y only)
                        plant_x, plant_y = self.plants_tracks[plant_id]
                        if abs(self.current_y - plant_y) <= 0.20:   # VALIDATION WINDOW
                            # mark plant as handled
                            self.right_last_broadcast_plant_id = plant_id

                            self.current_plant_idx = plant_id
                            self.current_shape = self.right_side_next_shape.index(max(self.right_side_next_shape))

                            self.stop_active = True
                            self.stop_start_time = time.time()
                            # reset counters (avoid repeating counting)
                            self.right_side_next_plant_index = [0]*9
                            self.right_side_next_shape = [0,0]

        #  ============> LEFT SIDE DETECTION <===========
        self.left_lines = self.extract_multiple_lines(self.left_buffer)

        line_dicts = []
        for i, (model, pts_line) in enumerate(self.left_lines):
            color = self.colors[i % len(self.colors)]
            p1, p2, m, k = self.compute_line_endpoints(model, pts_line)
            eq = f"{m:.2f}x + {k:.2f}" if m else f"x = {k:.2f}"
            line_dicts.append({
                "model": model,
                "p1": p1,
                "p2": p2,
                "slope": m,
                "intercept": k
            })
            self.ax.plot([p1[0],p2[0]], [p1[1], p2[1]],
                 color=color, linewidth=2, label=f"R{i+1}: {eq}")

        shape = self.detect_shape(line_dicts)
        if shape:
            shape_type, group = shape
            cx, cy, dist = self.compute_shape_position(group)
            is_valid = False
            self.get_logger().info(f"current status: {self.left_side_next_shape} {self.left_side_next_plant_index}")
            for idx, (px, py) in enumerate(self.plants_tracks):
                if abs(cy - py) <= 0.5 and abs(cx - px) <= 1.00 and idx != self.left_last_broadcast_plant_id:
                    is_valid = True
                    self.left_side_next_plant_index[idx] +=1
                    break 
            
            if is_valid:
                if shape_type == "triangle":
                    self.left_side_next_shape[0] += 1
                elif shape_type == "square":
                    self.left_side_next_shape[1] += 1
       
        # stop logic    
        max_count = max(self.left_side_next_plant_index)
        if max_count >= 5:  # minimum frequency threshold
                    plant_id = self.left_side_next_plant_index.index(max_count)

                    # Check if already used
                    if self.left_last_broadcast_plant_id != plant_id:
                    
                        # Check robot is near the plant (validate by Y only)
                        plant_x, plant_y = self.plants_tracks[plant_id]
                        if abs(self.current_y - plant_y) <= 0.20:   # VALIDATION WINDOW
                            # mark plant as handled
                            self.left_last_broadcast_plant_id = plant_id

                            self.current_plant_idx = plant_id
                            self.current_shape = self.left_side_next_shape.index(max(self.left_side_next_shape))

                            self.stop_active = True
                            self.stop_start_time = time.time()
                            # reset counters (avoid repeating counting)
                            self.left_side_next_plant_index = [0]*9
                            self.left_side_next_shape = [0,0]

        # self.graph()

    def graph(self):
        self.ax.clear()
        self.ax.set_title("RANSAC", fontsize=14, fontweight="bold")
        self.ax.set_xlabel("X (m)", fontsize=11)
        self.ax.set_ylabel("Y (m)", fontsize=11)
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-6, 6)
        self.ax.grid(True, alpha=0.3)

        # current buffer 
        if len(self.current_buffer) > 0:
            pts  = np.vstack(self.current_buffer)
            self.ax.scatter(pts[:, 0], pts[:, 1], s=8)

        # right buffer 
        if  len(self.right_buffer) > 0:
            rpts  = np.vstack(self.right_buffer)
            self.ax.scatter(rpts[:, 0], rpts[:, 1], s=8)

        # left buffer 
        if  len(self.left_buffer) > 0:
            lpts  = np.vstack(self.left_buffer)
            self.ax.scatter(lpts[:, 0], lpts[:, 1], s=8)

        # plot robot if known
        if self.current_x is not None and self.current_y is not None and self.current_yaw is not None:
            self.ax.plot(self.current_x, self.current_y, "bo", markersize=6)
            self.ax.arrow(
                self.current_x,
                self.current_y,
                0.5 * math.cos(self.current_yaw),
                0.5 * math.sin(self.current_yaw),
                head_width=0.05,
                color="blue",
            )

        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # Ransac
    def ransac_line(self, points):
        '''
        Purpose:
        ---
        Performs a single RANSAC iteration to find the best fitting line 
        model (ax + by + c = 0) for a set of points.

        Input Arguments:
        ---
        `points` :  [ numpy array ]
            Array of (x,y) points to fit a line to.

        Returns:
        ---
        `best_model` :  [ tuple ]
            Coefficients (a, b, c) of the best line.
        
        `best_inliers` :  [ boolean mask ]
            Mask indicating which points belong to the best line.

        Example call:
        ---
        model, inliers = self.ransac_line(pts)
        '''
        pts = np.array(points)
        best_inliers = None
        best_model = None
        n = len(pts)

        if n < self.min_inliers:
            return None, None

        for _ in range(self.iterations):
            idx = np.random.choice(n, 2, replace=False)
            p1, p2 = pts[idx]

            x1, y1 = p1
            x2, y2 = p2

            a = y2 - y1
            b = -(x2 - x1)
            c = x2*y1 - y2*x1

            norm = np.hypot(a, b)
            if norm < 1e-6:
                continue

            dist = np.abs(a*pts[:,0] + b*pts[:,1] + c) / norm
            mask = dist < self.threshold
            inliers = np.sum(mask)

            if inliers > self.min_inliers and (best_inliers is None or inliers > np.sum(best_inliers)):
                best_inliers = mask
                best_model = (a, b, c)

        return best_model, best_inliers

    def extract_multiple_lines(self, pts):
        '''
        Purpose:
        ---
        Iteratively extracts multiple lines from a point cloud using RANSAC. 
        Removes inliers of found lines and repeats the process.

        Input Arguments:
        ---
        `pts` :  [ list or array ]
            The input point cloud.

        Returns:
        ---
        `lines` :  [ list ]
            List of tuples, where each tuple contains (model, line_points).

        Example call:
        ---
        lines = self.extract_multiple_lines(buffer)
        '''
        if not pts or len(pts) == 0:
            return []
        pts = np.array(np.vstack(pts))
        lines = []
    
        while len(pts) >= self.min_inliers:
            model, inliers = self.ransac_line(pts)
            if model is None:
                break
            
            line_pts = pts[inliers]
            lines.append((model, line_pts))
    
            pts = pts[~inliers]
            if len(lines) >= self.max_lines:
                break
            
        return lines

    #  Shap detect
    def detect_shape(self, lines):
        '''
        Purpose:
        ---
        Analyzes a set of lines to detect geometric shapes (Triangle or Square) 
        based on side lengths and angles between lines.

        Input Arguments:
        ---
        `lines` :  [ list ]
            List of dictionaries containing line models and properties.

        Returns:
        ---
        `result` :  [ tuple or None ]
            ("triangle", group) or ("square", group) if detected, else None.

        Example call:
        ---
        shape = self.detect_shape(extracted_lines)
        '''

        if len(lines) < 3:
            return None

        # Precompute lengths and angles
        enriched = []
        for L in lines:
            Llen = self.segment_length(L)
            enriched.append({
                **L,
                "length": Llen
            })

        best_square = None
        best_triangle = None

        # test all combinations of 3 lines
        from itertools import combinations

        for group in combinations(enriched, 3):
            L1, L2, L3 = group

            # lengths
            len1, len2, len3 = L1["length"], L2["length"], L3["length"]
            lengths = [len1, len2, len3]

            # check near side_length
            near_len = [abs(L - self.side_length) < self.len_tol for L in lengths]
            near_count = sum(near_len)

            # angles between them
            ang12 = self.angle_between_lines(L1["model"], L2["model"])
            ang23 = self.angle_between_lines(L2["model"], L3["model"])
            ang31 = self.angle_between_lines(L3["model"], L1["model"])

            angles = [ang12, ang23, ang31]

            # ============================================================
            # SQUARE LOGIC (3 sides visible)
            # ============================================================
            # Need 2 perpendicular pairs
            right_angles = sum(abs(a - 90) < self.ang90_tol for a in angles)

            # Need 2 sides ≈ side_length
            if right_angles == 2 and near_count >= 2:
                best_square = group

            # ============================================================
            # TRIANGLE LOGIC
            # ============================================================
            # Need 2 sides ≈ side_length
            # Need 2 angles ≈ 60°
            sixty_angles = sum(abs(a - 60) < self.ang60_tol for a in angles)

            if near_count >= 2 and sixty_angles >= 2:
                best_triangle = group

        if best_triangle:
            return ("triangle", best_triangle)
        
        if best_square:
            return ("square", best_square)


        return None

    def line_direction(mself, model):
        a, b, c = model
        # direction vector (b, -a)
        d = np.array([b, -a], float)
        d /= (np.linalg.norm(d) + 1e-9)
        return d

    def angle_between_lines(self, model1, model2):
        d1 = self.line_direction(model1)
        d2 = self.line_direction(model2)

        dot = np.clip(np.dot(d1, d2), -1.0, 1.0)
        ang = math.degrees(math.acos(dot))

        # smallest angle
        if ang > 90:
            ang = 180 - ang
        return ang
    
    def compute_line_endpoints(self, model, inlier_pts):
       a, b, c = model
    
       # --------------------------------------
       # Compute slope + intercept (if possible)
       # --------------------------------------
       if abs(b) > 1e-8:  
           m = -a / b
           k = -c / b
       else:
           m = None          # vertical line → infinite slope
           k = -c / a        # x = k
                             # (store intercept as the x-constant)
    
       # --------------------------------------
       # Direction vector of the line
       # (b, -a) is perpendicular to normal (a, b)
       # --------------------------------------
       d = np.array([b, -a], dtype=float)
       d = d / np.linalg.norm(d)  # normalize to unit direction
    
       # --------------------------------------
       # Point on the line closest to origin
       # --------------------------------------
       denom = a*a + b*b
       point0 = np.array([-a*c/denom, -b*c/denom])
    
       # --------------------------------------
       # Project all inlier points onto the line
       # --------------------------------------
       projections = []
       for p in inlier_pts:
           v = p - point0
           t = np.dot(v, d)             # scalar projection
           projections.append((t, p))
    
       # Extract endpoints using min and max projection
       t_vals = [tp[0] for tp in projections]
       p_min = projections[np.argmin(t_vals)][1]
       p_max = projections[np.argmax(t_vals)][1]
    
       # Return endpoints + slope + intercept/constant
       return p_min, p_max, m, k
       
    def segment_length(sself, seg):
        return np.linalg.norm(seg["p2"] - seg["p1"])

    def compute_shape_position(self, group):
        """
        group: tuple of 3 lines -> each element is dict with keys:
               p1, p2, model, slope, intercept
        Returns (cx, cy, dist)
        """
        pts = []
        for L in group:
            pts.append(L["p1"])
            pts.append(L["p2"])

        pts = np.array(pts)
        cx, cy = np.mean(pts, axis=0)

        # distance from robot
        dist = math.sqrt( (cx - self.current_x)**2 + (cy - self.current_y)**2 )

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