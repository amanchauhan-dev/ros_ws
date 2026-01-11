#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
from std_msgs.msg import Bool, String
import time

# CONSTANTS
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
# Buffer parameters
BUFFER_LENGTH = 3
GLOBAL_BUFFER_DISTANCE_DIFFERENCE = 0.2

# Side filtering parameters
SIDE_DISTANCE_TOLERANCE = 1.5

# RANSAC parameters
RANSAC_ITERATIONS = 200
RANSAC_THRESHOLD = 0.01
RANSAC_MIN_INLIERS = 5
MAX_LINES = 6


# Shape detection parameters
SHAPE_SIDE_LENGTH = 0.25
SHAPE_LENGTH_TOLERANCE = 0.08
SHAPE_ANGLE_90_TOLERANCE = 15
SHAPE_ANGLE_60_TOLERANCE = 20

# Stop detection parameters
MIN_FREQUENCY_THRESHOLD = 5
PLANT_VALIDATION_WINDOW_Y = 0.20
PLANT_DETECTION_TOLERANCE_Y = 0.5
PLANT_DETECTION_TOLERANCE_X = 1.00

# MAD filtering parameters
MAD_MULTIPLIER = 2.5
MAD_EPSILON = 1e-6

# Noise filtering parameters
NEIGHBOR_DISTANCE_THRESHOLD = 0.15

# Misc constants
EPSILON = 1e-9
NORM_EPSILON = 1e-6

# LANE ENTERENCE
LANE_Y_MIN = -4.9
LANE_Y_MAX = 0.6

def check_in_lane(x, y):
    """Check if position (x, y) is within the lane boundaries."""
    return LANE_Y_MIN < y < LANE_Y_MAX


class Ransac(Node):
    def __init__(self):
        super().__init__("ransac_node")

        # subscribers
        self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
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

        # ransac
        self.right_lines = []
        self.left_lines = []

        self.stop_active = False
        self.stop_start_time = 0.0

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

        # timer
        self.create_timer(0.1, self.control_loop)

    def fertilizer_placement_status_callback(self, msg:Bool):
        if msg.data == True:
            self.at_dock = False

    # ---------------- callbacks ---------------- #
    def odom_callback(self, msg: Odometry):
        """Extract position and yaw from odometry (store as floats)."""
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        self.in_lane = check_in_lane(x, y)

        if not self.in_lane:
            self.right_last_broadcast_plant_id = None
            self.left_last_broadcast_plant_id = None
            return
        self.current_x = x
        self.current_y = y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = float(math.atan2(siny_cosp, cosy_cosp))

    def lidar_callback(self, msg: LaserScan):
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

        if dist >= GLOBAL_BUFFER_DISTANCE_DIFFERENCE:
            right_pts = pts[0:90]
            left_pts  = pts[270:360]

            self.right_buffer.append(self.get_side(right_pts))
            self.left_buffer.append(self.get_side(left_pts))

            self.last_point_buffered = (self.current_x, self.current_y)
                
        # self.global_buffer = self.global_buffer[-BUFFER_LENGTH:]
        self.right_buffer  = self.right_buffer[-BUFFER_LENGTH:]
        self.left_buffer   = self.left_buffer[-BUFFER_LENGTH:]

    # ---------------- helpers ---------------- #

    def get_side(self, pts):
    
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
        mask = dists < SIDE_DISTANCE_TOLERANCE
        pts = pts[mask]
        dists = dists[mask]
    
        if len(pts) == 0:
            return pts
    
        # -------------------------------
        # 3. Remove noisy outliers using median absolute deviation (MAD)
        # -------------------------------
        median_dist = np.median(dists)
        mad = np.median(np.abs(dists - median_dist)) + MAD_EPSILON  # prevent divide by zero
    
        # keep points close to trend
        stable_mask = np.abs(dists - median_dist) < MAD_MULTIPLIER * mad
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
            if left_d < NEIGHBOR_DISTANCE_THRESHOLD or right_d < NEIGHBOR_DISTANCE_THRESHOLD:
                clean_pts.append(pts[i])
    
        clean_pts.append(pts[-1])
    
        return np.array(clean_pts)

    # ---------------- main loop ---------------- #
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
            p1, p2, m, k = self.compute_line_endpoints(model, pts_line)
            eq = f"{m:.2f}x + {k:.2f}" if m else f"x = {k:.2f}"
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
                if abs(cy - py) <= PLANT_DETECTION_TOLERANCE_Y and abs(cx - px) <= PLANT_DETECTION_TOLERANCE_X and idx != self.right_last_broadcast_plant_id:
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
        if max_count >= MIN_FREQUENCY_THRESHOLD:  # minimum frequency threshold
                    plant_id = self.right_side_next_plant_index.index(max_count)

                    # Check if already used
                    if self.right_last_broadcast_plant_id != plant_id:
                    
                        # Check robot is near the plant (validate by Y only)
                        plant_x, plant_y = PLANTS_TRACKS[plant_id]
                        if abs(self.current_y - plant_y) <= PLANT_VALIDATION_WINDOW_Y:   # VALIDATION WINDOW
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
            p1, p2, m, k = self.compute_line_endpoints(model, pts_line)
            eq = f"{m:.2f}x + {k:.2f}" if m else f"x = {k:.2f}"
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
                if abs(cy - py) <= PLANT_DETECTION_TOLERANCE_Y and abs(cx - px) <= PLANT_DETECTION_TOLERANCE_X and idx != self.left_last_broadcast_plant_id:
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
        if max_count >= MIN_FREQUENCY_THRESHOLD:  # minimum frequency threshold
                    plant_id = self.left_side_next_plant_index.index(max_count)

                    # Check if already used
                    if self.left_last_broadcast_plant_id != plant_id:
                    
                        # Check robot is near the plant (validate by Y only)
                        plant_x, plant_y = PLANTS_TRACKS[plant_id]
                        if abs(self.current_y - plant_y) <= PLANT_VALIDATION_WINDOW_Y:   # VALIDATION WINDOW
                            # mark plant as handled
                            self.left_last_broadcast_plant_id = plant_id

                            self.current_plant_idx = plant_id
                            self.current_shape = self.left_side_next_shape.index(max(self.left_side_next_shape))

                            self.stop_active = True
                            self.stop_start_time = time.time()
                            # reset counters (avoid repeating counting)
                            self.left_side_next_plant_index = [0]*9
                            self.left_side_next_shape = [0,0]


    # Ransac
    def ransac_line(self, points):
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
            c = x2*y1 - y2*x1

            norm = np.hypot(a, b)
            if norm < NORM_EPSILON:
                continue

            dist = np.abs(a*pts[:,0] + b*pts[:,1] + c) / norm
            mask = dist < RANSAC_THRESHOLD
            inliers = np.sum(mask)

            if inliers > RANSAC_MIN_INLIERS and (best_inliers is None or inliers > np.sum(best_inliers)):
                best_inliers = mask
                best_model = (a, b, c)

        return best_model, best_inliers

    def extract_multiple_lines(self, pts):
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

    #  Shap detect
    def detect_shape(self, lines):

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
            near_len = [abs(L - SHAPE_SIDE_LENGTH) < SHAPE_LENGTH_TOLERANCE for L in lengths]
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
            right_angles = sum(abs(a - 90) < SHAPE_ANGLE_90_TOLERANCE for a in angles)

            # Need 2 sides ≈ side_length
            if right_angles == 2 and near_count >= 2:
                best_square = group

            # ============================================================
            # TRIANGLE LOGIC
            # ============================================================
            # Need 2 sides ≈ side_length
            # Need 2 angles ≈ 60°
            sixty_angles = sum(abs(a - 60) < SHAPE_ANGLE_60_TOLERANCE for a in angles)

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
        d /= (np.linalg.norm(d) + EPSILON)
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