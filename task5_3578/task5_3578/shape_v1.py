#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Side wall preview node

Logic:
- Check if robot is inside lane (based on X)
- Convert LiDAR scan to global frame
- Extract LEFT and RIGHT side points using angular sectors
- Filter only CURRENT frame points by distance (≤ 1.5 m)
- Accumulate points while robot stays in lane
- Reset buffers when robot exits lane
- Since walls are aligned with global X (yaw ≈ 0 or π),
  estimate wall as horizontal line using mean Y (no fitting)
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
# ============================================================
# ===================== CONSTANTS ============================
# ============================================================

# ---------- Lane limits (world frame) ----------
LANE_MIN_X = 0.8
LANE_MAX_X = 4.2

# ---------- Side sector definition (robot frame) ----------
LEFT_SECTOR_ANGLE   = math.pi / 2
RIGHT_SECTOR_ANGLE  = -math.pi / 2
SECTOR_HALF_WIDTH   = math.radians(40)

# ---------- Buffering control ----------
DISTANCE_STEP = 0.10          # store scan every 10 cm of motion

# ---------- Point filtering ----------
MIN_LIDAR_RANGE     = 0.05    # ignore invalid / too-close returns
MAX_POINT_DISTANCE  = 1.5     # keep only near points (current frame)
WALL_Y_CLEARANCE    = 0.02    # ignore wall too close to robot Y


# Plant and shape freq
SHAPE_PLANT_ID_MIN_FREQ = 4
ROBOT_PLANT_DIST_TOL = 0.15


# ============================================================
# ===================== HELPERS ==============================
# ============================================================

PLANTS_TRACKS = [
    [2.333, -1.877],   # 0
    [1.301, -0.8475],  # 1
    [2.0115, -0.8475], # 2
    [2.748, -0.8475],  # 3
    [3.573, -0.8475],  # 4
    
    [1.301,  0.856],   # 5
    [2.2115, 0.856],   # 6
    [3.048,  0.856],   # 7
    [4.0,  0.856],   # 8
]

DIST_TOL = 1.0

def get_region_id(x: float, y: float) -> int | None:
    """
    Return nearest PLANTS_TRACKS index (0–8) based on (x, y).

    Rules:
    - y < 0  → search IDs 0–4
    - y > 0  → search IDs 5–8
    """

    # decide valid ID range based on side
    if y < 0:
        valid_ids = range(0, 5)     # 0–4
    elif y > 0:
        valid_ids = range(5, 9)     # 5–8
    else:
        return None                 # exactly on centerline

    best_id = None
    best_dist = float("inf")

    for idx in valid_ids:
        px, py = PLANTS_TRACKS[idx]
        d = math.hypot(x - px, y - py)

        if d < best_dist:
            best_dist = d
            best_id = idx

    if best_dist <= DIST_TOL:
        return best_id

    return -1

def line_direction(model):
    """Unit direction vector of line ax+by+c=0"""
    a, b, _ = model
    d = np.array([b, -a], dtype=float)
    return d / (np.linalg.norm(d) + 1e-9)

def angle_to_horizontal(model):
    """
    Angle between line and horizontal wall (deg).
    Wall is assumed horizontal.
    """
    d = line_direction(model)
    ang = abs(math.degrees(math.atan2(d[1], d[0])))
    return min(ang, 180 - ang)

def detect_shape_from_lines(
    ransac_lines,
    angle_tol=15
):
    """
    Detect TRIANGLE or SQUARE using ONLY angle to horizontal wall.

    ransac_lines: [(model, inliers), ...]

    Returns:
        {
          "type": "triangle" | "square",
          "confidence": float,
          "lines": used_lines
        }
        or None
    """

    if not ransac_lines:
        return None

    # Compute angles for all lines
    angles = [
        {
            "model": model,
            "inliers": inliers,
            "angle": angle_to_horizontal(model)
        }
        for model, inliers in ransac_lines
    ]

    # =====================================================
    # TRIANGLE: ANY line near 60° (or 120°)
    # =====================================================
    for a in angles:
        if abs(a["angle"] - 60) < angle_tol or abs(a["angle"] - 120) < angle_tol:
            return {
                "type": 0,
                "confidence": max(0.0, 1.0 - abs(a["angle"] - 60) / angle_tol),
                "lines": [a]
            }

    # =====================================================
    # SQUARE: AT LEAST ONE strong ~90° line
    # (since triangle already excluded)
    # =====================================================
    verticals = [
        a for a in angles
        if abs(a["angle"] - 90) < angle_tol
    ]

    if verticals:
        return {
            "type": 1,
            "confidence": 1.0,
            "lines": verticals
        }

    return None


#  RANSAC

def ransac_line_fit(
    points,
    iterations=200,
    distance_thresh=0.02,
    min_inliers=5
):
    """
    Fit a single line using RANSAC.
    Line model: ax + by + c = 0

    Returns:
        model: (a, b, c) or None
        inliers: Nx2 array of inlier points or None
    """
    if points is None or len(points) < min_inliers:
        return None, None

    pts = np.asarray(points)
    best_model = None
    best_inliers = None
    best_count = 0

    for _ in range(iterations):
        i1, i2 = np.random.choice(len(pts), 2, replace=False)
        p1, p2 = pts[i1], pts[i2]

        # line from two points
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c = p2[0]*p1[1] - p2[1]*p1[0]

        norm = math.hypot(a, b)
        if norm < 1e-6:
            continue

        # distance of all points to line
        dist = np.abs(a*pts[:,0] + b*pts[:,1] + c) / norm
        mask = dist < distance_thresh
        count = np.sum(mask)

        if count > best_count:
            best_count = count
            best_model = (a, b, c)
            best_inliers = pts[mask]

    if best_count >= min_inliers:
        return best_model, best_inliers

    return None, None

def filter_points_between_robot_and_wall(
    points,
    robot_y,
    wall_y,
    wall_clearance=0.06
):
    """
    Keep points that are:
    - strictly between robot_y and wall_y
    - at least `wall_clearance` away from the wall
    """
    if points is None or len(points) == 0:
        return []

    pts = np.asarray(points)

    y_low  = min(robot_y, wall_y)
    y_high = max(robot_y, wall_y)

    # shrink interval near the wall
    if wall_y > robot_y:
        # wall is above robot
        valid_min = y_low
        valid_max = y_high - wall_clearance
    else:
        # wall is below robot
        valid_min = y_low + wall_clearance
        valid_max = y_high

    mask = (pts[:, 1] > valid_min) & (pts[:, 1] < valid_max)
    return pts[mask].tolist()


def is_inside_lane(x: float) -> bool:
    """Return True if robot X is inside lane bounds."""
    return LANE_MIN_X < x < LANE_MAX_X

def quaternion_to_yaw(q) -> float:
    """Convert quaternion to yaw angle (rad)."""
    return math.atan2(
        2 * (q.w * q.z + q.x * q.y),
        1 - 2 * (q.y * q.y + q.z * q.z)
    )


def normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def filter_points_by_distance(points, rx, ry, max_dist):
    """
    Remove points farther than max_dist from robot.
    Applied ONLY to current frame.
    """
    if not points:
        return []

    pts = np.asarray(points)
    dist_sq = (pts[:, 0] - rx)**2 + (pts[:, 1] - ry)**2
    return pts[dist_sq <= max_dist**2].tolist()

# ============================================================
# ===================== NODE ================================
# ============================================================

class SideWallPreview(Node):
    """
    ROS2 node that previews left and right lane walls using LiDAR.
    """

    def __init__(self):
        super().__init__("side_wall_preview")

        # Subscribers
        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.stop_pub = self.create_publisher(Bool, "/to_stop", 10)
        self.detection_pub = self.create_publisher(String, "/detection_status", 10)

        self.stop_active = False
        self.stop_end_time = None

        # Robot state
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        self.last_buffer_x = None
        self.last_buffer_y = None

        self.in_lane = False

        # Accumulated buffers (not trimmed)
        self.left_points_buffer  = []
        self.right_points_buffer = []
        self.cur_left_buffer = []
        self.cur_right_buffer = []

        self.left_shape = [[0,0], [0]*9]

        self.right_shape = [[0,0], [0]*9]

        self.last_id = -1
        self.last_shape = -1

        self.detected_hash = {
            "10":0,
            "11":0,
            "20":0,
            "21":0,
            "30":0,
            "31":0,
            "40":0,
            "41":0,
            "50":0,
            "51":0,
            "60":0,
            "61":0,
            "70":0,
            "71":0,
            "80":0,
            "81":0,
        }

        self.create_timer(0.1, self.plot_loop)

    # ------------------------------------------------
    # Odometry callback
    # ------------------------------------------------

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_yaw = quaternion_to_yaw(msg.pose.pose.orientation)

        if not is_inside_lane(self.robot_x):
            # Reset everything when leaving lane
            self.in_lane = False
            self.left_points_buffer.clear()
            self.right_points_buffer.clear()
            self.cur_left_buffer = []
            self.cur_right_buffer = []
            self.last_buffer_x = None
            self.last_buffer_y = None

            self.last_id = -1
            self.last_shape = -1
        else:
            self.in_lane = True

    # ------------------------------------------------
    # LiDAR callback
    # ------------------------------------------------

    def scan_callback(self, msg: LaserScan):
        if not self.in_lane or self.robot_x is None:
            return

        # Distance-based buffering
        if self.last_buffer_x is not None:
            moved = math.hypot(
                self.robot_x - self.last_buffer_x,
                self.robot_y - self.last_buffer_y
            )
            if moved < DISTANCE_STEP:
                return

        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        valid = np.isfinite(ranges) & (ranges > MIN_LIDAR_RANGE)
        ranges = ranges[valid]
        angles = angles[valid]

        # Local → global transform
        x_local = ranges * np.cos(angles)
        y_local = ranges * np.sin(angles)

        cy, sy = math.cos(self.robot_yaw), math.sin(self.robot_yaw)
        x_global = self.robot_x + x_local * cy - y_local * sy
        y_global = self.robot_y + x_local * sy + y_local * cy

        points = np.column_stack((x_global, y_global))

        left_frame  = []
        right_frame = []

        for px, py in points:
            angle = normalize_angle(
                math.atan2(py - self.robot_y, px - self.robot_x) - self.robot_yaw
            )

            if abs(angle - LEFT_SECTOR_ANGLE) <= SECTOR_HALF_WIDTH:
                left_frame.append([px, py])
            elif abs(angle - RIGHT_SECTOR_ANGLE) <= SECTOR_HALF_WIDTH:
                right_frame.append([px, py])

        # Filter ONLY current frame
        left_frame  = filter_points_by_distance(
            left_frame, self.robot_x, self.robot_y, MAX_POINT_DISTANCE
        )
        right_frame = filter_points_by_distance(
            right_frame, self.robot_x, self.robot_y, MAX_POINT_DISTANCE
        )
        self.cur_left_buffer = left_frame
        self.cur_right_buffer = right_frame
        # Accumulate
        self.left_points_buffer.extend(left_frame)
        self.right_points_buffer.extend(right_frame)

        self.last_buffer_x = self.robot_x
        self.last_buffer_y = self.robot_y

    # ------------------------------------------------
    #  helpers
    # ------------------------------------------------


    def get_wall_mean_y(self, points):
        if points is None or len(points) < 5:
            return None
        return float(np.mean(np.asarray(points)[:, 1]))
    

    def trigger_stop(self, id, shape):
        if self.stop_active:
            return
  
        self.last_id = id
        self.last_shape = shape

        msg = Bool()
        msg.data = True
        self.stop_pub.publish(msg)

        self.stop_active = True
        self.stop_end_time = self.get_clock().now().nanoseconds + int(2e9)
        
        self.get_logger().info("STOP triggered for 2 seconds")


    def check_and_trigger_stop_by_x(self):
        if self.stop_active or self.robot_x is None:
            return
    
        # ---------------- LEFT SIDE ----------------
        left_counts = np.array(self.left_shape[1])
    
        left_pid = int(np.argmax(left_counts))
        left_freq = left_counts[left_pid]

        if left_freq > SHAPE_PLANT_ID_MIN_FREQ:
            plant_x = PLANTS_TRACKS[left_pid][0]
            if abs(self.robot_x - plant_x) < ROBOT_PLANT_DIST_TOL:
                if self.left_shape[0][0] > self.left_shape[0][1]:
                    shape = 0
                else:
                    shape = 1 
                self.trigger_stop(left_pid, shape = shape)
                return
    
        # ---------------- RIGHT SIDE ----------------
        right_counts = np.array(self.right_shape[1])
    
        right_pid = int(np.argmax(right_counts))
        right_freq = right_counts[right_pid]
    
        if right_freq > SHAPE_PLANT_ID_MIN_FREQ:
            plant_x = PLANTS_TRACKS[right_pid][0]
            if abs(self.robot_x - plant_x) < ROBOT_PLANT_DIST_TOL:
                if self.right_shape[0][0] > self.right_shape[0][1]:
                    shape = 0
                else:
                    shape = 1 
                self.trigger_stop(right_pid,  shape=shape)
                return
    

    # ------------------------------------------------
    #  loop
    # ------------------------------------------------
    def publish_detection_status(self):
        msg = String()

        # decide label
        if self.last_id == 0:
            label = "DOCK_STATION"
        elif self.last_shape == 0:
            label = "FERTILIZER_REQUIRED"
        else:
            label = "BAD_HEALTH"

        if self.last_id !=0:
            self.detected_hash[f"{self.last_id}{self.last_shape}"] = 1

        msg.data = (
            f"{label},"
            f"{self.robot_x:.2f},"
            f"{self.robot_y:.2f},"
            f"{self.last_id}"
        )
        self.detection_pub.publish(msg)
        self.get_logger().info(msg.data)

    
    def plot_loop(self):
        # ---------- STOP release logic ----------
        if self.stop_active:
            now_ns = self.get_clock().now().nanoseconds
            # publish
            
            if now_ns >= self.stop_end_time:
                self.publish_detection_status()
                msg = Bool()
                msg.data = False
                self.stop_pub.publish(msg)

                self.stop_active = False
                self.stop_end_time = None

                # reset shape buffers AFTER stop
                self.left_shape = [[0, 0], [0]*9]
                self.right_shape = [[0, 0], [0]*9]

                self.get_logger().info("STOP released, buffers reset")
            return   # do NOTHING else while stopped
        self.check_and_trigger_stop_by_x()

        # ransac
        left_pts  = np.asarray(self.left_points_buffer)
        right_pts = np.asarray(self.right_points_buffer)

        cur_left = np.asanyarray(self.cur_left_buffer) 
        cur_right = np.asanyarray(self.cur_right_buffer) 

        # ---------- compute wall positions ----------
        left_wall_y  = self.get_wall_mean_y(left_pts)
        right_wall_y = self.get_wall_mean_y(right_pts)

        # ---------- filter CURRENT frame ----------
        if left_wall_y is not None:
            cur_left = np.asarray(
                filter_points_between_robot_and_wall(
                    cur_left, self.robot_y, left_wall_y
)
            )

        if right_wall_y is not None:
            cur_right = np.asarray(
                filter_points_between_robot_and_wall(
                    cur_right, self.robot_y, right_wall_y
                )
            )

        # Left side shape
        ransac_lines_L = []
        if len(cur_left) > 10:
            model_L, inliers_L = ransac_line_fit(cur_left)

            if model_L is not None and len(inliers_L) > 10:
                ransac_lines_L.append((model_L, inliers_L))

        shape_L = detect_shape_from_lines(ransac_lines_L)

        if shape_L:
            shape_type = shape_L["type"]
            used_lines = shape_L["lines"]
            shape_points = np.vstack([l["inliers"] for l in used_lines])
            cx, cy = shape_points.mean(axis=0)
            id = get_region_id(cx, cy)
            if id != -1 and self.last_id != id:
                if id == 0 or self.detected_hash[f"{id}{shape_type}"] == 0:
                    self.left_shape[0][shape_type] += 1
                    self.left_shape[1][id] += 1

                    self.get_logger().info(
                        f"LEFT SHAPE: {self.left_shape[0]} | {self.left_shape[1]}"
                    )            

        # right side shape
        ransac_lines_R = []
        if len(cur_right) > 10:
            model_R, inliers_R = ransac_line_fit(cur_right)

            if model_R is not None and len(inliers_R) > 10:
                ransac_lines_R.append((model_R, inliers_R))

        shape_R = detect_shape_from_lines(ransac_lines_R)

        if shape_R:
            shape_type = shape_R["type"]
            used_lines = shape_R["lines"]
            shape_points = np.vstack([l["inliers"] for l in used_lines])
            cx, cy = shape_points.mean(axis=0)
            id = get_region_id(cx, cy)
            if id !=-1 and self.last_id != id:
                if id == 0 or self.detected_hash[f"{id}{shape_type}"] == 0:
                    self.right_shape[0][shape_type] += 1
                    self.right_shape[1][id] += 1

                    self.get_logger().info(
                        f"RIGHT SHAPE: {self.right_shape[0]} | {self.right_shape[1]}"
                    )

# ============================================================

def main():
    rclpy.init()
    node = SideWallPreview()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()