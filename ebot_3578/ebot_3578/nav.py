#!/usr/bin/env python3
"""
Deterministic sequential line-by-line coverage with obstacle avoidance
"""

import math
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
# ============================================================
# CONFIG
# ============================================================

CMD_VEL_TOPIC = "/cmd_vel"
ODOM_TOPIC    = "/odom"
SCAN_TOPIC    = "/scan"
TO_SOP = "/to_stop"

CONTROL_DT = 0.1

LINES = [
    ((0.0, 0.0),    (0.0, -1.72)),       # 0
    ((0.0, -1.72),   (4.722, -1.72)),     # 1
    ((4.722, -1.72), (4.722, 0.0)),      # 2
    ((4.722, 0.0),  (0.0, 0.0)),        # 3
    ((0.0, 0.0),    (0.0, 1.72)),        # 4
    ((0.0, 1.72),    (4.722, 1.72)),      # 5
    ((4.722, 1.72),  (4.722, 0.0)),      # 6
    ((4.722, 0.0),  (0.0, 0.0)),        # 7
]

START_LINE_INDEX = 0

SPEED_FOLLOW   = 0.8
SPEED_APPROACH = 0.6
MAX_OMEGA      = 1.5

GAIN_HEADING    = 1.5
GAIN_CROSSTRACK = 2.5
MAX_CROSSTRACK  = 0.4

OBSTACLE_DIST = 0.7
OBSTACLE_GAIN = 1.0

# Tolerance for reaching next line's start point
POS_TOL = 0.2
YAW_TOL = 0.1

# Threshold for detecting we're on the next line (reached it through current line)
LINE_TRANSITION_TOL = 0.3

FRONT_ANGLE = 0.6
SIDE_ANGLE  = 0.4

LOG_THROTTLE = 1.0


# ============================================================
# HELPERS
# ============================================================

def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def distance_to_point(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def distance_to_line_segment(px, py, x1, y1, x2, y2):
    """Calculate perpendicular distance from point to line segment"""
    vx, vy = x2 - x1, y2 - y1
    L2 = vx*vx + vy*vy
    
    if L2 == 0:
        return distance_to_point(px, py, x1, y1)
    
    wx, wy = px - x1, py - y1
    t = clamp((wx*vx + wy*vy) / L2, 0.0, 1.0)
    
    qx = x1 + t * vx
    qy = y1 + t * vy
    
    return distance_to_point(px, py, qx, qy)


# ============================================================
# NODE
# ============================================================

class LineFollower(Node):

    MOVE_TO_START  = 0
    ROTATE_TO_LINE = 1
    FOLLOW_LINE    = 2
    TRANSITION_STOP = 3
    DONE           = 4

    def __init__(self):
        super().__init__("line_follower")

        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_cb, 10)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.scan_cb, 10)
        self.create_subscription(Bool, TO_SOP, self.to_stop_cb, 10)

        self.to_stop = False

        self.x = -1.5
        self.y = -5.6
        self.yaw = 0.0
        self.lidar_pts = []

        self.lines = LINES
        self.line_idx = START_LINE_INDEX
        self.total_lines = len(self.lines)

        # Always use line endpoints in order (start -> end)
        (x1, y1), (x2, y2) = self.lines[self.line_idx]
        self.line_start = (x1, y1)
        self.line_end = (x2, y2)
        self.target_point = self.line_start

        # Start with rotation, not movement (like line switching behavior)
        self.state = self.ROTATE_TO_LINE
        self.trajectory = []

        self.get_logger().info(
            f"[INIT] Line {self.line_idx}/{self.total_lines-1} | "
            f"start={self.line_start} end={self.line_end} | Starting with ROTATION"
        )

        self.create_timer(CONTROL_DT, self.control_loop)
        self.log_timer = 0

    # ========================================================
    def to_stop_cb(self, msg):
        self.to_stop = msg.data

    def control_loop(self):
        twist = Twist()

        if self.to_stop:
            self.cmd_pub.publish(twist)
            return

        self.trajectory.append((self.x, self.y))
        self.log_timer += CONTROL_DT

        # ---------------- MOVE TO START ----------------
        if self.state == self.MOVE_TO_START:
            tx, ty = self.target_point
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)

            if self.log_timer > LOG_THROTTLE:
                self.get_logger().info(
                    f"[MOVE_TO_START] Line {self.line_idx} | dist={dist:.2f}"
                )
                self.log_timer = 0

            if dist < POS_TOL:
                self.state = self.ROTATE_TO_LINE
                self.get_logger().info(
                    f"[→ ROTATE_TO_LINE] Reached start of line {self.line_idx}"
                )
                self.cmd_pub.publish(Twist())
                return

            yaw_goal = math.atan2(dy, dx)
            yaw_err = normalize_angle(yaw_goal - self.yaw)

            # Slow down as we approach
            speed_scale = min(1.0, dist / 0.5)
            twist.linear.x = SPEED_APPROACH * speed_scale
            twist.angular.z = clamp(2.0 * yaw_err, -MAX_OMEGA, MAX_OMEGA)
            self.cmd_pub.publish(twist)
            return

        # ---------------- ROTATE TO LINE ----------------
        if self.state == self.ROTATE_TO_LINE:
            sx, sy = self.line_start
            ex, ey = self.line_end
            line_yaw = math.atan2(ey - sy, ex - sx)
            yaw_err = normalize_angle(line_yaw - self.yaw)

            if self.log_timer > LOG_THROTTLE:
                self.get_logger().info(
                    f"[ROTATE_TO_LINE] Line {self.line_idx} | yaw_err={yaw_err:.2f}"
                )
                self.log_timer = 0

            if abs(yaw_err) < YAW_TOL:
                self.state = self.FOLLOW_LINE
                self.get_logger().info(
                    f"[→ FOLLOW_LINE] Aligned with line {self.line_idx}"
                )
                return

            twist.angular.z = clamp(2.0 * yaw_err, -MAX_OMEGA, MAX_OMEGA)
            self.cmd_pub.publish(twist)
            return

        # ---------------- FOLLOW LINE ----------------
        if self.state == self.FOLLOW_LINE:
            sx, sy = self.line_start
            ex, ey = self.line_end
            
            # Check if we're on the last line
            is_last_line = (self.line_idx == self.total_lines - 1)
            
            # If last line, check distance to endpoint
            if is_last_line:
                dist_to_end = distance_to_point(self.x, self.y, ex, ey)
                
                if dist_to_end < POS_TOL:
                    self.get_logger().info(
                        f"[DONE] Reached end of last line at ({ex:.2f}, {ey:.2f})"
                    )
                    self.cmd_pub.publish(Twist())
                    self.state = self.DONE
                    return
            else:
                # Check if we've reached the next line
                next_line_idx = self.line_idx + 1
                if next_line_idx < self.total_lines:
                    (nx1, ny1), (nx2, ny2) = self.lines[next_line_idx]
                    dist_to_next_line = distance_to_line_segment(
                        self.x, self.y, nx1, ny1, nx2, ny2
                    )
                    
                    if dist_to_next_line < LINE_TRANSITION_TOL:
                        self.get_logger().info(
                            f"[TRANSITION] Reached next line. STOPPING to transition from line "
                            f"{self.line_idx} to {next_line_idx}"
                        )
                        
                        # STOP the robot first
                        self.cmd_pub.publish(Twist())
                        
                        # Update to next line
                        self.line_idx = next_line_idx
                        self.line_start = (nx1, ny1)
                        self.line_end = (nx2, ny2)
                        self.target_point = self.line_start
                        
                        # Go to MOVE_TO_START to reposition if needed
                        self.state = self.MOVE_TO_START
                        return

            # Line following control
            vx, vy = ex - sx, ey - sy
            L2 = vx*vx + vy*vy
            
            if L2 == 0:
                self.get_logger().warn(f"Line {self.line_idx} has zero length!")
                return

            px, py = self.x - sx, self.y - sy
            t = clamp((px*vx + py*vy) / L2, 0.0, 1.0)

            qx = sx + t * vx
            qy = sy + t * vy

            line_yaw = math.atan2(vy, vx)
            yaw_err = normalize_angle(line_yaw - self.yaw)

            # Cross-track error (perpendicular distance to line)
            ct_err = (
                -(self.x - qx) * math.sin(line_yaw)
                + (self.y - qy) * math.cos(line_yaw)
            )

            # Obstacle detection
            front = left = right = float("inf")
            for gx, gy in self.lidar_pts:
                ang = math.atan2(gy - self.y, gx - self.x)
                rel = normalize_angle(ang - self.yaw)
                d = math.hypot(gx - self.x, gy - self.y)

                if abs(rel) < FRONT_ANGLE:
                    front = min(front, d)
                elif abs(rel - math.pi/2) < SIDE_ANGLE:
                    left = min(left, d)
                elif abs(rel + math.pi/2) < SIDE_ANGLE:
                    right = min(right, d)

            # Control law
            omega = (
                GAIN_HEADING * yaw_err
                - GAIN_CROSSTRACK * clamp(ct_err, -MAX_CROSSTRACK, MAX_CROSSTRACK)
            )

            # Obstacle avoidance
            if front < OBSTACLE_DIST:
                omega += OBSTACLE_GAIN if left > right else -OBSTACLE_GAIN
                self.get_logger().warn(
                    f"[OBSTACLE] front={front:.2f}, steering {'left' if left > right else 'right'}"
                )

            twist.linear.x = SPEED_FOLLOW
            twist.angular.z = clamp(omega, -MAX_OMEGA, MAX_OMEGA)
            self.cmd_pub.publish(twist)

            if self.log_timer > LOG_THROTTLE:
                self.get_logger().info(
                    f"[FOLLOW] Line {self.line_idx} | t={t:.2f} | ct_err={ct_err:.2f}"
                )
                self.log_timer = 0

    # ========================================================

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z)
        )

    def scan_cb(self, msg):
        self.lidar_pts.clear()
        ang = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                lx = r * math.cos(ang)
                ly = r * math.sin(ang)
                gx = self.x + math.cos(self.yaw)*lx - math.sin(self.yaw)*ly
                gy = self.y + math.sin(self.yaw)*lx + math.cos(self.yaw)*ly
                self.lidar_pts.append((gx, gy))
            ang += msg.angle_increment


# ============================================================
# MAIN
# ============================================================

def main():
    rclpy.init()
    node = LineFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    finally:
        # Ensure robot stops
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()