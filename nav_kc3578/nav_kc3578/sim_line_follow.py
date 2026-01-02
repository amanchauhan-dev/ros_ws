#!/usr/bin/env python3
"""
Deterministic line-by-line coverage with bounded obstacle avoidance

States:
- MOVE_TO_START : go to closest endpoint of active line
- ROTATE_TO_LINE: rotate in place to align with line direction
- FOLLOW_LINE   : follow the line with cross-track control
- DONE          : stop after all lines are covered

Guarantees:
- Each line is covered exactly once
- Direction of each line is chosen dynamically
- Obstacle avoidance is temporary and bounded
- Robot always re-aligns with the line
"""

# ============================================================
# =========================== IMPORTS ========================
# ============================================================

import math
import threading
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

import matplotlib.pyplot as plt

# ============================================================
# =========================== CONFIG =========================
# ============================================================

CMD_VEL_TOPIC = "/cmd_vel"
ODOM_TOPIC    = "/odom"
SCAN_TOPIC    = "/scan"

CONTROL_DT = 0.1

# ---------- LINES (ORDER = COVERAGE ORDER) ----------
LINES = [
    ((-1.5, -5.6), (0.32, -5.6)),
    ((0.32, -5.6), (0.32,  1.38)),
    ((0.32,  1.38), (-1.5, 1.38)),
    ((-1.5,  1.38), (-3.459, 1.38)),
    ((-3.459, 1.38), (-3.459, -5.6)),
    ((-3.459, -5.6), (-1.5, -5.6)),
]

START_LINE_INDEX = 0

# ---------- MOTION ----------
SPEED_FOLLOW   = 0.4
SPEED_APPROACH = 0.2
MAX_OMEGA      = 1.5

# ---------- LINE CONTROL ----------
GAIN_HEADING    = 1.2
GAIN_CROSSTRACK = 2.0
MAX_CROSSTRACK  = 0.4

# ---------- OBSTACLE ----------
OBSTACLE_DIST = 0.4
OBSTACLE_GAIN = 0.8

# ---------- TOLERANCES ----------
POS_TOL = 0.15
YAW_TOL = 0.08

# ---------- LIDAR SECTORS ----------
FRONT_ANGLE = 0.6
SIDE_ANGLE  = 0.4

LOG_THROTTLE = 1.0

# ============================================================
# =========================== HELPERS ========================
# ============================================================

def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def choose_nearest_endpoint(robot_xy, line):
    """Return (start, end) so that start is closest to robot."""
    (x1, y1), (x2, y2) = line
    d1 = math.hypot(robot_xy[0] - x1, robot_xy[1] - y1)
    d2 = math.hypot(robot_xy[0] - x2, robot_xy[1] - y2)
    return ((x1, y1), (x2, y2)) if d1 <= d2 else ((x2, y2), (x1, y1))

# ============================================================
# ============================= NODE =========================
# ============================================================

class LineFollower(Node):

    MOVE_TO_START  = 0
    ROTATE_TO_LINE = 1
    FOLLOW_LINE    = 2
    DONE           = 3

    def __init__(self):
        super().__init__("line_follower")

        # ROS I/O
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_cb, 10)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.scan_cb, 10)

        # Robot state
        self.x = -1.5
        self.y = -5.6
        self.yaw = 0.0
        self.lidar_pts = []

        # Line tracking
        self.lines = LINES
        self.total_lines = len(self.lines)
        self.line_idx = START_LINE_INDEX
        self.completed_lines = 0

        start, end = choose_nearest_endpoint(
            (self.x, self.y),
            self.lines[self.line_idx]
        )
        self.active_line = (start, end)
        self.target_point = start

        self.state = self.MOVE_TO_START

        # Trajectory (SAFE)
        self.trajectory = []

        self.get_logger().info(
            f"[INIT] Starting on line {self.line_idx} | start={start} end={end}"
        )

        self.create_timer(CONTROL_DT, self.control_loop)

    # ========================================================

    def control_loop(self):
        twist = Twist()

        # ---- trajectory logging (ONCE per tick) ----
        self.trajectory.append((self.x, self.y))

        # ====================================================
        # MOVE TO START
        # ====================================================
        if self.state == self.MOVE_TO_START:
            tx, ty = self.target_point
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)

            self.get_logger().info(
                f"[MOVE_TO_START] Target={self.target_point} Dist={dist:.2f}",
                throttle_duration_sec=LOG_THROTTLE
            )

            if dist < POS_TOL:
                self.get_logger().info("[MOVE_TO_START] Reached start point")
                self.state = self.ROTATE_TO_LINE
                return

            yaw_goal = math.atan2(dy, dx)
            yaw_err = normalize_angle(yaw_goal - self.yaw)

            twist.linear.x = SPEED_APPROACH
            twist.angular.z = clamp(1.5 * yaw_err, -MAX_OMEGA, MAX_OMEGA)
            self.cmd_pub.publish(twist)
            return

        # ====================================================
        # ROTATE TO LINE
        # ====================================================
        if self.state == self.ROTATE_TO_LINE:
            (ax, ay), (bx, by) = self.active_line
            line_yaw = math.atan2(by - ay, bx - ax)
            yaw_err = normalize_angle(line_yaw - self.yaw)

            self.get_logger().info(
                f"[ROTATE] YawErr={math.degrees(yaw_err):.1f} deg",
                throttle_duration_sec=LOG_THROTTLE
            )

            if abs(yaw_err) < YAW_TOL:
                self.get_logger().info("[ROTATE] Alignment complete")
                self.state = self.FOLLOW_LINE
                return

            twist.angular.z = clamp(1.8 * yaw_err, -MAX_OMEGA, MAX_OMEGA)
            self.cmd_pub.publish(twist)
            return

        # ====================================================
        # FOLLOW LINE
        # ====================================================
        if self.state == self.FOLLOW_LINE:
            (ax, ay), (bx, by) = self.active_line
            vx, vy = bx - ax, by - ay
            L2 = vx*vx + vy*vy

            px, py = self.x - ax, self.y - ay
            t = clamp((px*vx + py*vy)/L2, 0.0, 1.0)

            if t > 0.98:
                self.completed_lines += 1
                self.get_logger().info(
                    f"[FOLLOW] Line {self.line_idx} completed "
                    f"({self.completed_lines}/{self.total_lines})"
                )

                if self.completed_lines >= self.total_lines:
                    self.get_logger().info("[DONE] All lines covered")
                    self.cmd_pub.publish(Twist())
                    self.state = self.DONE
                    return

                self.line_idx += 1
                start, end = choose_nearest_endpoint(
                    (self.x, self.y),
                    self.lines[self.line_idx]
                )
                self.active_line = (start, end)
                self.target_point = start
                self.state = self.MOVE_TO_START
                return

            qx = ax + t * vx
            qy = ay + t * vy

            line_yaw = math.atan2(vy, vx)
            yaw_err = normalize_angle(line_yaw - self.yaw)

            ct_err = (
                -(self.x - qx) * math.sin(line_yaw)
                + (self.y - qy) * math.cos(line_yaw)
            )

            # ---- obstacle check ----
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

            self.get_logger().info(
                f"[FOLLOW] CT={ct_err:.2f} Front={front:.2f}",
                throttle_duration_sec=LOG_THROTTLE
            )

            omega = (
                GAIN_HEADING * yaw_err
                - GAIN_CROSSTRACK * clamp(ct_err, -MAX_CROSSTRACK, MAX_CROSSTRACK)
            )

            if front < OBSTACLE_DIST:
                omega += OBSTACLE_GAIN if left > right else -OBSTACLE_GAIN

            twist.linear.x = (
                SPEED_FOLLOW if abs(ct_err) < MAX_CROSSTRACK else SPEED_APPROACH
            )
            twist.angular.z = clamp(omega, -MAX_OMEGA, MAX_OMEGA)
            self.cmd_pub.publish(twist)
            return

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
# ============================== MAIN ========================
# ============================================================

def main():
    rclpy.init()
    node = LineFollower()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    plt.ion()
    fig, ax = plt.subplots()

    try:
        while rclpy.ok():
            ax.clear()

            # All lines
            for (x1,y1),(x2,y2) in LINES:
                ax.plot([x1,x2],[y1,y2],'k-',lw=2)

            # Active line
            if node.state != node.DONE:
                (a,b) = node.active_line
                ax.plot([a[0],b[0]],[a[1],b[1]],'g-',lw=4,label="Active Line")

            # Trajectory (SAFE)
            if len(node.trajectory) > 1:
                xs, ys = zip(*node.trajectory)
                ax.plot(xs, ys, 'b--', label="Trajectory")

            # Robot pose
            ax.scatter(node.x, node.y, c='red', s=60)
            ax.arrow(
                node.x, node.y,
                0.4*math.cos(node.yaw),
                0.4*math.sin(node.yaw),
                head_width=0.1,
                color='red'
            )
            x, y = np.vstack(node.lidar_pts).T if node.lidar_pts else ([], [])
            ax.scatter(x, y, c='orange', s=5, label="LIDAR Points")

            ax.set_aspect('equal')
            ax.set_xlim(node.x-5, node.x+5)
            ax.set_ylim(node.y-5, node.y+5)
            ax.legend()
            plt.pause(0.05)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
