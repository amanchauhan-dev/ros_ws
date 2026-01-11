#!/usr/bin/env python3
"""
sim_point_to_point_v1

- Point-to-point navigation with turn-in-place alignment
- Obstacle avoidance has highest priority
- Straight motion only when aligned
- NO yaw quantization (continuous yaw)
"""

# ============================================================
# ======================= IMPORTS =============================
# ============================================================

import math
import threading
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import matplotlib.pyplot as plt


# ============================================================
# ======================= CONFIG =============================
# ============================================================

CMD_VEL_TOPIC = "/cmd_vel"
SCAN_TOPIC    = "/scan"
ODOM_TOPIC    = "/odom"

CONTROL_PERIOD = 0.1

# -------- WAYPOINTS --------
P0 = (-1.5,  -5.6)
P1 = (0.32,  -5.6)
P2 = (0.32,   1.38)
P3 = (-1.5,   1.38)
P4 = (-3.459, 1.38)
P5 = (-3.459, -5.6)

WAYPOINTS = [P1, P2, P3, P0, P5, P4, P3, P0]
WAYPOINT_TOL = 0.2

# -------- MOTION --------
FORWARD_SPEED = 1.0
TURN_GAIN     = 1.2
AVOID_GAIN    = 0.7
MAX_ANG_Z     = 1.5

# -------- ALIGNMENT --------
YAW_ALIGN_TOL = 0.12  # radians

# -------- OBSTACLE --------
OBSTACLE_DIST = 0.5

# -------- LIDAR SECTORS (relative to current yaw) --------
FRONT_CENTER = 0.0
LEFT_CENTER  = math.pi / 2
RIGHT_CENTER = -math.pi / 2

FRONT_TOL = 1.0
SIDE_TOL  = 0.4

LOG_THROTTLE = 1.0

PLOT_RANGE = 5.0
PLOT_DT    = 0.05


# ============================================================
# ======================= HELPERS =============================
# ============================================================

def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def min_distance(pts, rx, ry):
    if not pts:
        return float("inf")
    return min(math.hypot(px - rx, py - ry) for px, py in pts)


# ============================================================
# ========================= NODE ==============================
# ============================================================

class Navigate(Node):

    def __init__(self):
        super().__init__("sim_point_to_point_v1")

        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.lidar_cb, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_cb, 10)

        # Robot pose
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # LiDAR storage
        self.lidar_pts = []
        self.front_pts = []
        self.left_pts  = []
        self.right_pts = []

        # Waypoints
        self.waypoints = WAYPOINTS
        self.wp_idx = 0

        self.running = True
        self.create_timer(CONTROL_PERIOD, self.control_loop)

    # ========================================================

    def control_loop(self):
        if not self.running:
            self.get_logger().info(f"(STOP) Self.running: {self.running}")
            return

        twist = Twist()

        self.front_pts.clear()
        self.left_pts.clear()
        self.right_pts.clear()

        # -------- SECTORING USING TRUE YAW --------
        for gx, gy in self.lidar_pts:
            ang = math.atan2(gy - self.y, gx - self.x)
            rel = normalize_angle(ang - self.yaw)

            if abs(rel - FRONT_CENTER) <= FRONT_TOL:
                self.front_pts.append((gx, gy))
            elif abs(rel - LEFT_CENTER) <= SIDE_TOL:
                self.left_pts.append((gx, gy))
            elif abs(rel - RIGHT_CENTER) <= SIDE_TOL:
                self.right_pts.append((gx, gy))

        d_front = min_distance(self.front_pts, self.x, self.y)
        d_left  = min_distance(self.left_pts,  self.x, self.y)
        d_right = min_distance(self.right_pts, self.x, self.y)

        self.get_logger().info(
            f"F:{d_front:.2f} L:{d_left:.2f} R:{d_right:.2f}",
            throttle_duration_sec=LOG_THROTTLE
        )

        # -------- WAYPOINT --------
        wx, wy = self.waypoints[self.wp_idx]
        dx, dy = wx - self.x, wy - self.y
        dist = math.hypot(dx, dy)

        if dist < WAYPOINT_TOL:
            self.get_logger().info(f"WAYPOINT REACHED :{self.wp_idx} ({wx:.2f}, {wy:.2f})")
            self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
            wx, wy = self.waypoints[self.wp_idx]
            self.get_logger().info(f"NEW WAYPOINT :{self.wp_idx} ({wx:.2f}, {wy:.2f})")
            return

        goal_yaw = math.atan2(dy, dx)
        yaw_err = normalize_angle(goal_yaw - self.yaw)

        # =====================================================
        # PRIORITY 1 — OBSTACLE AVOIDANCE
        # =====================================================
        if d_front < OBSTACLE_DIST:
            twist.linear.x = FORWARD_SPEED * 0.6
            twist.angular.z = AVOID_GAIN if d_left > d_right else -AVOID_GAIN
            self.cmd_pub.publish(twist)
            side = "LEFT" if d_left > d_right else "RIGHT" 
            self.get_logger().info(f"OBSTACLE AT FRONT AVOIDING: MOVING && TURNING {side}")
            return

        # =====================================================
        # PRIORITY 2 — TURN IN PLACE
        # =====================================================
        if abs(yaw_err) > YAW_ALIGN_TOL:
            twist.linear.x = 0.0
            twist.angular.z = TURN_GAIN * yaw_err
            twist.angular.z = max(-MAX_ANG_Z, min(MAX_ANG_Z, twist.angular.z))
            self.cmd_pub.publish(twist)
            self.get_logger().info(f"TURNING IN PLACE TOWARDS WAYPOINT")
            return

        # =====================================================
        # PRIORITY 3 — MOVE STRAIGHT
        # =====================================================
        self.get_logger().info(f"MOVING TOWARDS WAYPOINT")
        twist.linear.x = FORWARD_SPEED
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    # ========================================================

    def odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(siny, cosy)

    def lidar_cb(self, msg: LaserScan):
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

    def stop(self):
        self.running = False
        self.cmd_pub.publish(Twist())


# ============================================================
# ========================== MAIN =============================
# ============================================================

def main():
    rclpy.init()
    node = Navigate()

    threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    ).start()

    plt.ion()
    fig, ax = plt.subplots()

    try:
        while rclpy.ok():
            ax.clear()

            if node.lidar_pts:
                xs, ys = zip(*node.lidar_pts)
                ax.scatter(xs, ys, s=5, c="gray", label="LiDAR")

            for pts, c, name in [
                (node.front_pts, "red", "Front"),
                (node.left_pts, "green", "Left"),
                (node.right_pts, "orange", "Right"),
            ]:
                if pts:
                    x, y = zip(*pts)
                    ax.scatter(x, y, c=c, label=name)

            wx, wy = node.waypoints[node.wp_idx]
            ax.scatter(wx, wy, c="purple", s=80, marker="X", label="Waypoint")
            ax.scatter(node.x, node.y, c="blue", s=60, label="Robot")

            ax.set_aspect("equal")
            ax.set_xlim(node.x - PLOT_RANGE, node.x + PLOT_RANGE)
            ax.set_ylim(node.y - PLOT_RANGE, node.y + PLOT_RANGE)
            ax.legend(loc="upper right")

            plt.pause(PLOT_DT)

    except KeyboardInterrupt:
        pass

    node.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
