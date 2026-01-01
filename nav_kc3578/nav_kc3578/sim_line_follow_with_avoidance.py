#!/usr/bin/env python3
# """
# Line-following with sharp 90-degree turns and bounded obstacle avoidance

# Behavior:
# - Explicitly starts on line index 0
# - Move to line start → rotate in place → follow line
# - Rotate-in-place at every corner
# - Obstacle avoidance is temporary and bounded
# - Full matplotlib visualization
# """

# import math
# import threading
# import rclpy
# from rclpy.node import Node

# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import LaserScan
# from geometry_msgs.msg import Twist

# import matplotlib.pyplot as plt

# # ============================================================
# # ======================= ROS CONFIG =========================
# # ============================================================

# CMD_VEL_TOPIC = "/cmd_vel"
# ODOM_TOPIC    = "/odom"
# SCAN_TOPIC    = "/scan"

# CONTROL_PERIOD = 0.1  # seconds

# # ============================================================
# # ======================= PATH GEOMETRY ======================
# # ============================================================

# LINES = [
#     ((-1.5, -5.6), (0.32, -5.6)),   # START LINE (index 0)
#     ((0.32, -5.6), (0.32,  1.38)),
#     ((0.32,  1.38), (-1.5, 1.38)),
#     ((-1.5,  1.38), (-3.459, 1.38)),
#     ((-3.459, 1.38), (-3.459, -5.6)),
#     ((-3.459, -5.6), (-1.5, -5.6)),
# ]

# START_LINE_INDEX = 0

# # ============================================================
# # ======================= MOTION TUNING ======================
# # ============================================================

# FORWARD_SPEED        = 0.4
# APPROACH_SPEED       = 0.2
# MAX_ANGULAR_SPEED    = 1.5

# HEADING_GAIN         = 1.2
# CROSSTRACK_GAIN      = 2.0
# MAX_LINE_DEVIATION   = 0.4

# OBSTACLE_DISTANCE    = 0.4
# OBSTACLE_TURN_GAIN   = 0.8

# YAW_ALIGNMENT_TOL    = 0.08
# POSITION_TOLERANCE  = 0.15

# # ============================================================
# # ======================= LIDAR CONFIG =======================
# # ============================================================

# FRONT_ANGLE_TOL = 0.6
# SIDE_ANGLE_TOL  = 0.4

# # ============================================================
# # ======================= HELPERS ============================
# # ============================================================

# def normalize_angle(a: float) -> float:
#     return math.atan2(math.sin(a), math.cos(a))


# def clamp(v: float, lo: float, hi: float) -> float:
#     return max(lo, min(hi, v))


# # ============================================================
# # ======================= CONTROLLER =========================
# # ============================================================

# class LineFollower(Node):

#     MOVE_TO_START = 1
#     ROTATE_TO_LINE = 2
#     FOLLOW_LINE = 3

#     def __init__(self):
#         super().__init__("line_follower")

#         self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
#         self.create_subscription(Odometry, ODOM_TOPIC, self.odom_cb, 10)
#         self.create_subscription(LaserScan, SCAN_TOPIC, self.scan_cb, 10)

#         # Robot state
#         self.x = 0.0
#         self.y = 0.0
#         self.yaw = 0.0

#         self.lidar_points = []

#         # Path state
#         self.lines = LINES
#         self.line_index = START_LINE_INDEX
#         self.target_point = self.lines[self.line_index][0]

#         self.state = self.MOVE_TO_START

#         # Trajectory for plotting
#         self.traj_x = []
#         self.traj_y = []

#         self.create_timer(CONTROL_PERIOD, self.control_loop)

#     # ========================================================

#     def control_loop(self):
#         twist = Twist()
#         self.traj_x.append(self.x)
#         self.traj_y.append(self.y)

#         # ====================================================
#         # MOVE TO START POINT
#         # ====================================================
#         if self.state == self.MOVE_TO_START:
#             tx, ty = self.target_point
#             dx, dy = tx - self.x, ty - self.y
#             distance = math.hypot(dx, dy)

#             if distance < POSITION_TOLERANCE:
#                 self.state = self.ROTATE_TO_LINE
#                 return

#             desired_yaw = math.atan2(dy, dx)
#             yaw_error = normalize_angle(desired_yaw - self.yaw)

#             twist.linear.x = APPROACH_SPEED
#             twist.angular.z = clamp(1.5 * yaw_error, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
#             self.cmd_pub.publish(twist)
#             return

#         # ====================================================
#         # ROTATE IN PLACE TO LINE DIRECTION
#         # ====================================================
#         if self.state == self.ROTATE_TO_LINE:
#             A, B = self.lines[self.line_index]
#             line_yaw = math.atan2(B[1] - A[1], B[0] - A[0])
#             yaw_error = normalize_angle(line_yaw - self.yaw)

#             if abs(yaw_error) < YAW_ALIGNMENT_TOL:
#                 self.state = self.FOLLOW_LINE
#                 return

#             twist.angular.z = clamp(
#                 1.8 * yaw_error,
#                 -MAX_ANGULAR_SPEED,
#                 MAX_ANGULAR_SPEED
#             )
#             self.cmd_pub.publish(twist)
#             return

#         # ====================================================
#         # FOLLOW CURRENT LINE
#         # ====================================================
#         A, B = self.lines[self.line_index]
#         ax, ay = A
#         bx, by = B

#         vx, vy = bx - ax, by - ay
#         length_sq = vx * vx + vy * vy

#         px, py = self.x - ax, self.y - ay
#         t = clamp((px * vx + py * vy) / length_sq, 0.0, 1.0)

#         if t > 0.98:
#             self.line_index = (self.line_index + 1) % len(self.lines)
#             self.target_point = self.lines[self.line_index][0]
#             self.state = self.MOVE_TO_START
#             return

#         proj_x = ax + t * vx
#         proj_y = ay + t * vy

#         line_yaw = math.atan2(vy, vx)
#         yaw_error = normalize_angle(line_yaw - self.yaw)

#         cross_track_error = (
#             math.sin(line_yaw) * (self.x - proj_x)
#             - math.cos(line_yaw) * (self.y - proj_y)
#         )

#         # ---------- LIDAR ----------
#         front = left = right = float("inf")

#         for gx, gy in self.lidar_points:
#             ang = math.atan2(gy - self.y, gx - self.x)
#             rel = normalize_angle(ang - self.yaw)
#             dist = math.hypot(gx - self.x, gy - self.y)

#             if abs(rel) < FRONT_ANGLE_TOL:
#                 front = min(front, dist)
#             elif abs(rel - math.pi/2) < SIDE_ANGLE_TOL:
#                 left = min(left, dist)
#             elif abs(rel + math.pi/2) < SIDE_ANGLE_TOL:
#                 right = min(right, dist)

#         omega = (
#             HEADING_GAIN * yaw_error
#             - CROSSTRACK_GAIN * clamp(cross_track_error, -MAX_LINE_DEVIATION, MAX_LINE_DEVIATION)
#         )

#         if front < OBSTACLE_DISTANCE:
#             omega += OBSTACLE_TURN_GAIN if left > right else -OBSTACLE_TURN_GAIN

#         twist.linear.x = FORWARD_SPEED if abs(cross_track_error) < MAX_LINE_DEVIATION else APPROACH_SPEED
#         twist.angular.z = clamp(omega, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

#         self.cmd_pub.publish(twist)

#     # ========================================================

#     def odom_cb(self, msg: Odometry):
#         self.x = msg.pose.pose.position.x
#         self.y = msg.pose.pose.position.y
#         q = msg.pose.pose.orientation

#         self.yaw = math.atan2(
#             2 * (q.w * q.z + q.x * q.y),
#             1 - 2 * (q.y * q.y + q.z * q.z)
#         )

#     def scan_cb(self, msg: LaserScan):
#         self.lidar_points.clear()
#         angle = msg.angle_min

#         for r in msg.ranges:
#             if msg.range_min < r < msg.range_max:
#                 lx = r * math.cos(angle)
#                 ly = r * math.sin(angle)

#                 gx = self.x + math.cos(self.yaw) * lx - math.sin(self.yaw) * ly
#                 gy = self.y + math.sin(self.yaw) * lx + math.cos(self.yaw) * ly

#                 self.lidar_points.append((gx, gy))
#             angle += msg.angle_increment


# # ============================================================
# # =========================== MAIN ===========================
# # ============================================================

# def main():
#     rclpy.init()
#     node = LineFollower()

#     threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

#     plt.ion()
#     fig, ax = plt.subplots()

#     try:
#         while rclpy.ok():
#             ax.clear()

#             # draw all lines
#             for (x1, y1), (x2, y2) in LINES:
#                 ax.plot([x1, x2], [y1, y2], "k-", lw=2)

#             # highlight active line
#             A, B = LINES[node.line_index]
#             ax.plot([A[0], B[0]], [A[1], B[1]], "g-", lw=4, label="Active Line")

#             # trajectory
#             ax.plot(node.traj_x, node.traj_y, "b--", label="Trajectory")

#             # robot
#             ax.scatter(node.x, node.y, c="red", s=60)
#             ax.arrow(
#                 node.x,
#                 node.y,
#                 0.4 * math.cos(node.yaw),
#                 0.4 * math.sin(node.yaw),
#                 head_width=0.1,
#                 color="red"
#             )

#             ax.set_aspect("equal")
#             ax.set_xlim(node.x - 5, node.x + 5)
#             ax.set_ylim(node.y - 5, node.y + 5)
#             ax.legend(loc="upper right")

#             plt.pause(0.05)

#     except KeyboardInterrupt:
#         pass

#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
"""
Line-following with sharp 90-degree turns and bounded obstacle avoidance
"""

import math
import threading
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

import matplotlib.pyplot as plt

# ============================================================
# ======================= CONFIG =============================
# ============================================================

CMD_VEL_TOPIC = "/cmd_vel"
ODOM_TOPIC    = "/odom"
SCAN_TOPIC    = "/scan"

CONTROL_PERIOD = 0.1

LINES = [
    ((-1.5, -5.6), (0.32, -5.6)),   # START LINE (index 0)
    ((0.32, -5.6), (0.32,  1.38)),
    ((0.32,  1.38), (-1.5, 1.38)),
    ((-1.5,  1.38), (-3.459, 1.38)),
    ((-3.459, 1.38), (-3.459, -5.6)),
    ((-3.459, -5.6), (-1.5, -5.6)),
]

START_LINE_INDEX = 0

FORWARD_SPEED = 0.4
APPROACH_SPEED = 0.2
MAX_ANGULAR_SPEED = 1.5

HEADING_GAIN = 1.2
CROSSTRACK_GAIN = 2.0
MAX_LINE_DEVIATION = 0.4

OBSTACLE_DISTANCE = 0.4
OBSTACLE_TURN_GAIN = 0.8

YAW_ALIGNMENT_TOL = 0.08
POSITION_TOLERANCE = 0.15

FRONT_ANGLE_TOL = 0.6
SIDE_ANGLE_TOL  = 0.4

LOG_THROTTLE = 1.0  # seconds

# ============================================================
# ======================= HELPERS =============================
# ============================================================

def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ============================================================
# ========================= NODE ==============================
# ============================================================

class LineFollower(Node):

    MOVE_TO_START = 1
    ROTATE_TO_LINE = 2
    FOLLOW_LINE = 3

    def __init__(self):
        super().__init__("line_follower")

        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_cb, 10)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.scan_cb, 10)

        self.x = self.y = self.yaw = 0.0
        self.lidar_points = []

        self.lines = LINES
        self.line_index = START_LINE_INDEX
        self.target_point = self.lines[self.line_index][0]

        self.state = self.MOVE_TO_START

        self.traj_x = []
        self.traj_y = []

        self.get_logger().info(
            f"Initialized. Starting on line index {self.line_index}"
        )

        self.create_timer(CONTROL_PERIOD, self.control_loop)

    # ========================================================

    def control_loop(self):
        twist = Twist()
        self.traj_x.append(self.x)
        self.traj_y.append(self.y)

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

            if dist < POSITION_TOLERANCE:
                self.get_logger().info("[MOVE_TO_START] Reached start point")
                self.state = self.ROTATE_TO_LINE
                return

            yaw_goal = math.atan2(dy, dx)
            yaw_err = normalize_angle(yaw_goal - self.yaw)

            twist.linear.x = APPROACH_SPEED
            twist.angular.z = clamp(1.5 * yaw_err, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
            self.cmd_pub.publish(twist)
            return

        # ====================================================
        # ROTATE TO LINE
        # ====================================================
        if self.state == self.ROTATE_TO_LINE:
            A, B = self.lines[self.line_index]
            line_yaw = math.atan2(B[1] - A[1], B[0] - A[0])
            yaw_err = normalize_angle(line_yaw - self.yaw)

            self.get_logger().info(
                f"[ROTATE] Line={self.line_index} "
                f"YawErr={math.degrees(yaw_err):.1f} deg",
                throttle_duration_sec=LOG_THROTTLE
            )

            if abs(yaw_err) < YAW_ALIGNMENT_TOL:
                self.get_logger().info("[ROTATE] Alignment complete")
                self.state = self.FOLLOW_LINE
                return

            twist.angular.z = clamp(
                1.8 * yaw_err,
                -MAX_ANGULAR_SPEED,
                MAX_ANGULAR_SPEED
            )
            self.cmd_pub.publish(twist)
            return

        # ====================================================
        # FOLLOW LINE
        # ====================================================
        A, B = self.lines[self.line_index]
        ax, ay = A
        bx, by = B

        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy

        px, py = self.x - ax, self.y - ay
        t = clamp((px * vx + py * vy) / L2, 0.0, 1.0)

        if t > 0.98:
            self.get_logger().info(
                f"[FOLLOW] Line {self.line_index} completed → switching"
            )
            self.line_index = (self.line_index + 1) % len(self.lines)
            self.target_point = self.lines[self.line_index][0]
            self.state = self.MOVE_TO_START
            return

        proj_x = ax + t * vx
        proj_y = ay + t * vy

        line_yaw = math.atan2(vy, vx)
        yaw_err = normalize_angle(line_yaw - self.yaw)

        qx = ax + t * vx
        qy = ay + t * vy
        
        ct_err = -(self.x - qx) * math.sin(line_yaw) + (self.y - qy) * math.cos(line_yaw)


        # -------- LIDAR --------
        front = left = right = float("inf")

        for gx, gy in self.lidar_points:
            ang = math.atan2(gy - self.y, gx - self.x)
            rel = normalize_angle(ang - self.yaw)
            d = math.hypot(gx - self.x, gy - self.y)

            if abs(rel) < FRONT_ANGLE_TOL:
                front = min(front, d)
            elif abs(rel - math.pi / 2) < SIDE_ANGLE_TOL:
                left = min(left, d)
            elif abs(rel + math.pi / 2) < SIDE_ANGLE_TOL:
                right = min(right, d)

        self.get_logger().info(
            f"[FOLLOW] Line={self.line_index} "
            f"CT={ct_err:.2f} Front={front:.2f}",
            throttle_duration_sec=LOG_THROTTLE
        )

        omega = (
            HEADING_GAIN * yaw_err
            - CROSSTRACK_GAIN * clamp(ct_err, -MAX_LINE_DEVIATION, MAX_LINE_DEVIATION)
        )

        if front < OBSTACLE_DISTANCE:
            turn = "LEFT" if left > right else "RIGHT"
            self.get_logger().warn(
                f"[AVOID] Obstacle ahead → turning {turn}",
                throttle_duration_sec=LOG_THROTTLE
            )
            omega += OBSTACLE_TURN_GAIN if left > right else -OBSTACLE_TURN_GAIN

        twist.linear.x = (
            FORWARD_SPEED if abs(ct_err) < MAX_LINE_DEVIATION else APPROACH_SPEED
        )
        twist.angular.z = clamp(omega, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

        self.cmd_pub.publish(twist)

    # ========================================================

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        self.yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z)
        )

    def scan_cb(self, msg):
        self.lidar_points.clear()
        ang = msg.angle_min

        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                lx = r * math.cos(ang)
                ly = r * math.sin(ang)

                gx = self.x + math.cos(self.yaw) * lx - math.sin(self.yaw) * ly
                gy = self.y + math.sin(self.yaw) * lx + math.cos(self.yaw) * ly

                self.lidar_points.append((gx, gy))
            ang += msg.angle_increment


# ============================================================
# ============================ MAIN ==========================
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

            for (x1, y1), (x2, y2) in LINES:
                ax.plot([x1, x2], [y1, y2], "k-", lw=2)

            A, B = LINES[node.line_index]
            ax.plot([A[0], B[0]], [A[1], B[1]], "g-", lw=4, label="Active Line")

            ax.plot(node.traj_x, node.traj_y, "b--", label="Trajectory")
            ax.scatter(node.x, node.y, c="red", s=60)

            ax.arrow(
                node.x,
                node.y,
                0.4 * math.cos(node.yaw),
                0.4 * math.sin(node.yaw),
                head_width=0.1,
                color="red"
            )

            ax.set_aspect("equal")
            ax.set_xlim(node.x - 5, node.x + 5)
            ax.set_ylim(node.y - 5, node.y + 5)
            ax.legend(loc="upper right")

            plt.pause(0.05)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
