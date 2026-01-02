# #!/usr/bin/env python3
# """
# Docstring for testing_task4b.sim.point_to_point_v1

# 1. It moves point to point, turning allowed to fixed angles [0, 90, 270, 260]
# 2. No Shape Detection
# 3. Lidar used

# """

# P0 =  (0.0, 0.0) # initial
# P1 =  (0.0, -1.54)
# P2 = (4.7, -1.54)
# P3 = (4.7, 0.0)
# P4 =   (4.7, 1.70)
# P5 = (0.0, 1.70)



# WAYPOINTS = [P1, P2, P3, P0, P5, P4, P3, P0]



# import math
# import threading
# import rclpy
# from rclpy.node import Node

# from sensor_msgs.msg import LaserScan
# from nav_msgs.msg import Odometry
# from geometry_msgs.msg import Twist

# import matplotlib.pyplot as plt


# # ---------------- HELPERS ---------------- #

# def normalize_angle(a):
#     return math.atan2(math.sin(a), math.cos(a))


# ALLOWED_YAWS = [0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi]

# FRONT_CENTER = 0.0
# LEFT_CENTER  = math.pi / 2
# RIGHT_CENTER = -math.pi / 2

# TOL = 0.3   # sector half-width (rad)
# FRONT_TOL = 0.6

# def quantize_yaw(yaw):
#     return min(ALLOWED_YAWS, key=lambda a: abs(normalize_angle(yaw - a)))


# # ---------------- NODE ---------------- #

# class Navigate(Node):

#     def __init__(self):
#         super().__init__("real_point_to_point_v1")

#         # ROS I/O
#         self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
#         self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
#         self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

#         # Robot pose
#         self.x = 0.0
#         self.y = 0.0
#         self.yaw = 0.0

#         # LiDAR points
#         self.lidar_points = []
#         self.front_points = []
#         self.left_points = []
#         self.right_points = []

#         # Motion params
#         self.forward_speed = 0.4
#         self.turn_gain = 1.2
#         self.avoid_gain = 0.6
#         self.obstacle_dist = 0.6

#         # Waypoints (edit as needed)
#         self.waypoints = WAYPOINTS
#         self.wp_idx = 0
#         self.wp_tol = 0.4   # near enough is fine

#         self.running = True
#         self.create_timer(0.1, self.control_loop)

#     # ---------------- CONTROL ---------------- #

#     def control_loop(self):
#         if not self.running:
#             return

#         twist = Twist()

#         # clear sectors
#         self.front_points.clear()
#         self.left_points.clear()
#         self.right_points.clear()

#         # snap yaw only for sector reference
#         ref_yaw = quantize_yaw(self.yaw)

#         # -------- SECTOR CLASSIFICATION --------
#         for gx, gy in self.lidar_points:
#             angle = math.atan2(gy - self.y, gx - self.x)
#             rel = normalize_angle(angle - ref_yaw)

#             if abs(rel - FRONT_CENTER) <= FRONT_TOL:
#                 self.front_points.append((gx, gy))
#             elif abs(rel - LEFT_CENTER) <= TOL:
#                 self.left_points.append((gx, gy))
#             elif abs(rel - RIGHT_CENTER) <= TOL:
#                 self.right_points.append((gx, gy))

#         def min_dist(pts):
#             if not pts:
#                 return float("inf")
#             return min(math.hypot(px - self.x, py - self.y) for px, py in pts)

#         d_front = min_dist(self.front_points)
#         d_left  = min_dist(self.left_points)
#         d_right = min_dist(self.right_points)

#         # single log
#         self.get_logger().info(
#             f"Distances | Front: {d_front:.2f} | Left: {d_left:.2f} | Right: {d_right:.2f}",
#             throttle_duration_sec=1.0
#         )

#         # -------- WAYPOINT NAVIGATION --------
#         wx, wy = self.waypoints[self.wp_idx]
#         dx = wx - self.x
#         dy = wy - self.y
#         dist_wp = math.hypot(dx, dy)

#         # waypoint reached (near enough)
#         if dist_wp < self.wp_tol:
#             self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
#             return

#         # desired heading to waypoint
#         goal_yaw = math.atan2(dy, dx)
#         yaw_error = normalize_angle(goal_yaw - self.yaw)

#         # base motion
#         twist.linear.x = self.forward_speed
#         twist.angular.z = self.turn_gain * yaw_error

#         # -------- OBSTACLE AVOIDANCE (BACKUP) --------
#         if d_front < self.obstacle_dist:
#             if d_left > d_right:
#                 twist.angular.z += self.avoid_gain
#             else:
#                 twist.angular.z -= self.avoid_gain

#         # clamp angular velocity
#         twist.angular.z = max(-1.5, min(1.5, twist.angular.z))

#         self.cmd_pub.publish(twist)

#     # ---------------- CALLBACKS ---------------- #

#     def odom_callback(self, msg: Odometry):
#         self.x = msg.pose.pose.position.x
#         self.y = msg.pose.pose.position.y

#         q = msg.pose.pose.orientation
#         siny = 2.0 * (q.w * q.z + q.x * q.y)
#         cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
#         self.yaw = math.atan2(siny, cosy)

#     def lidar_callback(self, msg: LaserScan):
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

#     # ---------------- STOP ---------------- #

#     def stop(self):
#         self.running = False
#         if rclpy.ok():
#             self.cmd_pub.publish(Twist())


# # ---------------- MAIN ---------------- #

# def main():
#     rclpy.init()
#     node = Navigate()

#     spin_thread = threading.Thread(
#         target=rclpy.spin, args=(node,), daemon=True
#     )
#     spin_thread.start()

#     plt.ion()
#     fig, ax = plt.subplots()

#     try:
#         while rclpy.ok():
#             ax.clear()

#             if node.lidar_points:
#                 xs, ys = zip(*node.lidar_points)
#                 ax.scatter(xs, ys, s=5, c="gray", label="LiDAR")

#             if node.front_points:
#                 fx, fy = zip(*node.front_points)
#                 ax.scatter(fx, fy, s=18, c="red", label="Front")

#             if node.left_points:
#                 lx, ly = zip(*node.left_points)
#                 ax.scatter(lx, ly, s=18, c="green", label="Left")

#             if node.right_points:
#                 rx, ry = zip(*node.right_points)
#                 ax.scatter(rx, ry, s=18, c="orange", label="Right")

#             wx, wy = node.waypoints[node.wp_idx]
#             ax.scatter(wx, wy, c="purple", s=80, marker="X", label="Waypoint")

#             ax.scatter(node.x, node.y, c="blue", s=60, label="Robot")

#             ax.set_aspect("equal")
#             ax.set_xlim(node.x - 5, node.x + 5)
#             ax.set_ylim(node.y - 5, node.y + 5)
#             ax.legend(loc="upper right")

#             plt.pause(0.05)

#     except KeyboardInterrupt:
#         pass

#     node.stop()
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3

"""
point_to_point_v1 (no plotting)

1. Moves point to point
2. Fixed-angle turning
3. LiDAR-based obstacle avoidance
4. ROS logging instead of plotting
"""

import math
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# ---------------- WAYPOINTS ---------------- #

P0 = (0.0, 0.0)
P1 = (0.0, -1.54)
P2 = (4.7, -1.54)
P3 = (4.7, 0.0)
P4 = (4.7, 1.70)
P5 = (0.0, 1.70)

WAYPOINTS = [P1, P2, P3, P0, P5, P4, P3, P0]

# ---------------- HELPERS ---------------- #

def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))

ALLOWED_YAWS = [0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi]

FRONT_CENTER = 0.0
LEFT_CENTER  = math.pi / 2
RIGHT_CENTER = -math.pi / 2

TOL = 0.3
FRONT_TOL = 0.6

def quantize_yaw(yaw):
    return min(ALLOWED_YAWS, key=lambda a: abs(normalize_angle(yaw - a)))

# ---------------- NODE ---------------- #

class Navigate(Node):

    def __init__(self):
        super().__init__("real_point_to_point_v1")

        # ROS I/O
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        # Robot pose
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # LiDAR points
        self.lidar_points = []
        self.front_points = []
        self.left_points = []
        self.right_points = []

        # Motion params
        self.forward_speed = 0.4
        self.turn_gain = 1.2
        self.avoid_gain = 0.6
        self.obstacle_dist = 0.6

        # Waypoints
        self.waypoints = WAYPOINTS
        self.wp_idx = 0
        self.wp_tol = 0.4

        self.running = True
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Navigator started")

    # ---------------- CONTROL LOOP ---------------- #

    def control_loop(self):
        if not self.running:
            return

        twist = Twist()

        # Clear sectors
        self.front_points.clear()
        self.left_points.clear()
        self.right_points.clear()

        ref_yaw = quantize_yaw(self.yaw)

        # -------- SECTOR CLASSIFICATION --------
        for gx, gy in self.lidar_points:
            angle = math.atan2(gy - self.y, gx - self.x)
            rel = normalize_angle(angle - ref_yaw)

            if abs(rel - FRONT_CENTER) <= FRONT_TOL:
                self.front_points.append((gx, gy))
            elif abs(rel - LEFT_CENTER) <= TOL:
                self.left_points.append((gx, gy))
            elif abs(rel - RIGHT_CENTER) <= TOL:
                self.right_points.append((gx, gy))

        def min_dist(pts):
            if not pts:
                return float("inf")
            return min(math.hypot(px - self.x, py - self.y) for px, py in pts)

        d_front = min_dist(self.front_points)
        d_left  = min_dist(self.left_points)
        d_right = min_dist(self.right_points)

        self.get_logger().info(
            f"Pose x={self.x:.2f}, y={self.y:.2f}, yaw={math.degrees(self.yaw):.1f}°",
            throttle_duration_sec=1.5
        )

        self.get_logger().info(
            f"Obstacle dist | Front={d_front:.2f}, Left={d_left:.2f}, Right={d_right:.2f}",
            throttle_duration_sec=1.5
        )

        # -------- WAYPOINT NAVIGATION --------
        wx, wy = self.waypoints[self.wp_idx]
        dx = wx - self.x
        dy = wy - self.y
        dist_wp = math.hypot(dx, dy)

        self.get_logger().info(
            f"Target WP[{self.wp_idx}] x={wx:.2f}, y={wy:.2f}, dist={dist_wp:.2f}",
            throttle_duration_sec=1.5
        )

        if dist_wp < self.wp_tol:
            self.get_logger().info(f"Waypoint {self.wp_idx} reached → switching")
            self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
            return

        goal_yaw = math.atan2(dy, dx)
        yaw_error = normalize_angle(goal_yaw - self.yaw)

        twist.linear.x = self.forward_speed
        twist.angular.z = self.turn_gain * yaw_error

        # -------- OBSTACLE AVOIDANCE --------
        if d_front < self.obstacle_dist:
            if d_left > d_right:
                twist.angular.z += self.avoid_gain
                self.get_logger().warn("Obstacle ahead → steering LEFT")
            else:
                twist.angular.z -= self.avoid_gain
                self.get_logger().warn("Obstacle ahead → steering RIGHT")

        twist.angular.z = max(-1.5, min(1.5, twist.angular.z))

        self.cmd_pub.publish(twist)

    # ---------------- CALLBACKS ---------------- #

    def odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

    def lidar_callback(self, msg: LaserScan):
        self.lidar_points.clear()

        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                lx = r * math.cos(angle)
                ly = r * math.sin(angle)

                gx = self.x + math.cos(self.yaw) * lx - math.sin(self.yaw) * ly
                gy = self.y + math.sin(self.yaw) * lx + math.cos(self.yaw) * ly

                self.lidar_points.append((gx, gy))

            angle += msg.angle_increment

    # ---------------- STOP ---------------- #

    def stop(self):
        self.running = False
        self.cmd_pub.publish(Twist())
        self.get_logger().info("Navigator stopped")

# ---------------- MAIN ---------------- #

def main():
    rclpy.init()
    node = Navigate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
