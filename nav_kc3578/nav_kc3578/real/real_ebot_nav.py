#!/usr/bin/env python3

# Waypoints
# P0 = (0.0, 0.0)
# L1 = (1.5, -1.54)
# L2 = (3.2, 0.0)
# L3 = (1.5, 1.70)

# SAFE_X = [0, 4.70]
# SAFE_Y = [-1.54, 0.0, 1.70]      


import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from enum import Enum, auto
from std_msgs.msg import Bool, String

# ---------------- CONSTANTS ---------------- #

P0 = (0.0, 0.0)
L1 = (1.5, -1.54)
L2 = (3.2, 0.0)
L3 = (1.5, 1.70)

SAFE_X = [0, 4.70]
SAFE_Y = [-1.54, 0.0, 1.70]

# ---------------- ENUMS ---------------- #

class Actions(Enum):
    MOVE = auto()
    TURN = auto()

class Stage(Enum):
    ONE = auto()    # move to nearest_safe_x (now along Y)
    TWO = auto()    # cover Y via safe_x (now along Y)
    THREE = auto()  # cover X after Y (now along X)

# ---------------- NODE ---------------- #

class Navigate(Node):

    def __init__(self):
        super().__init__("waypoint_navigator")

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.ransac_pub = self.create_publisher(Bool, "/in_lane", 10)

        # Subscribers
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(Bool, "/to_stop", self.to_stop_shape_callback, 10)
        self.create_subscription(String, "/detection_status", self.shape_callback, 10)

        # Waypoints
        self.waypoints = [L1, L2, L3, L2, P0]
        self.next_point = 0

        # Safe lines
        self.safe_x = SAFE_X
        self.safe_y = SAFE_Y

        self.current_lane = 1
        self.move_error_points = None

        # Nearest safe_y selection (SWAPPED: based on X now)
        if self.waypoints:
            first_x = self.waypoints[0][0]
            d0 = abs(self.safe_y[0] - first_x)
            d1 = abs(self.safe_y[1] - first_x)
            self.nearest_safe_x = self.safe_x[0] if d0 < d1 else self.safe_x[1]
        else:
            self.nearest_safe_x = self.safe_x[0]

        # State machine
        self.stage = Stage.ONE
        self.action = Actions.TURN
        self.is_stop = False

        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Motion params
        self.linear_speed = 1.5
        self.angular_speed = 1.5
        self.position_tolerance = 0.15
        self.angular_tolerance = 0.05

        self.pre_turn_distance = 0.25
        self.pre_turn_start = None
        self.is_pre_turn = False

        self.create_timer(0.1, self.control_loop)

    # ---------------- CALLBACKS ---------------- #

    def shape_callback(self, msg: String):
        self.get_logger().info(msg.data)

    def to_stop_shape_callback(self, msg: Bool):
        self.is_stop = msg.data

    def publish_in_lane(self):
        msg = Bool()
        min_x = self.safe_x[0] + (self.position_tolerance * 1.5)
        max_x = 0.50
        msg.data = (min_x < self.current_x < max_x) and (self.action != Actions.TURN)
        self.ransac_pub.publish(msg)

    # ---------------- MAIN LOOP ---------------- #

    def control_loop(self):
        self.publish_in_lane()

        twist = Twist()

        if self.is_stop or self.next_point >= len(self.waypoints):
            self.cmd_pub.publish(Twist())
            return

        target_x, target_y = self.waypoints[self.next_point]

        if self.stage == Stage.ONE:
            self.handle_stage_one(twist, target_y)
        elif self.stage == Stage.TWO:
            self.handle_stage_two(twist, target_y)
        elif self.stage == Stage.THREE:
            self.handle_stage_three(twist, target_x)

        self.cmd_pub.publish(twist)

    # ---------------- STAGE 1 ---------------- #
    # Move to nearest_safe_x (SWAPPED → along X)

    def handle_stage_one(self, twist: Twist, target_y: float):

        # Y reached → go to stage THREE
        y_error = target_y - self.current_y
        if abs(y_error) <= self.position_tolerance:
            self.stage = Stage.THREE
            self.action = Actions.TURN
            self.is_pre_turn = True
            self.current_lane = self.get_lane_number()
            return

        x_error = self.nearest_safe_x - self.current_x
        if abs(x_error) <= self.position_tolerance:
            self.current_lane = None
            self.stage = Stage.TWO
            self.action = Actions.TURN
            self.is_pre_turn = True
            return

        direction = 1.0 if x_error > 0 else -1.0
        desired_yaw = 0.0 if direction > 0 else math.pi
        yaw_error = self.normalize_angle(desired_yaw - self.current_yaw)

        if self.action == Actions.TURN:
            if self.pre_turn_move(twist):
                return
            if abs(yaw_error) > self.angular_tolerance:
                twist.angular.z = self.angular_speed if yaw_error > 0 else -self.angular_speed
            else:
                self.action = Actions.MOVE
        else:
            self.move_horizontal(twist)

    # ---------------- STAGE 2 ---------------- #
    # Cover X (SWAPPED → move along Y)

    def handle_stage_two(self, twist: Twist, target_y: float):

        y_error = target_y - self.current_y
        if abs(y_error) <= self.position_tolerance:
            self.stage = Stage.THREE
            self.action = Actions.TURN
            self.is_pre_turn = True
            self.current_lane = self.get_lane_number()
            return

        direction = 1.0 if y_error > 0 else -1.0
        desired_yaw = math.pi/2 if direction > 0 else -math.pi/2
        yaw_error = self.normalize_angle(desired_yaw - self.current_yaw)

        if self.action == Actions.TURN:
            if self.pre_turn_move(twist):
                return
            if abs(yaw_error) > self.angular_tolerance:
                twist.angular.z = self.angular_speed if yaw_error > 0 else -self.angular_speed
            else:
                self.action = Actions.MOVE
        else:
            twist.linear.x = self.linear_speed

    # ---------------- STAGE 3 ---------------- #
    # Cover Y (SWAPPED → along X)

    def handle_stage_three(self, twist: Twist, target_x: float):

        x_error = target_x - self.current_x
        if abs(x_error) <= self.position_tolerance:
            self.next_point += 1
            self.cmd_pub.publish(Twist())

            if self.next_point < len(self.waypoints):
                wp_x, _ = self.waypoints[self.next_point]
                d0 = abs(self.safe_y[0] - wp_x)
                d1 = abs(self.safe_y[1] - wp_x)
                self.nearest_safe_x = self.safe_x[0] if d0 < d1 else self.safe_x[1]
                self.stage = Stage.ONE
                self.action = Actions.TURN
                self.is_pre_turn = True
            return

        direction = 1.0 if x_error > 0 else -1.0
        desired_yaw = 0.0 if direction > 0 else math.pi
        yaw_error = self.normalize_angle(desired_yaw - self.current_yaw)

        if self.action == Actions.TURN:
            if self.pre_turn_move(twist):
                return
            if abs(yaw_error) > self.angular_tolerance:
                twist.angular.z = self.angular_speed if yaw_error > 0 else -self.angular_speed
            else:
                self.action = Actions.MOVE
        else:
            self.move_horizontal(twist)

    # ---------------- HELPERS ---------------- #

    def pre_turn_move(self, twist: Twist):
        if not self.is_pre_turn:
            return False

        if self.pre_turn_start is None:
            self.pre_turn_start = (self.current_x, self.current_y)

        sx, sy = self.pre_turn_start
        dist = math.hypot(self.current_x - sx, self.current_y - sy)

        if dist < self.pre_turn_distance:
            twist.linear.x = self.linear_speed
            self.cmd_pub.publish(twist)
            return True
        else:
            self.pre_turn_start = None
            self.is_pre_turn = False
            return False

    def move_horizontal(self, twist: Twist):

        if self.current_lane is None:
            return False

        nearest_track_y = self.safe_y[self.current_lane]
        y_error = nearest_track_y - self.current_y

        if abs(y_error) > (self.position_tolerance / 3.0):
            direction = 1.0 if y_error > 0 else -1.0
            desired_yaw = math.pi/2 if direction > 0 else -math.pi/2
            yaw_error = self.normalize_angle(desired_yaw - self.current_yaw)

            if not self.move_error_points:
                self.move_error_points = [self.current_x, self.current_y]
                twist.angular.z = self.angular_speed if yaw_error > 0 else -self.angular_speed
            else:
                x, y = self.move_error_points
                if math.hypot(x - self.current_x, y - self.current_y) > (self.position_tolerance / 2.0):
                    self.move_error_points = None
        else:
            self.move_error_points = None

        twist.linear.x = self.linear_speed

    # ---------------- ODOM ---------------- #

    def odom_callback(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        self.current_yaw = math.atan2(siny, cosy)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2*math.pi
        while angle < -math.pi:
            angle += 2*math.pi
        return angle

    def stop(self):
        self.cmd_pub.publish(Twist())

    def get_lane_number(self):
        _, idx = min(
            ((v, i) for i, v in enumerate(self.safe_y)),
            key=lambda x: abs(x[0] - self.current_y)
        )
        return idx

# ---------------- MAIN ---------------- #

def main(args=None):
    rclpy.init(args=args)
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
