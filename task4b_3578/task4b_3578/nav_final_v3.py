#!/usr/bin/env python3

'''
# Team ID:          3578
# Theme:            Krishi coBot
# Author List:      Raghav Jibachha Mandal, Ashishkumar Rajeshkumar Jha,
#                   Aman Ratanlal Chauhan, Harshil Rahulbhai Mehta
# Filename:         ebot_nav_task3b.py
# Functions:        __init__, shape_callback, to_stop_shape_callback,
#                   publish_in_lane, control_loop, handle_stage_one,
#                   handle_stage_two, handle_stage_three, pre_turn_move,
#                   move_horizontal, odom_callback, normalize_angle,
#                   stop, get_lane_number, main
# Global variables: None
'''

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from enum import Enum, auto
from std_msgs.msg import Bool, String

class Actions(Enum):
    MOVE = auto()
    TURN = auto()


class Stage(Enum):
    ONE = auto()    # move to nearest_safe_x
    TWO = auto()    # cover y axis keeping x = nearest_safe_x
    THREE = auto()  # move to final waypoint (cover x)


class Navigate(Node):

    def __init__(self):
        super().__init__("waypoint_navigator")
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        # Subscribers
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(Bool, "/to_stop", self.to_stop_shape_callback, 10)

        self.create_subscription(String, "/detection_status", self.shape_callback, 10)

        # Waypoints [x, y]
        self.waypoints = [
            [1.1, -1.6],   # lane 0  0.4 or 0.5
            [3.5, 0.0],  # lane 1 
            [1.1, 1.6],  # lane 2
            [3.5, 0.0],  # lane 1 
            [0.0, 0.0],  # lane 1 (starting point of ebot)
        ]
        self.is_stop = False

        # Safe vertical y lines
        self.safe_y = [-1.6, 0.0, 1.6] # lane0, lane1, lane2
        self.safe_x = [0.0, 4.722] 

        self.current_lane = 1
        self.move_error_points = None
        self.next_point = 0

        # Choose nearest safe_y for first waypoint
        if self.waypoints:
            first_x = self.waypoints[0][0]
            dist0 = abs(self.safe_x[0] - first_x)
            dist1 = abs(self.safe_x[1] - first_x)
            self.nearest_safe_x = self.safe_x[0] if dist0 < dist1 else self.safe_x[1]
        else:
            self.nearest_safe_x = self.safe_x[0]
        

        # State machine
        self.action = Actions.TURN   # start by turning in Stage.ONE
        self.stage = Stage.ONE


        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # Motion params
        self.linear_speed = 1.5
        self.angular_speed = 1.5
        self.position_tolerance = 0.15
        self.angular_tolerance = 0.05
        self.pre_turn_distance = 0.25   # how much extra to move before turning (meters)
        self.pre_turn_start = None 
        self.is_pre_turn  = False

        self.create_timer(0.1, self.control_loop)

    def shape_callback(self, msg:String):
        #if msg.data:
        self.get_logger().info(msg.data)      

    def to_stop_shape_callback(self, msg):
        self.is_stop = msg.data

    def control_loop(self):

        twist = Twist()

        if self.is_stop:
            self.cmd_pub.publish(Twist())
            return


        # If no more waypoints are available → stop
        if self.next_point >= len(self.waypoints):
            self.cmd_pub.publish(twist)
            return
        target_x, target_y = self.waypoints[self.next_point]

        # Per-stage logic
        if self.stage == Stage.ONE: 
            self.handle_stage_one(twist, target_y)
        elif self.stage == Stage.TWO:
            self.handle_stage_two(twist, target_y)
        elif self.stage == Stage.THREE:
            self.handle_stage_three(twist, target_x)
        else:
       
            self.cmd_pub.publish(Twist())
            return

        self.cmd_pub.publish(twist)

    # ---------------- STAGE 1 ---------------- #
    def handle_stage_one(self, twist: Twist, target_y: float):
        '''
        Purpose:
        ---
        Handles navigation logic for Stage ONE: Moving in X-direction to the nearest_safe_x.
        It checks for completion of X or Y errors and transitions stages accordingly.

        Input Arguments:
        ---
        `twist` :  [ Twist ]
            The Twist message object to be modified for robot velocity control.

        `target_y` :  [ float ]
            The target Y coordinate for the current waypoint.

        Returns:
        ---
        None

        Example call:
        ---
        self.handle_stage_one(twist_msg, 0.35)
        '''
        """Move in Y to nearest_safe_y."""

        # X reached → go to stage THREE
        y_error = target_y - self.current_y
        if abs(y_error) <= self.position_tolerance:
            
            self.stage = Stage.THREE
            self.action = Actions.TURN
            self.is_pre_turn = True
            self.current_lane = self.get_lane_number()
            return


        x_error = self.nearest_safe_x - self.current_x
        # Already at safe_x → go to stage TWO
        if abs(x_error) <= self.position_tolerance:
  
            self.current_lane = None
            self.stage = Stage.TWO
            self.action = Actions.TURN
            self.is_pre_turn = True
            return
        
        direction = 1.0 if x_error > 0.0 else -1.0
        desired_yaw = math.pi / 2.0 if direction > 0.0 else -math.pi / 2.0
        yaw_error = self.normalize_angle(desired_yaw - self.current_yaw)

        if self.action == Actions.TURN:
            if self.pre_turn_move(twist):
                return
            # Direction along X
            if abs(yaw_error) > self.angular_tolerance:
                
                twist.angular.z = self.angular_speed if yaw_error > 0 else -self.angular_speed
            else:
          
                self.action = Actions.MOVE
        else:  
            # MOVE
            self.move_horizontal(twist)

    # ---------------- STAGE 2 ---------------- #
    def handle_stage_two(self, twist: Twist, target_y: float):
        '''
        Purpose:
        ---
        Handles navigation logic for Stage TWO: Moving along Y-axis at nearest_safe_x 
        towards the target_y.

        Input Arguments:
        ---
        `twist` :  [ Twist ]
            The Twist message object to be modified for robot velocity control.

        `target_y` :  [ float ]
            The target Y coordinate for the current waypoint.

        Returns:
        ---
        None

        Example call:
        ---
        self.handle_stage_two(twist_msg, 0.35)
        '''
        """Move along Y at nearest_safe_x to target_y."""

        y_error = target_y - self.current_y

        # X reached → go to stage THREE
        if abs(y_error) <= self.position_tolerance:
       
            self.stage = Stage.THREE
            self.action = Actions.TURN
            self.is_pre_turn = True
            self.current_lane = self.get_lane_number()
            return

        direction = 1.0 if y_error > 0.0 else -1.0
        desired_yaw = 0.0 if direction > 0.0 else math.pi
        yaw_error = self.normalize_angle(desired_yaw - self.current_yaw)

        if self.action == Actions.TURN:
            if self.pre_turn_move(twist):
                return
            if abs(yaw_error) > self.angular_tolerance:
                
                twist.angular.z = self.angular_speed if yaw_error > 0 else -self.angular_speed
            else:
            
                self.action = Actions.MOVE
        else:  
            # MOVE
            twist.linear.x = self.linear_speed
      

    # ---------------- STAGE 3 ---------------- #
    def handle_stage_three(self, twist: Twist, target_x: float):
        '''
        Purpose:
        ---
        Handles navigation logic for Stage THREE: Moving from the safe_x row to 
        the final waypoint X coordinate. Updates to next waypoint upon completion.

        Input Arguments:
        ---
        `twist` :  [ Twist ]
            The Twist message object to be modified for robot velocity control.

        `target_x` :  [ float ]
            The target X coordinate for the current waypoint.

        Returns:
        ---
        None

        Example call:
        ---
        self.handle_stage_three(twist_msg, -4.20)
        '''
        """Move from safe_x row to final waypoint X."""

        x_error = target_x - self.current_x

        # mark it reached
        if abs(x_error) <= self.position_tolerance:
          
            self.next_point += 1
            self.cmd_pub.publish(Twist())

            # Setup for next waypoint
            if self.next_point < len(self.waypoints):
                wp_x, _ = self.waypoints[self.next_point]
                dist0 = abs(self.safe_x[0] - wp_x)
                dist1 = abs(self.safe_x[1] - wp_x)
                self.nearest_safe_x = self.safe_x[0] if dist0 < dist1 else self.safe_x[1]
           
                self.stage = Stage.ONE
                self.action = Actions.TURN
                self.is_pre_turn = True
            return

        direction = 1.0 if x_error > 0.0 else -1.0
        desired_yaw = math.pi / 2.0 if direction > 0.0 else -math.pi / 2.0
        yaw_error = self.normalize_angle(desired_yaw - self.current_yaw)

        if self.action == Actions.TURN:
            if self.pre_turn_move(twist):
                return
            if abs(yaw_error) > self.angular_tolerance:
               
                twist.angular.z = self.angular_speed if yaw_error > 0 else -self.angular_speed
            else:
              
                self.action = Actions.MOVE
        else:  
            # MOVE
            self.move_horizontal(twist)

    def pre_turn_move(self, twist: Twist):
        '''
        Purpose:
        ---
        Moves the robot a small extra distance before allowing a turn.
        Checks if the robot has moved `pre_turn_distance` from the start of the turn.

        Input Arguments:
        ---
        `twist` :  [ Twist ]
            The Twist message object to be modified for robot velocity control.

        Returns:
        ---
        `result` :  [ bool ]
            Returns True if still moving (do not turn yet), False if movement complete (turn allowed).

        Example call:
        ---
        if self.pre_turn_move(twist): return
        '''
        """
        Moves the robot a small extra distance before allowing a turn.
        Returns True when extra movement is complete (turn is allowed).
        """
        if not self.is_pre_turn:
            return

        if self.pre_turn_start is None:
            # Store starting position when pre-turn starts
            self.pre_turn_start = (self.current_x, self.current_y)
            # self.get_logger().info("[PRE-TURN] Starting extra move before turn")

        start_x, start_y = self.pre_turn_start
        moved_dist = math.hypot(self.current_x - start_x,
                                 self.current_y - start_y)

        if moved_dist < self.pre_turn_distance:
            # Keep moving forward
            twist.linear.x = self.linear_speed
       
            self.cmd_pub.publish(twist)
            return True   # still moving, DO NOT turn yet
        else:
         
            self.pre_turn_start = None
            self.is_pre_turn = False
            return False    # now turning is allowed

    def move_horizontal(self, twist: Twist):
        '''
        Purpose:
        ---
        Aligns the robot horizontally while moving. Corrects angular deviation if 
        the robot drifts from the current lane's safe_y position.

        Input Arguments:
        ---
        `twist` :  [ Twist ]
            The Twist message object to be modified for robot velocity control.

        Returns:
        ---
        None

        Example call:
        ---
        self.move_horizontal(twist)
        '''
        if self.current_lane is None:
          
            return False
        
        nearest_track_y = self.safe_y[self.current_lane]
        y_error = nearest_track_y - self.current_y
        
        if abs(y_error) > (self.position_tolerance / 3.0):

            # Decide whether we need to go +X (0 rad) or -X (pi rad)
            direction = 1.0 if y_error > 0.0 else -1.0
            desired_yaw = 0.0 if direction > 0.0 else math.pi
            yaw_error = self.normalize_angle(desired_yaw - self.current_yaw)

            if not self.move_error_points:
                self.move_error_points = [self.current_x, self.current_y]

                twist.angular.z = self.angular_speed if yaw_error > 0 else -self.angular_speed
           
            else:
                x, y = self.move_error_points
                dist = math.hypot(x - self.current_x, y - self.current_y)
                if dist > (self.position_tolerance / 2.0):
                    self.move_error_points = None

        else:
            self.move_error_points = None

        twist.linear.x = self.linear_speed

    # ---------------- ODOM + HELPERS ---------------- #

    def odom_callback(self, msg: Odometry):
        """Extract position and yaw from odometry"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
            
    def normalize_angle(self, angle):
        """Keep angle in [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def stop(self):
        """Stop robot"""
        self.cmd_pub.publish(Twist())

    def get_lane_number(self):
        nearest_track_x_idx, _ = min(
                            ((v, i) for i, v in enumerate(self.safe_y)),
                            key=lambda y: abs(y[0] - self.current_y)
                        )
        return nearest_track_x_idx
    

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