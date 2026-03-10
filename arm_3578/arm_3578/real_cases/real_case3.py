#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
* Team Id :          3578
* Author List :      Raghav Jibachha Mandal, Ashishkumar Rajeshkumar Jha,
* Filename:          task6_arm_manipulation.py
* Theme:             Krishi coBot -- eYRC 2025-26
* Functions:         MATH UTILITIES  : [euler_to_quaternion, normalize_quaternion, conjugate_quaternion,
*                    multiply_quaternion, quaternion_to_euler,]
*                    Task6.__init__, Task6.joint_state_callback, Task6.tcp_pose_callback,
*                    Task6.force_callback, Task6.ebot_status_callback,
*                    Task6.set_gripper_state, Task6.stop_joint, Task6.publish_twist,
*                    Task6.stop_all, Task6.move_to_tcp_target, Task6.orient_to_target,
*                    Task6.move_joint_to_angle, Task6.move_joint_group,
*                    Task6.align_joint_to_pose, Task6.lookup_tf,
*                    Task6.scan_for_bad_fruit_frames, Task6.wait_for_timer,
*                    Task6.norm, Task6.main_loop, main
* Global Variables:  None
'''

import rclpy
from rclpy.node import Node
import numpy as np
import time
# Interface Imports for ROS2 messages and services
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray, Float32, Bool
from control_msgs.msg import JointJog
from std_srvs.srv import SetBool
from sensor_msgs.msg import JointState
from rclpy.duration import Duration
import tf2_ros
from rclpy.callback_groups import ReentrantCallbackGroup

# ==================================================================
# --------------------- MATH UTILITIES -----------------------------
# ==================================================================

'''
* Function Name: euler_to_quaternion
* Input:         roll  -> float, rotation about X-axis in radians
*                pitch -> float, rotation about Y-axis in radians
*                yaw   -> float, rotation about Z-axis in radians
* Output:        numpy array of shape (4,) -> [qx, qy, qz, qw] quaternion representation
* Logic:         Converts Euler angles (roll, pitch, yaw) to a quaternion using the
*                standard trigonometric formulation. The resulting quaternion represents
*                the same 3D orientation as the input Euler angles.
* Example Call:  q = euler_to_quaternion(0.0, 0.0, 1.57)
'''
def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return np.array([qx, qy, qz, qw])

'''
* Function Name: normalize_quaternion
* Input:         quaternion -> array-like of 4 floats [qx, qy, qz, qw]
* Output:        numpy array of shape (4,) -> unit quaternion (magnitude = 1)
* Logic:         Divides each component by the quaternion's magnitude (L2 norm).
*                If the magnitude is zero, returns the original array unchanged
*                to avoid division by zero.
* Example Call:  unit_q = normalize_quaternion([0.5, 0.5, 0.5, 0.5])
'''
def normalize_quaternion(quaternion):
    quaternion_array = np.array(quaternion, dtype=float)
    magnitude = np.linalg.norm(quaternion_array)
    if magnitude > 0.0:
        return quaternion_array / magnitude
    else:
        return quaternion_array

'''
* Function Name: conjugate_quaternion
* Input:         quaternion -> array-like of 4 floats [qx, qy, qz, qw]
* Output:        numpy array of shape (4,) -> conjugate quaternion [-qx, -qy, -qz, qw]
* Logic:         Negates the vector (x, y, z) components while keeping the scalar (w)
*                component unchanged. For a unit quaternion, the conjugate equals
*                the inverse and represents the opposite rotation.
* Example Call:  q_inv = conjugate_quaternion([0.0, 0.0, 0.707, 0.707])
'''
def conjugate_quaternion(quaternion):
    x_value, y_value, z_value, w_value = quaternion
    return np.array([-x_value, -y_value, -z_value, w_value], dtype=float)

'''
* Function Name: multiply_quaternion
* Input:         first  -> array-like of 4 floats [qx, qy, qz, qw], first quaternion
*                second -> array-like of 4 floats [qx, qy, qz, qw], second quaternion
* Output:        numpy array of shape (4,) -> product quaternion (first * second)
* Logic:         Applies the Hamilton product formula for quaternion multiplication.
*                The result represents the combined rotation of first followed by second.
*                Note: quaternion multiplication is NOT commutative.
* Example Call:  q_combined = multiply_quaternion(q1, q2)
'''
def multiply_quaternion(first, second):
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    x_out = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_out = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_out = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w_out = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x_out, y_out, z_out, w_out], dtype=float)

'''
* Function Name: quaternion_to_euler
* Input:         quaternion -> array-like of 4 floats [qx, qy, qz, qw]
* Output:        tuple of 3 floats -> (roll_angle, pitch_angle, yaw_angle) in radians
* Logic:         Converts a quaternion to Euler angles (roll-pitch-yaw convention).
*                Uses arctan2 for roll and yaw to handle the full 2*pi range.
*                Clamps the sin_pitch value to [-1, 1] to handle numerical singularities
*                (gimbal lock) before applying arcsin for pitch.
* Example Call:  roll, pitch, yaw = quaternion_to_euler([0.0, 0.0, 0.707, 0.707])
'''
def quaternion_to_euler(quaternion):
    x_value, y_value, z_value, w_value = quaternion

    # Roll (x-axis rotation)
    sin_roll_cos_pitch = 2.0 * (w_value * x_value + y_value * z_value)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x_value * x_value + y_value * y_value)
    roll_angle = np.arctan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    # Pitch (y-axis rotation)
    sin_pitch = 2.0 * (w_value * y_value - z_value * x_value)
    # Handle singularity cases (gimbal lock) by clamping to [-1, 1]
    if sin_pitch < -1.0:
        sin_pitch = -1.0
    if sin_pitch > 1.0:
        sin_pitch = 1.0
    pitch_angle = np.arcsin(sin_pitch)

    # Yaw (z-axis rotation)
    sin_yaw_cos_pitch = 2.0 * (w_value * z_value + x_value * y_value)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y_value * y_value + z_value * z_value)
    yaw_angle = np.arctan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)

    return roll_angle, pitch_angle, yaw_angle


class Task6(Node):

    '''
    * Function Name: __init__
    * Input:         None (self)
    * Output:        None
    * Logic:         Initializes the Task6 ROS2 node. Sets up all publishers, subscribers,
    *                service clients, TF listener, timers, and state variables required
    *                for the full fertilizer pick-and-place and bad-fruit sorting task.
    *                The main control loop runs at 50 Hz (0.02s timer).
    * Example Call:  node = Task6()
    '''
    def __init__(self):
        super().__init__('Task6')

        self.service_callback_group = ReentrantCallbackGroup()

        # ===================publishers========================

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)

        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)
        self.fertilizer_status_publisher = self.create_publisher(Bool, '/fertilizer_placement_status', 10)

        # joint_names_list: ordered list of all 6 UR5 joint names used for velocity commands
        self.joint_names_list = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        # ===================subscribers========================
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Subscribe to TCP pose published as Float64MultiArray [x, y, z, rx, ry, rz]
        self.tcp_pose_sub = self.create_subscription(
            Float64MultiArray,
            '/tcp_pose_raw',
            self.tcp_pose_callback,
            10
        )

        # Subscribe to net wrench (force sensor) to detect contact
        self.force_sub = self.create_subscription(
            Float32,
            '/net_wrench',
            self.force_callback,
            10
        )

        # Service client for electromagnet control (SetBool: True=ON, False=OFF)
        self.magnet_client = self.create_client(
                SetBool,
                '/magnet',
                callback_group=self.service_callback_group
            )

        self.ebot_docked = False
        self.ebot_dock_subscriber = self.create_subscription(
            Bool,
            '/ebot_dock_status',
            self.ebot_status_callback,
            10,
            callback_group=self.service_callback_group
        )

        # Main control loop timer running at 50 Hz
        self.timer = self.create_timer(
                        0.02,
                        self.main_loop,
                        callback_group=self.service_callback_group
                    )

        # ===================configurations========================

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # teamIdentifier: unique 4-digit team identifier used to prefix TF frame names
        self.teamIdentifier = '3578'
        self.base_link_name = 'base_link'

        # joint_pos: dictionary mapping joint name -> current angle in radians
        self.joint_pos = {}

        # tcp_pose variables — all None until first message is received
        self.current_tcp_pos = None
        self.current_tcp_orient = None
        self.current_euler = None
        self.initial_arm_pos = None
        self.ferti_align_joint_state = None
        self.fruits_tray_hover_pos = None
        self.ferti_unload_pose = None

        # max_tol: joint arrival tolerance in radians (~3 degrees)
        self.max_tol = np.deg2rad(3)
        # base_max_speed: maximum linear TCP speed in m/s
        self.base_max_speed = 0.5
        # base_max_angular: maximum angular speed in rad/s

        self.base_max_angular = 1.5    #   i am not yet this because when i try to arm iny particular orenation the get unusal behaviour so i have t check further to wite a def to used this

        # phase: string label for the current state in the finite state machine
        self.phase = 'START'

        self.ferti_pose = None
        # fertilizerTFname: TF frame name of the fertilizer object in the simulation
        self.fertilizerTFname = f'{self.teamIdentifier}_fertilizer_1'
        # ebotTransformName: TF frame name of the eBot's ArUco marker
        self.ebotTransformName = f'{self.teamIdentifier}_ebot_marker'
        # pickupFertiOrientationQuaternion: target end-effector orientation for fertilizer pickup
        self.pickupFertiOrientationQuaternion = np.array([0.707, 0.028, 0.034, 0.707])
        # { i  am not able to used this self.pickupFertiOrientationQuaternion  because my orenation function behave unusal
        # so is used joint state for particular orenation}

        # ebotWorldPosition: 3D position [x, y, z] of the eBot in the world frame
        self.ebotWorldPosition = None

        # badFruitTable: list of dicts {tf_name, pos} for all detected bad fruit frames
        self.badFruitTable = []
        # badFruitFrameList: TF frame names of the two bad fruits to be removed
        self.badFruitFrameList = [
            f'{self.teamIdentifier}_bad_fruit_1',
            f'{self.teamIdentifier}_bad_fruit_2',
            f'{self.teamIdentifier}_bad_fruit_3',
        ]

        # fruitHomePosition: fixed 3D hover position above the fruits tray [x, y, z] in meters
        self.fruitHomePosition = np.array([-0.159, 0.501, 0.415])

        # dustbinPosition: fixed 3D position of the dustbin [x, y, z] in meters
        self.dustbinPosition = np.array([-0.806, 0.010, 0.182])

        self.phase_initialized = False
        self.current_fruits_pose = None
        self.wait_start_time = None
        # wrist1_delta_down: wrist_1_joint angle offset (~90 deg) to orient gripper downward
        self.wrist1_delta_down = 1.36
        # current_fruit_index: index into badFruitTable tracking which fruit is being processed
        self.current_fruit_index = 0
        # current_force_z: latest force reading from the net_wrench topic (Newtons)
        self.current_force_z = 0.0

        self.dustbin_hover_pose = None

# ====================callbacks===============================================

    '''
    * Function Name: joint_state_callback
    * Input:         msg -> sensor_msgs/JointState, contains joint names and positions
    * Output:        None (updates self.joint_pos dictionary)
    * Logic:         Iterates through the received joint names and positions,
    *                storing each joint's current angle in the joint_pos dictionary
    *                for use by motion control functions.
    * Example Call:  Called automatically by ROS2 subscription to /joint_states
    '''
    def joint_state_callback(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.joint_pos[n] = p

# --------------------------------------------------------------------------

    '''
    * Function Name: tcp_pose_callback
    * Input:         msg -> std_msgs/Float64MultiArray, data = [x, y, z, roll, pitch, yaw]
    * Output:        None (updates self.current_tcp_pos and self.current_tcp_orient)
    * Logic:         Extracts the 3D position [x, y, z] and Euler angles [roll, pitch, yaw]
    *                from the flat array. Converts the Euler angles to a quaternion and
    *                stores both for use by position and orientation controllers.
    * Example Call:  Called automatically by ROS2 subscription to /tcp_pose_raw
    '''
    def tcp_pose_callback(self, msg):
        # Validate that the message contains all 6 required fields
        if len(msg.data) >= 6:
            self.current_tcp_pos = np.array([msg.data[0], msg.data[1], msg.data[2]])

            # Convert Euler to Quaternion for orientation control logic
            roll = msg.data[3]
            pitch = msg.data[4]
            yaw = msg.data[5]
            self.current_euler = (roll ,pitch ,yaw)
            self.current_tcp_orient = euler_to_quaternion(roll, pitch, yaw)

    '''
    * Function Name: force_callback
    * Input:         msg -> std_msgs/Float32, net wrench magnitude from the force sensor
    * Output:        None (updates self.current_force_z)
    * Logic:         Stores the latest force sensor reading for use in contact detection
    *                during pick-and-place operations.
    * Example Call:  Called automatically by ROS2 subscription to /net_wrench
    '''
    def force_callback(self, msg):
        self.current_force_z = msg.data

# -------------------------------------------------------------------------

    '''
    * Function Name: ebot_status_callback
    * Input:         msg -> std_msgs/Bool, True when eBot has docked at the station
    * Output:        None (updates self.ebot_docked flag)
    * Logic:         Sets the ebot_docked flag to True on the first rising edge of the
    *                dock status signal, and logs a confirmation message.
    * Example Call:  Called automatically by ROS2 subscription to /ebot_dock_status
    '''
    def ebot_status_callback(self, msg):
        if msg.data:
            if not self.ebot_docked:
                self.get_logger().info("✓ eBot has reached the dock station!")
                self.ebot_docked = True

# ==================Attach/Detach=========================================

    '''
    * Function Name: set_gripper_state
    * Input:         action -> str, either 'attach' to energize magnet or 'detach' to release
    * Output:        bool -> True if the command was sent successfully, False if blocked
    * Logic:         For 'attach': checks current force reading against a threshold (3.0 N)
    *                to confirm contact before energizing the electromagnet.
    *                For 'detach': immediately sends the OFF command without any check.
    *                Sends the SetBool service request asynchronously.
    * Example Call:  success = self.set_gripper_state('attach')
    '''
    def set_gripper_state(self, action):
        req = SetBool.Request()

        if action == 'attach':
            if self.current_force_z is None:
                self.get_logger().warn("Force sensor data None.")
                return False

            # Only energize magnet if force reading confirms physical contact
            if self.current_force_z > 25.0:
                req.data = True
                self.get_logger().info(f"Force Good ({self.current_force_z:.2f} > 25). Magnet ON.")
                self.magnet_client.call_async(req)
                return True
            else:
                self.get_logger().warn(f"Force Low ({self.current_force_z:.2f}). Waiting...")
                return False
        else:
            req.data = False
            self.get_logger().info("Magnet OFF.")
            self.magnet_client.call_async(req)
            return True

# =================  MOTION_COMMAND_FUNCTION  ===================================================================

    '''
    * Function Name: stop_joint
    * Input:         None (self)
    * Output:        None
    * Logic:         Publishes a JointJog message with zero velocities for all 6 joints,
    *                bringing any active joint motion to an immediate stop.
    * Example Call:  self.stop_joint()
    '''"leetcode"
    def stop_joint(self):
        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities = [0.0] * 6
        self.joint_pub.publish(msg)

    '''
    * Function Name: publish_twist
    * Input:         linear_vel  -> array-like of 3 floats [vx, vy, vz] in m/s
    *                angular_vel -> array-like of 3 floats [wx, wy, wz] in rad/s (optional)
    * Output:        None
    * Logic:         Constructs and publishes a TwistStamped message to the servo topic.
    *                The header is stamped with the current ROS time and frame set to
    *                'base_link'. Angular velocity is only applied when provided.
    * Example Call:  self.publish_twist([0.1, 0.0, 0.0], [0.0, 0.0, 0.0])
    '''
    def publish_twist(self, linear_vel, angular_vel=None):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Linear Velocity (Expects [x, y, z] array)
        msg.twist.linear.x = float(linear_vel[0])
        msg.twist.linear.y = float(linear_vel[1])
        msg.twist.linear.z = float(linear_vel[2])

        # Angular Velocity (Expects [rx, ry, rz] array)
        if angular_vel is not None:
            msg.twist.angular.x = float(angular_vel[0])
            msg.twist.angular.y = float(angular_vel[1])
            msg.twist.angular.z = float(angular_vel[2])

        self.twist_pub.publish(msg)

    '''
    * Function Name: stop_all
    * Input:         None (self)
    * Output:        None
    * Logic:         Calls both stop_joint() and publish_twist() with zero vectors to
    *                halt all joint and Cartesian motion simultaneously.
    * Example Call:  self.stop_all()
    '''
    def stop_all(self):
        self.stop_joint()
        self.publish_twist([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

# =============================DELTA_TWIST_CMD===================================================

    '''
    * Function Name: move_to_tcp_target
    * Input:         target -> numpy array [x, y, z], desired TCP position in meters
    *                tol    -> float, arrival tolerance in meters (default 0.01)
    *                slow   -> bool, use reduced speed profile when True (default False)
    * Output:        bool -> True when TCP is within tol of target, False while moving
    * Logic:         Distance-adaptive P-controller. Computes the error vector between
    *                target and current TCP position. Selects gain (kp) and speed cap (max_s)
    *                based on distance zones. Speed tapers naturally to zero near the target
    *                with no hard minimum, allowing smooth deceleration.
    *                Force/contact logic is intentionally excluded — handle in calling phase.
    * Example Call:  reached = self.move_to_tcp_target(np.array([0.3, 0.2, 0.4]), tol=0.01)
    '''
    def move_to_tcp_target(self, target, tol=0.01, slow=False):
        if self.current_tcp_pos is None:
            return False

        err_vec = target - self.current_tcp_pos
        dist    = np.linalg.norm(err_vec)

        # --- Arrival check ---
        if dist < tol:
            self.stop_all()
            self.get_logger().info(f"✓ TCP reached. dist={dist*1000:.1f}mm")
            return True

        # --- Speed zones: tighter limits in slow mode to protect against collisions ---
        if slow:
            if dist > 0.10:
                kp, max_s = 1.5, 0.12
            elif dist > 0.03:
                kp, max_s = 1.2, 0.06
            else:
                # Very close: proportional only, NO floor → creeps to stop
                kp, max_s = 1.0, 0.01
        else:
            if dist > 0.20:
                kp, max_s = 2.0, 0.40
            elif dist > 0.10:
                kp, max_s = 1.8, 0.20
            elif dist > 0.03:
                kp, max_s = 1.2, 0.08
            else:
                kp, max_s = 1.0, 0.01

        speed      = min(kp * dist, max_s)
        direction  = err_vec / dist
        linear_vel = direction * speed

        self.get_logger().info(
            f"TCP move | dist={dist:.3f}m | spd={speed:.3f} | slow={slow}",
            throttle_duration_sec=0.4
        )
        self.publish_twist(linear_vel, [0.0, 0.0, 0.0])
        return False
# --------------------------------------------------------------------------------------------------------------------------------
    '''
    * Function Name: orient_to_target
    * Input:         target_quat -> numpy array [qx, qy, qz, qw], desired end-effector orientation
    *                tol         -> float, tolerance on quaternion xyz-part magnitude (default 0.05)
    *                              (tol=0.05 ≈ 5.7 deg; use 0.03 for ~3.4 deg)
    * Output:        bool -> True when orientation is within tol, False while rotating
    * Logic:         Computes the error quaternion q_err = q_target * inv(q_current).
    *                The xyz part of q_err gives the rotation axis scaled by sin(half_angle).
    *                A P-controller with distance-based gain zones drives angular velocity.
    *                Speed tapers naturally to zero — no hard floor prevents overshoot.
    * Example Call:  aligned = self.orient_to_target(np.array([0.0, 0.0, 0.707, 0.707]))
    '''
#  i never test this but i used in the future for any particular orenation brfore pickup and place any object

    def orient_to_target(self, target_roll, target_pitch, target_yaw, tol=0.05):
        """
        * Function Name: orient_to_target
        * Input:         target_roll  -> float, desired roll  in radians
        *                target_pitch -> float, desired pitch in radians
        *                target_yaw   -> float, desired yaw   in radians
        *                tol          -> float, tolerance on quaternion xyz-part magnitude (default 0.05)
        * Output:        bool -> True when orientation is within tol, False while rotating
        * Logic:         Converts input Euler angles to quaternion internally, then computes
        *                the error quaternion q_err = q_target * inv(q_current).
        *                The xyz part of q_err gives the rotation axis scaled by sin(half_angle).
        *                A P-controller with distance-based gain zones drives angular velocity
        *                published to /delta_twist_cmds via publish_twist.
        * Example Call:  aligned = self.orient_to_target(0.0, 0.0, 1.57)
        """
        if self.current_tcp_orient is None:
            return False

        # --- Convert input Euler angles to quaternion ---
        target_quat = euler_to_quaternion(target_roll, target_pitch, target_yaw)
        target_quat = normalize_quaternion(target_quat)

        # --- Log what we're targeting (throttled) ---
        self.get_logger().info(
            f"Orient target | R={np.degrees(target_roll):.1f}° "
            f"P={np.degrees(target_pitch):.1f}° "
            f"Y={np.degrees(target_yaw):.1f}°",
            throttle_duration_sec=1.0
        )

        # --- Error quaternion: q_err = q_target * inv(q_current) ---
        q_curr_inv = conjugate_quaternion(self.current_tcp_orient)
        q_err      = multiply_quaternion(target_quat, q_curr_inv)

        # Enforce shortest-path rotation by flipping if scalar (w) is negative
        if q_err[3] < 0:
            q_err = -q_err

        # xyz part of q_err = rotation_axis * sin(half_angle)
        xyz_err   = q_err[:3]
        error_mag = np.linalg.norm(xyz_err)
        deg_err   = np.degrees(2.0 * np.arcsin(np.clip(error_mag, 0, 1)))

        # --- Arrival check ---
        if error_mag < tol:
            self.stop_all()
            self.get_logger().info(f"✓ Orientation aligned. err≈{deg_err:.1f}°")
            return True

        # --- Zone-based P-controller ---
        if error_mag > 0.3:
            kp_rot, max_rot = 3.0, 0.8
        elif error_mag > 0.1:
            kp_rot, max_rot = 2.5, 0.4
        else:
            # Close to target: natural deceleration, no speed floor
            kp_rot, max_rot = 2.0, 0.15

        ang_speed = min(kp_rot * error_mag, max_rot)

        # Normalize to get unit rotation axis; guard against near-zero magnitude
        ang_axis    = xyz_err / error_mag if error_mag > 1e-6 else np.zeros(3)
        angular_vel = ang_axis * ang_speed

        self.get_logger().info(
            f"Orienting | err≈{deg_err:.1f}° | spd={ang_speed:.3f} "
            f"| axis=[{ang_axis[0]:.2f},{ang_axis[1]:.2f},{ang_axis[2]:.2f}]",
            throttle_duration_sec=0.4
        )

        # --- Publish pure rotation to /delta_twist_cmds ---
        self.publish_twist([0.0, 0.0, 0.0], angular_vel)
        return False

# ============================ JOINT DELTA CMD =====================================================

    '''
    * Function Name: move_joint_to_angle
    * Input:         target_angle -> float, desired joint angle in radians
    *                joint_name   -> str, name of the joint to move
    *                joint_index  -> int, index of the joint in joint_names_list (0-5)
    *                tol          -> float, arrival tolerance in radians (default 0.02)
    * Output:        bool -> True when joint is within tol of target, False while moving
    * Logic:         Reads the current joint angle from joint_pos, computes the normalized
    *                angular error, and applies a zone-based P-controller. Speed tapers
    *                naturally to zero near the target to prevent overshoot.
    * Example Call:  done = self.move_joint_to_angle(1.57, 'wrist_1_joint', 3)
    '''
    def move_joint_to_angle(self, target_angle, joint_name, joint_index, tol=0.02):
        if joint_name not in self.joint_pos:
            return False

        current = self.joint_pos[joint_name]
        err     = self.norm(target_angle - current)
        abs_err = abs(err)

        if abs_err < tol:
            self.stop_joint()
            self.get_logger().info(f"✓ Joint {joint_name} at target. err={np.degrees(err):.2f}°")
            return True

        # Zone-based speed control — no hard floor to avoid overshoot near target
        if abs_err > 1.5:
            kp, max_s = 2.0, 1.0
        elif abs_err > 0.5:
            kp, max_s = 1.5, 0.5
        elif abs_err > 0.17:
            kp, max_s = 1.0, 0.15
        else:
            # Natural deceleration zone: very close to target
            kp, max_s = 0.8, 0.05

        speed = kp * err                        # signed velocity (direction encoded)
        speed = max(min(speed, max_s), -max_s)  # clip magnitude while preserving sign

        self.get_logger().info(
            f"Joint {joint_name} | err={np.degrees(err):.1f}° | cmd={speed:.3f}",
            throttle_duration_sec=0.4
        )

        cmd = [0.0] * 6
        cmd[joint_index] = float(speed)

        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities  = cmd
        self.joint_pub.publish(msg)
        return False

# -----------------------------------------------------------------------------------------------------------------------------------

    '''
    * Function Name: move_joint_group
    * Input:         targets     -> dict {joint_name: target_angle_rad}, joints to move
    *                speed_scale -> dict {joint_name: float}, per-joint speed multiplier
    * Output:        bool -> True when ALL joints have reached their targets, False otherwise
    * Logic:         Iterates over all target joints, computes per-joint normalized error,
    *                and applies zone-based P-control with the provided speed scale.
    *                Joints within max_tol get zero velocity. All velocities are published
    *                in a single JointJog message for simultaneous motion.
    * Example Call:  done = self.move_joint_group({'shoulder_pan_joint': 0.5}, {'shoulder_pan_joint': 1.0})
    '''
    def move_joint_group(self, targets, speed_scale):
        if not self.joint_pos:
            return False

        # Mapping from joint name to its index in the velocity command array
        joint_map = {
            'shoulder_pan_joint': 0, 'shoulder_lift_joint': 1, 'elbow_joint': 2,
            'wrist_1_joint': 3,      'wrist_2_joint': 4,       'wrist_3_joint': 5,
        }

        cmd         = [0.0] * 6
        all_reached = True
        max_err_dbg = 0.0

        for joint, target in targets.items():
            if joint not in self.joint_pos:
                all_reached = False
                continue

            idx     = joint_map[joint]
            err     = self.norm(target - self.joint_pos[joint])
            abs_err = abs(err)

            if abs_err > max_err_dbg:
                max_err_dbg = abs_err

            # Joint is within tolerance — no velocity needed
            if abs_err < self.max_tol:
                cmd[idx] = 0.0
                continue

            all_reached = False

            # Zone-based speed control — no hard floor
            if abs_err > 1.5:
                kp, local_max = 2.0, 1.0
            elif abs_err > 0.5:
                kp, local_max = 1.5, 0.5
            elif abs_err > 0.17:
                kp, local_max = 1.0, 0.15
            else:
                kp, local_max = 1.0, 0.1

            speed    = kp * err * speed_scale.get(joint, 1.0)
            speed    = max(min(speed, local_max), -local_max)
            cmd[idx] = float(speed)

        if not all_reached:
            self.get_logger().info(
                f"Group move | max_err={np.degrees(max_err_dbg):.1f}°",
                throttle_duration_sec=0.4
            )

        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities  = cmd
        self.joint_pub.publish(msg)

        if all_reached:
            self.stop_all()
            self.get_logger().info("✓ Joint group reached.")
        return all_reached
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    * Function Name: align_joint_to_pose
    * Input:         target_pose  -> numpy array [x, y, z], 3D world position to align toward
    *                target_label -> str, human-readable label for log messages
    *                joint_name   -> str, name of the joint to rotate
    *                joint_index  -> int, index of the joint in joint_names_list
    * Output:        bool -> True when the joint is aligned to the computed angle, False otherwise
    * Logic:         Computes the desired joint angle using arctan2 on the X-Y plane of
    *                target_pose, then delegates motion to move_joint_to_angle.
    *                Adds pi to arctan2 result to account for arm orientation convention.
    * Example Call:  done = self.align_joint_to_pose(ferti_pos, 'fertilizer', 'shoulder_pan_joint', 0)
    '''
    def align_joint_to_pose(self, target_pose, target_label, joint_name, joint_index):
        if target_pose is None:
            return False

        # Compute the angle from the base origin to the XY projection of the target
        x = target_pose[0]
        y = target_pose[1]
        desired_angle = self.norm(np.arctan2(y, x) + np.pi)

        # Delegate motion to the single-joint controller
        success = self.move_joint_to_angle(desired_angle, joint_name, joint_index)

        if success:
            self.get_logger().info(f"✓ Aligned {target_label} (Joint {joint_index})")

        return success

# =====================================extra function =============================================================================

    '''
    * Function Name: lookup_tf
    * Input:         target -> str, name of the target TF frame
    *                source -> str, name of the source TF frame
    * Output:        numpy array [x, y, z] if transform found, None on failure
    * Logic:         Queries the TF buffer for the latest transform from source to target
    *                with a 0.5-second timeout. Returns the translation component as a
    *                numpy array. Returns None silently if the lookup fails (e.g., TF not yet available).
    * Example Call:  pos = self.lookup_tf('base_link', '3578_fertilizer_1')
    '''
    def lookup_tf(self, target, source):
        try:
            tf = self.tf_buffer.lookup_transform(
                target, source, rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )
            t = tf.transform.translation
            return np.array([t.x, t.y, t.z])
        except Exception:
            return None

    '''
    * Function Name: scan_for_bad_fruit_frames
    * Input:         None (self)
    * Output:        list of dicts [{'tf_name': str, 'pos': numpy array}] if ALL fruits found,
    *                None if any frame is not yet available in TF
    * Logic:         Iterates over badFruitFrameList and attempts a TF lookup for each.
    *                Returns the complete list only when every frame has been successfully
    *                located. If any single lookup fails, returns None to trigger a retry.
    * Example Call:  records = self.scan_for_bad_fruit_frames()
    '''
    def scan_for_bad_fruit_frames(self):
        found_records = []
        all_detected = True

        for frame_name in self.badFruitFrameList:
            position = self.lookup_tf(self.base_link_name, frame_name)
            if position is not None:
                found_records.append({'tf_name': frame_name, 'pos': position})
            else:
                all_detected = False
                break

        if all_detected:
            return found_records
        else:
            return None

    '''
    * Function Name: wait_for_timer
    * Input:         seconds -> float, duration to wait in seconds
    * Output:        bool -> True when the specified duration has elapsed, False otherwise
    * Logic:         Non-blocking timer implemented using ROS clock. On the first call,
    *                records the start time. On subsequent calls, checks elapsed nanoseconds
    *                converted to seconds. Resets the timer on completion so it can be reused.
    * Example Call:  if self.wait_for_timer(2.0): self.phase = 'NEXT_PHASE'
    '''
    def wait_for_timer(self, seconds):
        # Start the timer on the first call in this wait cycle
        if self.wait_start_time is None:
            self.wait_start_time = self.get_clock().now()
            self.get_logger().info(f" Starting {seconds}s wait...")
            return False

        # Compute elapsed time in seconds from nanosecond difference
        current_time = self.get_clock().now()
        time_diff = (current_time - self.wait_start_time).nanoseconds / 1e9

        # Log progress while still waiting (throttled to avoid terminal spam)
        if time_diff < seconds:
            self.get_logger().info(
                f"Waiting... ({time_diff:.1f}/{seconds:.1f}s)",
                throttle_duration_sec=0.5
            )
            return False

        # Reset timer so it can be reused in the next wait call
        self.wait_start_time = None
        self.get_logger().info("Wait Complete.")
        return True

    '''
    * Function Name: norm
    * Input:         a -> float, angle in radians (any value)
    * Output:        float -> angle wrapped to the range (-pi, pi]
    * Logic:         Repeatedly adds or subtracts 2*pi until the angle falls within
    *                the standard range. Used to compute minimal angular errors for
    *                joint control.
    * Example Call:  wrapped = Task6.norm(4.0)   # returns ~-2.28
    '''
    @staticmethod
    def norm(a):
        while a > np.pi:
            a -= 2 * np.pi
        while a < -np.pi:
            a += 2 * np.pi
        return a

# =============================main loop======================================================================

    '''
    * Function Name: main_loop
    * Input:         None (self)
    * Output:        None
    * Logic:         Central finite state machine (FSM) that runs at 50 Hz via ROS timer.
    *                Each iteration checks self.phase and executes the corresponding action.
    *                Phases cover: initialization, TF acquisition, fertilizer pick-and-place,
    *                eBot interaction, bad-fruit sorting to dustbin, and fertilizer unloading.
    *                State transitions are triggered by motion completion, timer expiry,
    *                or sensor conditions (force, TF availability, dock status).
    *In simle world  (aling base => move clouser => check orenation => with use of net wench and pose grab object )
    * Example Call:  Called automatically by self.timer at 50 Hz
    '''
    def main_loop(self):
        if self.phase == 'START':
            self.get_logger().info("phase machine in the gazebo is start ")
            self.phase = 'PHASE_GETTING_TF'

        elif self.phase == 'PHASE_GETTING_TF':
            # Wait for both Joint States AND TCP Pose before proceeding
            if self.initial_arm_pos is None:
                if 'shoulder_pan_joint' not in self.joint_pos:
                    self.get_logger().info("Waiting for joint_states...", throttle_duration_sec=2.0)
                    return
                if self.current_tcp_pos is None:
                    self.get_logger().info("Waiting for TCP pose...", throttle_duration_sec=2.0)
                    return

                self.initial_arm_pos = self.joint_pos.copy()

                self.initial_cartesian_pos = self.current_tcp_pos.copy()
                self.initial_cartesian_orient = self.current_tcp_orient.copy()

                self.get_logger().info(f"✓ Stored Initial Pose: {self.initial_cartesian_pos}  euler angel {self.current_euler}")

            if self.ferti_pose is None:
                self.ferti_pose = self.lookup_tf(self.base_link_name, self.fertilizerTFname)

            if self.ferti_pose is None:
                self.get_logger().info("Waiting for fertilizer TF...", throttle_duration_sec=2.0)
                return

            if not self.badFruitTable:
                fruit_records = self.scan_for_bad_fruit_frames()
                if fruit_records:
                    self.badFruitTable = fruit_records
                    self.get_logger().info(f"✓ Found all {len(fruit_records)} bad fruits")
                else:
                    self.get_logger().info("Scanning for bad fruits...", throttle_duration_sec=2.0)
                    return
            self.get_logger().info("✓ All TFs acquired. Starting Phase 2.")
            self.phase = 'PHASE_ALIGN_TO_FERTI'

# ----------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'PHASE_ALIGN_TO_FERTI':
            if self.align_joint_to_pose(self.ferti_pose, 'shoulder', self.joint_names_list[0], 0):
                self.get_logger().info("Aligned shoulder to fertilizer. Transitioning to PRE_APPROACH.")
                self.phase = 'ALING_WAIT_FERTI'

# ---------------------------------------------------------------------------------------------------------------------
    # this type of timer i used for the settel down  in differnt place in code
        elif self.phase == 'ALING_WAIT_FERTI':
            if self.wait_for_timer(2.0):
                self.phase = 'PHASE_PRE_APPROACH'

# ----------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_PRE_APPROACH':
            # Approach a position 15 cm in front of the fertilizer (along Y-axis)
            target_pre = self.ferti_pose.copy()
            target_pre[1] += 0.15

            reached = self.move_to_tcp_target(target_pre, tol=0.01, slow=False)

            if reached:
                self.get_logger().info(" Reached +0.15 offset. now ckeck orenation first.euler  current  {self.current_euler}")
                self.phase = 'PREAPPROACH_WAIT_FERTI'

# ------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PREAPPROACH_WAIT_FERTI':
            if self.wait_for_timer(1.0):
                self.phase = 'MOVE_ORENATION_REQUIRED'

# ------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'MOVE_ORENATION_REQUIRED':
            # Adjust wrist joints to orient the gripper correctly for fertilizer pickup
            #  acuatlly  i used quatration angle to orenation  but not working so i find out that on adjust of wrist we got proper orenation so better to pickup
            targets = {
                'wrist_1_joint': -0.110,
                'wrist_2_joint': 1.95,
            }

            speed_scale = {
                'wrist_1_joint': 0.5,
                'wrist_2_joint': 0.5,
            }

            if self.move_joint_group(targets, speed_scale):
                self.get_logger().info(f" orentaion wala hai   {self.current_tcp_orient} euler {self.current_euler}, pose current {self.current_tcp_pos} , frti {self.ferti_pose}")
                self.phase = 'PHASE_FINAL_APPROACH_WAIT'


        elif self.phase == 'NEW_ORIENTATION_METHOD':
            if self.orient_to_target(np.pi/2,0,np.pi/2):
                self.phase = 'NEXT_PHASE'

# -----------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'PHASE_FINAL_APPROACH_WAIT':
            if self.wait_for_timer(2.0):
                self.get_logger().info("Fertilizer MAgnet start here ")
                self.set_gripper_state('attach')
                self.phase = 'PHASE_FINAL_APPROACH'

# -------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'PHASE_FINAL_APPROACH':
            self.get_logger().info(f"current magnet force  ({self.current_force_z:.2f} )")

            # Fine-tuned final target position with small X/Y offsets to center on fertilizer
            final_ferti_target = self.ferti_pose.copy()
            final_ferti_target[0] += 0.055
            final_ferti_target[1] -= 0.03
            final_ferti_target[2] -= 0.01


            reached = self.move_to_tcp_target(final_ferti_target, tol=0.001, slow=True)

            # Transition on position arrival OR force contact (whichever comes first)
            if reached or (self.current_force_z > 25.0):
                self.stop_all()
                self.get_logger().info(f"{self.current_tcp_orient} ,euler {self.current_euler} pose current {self.current_tcp_pos} , frti {self.ferti_pose}")
                self.get_logger().info("Reached fertilizer hover position. Waiting before attach...")
                self.phase = 'ATTACH_FERTI_PRE_WAIT'

# ---------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ATTACH_FERTI_PRE_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'ATTACH_FERTI_ACTION'

        elif self.phase == 'ATTACH_FERTI_ACTION':
                self.set_gripper_state('attach')
                self.phase = 'PHASE_LIFT_FERTILIZER'

# ---------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_LIFT_FERTILIZER':
            if not self.phase_initialized:
                 self.lift_target = self.current_tcp_pos.copy()
                 # Lift 5 cm upward to clear the fertilizer from its resting surface
                 self.lift_target[2] += 0.05
                 self.phase_initialized = True
                 self.get_logger().info(f"Lifting to: {self.lift_target}")

            if self.move_to_tcp_target(self.lift_target, tol=0.001, slow=True):
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'PHASE_REVERSE_FROM_FERTI_WAIT'

# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_REVERSE_FROM_FERTI_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'PHASE_REVERSE_FROM_FERTI'

# -----------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_REVERSE_FROM_FERTI':
            if not self.phase_initialized:
                self.reverse_target = self.current_tcp_pos.copy()
                # Pull back 30 cm along Y to safely clear the fertilizer zone
                self.reverse_target[1] += 0.30
                self.phase_initialized = True
                self.get_logger().info("Reversing safely...")

            if self.move_to_tcp_target(self.reverse_target, tol=0.02, slow=True):
                self.get_logger().info("Reverse Complete.")
                self.phase_initialized = False
                self.phase = 'REVRSE_TO_ALING_FERTI_WAIT'

# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'REVRSE_TO_ALING_FERTI_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'REVRSE_TO_ALING_FERTI'

# ------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'REVRSE_TO_ALING_FERTI':
            # Return all joints to the recorded initial arm position
            targets = {
                'shoulder_pan_joint': self.initial_arm_pos['shoulder_pan_joint'],
                'shoulder_lift_joint': self.initial_arm_pos['shoulder_lift_joint'],
                'elbow_joint': self.initial_arm_pos['elbow_joint'],
                'wrist_1_joint': self.initial_arm_pos['wrist_1_joint'],
                'wrist_2_joint': self.initial_arm_pos['wrist_2_joint'],
                'wrist_3_joint': self.initial_arm_pos['wrist_3_joint'],
            }

            speed_scale = {
                'shoulder_pan_joint': 1.0,
                'shoulder_lift_joint': 1.0,
                'elbow_joint': 1.0,
                'wrist_1_joint': 1.0,
                'wrist_2_joint': 1.0,
                'wrist_3_joint': 1.0,
            }

            if self.move_joint_group(targets, speed_scale):
                self.get_logger().info("Fertilizer alignment restored (FAST).")
                self.phase = 'ALING_TO_INTI_WAIT'

# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ALING_TO_INTI_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'PHASE_ALIGN_TO_INIT'

# ---------------------------------------------------------------------------------------------------------------------------
#  this i used because some time arm base not aling so just for the safety
        elif self.phase == 'PHASE_ALIGN_TO_INIT':
            initial_shoulder_pan = self.initial_arm_pos['shoulder_pan_joint']
            reached = self.move_joint_to_angle(
                initial_shoulder_pan,
                'shoulder_pan_joint',
                0
            )

            if reached:
                self.get_logger().info(" Returned to initial shoulder pan.")
                self.stop_all()
                self.get_logger().info(f" this is when  intint {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.phase = 'PHASE_GRIPPER_ORIENTATION_DOWN'

# -------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_GRIPPER_ORIENTATION_DOWN':
            joint_name = 'wrist_1_joint'
            joint_idx = 3

            if not self.phase_initialized:
                if joint_name not in self.joint_pos:
                    return

                # Subtract wrist delta to rotate gripper ~90 deg downward for placement
                self.target_wrist_val = self.joint_pos[joint_name] - self.wrist1_delta_down

                self.phase_initialized = True
                self.get_logger().info(f"Rotating Wrist Down... Target: {self.target_wrist_val:.2f}")

            if self.move_joint_to_angle(self.target_wrist_val, joint_name, joint_idx):
                self.get_logger().info(" Wrist Oriented Down.")
                self.get_logger().info(f"gripper down phase {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.phase_initialized = False
                self.phase = 'GETTING_TF_EBOT'

# ------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'GETTING_TF_EBOT':
            # Step 1: Block until the eBot has physically docked
            if not self.ebot_docked:
                self.get_logger().info("Waiting for eBot to arrive at dock...", throttle_duration_sec=2.0)
                return
            if self.wait_for_timer(1.0):
                if self.ebotWorldPosition is None:
                    self.ebotWorldPosition = self.lookup_tf(self.base_link_name, self.ebotTransformName)

                if self.ebotWorldPosition is None:
                    self.get_logger().info("Waiting for ebot TF...", throttle_duration_sec=2.0)
                    return
                else:
                    self.get_logger().info("✓ Found ebot TF. .")
                    self.phase = 'MOVED_FOR_EBOT_HOVER'

# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'MOVED_FOR_EBOT_HOVER':
            # Hover 25 cm above the eBot before descending to drop the fertilizer
            target = self.ebotWorldPosition.copy()
            target[2] += 0.25

            if self.move_to_tcp_target(target, 0.01):
                self.get_logger().info("Hovered over eBot. .")
                self.phase = 'FINAL_APPROACH_EBOT_WAIT'
# --------------------------------------------------------------------------------------------------------------------------------------
#  rotate the wrist 3 np.pi degree clock wise

        elif self.phase == 'ROTATE_WRIST3':
            joint_name = 'wrist_3_joint'
            joint_idx = 5

            if not self.phase_initialized:
                if joint_name not in self.joint_pos:
                    return

                # Subtract wrist delta to rotate gripper ~90 deg downward for placement
                self.target_wrist_val = self.joint_pos[joint_name] + np.pi

                self.phase_initialized = True
                self.get_logger().info(f"Rotating Wrist Down... Target: {self.target_wrist_val:.2f}")

            if self.move_joint_to_angle(self.target_wrist_val, joint_name, joint_idx):
                self.get_logger().info(" wrist 3 orentation.")
                self.get_logger().info(f"gripper rotation  phase {self.current_tcp_orient} euler {self.current_euler}, pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.phase_initialized = False
                self.phase = 'ROTATION_AFTER'
# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'FINAL_APPROACH_EBOT_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'DROP_FERTI_ON_EBOT'

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'DROP_FERTI_ON_EBOT':
            self.set_gripper_state('detach')
            self.get_logger().info("detach complete ferti on ebot ")
            self.phase = 'RETARCT_FROM_EBOT_WAIT'

# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'RETARCT_FROM_EBOT_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'RETARCT_FROM_EBOT'

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'RETARCT_FROM_EBOT':
            self.get_logger().info("we are retract from intial pose")
            if not self.phase_initialized:
                 self.Retract_target = self.current_tcp_pos.copy()
                 # Move slightly back and up to safely retract from the eBot drop zone
                 self.Retract_target[0] -= 0.05
                 self.Retract_target[2] += 0.05
                 self.phase_initialized = True
                 self.get_logger().info(f"Retarct to: {self.Retract_target}")

            if self.move_to_tcp_target(self.Retract_target, tol=0.02, slow=True):
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'EBOT_MOVEMENT_ALLOW'

# ------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'EBOT_MOVEMENT_ALLOW':
            self.get_logger().info(" signal Ebot to move.")

            # Publish True on fertilizer status topic to signal eBot it can now leave
            ferti_placed_msg = Bool()
            ferti_placed_msg.data = True
            self.fertilizer_status_publisher.publish(ferti_placed_msg)

            # Reset dock flag so the node can detect the next eBot arrival
            self.ebot_docked = False

            self.get_logger().info("Dock flag reset. Moving to Initial Phase.")

            self.phase = 'FRUITS_TRAY_ALING'

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase  == 'FRUITS_TRAY_ALING':
            if self.move_to_tcp_target(self.fruitHomePosition, tol=0.02, slow=True):
                # Record the joint configuration at the fruits tray hover for later return
                self.fruits_tray_hover_pos = self.joint_pos.copy()
                self.get_logger().info(f" hover to the fruits tray{self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.get_logger().info(" we are at the hover at the fruits tary ")
                self.phase = 'SETTEL_FRUITS_TRAY'

# -----------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'SETTEL_FRUITS_TRAY':
            if self.wait_for_timer(2.0):
                self.phase = 'APPROACH_FRUITS'

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'APPROACH_FRUITS':
            # Check if all fruits have been processed
            if self.current_fruit_index >= len(self.badFruitTable):
                self.get_logger().info("All fruits sorted. Stopping.")
                self.stop_joint()
                self.phase = 'REVERSE_ALING_INIT_POSE_WAIT'
                return

            if not self.phase_initialized:
                fruit_record = self.badFruitTable[self.current_fruit_index]

                self.current_fruits_pose = fruit_record['pos'].copy()

                # Hover 15 cm above the fruit before descending for pickup
                self.hover_target = self.current_fruits_pose.copy()
                self.hover_target[2] += 0.15
                self.phase_initialized = True

            reached = self.move_to_tcp_target(self.hover_target, tol=0.01)

            if reached:
                self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.current_fruits_pose},  jointState {self.joint_pos}")
                self.get_logger().info(" we are at the hover at the fruits   now go  check the orenation  of pickup ")
                self.phase_initialized = False
                self.phase = 'FRUIST_HOVER_WAIT'

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'FRUIST_HOVER_WAIT':
            if self.wait_for_timer(1.0):
                # Pre-activate magnet while descending to ensure capture on contact
                self.set_gripper_state('attach')
                self.get_logger().info("magnet start here so get attach the cane ")
                self.phase = 'CURRECT_FRUITS_POSE_FINAL_APPROACH'

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'CURRECT_FRUITS_POSE_FINAL_APPROACH':
            self.get_logger().info(f"current magnet force  ({self.current_force_z:.2f} )")
            if not self.phase_initialized:
                self.final_target = self.current_fruits_pose.copy()
                # Fine offset to center gripper on the fruit stem
                self.final_target[0]  -=0.05
                self.final_target[1] += 0.01
                self.phase_initialized = True

            reached = self.move_to_tcp_target(self.final_target, tol=0.001, slow=True)

            # Accept arrival or force contact as success condition
            if reached or (self.current_force_z > 50.0):
                self.stop_all()
                self.phase_initialized = False
                self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos} , fruits {self.final_target}, fruits {self.current_fruits_pose}   jointState {self.joint_pos}")
                self.get_logger().info(" arm on the fruits call attach")
                self.phase = 'ATTACH_FRUITS_PRE_WAIT'

# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ATTACH_FRUITS_PRE_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'ATTACH_FRUITS_ACTION'

        elif self.phase == 'ATTACH_FRUITS_ACTION':
                self.set_gripper_state('attach')
                self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos}    jointState {self.joint_pos}")
                self.phase = 'LIFT_FRUIRTS_ATTACH'

# ---------------------------------------------------------------

        elif self.phase == 'LIFT_FRUIRTS_ATTACH':
            if not self.phase_initialized:
                 self.lift_target = self.current_tcp_pos.copy()
                 # Lift 15 cm to fully clear the tray before lateral movement
                 self.lift_target[2] += 0.15
                 self.phase_initialized = True
                 self.get_logger().info(f"Lifting to: {self.lift_target}")

            reached = self.move_to_tcp_target(self.lift_target, tol=0.02, slow=True)

            if reached:
                self.get_logger().info(f"lift up done {self.current_tcp_orient} , pose current {self.current_tcp_pos}    jointState {self.joint_pos}")
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'LIFT_FRUITS_WAIT'

# ---------------------------------------------------------------------------------------
        elif self.phase == 'LIFT_FRUITS_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'SAFE_HOVER_POSE'

# ----------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'SAFE_HOVER_POSE':
            if not self.phase_initialized:
                 self.lift_target = self.current_tcp_pos.copy()
                 # Move slightly back and further up before rotating to avoid collision
                 self.lift_target[1] -= 0.05
                 self.lift_target[2] += 0.10
                 self.phase_initialized = True
                 self.get_logger().info(f"Lifting to: {self.lift_target}")

            reached = self.move_to_tcp_target(self.lift_target, tol=0.02, slow=False)

            if reached:
                self.get_logger().info(f"lift up done {self.current_tcp_orient} , pose current {self.current_tcp_pos}    jointState {self.joint_pos}")
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'BASE_ALING_TO_DUSTBIN'

# ----------------------------------------------------------------------------------------------------
        elif self.phase == 'BASE_ALING_TO_DUSTBIN':
            # Rotate shoulder_pan_joint by +50 deg to swing arm toward the dustbin
            if not self.phase_initialized:
                start_angle = self.joint_pos['shoulder_pan_joint']
                # target_angle: current pan + 50 degrees, normalized to (-pi, pi]
                self.target_angle = self.norm(start_angle + np.radians(50))
                self.phase_initialized = True

            if self.move_joint_to_angle(self.target_angle, self.joint_names_list[0], 0):
                self.get_logger().info(f"Base rotated +75° to {np.degrees(self.target_angle):.2f}°. Moving to dustbin.")
                self.phase_initialized = False
                # Save joint state at dustbin hover so we can return here after drop
                self.dustbin_hover_pose = self.joint_pos.copy()
                self.phase = 'MOVED_FOR_DUSTBIN_WAIT'

# ---------------------------------------------------------------------------------------
        elif self.phase == 'MOVED_FOR_DUSTBIN_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'MOVED_FOR_DUSTBIN'

# ----------------------------------------------------------------------------------------------------------
        elif self.phase == 'MOVED_FOR_DUSTBIN':
            if self.current_tcp_pos is None:
                self.get_logger().info("we tcp pose miissing here ")
                return

            if not self.phase_initialized:
                self.target_dustbin = self.current_tcp_pos.copy()
                # Move 25 cm back along X to position over the dustbin opening
                self.target_dustbin[0] -= 0.16
                self.phase_initialized = True

                # Log distance from current position to the fixed dustbin position for debugging
                dist_x = self.dustbinPosition[0] - self.current_tcp_pos[0]
                dist_y = self.dustbinPosition[1] - self.current_tcp_pos[1]
                dist_z = self.dustbinPosition[2] - self.current_tcp_pos[2]

                self.get_logger().info(
                    f"Distance to Dustbin -> X: {dist_x:.3f}m, Y: {dist_y:.3f}m, Z: {dist_z:.3f}m err : { np.linalg.norm(self.dustbinPosition - self.current_tcp_pos)}   target by use {self.target_dustbin}",
                    throttle_duration_sec=0.5
                )

            reached = self.move_to_tcp_target(self.target_dustbin, tol=0.02, slow=True)
            if reached:
                self.get_logger().info("Target reached. WE ARE AT DUSTBIN.")
                self.phase_initialized = False
                self.phase = 'DUSTBIN_WAIT'

# ---------------------------------------------------------------------------------------
        elif self.phase == 'DUSTBIN_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'CALL_DEATTACH_FRUITS'

# ----------------------------------------------------------------------------------------------------
        elif self.phase == 'CALL_DEATTACH_FRUITS':
                self.get_logger().info("Deactivating Gripper...")
                self.set_gripper_state('detach')
                # Increment index to process the next bad fruit on the next cycle
                self.current_fruit_index += 1
                self.phase = 'RETRACT_DUSTBIN_POSE'

# ----------------------------------------------------------------------------------------------------------
        elif self.phase == 'RETRACT_DUSTBIN_POSE':
            if not self.phase_initialized:
                # Return shoulder, lift, and elbow to the saved dustbin hover configuration
                self.targets = {
                    'shoulder_pan_joint': self.dustbin_hover_pose['shoulder_pan_joint'],
                    'shoulder_lift_joint': self.dustbin_hover_pose['shoulder_lift_joint'],
                    'elbow_joint': self.dustbin_hover_pose['elbow_joint'],
                }

                self.speed_scale = {
                    'shoulder_pan_joint': 1.0,
                    'shoulder_lift_joint': 1.0,
                    'elbow_joint': 0.9,
                }
                self.phase_initialized = True

            if self.move_joint_group(self.targets, self.speed_scale):
                self.phase_initialized  = False
                self.get_logger().info("Fertilizer alignment restored (FAST).")
                self.phase = 'RETURN_TRAY_WAIT'

# -----------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'RETURN_TRAY_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'RETURN_TRAY_POSE'

# ----------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'RETURN_TRAY_POSE':
            if not self.phase_initialized:
                # Restore all 6 joints to the fruits tray hover configuration
                self.targets = {
                    'shoulder_pan_joint': self.fruits_tray_hover_pos['shoulder_pan_joint'],
                    'shoulder_lift_joint': self.fruits_tray_hover_pos['shoulder_lift_joint'],
                    'elbow_joint': self.fruits_tray_hover_pos['elbow_joint'],
                    'wrist_1_joint': self.fruits_tray_hover_pos['wrist_1_joint'],
                    'wrist_2_joint': self.fruits_tray_hover_pos['wrist_2_joint'],
                    'wrist_3_joint': self.fruits_tray_hover_pos['wrist_3_joint'],
                }

                self.speed_scale = {
                    'shoulder_pan_joint': 1.0,
                    'shoulder_lift_joint': 1.0,
                    'elbow_joint': 1.0,
                    'wrist_1_joint': 1.0,
                    'wrist_2_joint': 1.0,
                    'wrist_3_joint': 1.0,
                }
                self.phase_initialized = True

            if self.move_joint_group(self.targets, self.speed_scale):
                self.phase_initialized  = False
                self.get_logger().info("Fertilizer alignment restored (FAST).")
                # Loop back to pick the next fruit
                self.phase = 'SETTEL_FRUITS_TRAY'

# =============================================  UNLOAD FERTILZER PHASE===================================================

#   1. come to intial phase
# ------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'REVERSE_ALING_INIT_POSE_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'REVERSE_TO_FERTI_INTIAL_ALING'

# ----------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'REVERSE_TO_FERTI_INTIAL_ALING':
            # Return all joints to the original startup pose to complete the task cycle
            targets = {
                'shoulder_pan_joint': self.initial_arm_pos['shoulder_pan_joint'],
                'shoulder_lift_joint': self.initial_arm_pos['shoulder_lift_joint'],
                'elbow_joint': self.initial_arm_pos['elbow_joint'],
                'wrist_1_joint': self.initial_arm_pos['wrist_1_joint'],
                'wrist_2_joint': self.initial_arm_pos['wrist_2_joint'],
                'wrist_3_joint': self.initial_arm_pos['wrist_3_joint'],
            }

            speed_scale = {
                'shoulder_pan_joint': 1.0,
                'shoulder_lift_joint': 1.0,
                'elbow_joint': 0.9,
                'wrist_1_joint': 1.0,
                'wrist_2_joint': 1.0,
                'wrist_3_joint': 1.0,
            }

            if self.move_joint_group(targets, speed_scale):
                self.get_logger().info("init pose again check sucess ")
                self.phase = 'WAIT_4_EBOT_FERTI_UNLOAD'  #  finish here because i  below code i wrote but not used therefore
# =======================================================  MOVED FOR UNLOAD ==================================

#   from here i wrote the unload  phase of logic   but not used  in this evalution    the rreason i not able to run because i busy with to fruits pickup strategy
# -----------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'WAIT_4_EBOT_FERTI_UNLOAD':
            if not self.ebot_docked:
                self.get_logger().info("Waiting for eBot to return with fertilizer...", throttle_duration_sec=2.0)
                return

            # Always look up the CURRENT fertilizer TF (position may have changed on eBot)
            self.ferti_unload_pose = self.lookup_tf(self.base_link_name, self.fertilizerTFname)

            if self.ferti_unload_pose is None:
                self.get_logger().info("Waiting for fertilizer TF  new ...", throttle_duration_sec=2.0)
                return

            self.get_logger().info("getting fertilzer here again ")
            self.phase = 'GRIPPER_UNLOAD_FERTI_DOWN'

# ---------------------------------------------------------------------------------------------------------------
        elif self.phase == 'GRIPPER_UNLOAD_FERTI_DOWN':
            joint_name = 'wrist_1_joint'
            joint_idx = 3

            if not self.phase_initialized:
                if joint_name not in self.joint_pos:
                    return

                # Subtract wrist delta to point gripper downward for unloading
                self.target_wrist_val = self.joint_pos[joint_name] - self.wrist1_delta_down

                self.phase_initialized = True
                self.get_logger().info(f"Rotating Wrist Down... Target: {self.target_wrist_val:.2f}")

            if self.move_joint_to_angle(self.target_wrist_val, joint_name, joint_idx):
                self.get_logger().info(" Wrist Oriented Down.")
                self.get_logger().info(f"gripper down phase {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.phase_initialized = False
                self.phase = 'UNLOAD_PRE_FERTI_WAIT'

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_PRE_FERTI_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'UNLOAD_PRE_FERTI'

# ----------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_PRE_FERTI':
            if not self.phase_initialized:
                 self.ferti_unload_target = self.ferti_unload_pose.copy()
                 # Hover 5 cm above the fertilizer before final descent
                 self.ferti_unload_target[2] += 0.05
                 self.phase_initialized = True
                 self.get_logger().info(f"unload target approach: {self.ferti_unload_target}   target : {self.ferti_unload_pose}")

            if self.move_to_tcp_target(self.ferti_unload_target, tol=0.01, slow=False):
                self.get_logger().info(f"  ferti unload pre {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'UNLOAD_FERTI_FINAL_WAIT'

# ------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_FERTI_FINAL_WAIT':
            if self.wait_for_timer(2.0):
                self.set_gripper_state('attach')
                self.get_logger().info(" magnet start ")
                self.phase = 'UNLOAD_FERTI_FINAL'

# ------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_FERTI_FINAL':
            if not self.phase_initialized:
                 self.ferti_unload_target_final = self.ferti_unload_pose.copy()
                 self.phase_initialized = True
                 self.get_logger().info(f"unload target approach: {self.ferti_unload_target_final} , current magnet {self.current_force_z} ")

            reached = self.move_to_tcp_target(self.ferti_unload_target_final, tol=0.01, slow=True)
            self.get_logger().info(f", current magnet {self.current_force_z} ")

            # Accept position arrival or force contact to confirm grasp
            if reached or (self.current_force_z > 20):
                self.stop_all()
                self.get_logger().info(f"  ferti unload  final {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'LIFT_TARGET_UNLOAD_WAIT'

# --------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'LIFT_TARGET_UNLOAD_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'LIFT_TARGET_UNLOAD'

# ---------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'LIFT_TARGET_UNLOAD':
             if not self.phase_initialized:
                self.lift_unload_target = self.current_tcp_pos.copy()
                self.get_logger().info(f"curren_pose {self.current_tcp_pos} target {self.lift_unload_target}")
                self.phase_initialized = True

             reached = self.move_to_tcp_target(self.lift_unload_target, tol=0.01, slow=True)
             if reached:
                self.get_logger().info(f" ferti unload wala hai   {self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.ferti_pose}")
                self.get_logger().info("we are lif success fully ")
                self.phase_initialized = False
                self.phase = 'ALLOW_EBOT_AFTER_UNLOAD'

# --------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ALLOW_EBOT_AFTER_UNLOAD':
            self.get_logger().info(" Signal Ebot to move.")

            # Publish True to notify eBot that fertilizer has been unloaded
            ferti_placed_msg = Bool()
            ferti_placed_msg.data = True
            self.fertilizer_status_publisher.publish(ferti_placed_msg)

            # Reset dock flag to detect the next eBot arrival
            self.ebot_docked = False

            self.get_logger().info("Dock flag reset. Moving to Initial Phase.")

            self.phase = 'ALING_TO_TRAY_UNLOAD_WAIT'

# ------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ALING_TO_TRAY_UNLOAD_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'ALING_TO_TRAY_UNLOAD'

# --------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ALING_TO_TRAY_UNLOAD':
            if self.move_to_tcp_target(self.fruitHomePosition, tol=0.02, slow=True):
                self.get_logger().info(f" hover to the fruits tray  for unload the ferti {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.phase = 'ALING_TO_DUSTBIN_UNLOAD'

# ----------------------------------------------------------------------------------------------------
        elif self.phase == 'ALING_TO_DUSTBIN_UNLOAD':
            # Rotate shoulder_pan_joint +75 deg to align toward dustbin for unload cycle
            if not self.phase_initialized:
                start_angle = self.joint_pos['shoulder_pan_joint']
                # target_angle: current pan + 75 degrees, normalized to (-pi, pi]
                self.target_angle = self.norm(start_angle + np.radians(75))
                self.phase_initialized = True

            if self.move_joint_to_angle(self.target_angle, self.joint_names_list[0], 0):
                self.get_logger().info(f"Base rotated +75° to {np.degrees(self.target_angle):.2f}°. Moving to dustbin.")
                self.phase_initialized = False
                self.dustbin_hover_pose = self.joint_pos.copy()
                self.phase = 'UNLOAD_MOVED_FOR_DUSTBIN_WAIT'

# ---------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_MOVED_FOR_DUSTBIN_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'UNLOAD_MOVED_FOR_DUSTBIN'

# ----------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_MOVED_FOR_DUSTBIN':
            if self.current_tcp_pos is None:
                self.get_logger().info("we tcp pose miissing here ")
                return

            if not self.phase_initialized:
                self.target_dustbin = self.current_tcp_pos.copy()
                # Move 10 cm back along X to position over the dustbin for the unload cycle
                self.target_dustbin[0] -= 0.10
                self.phase_initialized = True# -----------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'WAIT_4_EBOT_FERTI_UNLOAD':
            if not self.ebot_docked:
                self.get_logger().info("Waiting for eBot to return with fertilizer...", throttle_duration_sec=2.0)
                return

            # Always look up the CURRENT fertilizer TF (position may have changed on eBot)
            self.ferti_unload_pose = self.lookup_tf(self.base_link_name, self.fertilizerTFname)

            if self.ferti_unload_pose is None:
                self.get_logger().info("Waiting for fertilizer TF  new ...", throttle_duration_sec=2.0)
                return

            self.get_logger().info("getting fertilzer here again ")
            self.phase = 'GRIPPER_UNLOAD_FERTI_DOWN'

# ---------------------------------------------------------------------------------------------------------------
        elif self.phase == 'GRIPPER_UNLOAD_FERTI_DOWN':
            joint_name = 'wrist_1_joint'
            joint_idx = 3

            if not self.phase_initialized:
                if joint_name not in self.joint_pos:
                    return

                # Subtract wrist delta to point gripper downward for unloading
                self.target_wrist_val = self.joint_pos[joint_name] - self.wrist1_delta_down

                self.phase_initialized = True
                self.get_logger().info(f"Rotating Wrist Down... Target: {self.target_wrist_val:.2f}")

            if self.move_joint_to_angle(self.target_wrist_val, joint_name, joint_idx):
                self.get_logger().info(" Wrist Oriented Down.")
                self.get_logger().info(f"gripper down phase {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.phase_initialized = False
                self.phase = 'UNLOAD_PRE_FERTI_WAIT'

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_PRE_FERTI_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'UNLOAD_PRE_FERTI'

# ----------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_PRE_FERTI':
            if not self.phase_initialized:
                 self.ferti_unload_target = self.ferti_unload_pose.copy()
                 # Hover 5 cm above the fertilizer before final descent
                 self.ferti_unload_target[2] += 0.05
                 self.phase_initialized = True
                 self.get_logger().info(f"unload target approach: {self.ferti_unload_target}   target : {self.ferti_unload_pose}")

            if self.move_to_tcp_target(self.ferti_unload_target, tol=0.01, slow=False):
                self.get_logger().info(f"  ferti unload pre {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'UNLOAD_FERTI_FINAL_WAIT'

# ------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_FERTI_FINAL_WAIT':
            if self.wait_for_timer(2.0):
                self.set_gripper_state('attach')
                self.get_logger().info(" magnet start ")
                self.phase = 'UNLOAD_FERTI_FINAL'

# ------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_FERTI_FINAL':
            if not self.phase_initialized:
                 self.ferti_unload_target_final = self.ferti_unload_pose.copy()
                 self.phase_initialized = True
                 self.get_logger().info(f"unload target approach: {self.ferti_unload_target_final} , current magnet {self.current_force_z} ")

            reached = self.move_to_tcp_target(self.ferti_unload_target_final, tol=0.01, slow=True)
            self.get_logger().info(f", current magnet {self.current_force_z} ")

            # Accept position arrival or force contact to confirm grasp
            if reached or (self.current_force_z > 20):
                self.stop_all()
                self.get_logger().info(f"  ferti unload  final {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'LIFT_TARGET_UNLOAD_WAIT'

# --------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'LIFT_TARGET_UNLOAD_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'LIFT_TARGET_UNLOAD'

# ---------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'LIFT_TARGET_UNLOAD':
             if not self.phase_initialized:
                self.lift_unload_target = self.current_tcp_pos.copy()
                self.get_logger().info(f"curren_pose {self.current_tcp_pos} target {self.lift_unload_target}")
                self.phase_initialized = True

             reached = self.move_to_tcp_target(self.lift_unload_target, tol=0.01, slow=True)
             if reached:
                self.get_logger().info(f" ferti unload wala hai   {self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.ferti_pose}")
                self.get_logger().info("we are lif success fully ")
                self.phase_initialized = False
                self.phase = 'ALLOW_EBOT_AFTER_UNLOAD'

# --------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ALLOW_EBOT_AFTER_UNLOAD':
            self.get_logger().info(" Signal Ebot to move.")

            # Publish True to notify eBot that fertilizer has been unloaded
            ferti_placed_msg = Bool()
            ferti_placed_msg.data = True
            self.fertilizer_status_publisher.publish(ferti_placed_msg)

            # Reset dock flag to detect the next eBot arrival
            self.ebot_docked = False

            self.get_logger().info("Dock flag reset. Moving to Initial Phase.")

            self.phase = 'ALING_TO_TRAY_UNLOAD_WAIT'

# ------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ALING_TO_TRAY_UNLOAD_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'ALING_TO_TRAY_UNLOAD'

# --------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ALING_TO_TRAY_UNLOAD':
            if self.move_to_tcp_target(self.fruitHomePosition, tol=0.02, slow=True):
                self.get_logger().info(f" hover to the fruits tray  for unload the ferti {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.phase = 'ALING_TO_DUSTBIN_UNLOAD'

# ----------------------------------------------------------------------------------------------------
        elif self.phase == 'ALING_TO_DUSTBIN_UNLOAD':
            # Rotate shoulder_pan_joint +75 deg to align toward dustbin for unload cycle
            if not self.phase_initialized:
                start_angle = self.joint_pos['shoulder_pan_joint']
                # target_angle: current pan + 75 degrees, normalized to (-pi, pi]
                self.target_angle = self.norm(start_angle + np.radians(75))
                self.phase_initialized = True

            if self.move_joint_to_angle(self.target_angle, self.joint_names_list[0], 0):
                self.get_logger().info(f"Base rotated +75° to {np.degrees(self.target_angle):.2f}°. Moving to dustbin.")
                self.phase_initialized = False
                self.dustbin_hover_pose = self.joint_pos.copy()
                self.phase = 'UNLOAD_MOVED_FOR_DUSTBIN_WAIT'

# ---------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_MOVED_FOR_DUSTBIN_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'UNLOAD_MOVED_FOR_DUSTBIN'

# ----------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_MOVED_FOR_DUSTBIN':
            if self.current_tcp_pos is None:
                self.get_logger().info("we tcp pose miissing here ")
                return

            if not self.phase_initialized:
                self.target_dustbin = self.current_tcp_pos.copy()
                # Move 10 cm back along X to position over the dustbin for the unload cycle
                self.target_dustbin[0] -= 0.10
                self.phase_initialized = True

                self.get_logger().info(f" target by use {self.target_dustbin}")

            reached = self.move_to_tcp_target(self.target_dustbin, tol=0.02, slow=True)
            if reached:
                self.get_logger().info("Target reached. WE ARE AT DUSTBIN.")
                self.phase_initialized = False
                self.phase = 'UNLOAD_DUSTBIN_WAIT'

# ---------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_DUSTBIN_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'UNLOAD_CALL_DEATTACH_FRUITS'

# ----------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_CALL_DEATTACH_FRUITS':
                self.get_logger().info("Deactivating Gripper...")
                self.set_gripper_state('detach')
                self.current_fruit_index += 1
                self.phase = 'UNLOAD_RETRACT_DUSTBIN_POSE'

# ----------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_RETRACT_DUSTBIN_POSE':
            if not self.phase_initialized:
                # Return shoulder, lift, and elbow to dustbin hover configuration
                self.targets = {
                    'shoulder_pan_joint': self.dustbin_hover_pose['shoulder_pan_joint'],
                    'shoulder_lift_joint': self.dustbin_hover_pose['shoulder_lift_joint'],
                    'elbow_joint': self.dustbin_hover_pose['elbow_joint'],
                }

                self.speed_scale = {
                    'shoulder_pan_joint': 1.0,
                    'shoulder_lift_joint': 1.0,
                    'elbow_joint': 0.9,
                }
                self.phase_initialized = True

            if self.move_joint_group(self.targets, self.speed_scale):
                self.phase_initialized  = False
                self.get_logger().info("Fertilizer alignment restored (FAST).")
                self.phase = 'UNLOAD_RETURN_TRAY_WAIT'

# -----------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_RETURN_TRAY_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'UNLOAD_RETURN_TRAY_POSE'

# ----------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_RETURN_TRAY_POSE':
            if not self.phase_initialized:
                # Restore all 6 joints to the fruits tray hover configuration
                self.targets = {
                    'shoulder_pan_joint': self.fruits_tray_hover_pos['shoulder_pan_joint'],
                    'shoulder_lift_joint': self.fruits_tray_hover_pos['shoulder_lift_joint'],
                    'elbow_joint': self.fruits_tray_hover_pos['elbow_joint'],
                    'wrist_1_joint': self.fruits_tray_hover_pos['wrist_1_joint'],
                    'wrist_2_joint': self.fruits_tray_hover_pos['wrist_2_joint'],
                    'wrist_3_joint': self.fruits_tray_hover_pos['wrist_3_joint'],
                }

                self.speed_scale = {
                    'shoulder_pan_joint': 1.0,
                    'shoulder_lift_joint': 1.0,
                    'elbow_joint': 1.0,
                    'wrist_1_joint': 1.0,
                    'wrist_2_joint': 1.0,
                    'wrist_3_joint': 1.0,
                }
                self.phase_initialized = True

            if self.move_joint_group(self.targets, self.speed_scale):
                self.phase_initialized  = False
                self.get_logger().info("Fertilizer alignment restored (FAST).")
                self.phase = 'UNLOAD_REVERSE_TO_FERTI_INTIAL_ALING'

# ----------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_REVERSE_TO_FERTI_INTIAL_ALING':
            # Final return to the original startup pose — unload cycle complete
            targets = {
                'shoulder_pan_joint': self.initial_arm_pos['shoulder_pan_joint'],
                'shoulder_lift_joint': self.initial_arm_pos['shoulder_lift_joint'],
                'elbow_joint': self.initial_arm_pos['elbow_joint'],
                'wrist_1_joint': self.initial_arm_pos['wrist_1_joint'],
                'wrist_2_joint': self.initial_arm_pos['wrist_2_joint'],
                'wrist_3_joint': self.initial_arm_pos['wrist_3_joint'],
            }

            speed_scale = {
                'shoulder_pan_joint': 1.0,
                'shoulder_lift_joint': 1.0,
                'elbow_joint': 0.9,
                'wrist_1_joint': 1.0,
                'wrist_2_joint': 1.0,
                'wrist_3_joint': 1.0,
            }

            if self.move_joint_group(targets, speed_scale):
                self.get_logger().info("init pose again check sucess ")
                self.phase = 'DONE'


                self.get_logger().info(f" target by use {self.target_dustbin}")

            reached = self.move_to_tcp_target(self.target_dustbin, tol=0.02, slow=True)
            if reached:
                self.get_logger().info("Target reached. WE ARE AT DUSTBIN.")
                self.phase_initialized = False
                self.phase = 'UNLOAD_DUSTBIN_WAIT'

# ---------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_DUSTBIN_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'UNLOAD_CALL_DEATTACH_FRUITS'

# ----------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_CALL_DEATTACH_FRUITS':
                self.get_logger().info("Deactivating Gripper...")
                self.set_gripper_state('detach')
                self.current_fruit_index += 1
                self.phase = 'UNLOAD_RETRACT_DUSTBIN_POSE'

# ----------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_RETRACT_DUSTBIN_POSE':
            if not self.phase_initialized:
                # Return shoulder, lift, and elbow to dustbin hover configuration
                self.targets = {
                    'shoulder_pan_joint': self.dustbin_hover_pose['shoulder_pan_joint'],
                    'shoulder_lift_joint': self.dustbin_hover_pose['shoulder_lift_joint'],
                    'elbow_joint': self.dustbin_hover_pose['elbow_joint'],
                }

                self.speed_scale = {
                    'shoulder_pan_joint': 1.0,
                    'shoulder_lift_joint': 1.0,
                    'elbow_joint': 0.9,
                }
                self.phase_initialized = True

            if self.move_joint_group(self.targets, self.speed_scale):
                self.phase_initialized  = False
                self.get_logger().info("Fertilizer alignment restored (FAST).")
                self.phase = 'UNLOAD_RETURN_TRAY_WAIT'

# -----------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_RETURN_TRAY_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'UNLOAD_RETURN_TRAY_POSE'

# ----------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_RETURN_TRAY_POSE':
            if not self.phase_initialized:
                # Restore all 6 joints to the fruits tray hover configuration
                self.targets = {
                    'shoulder_pan_joint': self.fruits_tray_hover_pos['shoulder_pan_joint'],
                    'shoulder_lift_joint': self.fruits_tray_hover_pos['shoulder_lift_joint'],
                    'elbow_joint': self.fruits_tray_hover_pos['elbow_joint'],
                    'wrist_1_joint': self.fruits_tray_hover_pos['wrist_1_joint'],
                    'wrist_2_joint': self.fruits_tray_hover_pos['wrist_2_joint'],
                    'wrist_3_joint': self.fruits_tray_hover_pos['wrist_3_joint'],
                }

                self.speed_scale = {
                    'shoulder_pan_joint': 1.0,
                    'shoulder_lift_joint': 1.0,
                    'elbow_joint': 1.0,
                    'wrist_1_joint': 1.0,
                    'wrist_2_joint': 1.0,
                    'wrist_3_joint': 1.0,
                }
                self.phase_initialized = True

            if self.move_joint_group(self.targets, self.speed_scale):
                self.phase_initialized  = False
                self.get_logger().info("Fertilizer alignment restored (FAST).")
                self.phase = 'UNLOAD_REVERSE_TO_FERTI_INTIAL_ALING'

# ----------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_REVERSE_TO_FERTI_INTIAL_ALING':
            # Final return to the original startup pose — unload cycle complete
            targets = {
                'shoulder_pan_joint': self.initial_arm_pos['shoulder_pan_joint'],
                'shoulder_lift_joint': self.initial_arm_pos['shoulder_lift_joint'],
                'elbow_joint': self.initial_arm_pos['elbow_joint'],
                'wrist_1_joint': self.initial_arm_pos['wrist_1_joint'],
                'wrist_2_joint': self.initial_arm_pos['wrist_2_joint'],
                'wrist_3_joint': self.initial_arm_pos['wrist_3_joint'],
            }

            speed_scale = {
                'shoulder_pan_joint': 1.0,
                'shoulder_lift_joint': 1.0,
                'elbow_joint': 0.9,
                'wrist_1_joint': 1.0,
                'wrist_2_joint': 1.0,
                'wrist_3_joint': 1.0,
            }

            if self.move_joint_group(targets, speed_scale):
                self.get_logger().info("init pose again check sucess ")
                self.phase = 'DONE'

# -----------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'DONE':
            self.get_logger().info(f"all task done bro look at the tcp pose ", throttle_duration_sec=2.0)
            pass

# --------------------------------------------------------------------------------------------------------------


'''
* Function Name: main
* Input:         None
* Output:        None
* Logic:         Entry point for the ROS2 node. Initializes the rclpy library,
*                creates an instance of Task6, and spins it until a KeyboardInterrupt.
*                Ensures proper cleanup by destroying the node and shutting down rclpy.
* Example Call:  Called automatically by the Python interpreter when the script is run directly
'''
def main():
    rclpy.init()
    node = Task6()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

# Entry point guard — only run main() when executed as a script, not when imported
if __name__ == '__main__':
    main()
