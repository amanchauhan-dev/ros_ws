#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import numpy as np
import time
# CHANGED: Interface Imports
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import Float64MultiArray, Float32
from control_msgs.msg import JointJog
from std_srvs.srv import SetBool
from sensor_msgs.msg import JointState
from rclpy.duration import Duration
import tf2_ros
from rclpy.callback_groups import ReentrantCallbackGroup

# ==================================================================
# --------------------- MATH UTILITIES -----------------------------
# ==================================================================

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return np.array([qx, qy, qz, qw])

def normalize_quaternion(quaternion):
    quaternion_array = np.array(quaternion, dtype=float)
    magnitude = np.linalg.norm(quaternion_array)
    if magnitude > 0.0:
        return quaternion_array / magnitude
    else:
        return quaternion_array

def conjugate_quaternion(quaternion):
    x_value, y_value, z_value, w_value = quaternion
    return np.array([-x_value, -y_value, -z_value, w_value], dtype=float)

def multiply_quaternion(first, second):
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    x_out = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_out = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_out = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w_out = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x_out, y_out, z_out, w_out], dtype=float)

def quaternion_to_euler(quaternion):
    x_value, y_value, z_value, w_value = quaternion

    # Roll (x-axis rotation)
    sin_roll_cos_pitch = 2.0 * (w_value * x_value + y_value * z_value)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x_value * x_value + y_value * y_value)
    roll_angle = np.arctan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    # Pitch (y-axis rotation)
    sin_pitch = 2.0 * (w_value * y_value - z_value * x_value)
    # Handle singularity cases
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


class Task5c(Node):
    def __init__(self):
        super().__init__('Task5c')

        self.service_callback_group = ReentrantCallbackGroup()

        # ===================publishers========================
        # CHANGED: TwistStamped
        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        # CHANGED: JointJog
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

        # DEFINED: Joint Names List
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

        # CHANGED: Float64MultiArray for TCP
        self.tcp_pose_sub = self.create_subscription(
            Float64MultiArray,
            '/tcp_pose_raw',
            self.tcp_pose_callback,
            10
        )

        # ADDED: Force Subscriber
        self.force_sub = self.create_subscription(
            Float32,
            '/net_wrench',
            self.force_callback,
            10
        )

        # 2. CREATE SERVICE CLIENTS FOR GRIPPER
        # CHANGED: SetBool
        self.magnet_client = self.create_client(SetBool, '/magnet')

        self.timer = self.create_timer(0.02, self.main_loop)

        # ===================configurations========================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.teamIdentifier = '3578'
        self.base_link_name = 'base_link'
        self.joint1 = 'shoulder_pan_joint'
        self.wrist2 = 'wrist_2_joint'
        self.joint_pos = {}

        # tcp_pose variables
        self.current_tcp_pos = None
        self.current_tcp_orient = None
        self.initial_arm_pos = None
        self.ferti_align_joint_state = None

        self.max_tol = np.deg2rad(2)
        self.base_kp = 2.0
        self.base_max_speed = 2.5

        # phases config
        self.safe_lift_angle = None
        self.joint2_name = 'shoulder_lift_joint'

        self.phase = 'PHASE_1_GETTING_TF'

        self.ferti_pose = None
        self.fertilizerTFname = f'{self.teamIdentifier}_fertilizer_1'
        self.ebotTransformName = f'{self.teamIdentifier}_ebot_marker'
        self.pickupOrientationQuaternion = np.array([0.707, 0.028, 0.034, 0.707])
        self.horizontalCarryQuaternion = np.array([-0.684, 0.726, 0.05, 0.008])
        # HERE NEED TO CHNAGE WHEN I WANT TO DET TF OF EBOT
        self.ebotWorldPosition = None
        self.ebotWorldPosition = np.array([0.711, 0.006, 0.145])
        self.approachSideOffsetYmeters = 0.05
        self.fertilizerLiftDistanceMeters = 0.05
        self.reverseAwayDistanceMeters = 0.20
        self.dropHeightMetersAboveEbot = 0.18
        self.dropOffsetXmeters = 0.0
        self.dropOffsetYmeters = 0.0

       # --- FERTILIZER SERVO PARAMETERS ---
        self.fertilizerMaximumLinearVelocity = 1.0
        self.fertilizerMaximumAngularVelocity = 2.0
        self.fertilizerPositionTolerance = 0.001
        self.fertilizerAngularToleranceRadians = np.radians(1.0)
        self.dropPositionTolerance = 0.05
        self.fruitTrayPositionTolerance = 0.05

        self.badFruitTable = []
        self.badFruitFrameList = [
            f'{self.teamIdentifier}_bad_fruit_1',
        ]


        # common variables for phases
        self.phase_initialized = False
        self.wrist1_delta_down = -1.36
        self.approach_offset_z = 0.10
        self.approach_tol = 0.03

        self.fruitHomePosition = np.array([-0.159, 0.501, 0.415])
        self.fruitHomeOrientation = normalize_quaternion([0.029, 0.997, 0.045, 0.033])
        self.highRetractPosition = np.array([-0.159, 0.501, 0.50])
        self.highRetractOrientation = self.fruitHomeOrientation
        self.dustbinPosition = np.array([-0.806, 0.010, 0.182])
        self.dustbinOrientation = normalize_quaternion([0.0, 0.0, 0.0, 1.0])

        # --- FRUIT MOVEMENT PARAMETERS ---
        self.fruitApproachOffsetZmeters = 0.1
        self.fruitFinalApproachOffsetZmeters = 0.01
        self.fruitLiftDistanceMeters = 0.10
        self.fruitStrafeRightMeters = 0.05

        # --- FRUIT SERVO PARAMETERS ---
        self.fruitMaximumLinearVelocity = 0.8
        self.fruitMaximumAngularVelocity = 2.5
        self.fruitPositionTolerance = 0.025
        self.fruitAngularToleranceRadians = np.radians(3.5)

        self.current_fruit_index = 0
        self.current_fruits_pose = None # Initialize this
        self.wait_start_time = None
        self.settleStartTime = None
        self.current_force_z = None # CHANGED



# ====================callbacks===============================================
    def joint_state_callback(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.joint_pos[n] = p
# --------------------------------------------------------------------------
    def tcp_pose_callback(self, msg):
        # CHANGED: Handle Float64MultiArray [x, y, z, rx, ry, rz]
        if len(msg.data) >= 6:
            self.current_tcp_pos = np.array([msg.data[0], msg.data[1], msg.data[2]])

            # Convert Euler to Quaternion for logic
            roll = msg.data[3]
            pitch = msg.data[4]
            yaw = msg.data[5]
            self.current_tcp_orient = euler_to_quaternion(roll, pitch, yaw)

    def force_callback(self, msg):
        self.current_force_z = msg.data

# ==================Attach/Detach=========================================
    def set_gripper_state(self, action):
        """
        Controls the electromagnet using SetBool.
        Only activates if Force > 30.
        Returns True if command was sent, False if blocked.
        """
        req = SetBool.Request()

        if action == 'attach':
            if self.current_force_z is None:
                self.get_logger().warn("Force sensor data None.")
                return False

            if self.current_force_z > 30.0:
                req.data = True
                self.get_logger().info(f"Force Good ({self.current_force_z:.2f} > 30). Magnet ON.")
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

# =================motion commands==========================================
    def stop_joint(self):
        # CHANGED: JointJog
        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities = [0.0] * 6
        self.joint_pub.publish(msg)

    def publish_twist(self, direction, speed):
        # CHANGED: TwistStamped
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.twist.linear.x = float(direction[0] * speed)
        msg.twist.linear.y = float(direction[1] * speed)
        msg.twist.linear.z = float(direction[2] * speed)

        self.twist_pub.publish(msg)

    def stop_all(self):
        self.stop_joint()
        self.publish_twist([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def halt_motion(self):
        self.stop_all()
# -----------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------
    def servo_to_goal(self,
                        currentPosition,
                        currentQuaternion,
                        targetPosition,
                        targetQuaternion,
                        positionTolerance,
                        angularTolerance,
                        maximumLinearVelocity,
                        maximumAngularVelocity,
                        checkOrientation=True):
    # ------  start -----------

        position_error = targetPosition - currentPosition
        distance_value = float(np.linalg.norm(position_error))
        position_ok = (distance_value <= positionTolerance)

        orientation_ok = True
        if checkOrientation:
            error_quaternion = multiply_quaternion(targetQuaternion, conjugate_quaternion(currentQuaternion))
            roll_angle, pitch_angle, yaw_angle = quaternion_to_euler(error_quaternion)
            if abs(roll_angle) <= angularTolerance and abs(pitch_angle) <= angularTolerance and abs(yaw_angle) <= angularTolerance:
                orientation_ok = True
            else:
                orientation_ok = False

        if position_ok and orientation_ok:
            return True

        if distance_value > 1e-6:
            proportional_gain_linear = 2.5
            chosen_speed = proportional_gain_linear * distance_value

            if chosen_speed > maximumLinearVelocity:
                chosen_speed = maximumLinearVelocity

            if distance_value < (positionTolerance * 2.0):
                limited_speed = maximumLinearVelocity * 0.35
                if chosen_speed > limited_speed:
                    chosen_speed = limited_speed

            linear_velocity_vector = (position_error / distance_value) * chosen_speed
        else:
            linear_velocity_vector = np.zeros(3)

        if checkOrientation:
            error_quaternion = multiply_quaternion(targetQuaternion, conjugate_quaternion(currentQuaternion))
            roll_angle, pitch_angle, yaw_angle = quaternion_to_euler(error_quaternion)
            angular_velocity_vector = np.array([roll_angle, pitch_angle, yaw_angle]) * maximumAngularVelocity
        else:
            angular_velocity_vector = np.zeros(3)

        # CHANGED: TwistStamped
        twist_message = TwistStamped()
        twist_message.header.stamp = self.get_clock().now().to_msg()
        twist_message.header.frame_id = 'base_link'

        twist_message.twist.linear.x = float(linear_velocity_vector[0])
        twist_message.twist.linear.y = float(linear_velocity_vector[1])
        twist_message.twist.linear.z = float(linear_velocity_vector[2])
        twist_message.twist.angular.x = float(angular_velocity_vector[0])
        twist_message.twist.angular.y = float(angular_velocity_vector[1])
        twist_message.twist.angular.z = float(angular_velocity_vector[2])

        self.twist_pub.publish(twist_message)

        return False

# ------------------------------------------------------------------------------------------------------------
    def align_joint_to_pose(self, target_pose, target_label, joint_name, joint_index):
        if target_pose is None:
            return False

        x = target_pose[0]
        y = target_pose[1]

        desired = self.norm(np.arctan2(y, x) + np.pi)

        if joint_name not in self.joint_pos:
            return False

        current = self.joint_pos[joint_name]
        err = self.norm(desired - current)

        if abs(err) < self.max_tol:
            self.stop_all()
            self.get_logger().info(f"✓ Aligned {target_label}. Switching phase.")
            return True

        speed = max(min(self.base_kp * err, self.base_max_speed), -self.base_max_speed)

        # CHANGED: JointJog
        cmd = [0.0] * 6
        cmd[joint_index] = float(speed)

        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities = cmd
        self.joint_pub.publish(msg)

        return False

# -----------------------------------------------------------------------------------------------------------------------
    def move_to_tcp_target(self, target, tol=0.01, slow=False):
        if self.current_tcp_pos is None:
            return False

        err = target - self.current_tcp_pos
        dist = np.linalg.norm(err)

        if dist < tol:
            self.stop_all()
            return True

        if dist < 1e-5:
            self.stop_all()
            return True

        direction = err / dist
        max_s = 0.25 if slow else 0.8
        speed = min(dist * 5.0, max_s)

        self.publish_twist(direction, speed)
        return False
# ----------------------------------------------------------------------------------------
    def move_joint_to_angle(self, target_angle, joint_name, joint_index):
        if joint_name not in self.joint_pos: return False

        current = self.joint_pos[joint_name]
        err = self.norm(target_angle - current)

        if abs(err) < self.max_tol:
            self.stop_all()
            return True

        speed = max(min(self.base_kp * err, self.base_max_speed), -self.base_max_speed)

        # CHANGED: JointJog
        cmd = [0.0] * 6
        cmd[joint_index] = float(speed)

        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities = cmd
        self.joint_pub.publish(msg)
        return False

    # ------------------------------------------------------------------------------------------------------
    def move_joint_group(self, targets, speed_scale):
        if not self.joint_pos:
            return False

        joint_map = {
            'shoulder_pan_joint': 0,
            'shoulder_lift_joint': 1,
            'elbow_joint': 2,
            'wrist_1_joint': 3,
            'wrist_2_joint': 4,
            'wrist_3_joint': 5,
        }

        cmd = [0.0] * 6
        all_reached = True

        for joint, target in targets.items():
            if joint not in self.joint_pos:
                all_reached = False
                continue

            idx = joint_map[joint]
            current = self.joint_pos[joint]
            err = self.norm(target - current)

            if abs(err) < self.max_tol:
                cmd[idx] = 0.0
            else:
                all_reached = False
                speed = self.base_kp * err
                speed *= speed_scale.get(joint, 1.0)
                speed = max(min(speed, self.base_max_speed), -self.base_max_speed)
                cmd[idx] = float(speed)

        # CHANGED: JointJog
        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities = cmd
        self.joint_pub.publish(msg)

        if all_reached:
            self.stop_all()

        return all_reached

# -------------------------------------------------------------------------------------------
    def wrist_orientation(self, direction, next_phase, angle=None):
        if 'wrist_1_joint' not in self.joint_pos:
            return

        if not self.phase_initialized:
            current_w1 = self.joint_pos['wrist_1_joint']
            rotation_amount = angle if angle is not None else self.wrist1_delta_down

            if direction == 'down':
                self.target_w1 = current_w1 + rotation_amount
            elif direction == 'up':
                self.target_w1 = current_w1 - rotation_amount
            else:
                return

            self.phase_initialized = True
            return

        current_w1 = self.joint_pos['wrist_1_joint']
        err = self.target_w1 - current_w1

        if abs(err) < self.max_tol:
            self.stop_all()
            self.phase_initialized = False
            self.phase = next_phase
            return

        # CHANGED: JointJog
        cmd = [0.0] * 6
        speed = max(min(self.base_kp * err, self.base_max_speed), -self.base_max_speed)
        cmd[3] = float(speed)

        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities = cmd
        self.joint_pub.publish(msg)

# ------------------------------------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------------------

    def wait_for_timer(self, seconds):
        if self.wait_start_time is None:
            self.wait_start_time = self.get_clock().now()
            return False

        current_time = self.get_clock().now()
        time_diff = (current_time - self.wait_start_time).nanoseconds / 1e9

        if time_diff < seconds:
            return False

        self.wait_start_time = None
        return True
# ---------------------------------------------------------------------------------
    def orient_to_target(self, target_quat, tol=0.03):
            """
            Calculates error between current orientation and target quaternion
            and publishes TwistStamped to correct it.
            """
            if self.current_tcp_orient is None:
                return False

            q_curr_inv = conjugate_quaternion(self.current_tcp_orient)
            q_err = multiply_quaternion(target_quat, q_curr_inv)

            if q_err[3] < 0:
                q_err = -q_err

            xyz_err = q_err[:3]
            error_mag = np.linalg.norm(xyz_err)

            if error_mag < tol:
                self.stop_all()
                return True

            kp_rot = 2.0
            max_rot_speed = 0.5

            ang_speed = min(kp_rot * error_mag, max_rot_speed)

            if error_mag > 0:
                ang_dir = xyz_err / error_mag
            else:
                ang_dir = np.zeros(3)

            # CHANGED: TwistStamped
            t = TwistStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'base_link'

            t.twist.linear.x = 0.0
            t.twist.linear.y = 0.0
            t.twist.linear.z = 0.0
            t.twist.angular.x = float(ang_dir[0] * ang_speed)
            t.twist.angular.y = float(ang_dir[1] * ang_speed)
            t.twist.angular.z = float(ang_dir[2] * ang_speed)

            self.twist_pub.publish(t)
            return False
# --------------------------------------------------------------------------------------------
    @staticmethod
    def norm(a):
        while a > np.pi:
            a -= 2 * np.pi
        while a < -np.pi:
            a += 2 * np.pi
        return a

# =============================main loop======================================================================
    def main_loop(self):
        if self.phase == 'PHASE_1_GETTING_TF':
            # Wait for both Joint States AND TCP Pose
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

                self.get_logger().info(f"✓ Stored Initial Pose: {self.initial_cartesian_pos}")

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
            self.phase = 'PHASE_SAFE_LIFT_SHOULDER'
# ----------------------------------------------------------------------------------------------------------------------------

 # ---------------------------------------------------------------------------------------------------------------------------------

# real world this phase not need
#  -------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_SAFE_LIFT_SHOULDER':
            # 1. Calculate the target ONLY ONCE
            if self.safe_lift_angle is None:
                # Example: Lift shoulder up by 0.2 radians
                current_val = self.joint_pos[self.joint2_name]
                self.safe_lift_angle = current_val - 0.2
                self.get_logger().info(f"Lifting Shoulder... Target: {self.safe_lift_angle:.2f}")
# -------------------------------------------------------------------------------------------------------------------------------------------------
            # 2. Call the helper function (Index 1 is shoulder_lift_joint)
            if self.move_joint_to_angle(self.safe_lift_angle, self.joint2_name, 1):
                self.get_logger().info("✓ Safe Lift Complete. Starting Phase 2.")
                self.phase = 'PHASE_3_ALIGN_TO_FERTI'

        elif self.phase == 'PHASE_3_ALIGN_TO_FERTI':
            # 1. Align Shoulder to Fertilizer
            if self.align_joint_to_pose(self.ferti_pose, 'shoulder', self.joint1, 0):
                self.get_logger().info("✓ Aligned shoulder to fertilizer. Transitioning to PRE_APPROACH.")
                self.phase = 'PHASE_4_PRE_APPROACH'

        # --- UPDATED PHASES FOR APPROACH ---
        elif self.phase == 'PHASE_4_PRE_APPROACH':
            # 1. Create a temp target with +0.10 offset in Y
            target_pre = self.ferti_pose.copy()
            target_pre[1] += 0.10

            # 2. Move there (Normal speed / slow=False)
            reached = self.move_to_tcp_target(target_pre, tol=0.04, slow=False)
            if reached:
                self.get_logger().info("✓ Reached +0.10 offset. Starting slow final approach.")
                self.phase = 'CHECK_ORIENATION'
# ------------------------------------------------------------------------------------------------------------------------------------
# i added this phase  to aling with ferti
        elif self.phase == 'WRIST_ALING_TO_FERTI':
            target_ferti = self.ferti_pose.copy()
            if self.align_joint_to_pose(target_ferti, "Fertilizer", 'wrist_3_joint', 5):
                self.get_logger().info("Wrist aligned to Fertilizer. Proceeding to Final Approach.")
                self.phase = 'CHECK_ORIENATION'
# ---------------------------------------------------------------------------------------------------------------

        elif self.phase == 'CHECK_ORIENATION':
            self.get_logger().info("Checking pickup orientation...", throttle_duration_sec=2.0)

            self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.ferti_pose}")
            if self.orient_to_target(self.pickupOrientationQuaternion, tol=0.01):
                self.get_logger().info("✓ Orientation Correct. Moving to Target.")
                self.phase = 'PHASE_4_FINAL_APPROACH'
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_4_FINAL_APPROACH':
            # self.get_logger().info(f"Force met ({self.current_force_z:.2f}). Attaching Gripper...")
            # 1. Move to actual fertilizer pose (slow=True)
            reached = self.move_to_tcp_target(self.ferti_pose, tol=0.01, slow=True)
            self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.ferti_pose}")
# ================================================================================================================================================================
# this line can me used when there is is collision happend
            # if reached or self.current_force_z > 30.0:
            if reached:
                self.get_logger().info("✓ Reached fertilizer hover position. Waiting before attach...")
                self.publish_twist([0.0, 0.0, 0.0], 0.0) # Stop twist
                self.phase = 'ATTACH_FERTI_PRE_WAIT'
        # -----------------------------------

        # --- NEW ATTACH SEQUENCE WITH FORCE CHECK ---
        elif self.phase == 'ATTACH_FERTI_PRE_WAIT':
            self.get_logger().info("Waiting 3s before attaching...", throttle_duration_sec=1.0)
            if self.wait_for_timer(3.0):
                self.phase = 'ATTACH_FERTI_ACTION'

        elif self.phase == 'ATTACH_FERTI_ACTION':
            # 1. Safety check
            if self.current_force_z is None:
                self.get_logger().warn("Waiting for force sensor data...")
                return

            # 2. Check Force Condition
            if self.current_force_z > 30.0:
                # self.get_logger().info(f"Force met ({self.current_force_z:.2f}). Attaching Gripper...")
                self.set_gripper_state('attach')
                self.phase = 'ATTACH_FERTI_POST_WAIT'
            # else:
                # self.get_logger().info(f"Force too low ({self.current_force_z:.2f}). Waiting for contact...", throttle_duration_sec=1.0)

        elif self.phase == 'ATTACH_FERTI_POST_WAIT':
            self.get_logger().info("Waiting 3s after attaching...", throttle_duration_sec=1.0)
            if self.wait_for_timer(3.0):
                 self.phase = 'PHASE_5_LIFT_FERTILIZER'
        # ---------------------------

        elif self.phase == 'PHASE_5_LIFT_FERTILIZER':
            # 1. Lift Fertilizer
            target = self.ferti_pose.copy()
            target[2] += 0.07
            reached = self.move_to_tcp_target(target, tol=0.01, slow=True)

            if reached:
                self.get_logger().info("✓ Lifted fertilizer. Transitioning to PHASE_6_Wrist_1_Up.")
                self.phase = 'PHASE_6_REVERSE_FERTILZER'

        elif self.phase == 'PHASE_6_REVERSE_FERTILZER':
            # 1. Reverse Fertilizer
            target = self.ferti_pose.copy()
            target[1] += 0.25
            reached = self.move_to_tcp_target(target, tol=0.05, slow=True)

            if reached:
                self.get_logger().info("✓ Reversed fertilizer. Transitioning to PHASE_7_Wrist_1_Up.")
                self.phase = 'PHASE_ALIGN_TO_BIN'

        elif self.phase == 'PHASE_ALIGN_TO_BIN':
            if self.align_joint_to_pose(self.dustbinPosition, "Dustbin", self.joint1, 0):
                self.get_logger().info("Base aligned to Dustbin. Reaching Bin.")
                self.dustbin_hover_pose = self.current_tcp_pos.copy()
                self.phase = 'GRIPPER_ORIENTATION_DOWN_FOR_FERTI'

        elif self.phase == 'GRIPPER_ORIENTATION_DOWN_FOR_FERTI':
            self.wrist_orientation('down', 'PHASE_REACH_BIN', angle=-1.0)

        elif self.phase == 'PHASE_REACH_BIN':
            # 1. Move Linear TCP to the specific Dustbin XYZ
            if self.move_to_tcp_target(self.dustbinPosition, self.approach_tol):
                self.get_logger().info("Reached Dustbin. Stopping and Waiting to Detach.")
                self.publish_twist([0.0, 0.0, 0.0], 0.0) # Stop
                self.phase = 'DETACH_FERTILIZER'

        elif self.phase == 'DETACH_FERTILIZER':
            self.get_logger().info("Waiting 3s before detaching...", throttle_duration_sec=1.0)
            if self.wait_for_timer(3.0):
                # REMOVED: Object Name
                self.set_gripper_state('detach')
                self.phase = 'TASK_COMPLETE'
        # ---------------------------

        elif self.phase == 'TASK_COMPLETE':
            self.get_logger().info("All tasks completed successfully. Shutting down.", throttle_duration_sec=2.0)
            self.stop_joint()
# -------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'DONE':
            self.get_logger().info(f"all task done bro look at the tcp pose ")
            self.get_logger().info(f"   >> Position (XYZ): {self.current_tcp_orient}")
            self.get_logger().info(f"   >> Quaternion (XYZW): {self.joint_pos}")
            pass
# -----------------------------------------------------------------------------------------------------------------------------------------------


def main():
    rclpy.init()
    node = Task5c()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

# -------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()