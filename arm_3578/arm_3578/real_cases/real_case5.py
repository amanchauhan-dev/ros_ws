#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import numpy as np
import time
# CHANGED: Interface Imports
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
        self.fertilizer_status_publisher = self.create_publisher(Bool, '/fertilizer_placement_status', 10)
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
        self.magnet_client = self.create_client(
                SetBool,
                '/magnet',
                callback_group=self.service_callback_group
            )


        self.ebot_docked = True
        self.ebot_dock_subscriber = self.create_subscription(
            Bool,
            '/ebot_dock_status',
            self.ebot_status_callback,
            10,
            callback_group=self.service_callback_group
        )


        self.timer = self.create_timer(
                        0.02,
                        self.main_loop,
                        callback_group=self.service_callback_group
                    )

        # ===================configurations========================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.teamIdentifier = '3578'
        self.base_link_name = 'base_link'
        self.joint_pos = {}

        #  variables
        self.safe_lift_angle = None
        self.current_tcp_pos = None
        self.current_tcp_orient = None
        self.initial_arm_pos = None


        self.max_tol = np.deg2rad(3)
        self.base_kp = 1.0
        self.base_max_speed = 0.5
        self.base_max_angular = 1.5
        self.wrist1_delta_down = -1.36

        # phases config

        self.phase = 'START'

        self.ferti_pose = None
        self.ferti_unload_pose = None
        self.fertilizerTFname = f'{self.teamIdentifier}_fertilizer_1'
        self.ebotTransformName = f'{self.teamIdentifier}_ebot_marker'

        # --------------------------------------------------------------
        # self.ebotWorldPosition = None
        self.ebotWorldPosition = np.array([0.711, 0.006, 0.145])
        # ------------------------------------------------------------

        self.badFruitTable = []
        self.badFruitFrameList = [
            f'{self.teamIdentifier}_bad_fruit_1',
              f'{self.teamIdentifier}_bad_fruit_2',
                f'{self.teamIdentifier}_bad_fruit_3',
        ]

        self.fruitHomePosition = np.array([-0.159, 0.501, 0.415])
        # -----------------------------------------------------------------------------------
        self.dustbinPosition = np.array([-0.806, 0.010, 0.182])
        # -----------------------------------------------------------------------------------
        self.phase_initialized = False
        self.current_fruits_pose = None
        self.wait_start_time = None
        self.current_fruit_index = 0
        self.current_force_z = 0.0
        # -----------------------------------------------------------------------------------------



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


# -------------------------------------------------------------------------
    def ebot_status_callback(self, msg):

        if msg.data :
            if not self.ebot_docked:
                self.get_logger().info("✓ eBot has reached the dock station!")
                self.ebot_docked = True

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

            req.data = True
            self.get_logger().info(f"Force Good ({self.current_force_z:.2f} > 30). Magnet ON.")
            self.magnet_client.call_async(req)
            return True

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

    def publish_twist(self, linear_vel, angular_vel=None):
        # CHANGED: Use TwistStamped
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

    def stop_all(self):
        self.stop_joint()
        self.publish_twist([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

# =============================DELTA_TWIST_CMD================================


# --------------------------------------------------------------------------------------------------------
# this is simple version of servo to goal   for only to move x,y,z
    def move_to_tcp_target(self, target, tol=0.01, slow=False):
        """
        A simplified position-only mover.
        Useful as a fallback or for simple XYZ adjustments.
        """
        if self.current_tcp_pos is None:
            return False

        # 1. Calculate Error
        err = target - self.current_tcp_pos
        dist = np.linalg.norm(err)

        if dist < tol:
            self.stop_all()
            self.get_logger().info(f"✓ Reached Target (XYZ Only). Final Error: {dist:.4f}m")
            return True

        # 3. Determine Speed Zone (The Creep Approach)
        if slow:
            max_s = 0.15
            kp = 1.5
        else:
            max_s = 0.4
            kp = 2.0

        if dist < 0.05:
            max_s = 0.05
            kp = 1.0

        # 4. Calculate Velocity
        desired_speed = kp * dist
        speed = min(desired_speed, max_s)

        if speed < 0.05:
            speed = 0.05

        direction = err / dist
        linear_vel = direction * speed

        # 5. Logging (Throttled)
        self.get_logger().info(
            f"Fallback Move | Dist: {dist:.3f}m | Speed: {speed:.3f}",
            throttle_duration_sec=0.5
        )

        # 6. Publish
        self.publish_twist(linear_vel, angular_vel=None)

        return False


# ============================ JOINT DELTA CMD =====================================================


# ----------------------------------------------------------------------------------------
    def move_joint_to_angle(self, target_angle, joint_name, joint_index, tol=0.05):
        """
        Master function to move any single joint to a specific angle.
        Includes Speed Ramping and Debug Logging.
        """
        if joint_name not in self.joint_pos:
            return False

        current = self.joint_pos[joint_name]
        err = self.norm(target_angle - current)

        if abs(err) < tol:
            self.stop_joint()
            self.get_logger().info(f"✓ Joint {joint_name} Reached. Err: {err:.4f}")
            return True

        if abs(err) > 1.5:
            kp = 2.0
            max_s = self.base_max_speed
        elif abs(err) > 0.17 or abs(err) > 1.0 :
            kp = 1.5
            max_s = 0.5
        else:
            kp = 1.0
            max_s = 0.1

        speed = kp * err
        speed = max(min(speed, max_s), -max_s)

        # If speed is too small (but not zero), boost it
        if abs(speed) < 0.05:
            if speed > 0:
                speed = 0.05
            else:
                speed = -0.05
        # --------------------------------

        deg_err = np.degrees(err)
        self.get_logger().info(
            f"Joint: {joint_name} | Err: {deg_err:.1f}° | Cmd: {speed:.3f}",
            throttle_duration_sec=0.5
        )
        # ==========================================================

        cmd = [0.0] * 6
        cmd[joint_index] = float(speed)

        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities = cmd
        self.joint_pub.publish(msg)

        return False

# ------------------------------------------------------------------------------------------------------
    def move_joint_group(self, targets, speed_scale):
        """
        Moves multiple joints simultaneously.
        Includes safety ramping for each individual joint.
        """
        if not self.joint_pos:
            return False

        joint_map = {
            'shoulder_pan_joint': 0, 'shoulder_lift_joint': 1, 'elbow_joint': 2,
            'wrist_1_joint': 3, 'wrist_2_joint': 4, 'wrist_3_joint': 5,
        }

        cmd = [0.0] * 6
        all_reached = True
        max_err_debug = 0.0 # Track the worst error for logging

        for joint, target in targets.items():
            if joint not in self.joint_pos:
                # If we can't see the joint, assume we aren't there yet
                all_reached = False
                continue

            idx = joint_map[joint]
            current = self.joint_pos[joint]
            err = self.norm(target - current)
            abs_err = abs(err)

            # Update debug stat
            if abs_err > max_err_debug:
                max_err_debug = abs_err

            # 1. Check if this specific joint is done
            if abs_err < self.max_tol:
                cmd[idx] = 0.0
            else:
                all_reached = False

                # 2. Speed Ramping (Per Joint)
                if abs_err > 1.5:          # Far
                    kp = 2.0
                    local_max = self.base_max_speed
                elif abs(err) > 0.17 or abs(err) > 1.0 :         # Medium
                    kp = 1.5
                    local_max = 0.5
                else:                      # Close (Creep)
                    kp = 1.0
                    local_max = 0.1

                # Calculate Base Speed
                speed = kp * err

                # Apply custom scale (if you want one joint slower than others)
                speed *= speed_scale.get(joint, 1.0)


                # Clip to safe limits
                speed = max(min(speed, local_max), -local_max)
            # lower speed control
                if abs(speed) < 0.05:
                    if speed > 0:
                        speed = 0.05
                    else:
                        speed = -0.05
                # --------------------------------

                cmd[idx] = float(speed)

        # 3. Log Status (Throttled)
        if not all_reached:
            self.get_logger().info(
                f"Group Move... Max Joint Err: {np.degrees(max_err_debug):.1f}°",
                throttle_duration_sec=0.5
            )

        # 4. Publish Command
        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities = cmd
        self.joint_pub.publish(msg)

        # 5. Stop if everyone is done
        if all_reached:
            self.stop_all()
            self.get_logger().info("✓ Joint Group Target Reached.")

        return all_reached


# ------------------------------------------------------------------------------------------------------------
    def align_joint_to_pose(self, target_pose, target_label, joint_name, joint_index):
        """
        Calculates required angle to look at a target (X,Y) and moves the joint.
        """
        if target_pose is None: return False

        # 1. The Math (Calculate desired angle)
        x = target_pose[0]
        y = target_pose[1]
        desired_angle = self.norm(np.arctan2(y, x) + np.pi)

        # 2. The Movement (Delegate to the master function)
        # We return whatever the master function returns (True if done, False if moving)
        success = self.move_joint_to_angle(desired_angle, joint_name, joint_index)

        if success:
            self.get_logger().info(f"✓ Aligned {target_label} (Joint {joint_index})")

        return success

# =====================================extra function ====================================

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
        """
        Non-blocking timer. Returns True when 'seconds' have passed.
        """
        # 1. Start the timer if not started
        if self.wait_start_time is None:
            self.wait_start_time = self.get_clock().now()
            self.get_logger().info(f" Starting {seconds}s wait...")
            return False

        # 2. Check elapsed time
        current_time = self.get_clock().now()
        time_diff = (current_time - self.wait_start_time).nanoseconds / 1e9

        # 3. Log status while waiting (only every 0.5s to avoid spam)
        if time_diff < seconds:
            self.get_logger().info(
                f"Waiting... ({time_diff:.1f}/{seconds:.1f}s)",
                throttle_duration_sec=0.5
            )
            return False

        # 4. Timer Finished
        self.wait_start_time = None
        self.get_logger().info("Wait Complete.")
        return True


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
        if self.phase == 'START':
            self.get_logger().info("phase machine in the gazebo is start ")
            self.phase = 'PHASE_GETTING_TF'

        elif self.phase == 'PHASE_GETTING_TF':
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
            self.phase = 'PHASE_ALIGN_TO_FERTI'

# -----------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'PHASE_SAFE_LIFT_SHOULDER':
            if self.safe_lift_angle is None:
                current_val = self.joint_pos[self.joint_names_list[1]]
                self.safe_lift_angle = current_val - 0.2
                self.get_logger().info(f"Lifting Shoulder...  Note only for the gazebo because the elbow get colied with the fruits tray   Target: {self.safe_lift_angle:.2f}")
            # 2  we are approach the target
            if self.move_joint_to_angle(self.safe_lift_angle, self.joint_names_list[1], 1):
                self.get_logger().info(" Safe Lift Complete. we are moving into next phase ")
                self.phase = 'PHASE_ALIGN_TO_FERTI'

# ----------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'PHASE_ALIGN_TO_FERTI':
            # 1. Align Shoulder to Fertilizer
            if self.align_joint_to_pose(self.ferti_pose, 'shoulder', self.joint_names_list[0], 0):
                self.get_logger().info("Aligned shoulder to fertilizer. Transitioning to PRE_APPROACH.")

                self.phase = 'ALING_WAIT_FERTI'
# ---------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ALING_WAIT_FERTI':
            if self.wait_for_timer(3.0):
                self.phase = 'PHASE_PRE_APPROACH'
# ----------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_PRE_APPROACH':
            target_pre = self.ferti_pose.copy()
            target_pre[1] += 0.15

            reached = self.move_to_tcp_target(target_pre, tol=0.01, slow=True)

            if reached:
                self.get_logger().info(f" this is when  intint {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.get_logger().info(" Reached +0.10 offset. n ow ckeck orenation first .")
                self.phase = 'PREAPPROACH_WAIT_FERTI'

# ---------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PREAPPROACH_WAIT_FERTI':
            if self.wait_for_timer(3.0):
                self.phase = 'MOVE_ORENATION_REQUIRED'
# ----------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'MOVE_ORENATION_REQUIRED':

            targets = {


                'wrist_1_joint': -0.100,
                'wrist_2_joint': 1.9,
            }

            speed_scale = {

                'wrist_1_joint': 0.5,
                'wrist_2_joint': 0.5,
            }



            if self.move_joint_group(targets, speed_scale):
                self.get_logger().info(f" orentaion wala hai   {self.current_tcp_orient} , pose current {self.current_tcp_pos} \n, frti {self.ferti_pose} \n,jointState {self.joint_pos}")
                self.phase = 'PHASE_FINAL_APPROACH_WAIT'

        elif self.phase == 'PHASE_FINAL_APPROACH_WAIT':
            if self.wait_for_timer(3.0):
                self.get_logger().info("Fertilizer MAgnet start hai ")
                self.set_gripper_state('attach')
                self.phase = 'PHASE_FINAL_APPROACH'

# -----------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_FINAL_APPROACH':

            self.get_logger().info(f"current magnet force  ({self.current_force_z:.2f} )")

            if not self.phase_initialized:
                self.final_ferti_target = self.ferti_pose.copy()
                self.final_ferti_target[0] += 0.05
                self.final_ferti_target[1] -= 0.015
                self.final_ferti_target[2] -= 0.01
                self.phase_initialized = True
                self.get_logger().info(f"we are set offset now move to target ")
                self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.ferti_pose} , jointState {self.joint_pos}")

            reached = self.move_to_tcp_target(self.final_ferti_target, tol=0.001, slow=True)

            if reached or (self.current_force_z > 12.0) :
                self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.ferti_pose} , jointState {self.joint_pos}")
                self.get_logger().info("Reached fertilizer position. Waiting before attach...")
                self.phase_initialized = False
                self.phase = 'ATTACH_FERTI_PRE_WAIT'

# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ATTACH_FERTI_PRE_WAIT':
            if self.wait_for_timer(5.0):
                self.phase = 'ATTACH_FERTI_ACTION'
#  the below phase is garbage nut the old concept therfore i kept here
        elif self.phase == 'ATTACH_FERTI_ACTION':
                self.set_gripper_state('attach')
                self.phase = 'PHASE_LIFT_FERTILIZER'

# -----------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_LIFT_FERTILIZER':
            if not self.phase_initialized:
                 self.lift_target = self.current_tcp_pos.copy()
                 self.lift_target[2] += 0.05
                 self.phase_initialized = True
                 self.get_logger().info(f"Lifting to: {self.lift_target}")

            if self.move_to_tcp_target(self.lift_target, tol=0.02, slow=True):
                self.get_logger().info(f" this is when  lift {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'PHASE_REVERSE_FROM_FERTI'

# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_REVERSE_FROM_FERTI_WAIT':
            if self.wait_for_timer(3.0):
                self.phase = 'PHASE_REVERSE_FROM_FERTI'

# -----------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_REVERSE_FROM_FERTI':
            if not self.phase_initialized:
                self.reverse_target = self.current_tcp_pos.copy()
                self.reverse_target[1] += 0.15
                self.phase_initialized = True
                self.get_logger().info("Reversing safely...")

            if self.move_to_tcp_target(self.reverse_target, tol=0.02, slow=True):
                self.get_logger().info("Reverse Complete.")
                self.phase_initialized = False
                self.phase = 'REVRSE_TO_ALING_FERTI_WAIT'
# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'REVRSE_TO_ALING_FERTI_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'REVRSE_TO_ALING_FERTI'
# --------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'REVRSE_TO_ALING_FERTI':


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
                self.get_logger().info("Fertilizer alignment restored (FAST).")
                self.phase = 'ALING_TO_INTI_WAIT'

# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ALING_TO_INTI_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'PHASE_ALIGN_TO_INIT'
# ----------------------------------------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'PHASE_GRIPPER_ORIENTATION_DOWN':

            joint_name = 'wrist_1_joint'
            joint_idx = 3

            if not self.phase_initialized:
                if joint_name not in self.joint_pos:
                    return

                self.target_wrist_val = self.joint_pos[joint_name] + self.wrist1_delta_down

                self.phase_initialized = True
                self.get_logger().info(f"Rotating Wrist Down... Target: {self.target_wrist_val:.2f}")

            if self.move_joint_to_angle(self.target_wrist_val, joint_name, joint_idx):
                self.get_logger().info(" Wrist Oriented Down.")
                self.get_logger().info(f"gripper down phase {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.phase_initialized = False
                self.phase = 'GETTING_TF_EBOT'

# ------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'GETTING_TF_EBOT':

            # 1. WAIT FOR DOCK SIGNAL
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
                    self.get_logger().info("✓ Found ebot TF. Transitioning to PHASE_7_ALING_TO_IN.")
                    self.phase = 'MOVED_FOR_EBOT_HOVER'

# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'MOVED_FOR_EBOT_HOVER':
            # Move to HOVER position
            target = self.ebotWorldPosition.copy()
            target[2] += 0.20

            if self.move_to_tcp_target(target, 0.01):
                self.get_logger().info("Hovered over eBot. Descending to DROP.")
                self.phase = 'FINAL_APPROACH_EBOT_WAIT'


# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'FINAL_APPROACH_EBOT_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'DROP_FERTI_ON_EBOT'

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
#   BELOW  pHASE Might be issue remove if need
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

#         elif self.phase == 'FINAL_APPROACH_EBOT':
#             target =  self.ebotWorldPosition.copy()
#             target[2] -= 0.05
#             if self.move_to_tcp_target(target , 0.01):
#                 self.get_logger().info("eBot is  Descending  now going to drop")
#                 self.phase = 'DROP_FERTI_ON_EBOT'
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'DROP_FERTI_ON_EBOT':
            self.set_gripper_state('detach')
            self.get_logger().info("detach hogaya hai bhai ")
            self.phase = 'RETARCT_FROM_EBOT_WAIT'

# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'RETARCT_FROM_EBOT_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'RETARCT_FROM_EBOT'

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'RETARCT_FROM_EBOT':
            self.get_logger().info("we are retract from intial pose")
            if not self.phase_initialized:
                 self.Retract_target = self.current_tcp_pos.copy()
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
            self.get_logger().info(" Signaling Ebot to move.")

            ferti_placed_msg = Bool()
            ferti_placed_msg.data = True
            self.fertilizer_status_publisher.publish(ferti_placed_msg)

            self.ebot_docked = False

            self.get_logger().info("Dock flag reset. Moving to Initial Phase.")

            self.phase = 'FRUITS_TRAY_ALING'

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# =======================================================================================================================================================================================
# ------------------------------------------------------------fruits sort phase ------------------------------------------------------------------------------------------------
# ========================================================================================================================================================================================



        elif self.phase  == 'FRUITS_TRAY_ALING':
            if self.move_to_tcp_target(self.fruitHomePosition, tol=0.02, slow=True):
                self.get_logger().info(f" hover to the fruits tray{self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.get_logger().info(" we are at the hover at the fruits tary ")
                self.phase = 'SETTEL_FRUITS_TRAY'

# -----------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'SETTEL_FRUITS_TRAY':
            if self.wait_for_timer(3.0):
                self.phase = 'APPROACH_FRUITS'

# ---------------------------------------------------------------------------------------------------------------------------------
# THERE IS MIGHT BE SOME ISSUE CAN BE HAPPEND WITH THE GRIPPER BETTTER IF WE USED THE ARM POSE CHECKER LIKE FERTILIZER

# -------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'APPROACH_FRUITS':

            if self.current_fruit_index >= len(self.badFruitTable):
                self.get_logger().info("All fruits sorted. Stopping.")
                self.stop_joint()
                self.phase = 'UNLOAD_BASE_ALING'
                return

            fruit_record = self.badFruitTable[self.current_fruit_index]
            original_fruit_pose = fruit_record['pos']

            self.current_fruits_pose = original_fruit_pose.copy()

            hover_target = original_fruit_pose.copy()
            hover_target[2] += 0.10

            reached = self.move_to_tcp_target(hover_target,tol=0.01)

            if reached:
                self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.current_fruits_pose},  jointState {self.joint_pos}")
                self.get_logger().info(" we are at the hover at the fruits   now go  check the orenation  of pickup ")
                self.phase = 'FRUIST_HOVER_WAIT'

# ---------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'FRUIST_HOVER_WAIT':
            if self.wait_for_timer(3.0):
                self.set_gripper_state('attach')
                self.get_logger().info("magnet start here so get attach the cane ")
                self.phase = 'CURRECT_FRUITS_POSE_FINAL_APPROACH'

# -----------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'CURRECT_FRUITS_POSE_FINAL_APPROACH':

            self.get_logger().info(f"current magnet force  ({self.current_force_z:.2f} )")

            final_target = self.current_fruits_pose.copy()
            final_target[0]  -=0.04
            final_target[1] -= 0.01
            # final_target[2] = -0.005
            self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos} , fruits {final_target}, fruits {self.current_fruits_pose}   jointState {self.joint_pos}")
            reached = self.move_to_tcp_target(final_target,tol=0.001,slow=True)

            if reached or (self.current_force_z > 30.0):
                self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos} , fruits {final_target}, fruits {self.current_fruits_pose}   jointState {self.joint_pos}")
                self.get_logger().info(" arm on the fruits call attach")
                self.phase = 'ATTACH_FRUITS_PRE_WAIT'

# -------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ATTACH_FRUITS_PRE_WAIT':
            if self.wait_for_timer(3.0):
                self.phase = 'ATTACH_FRUITS_ACTION'

        elif self.phase == 'ATTACH_FRUITS_ACTION':
                self.set_gripper_state('attach')
                self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos}    jointState {self.joint_pos}")
                self.phase = 'LIFT_FRUIRTS_ATTACH'


# -------------------------------------------------------------------------------------------------------------
        elif self.phase == 'LIFT_FRUIRTS_ATTACH':
             if not self.phase_initialized:
                self.temp = self.joint_pos.copy()
                self.phase_initialized = True

             targets = {


                'elbow_joint': self.temp['elbow_joint'] + 0.34,
                'wrist_1_joint': self.temp['wrist_1_joint'] - 0.30,
                # 'wrist_3_joint': self.temp['wrist_3_joint'] - 0.30,

            }

             speed_scale = {

                'elbow_joint': 0.5,
                'wrist_1_joint': 0.5,
                # 'wrist_3_joint': 0.5,

            }

            #  if this fail the use indivdual joint
            # self.move_joint_to_angle(self.safe_lift_angle, self.joint_names_list[1], 1):

             reached = self.move_joint_group(targets, speed_scale)
             if reached :
                self.get_logger().info(f" lift fruits  wala hai   {self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.ferti_pose}")
                self.get_logger().info("we are lif success fully ")
                self.temp = None
                self.phase_initialized = False
                self.phase = 'LIFT_FRUITS_WAIT'

# ------------------------------------------------------------------------------------------------------------

        elif self.phase == 'LIFT_FRUITS_WAIT':
            if self.wait_for_timer(3.0):
                self.phase = 'MOVED_FOR_DUSTBIN'

# ------------------------------------------------------------------------------------------------------------

        elif self.phase == 'MOVED_FOR_DUSTBIN':

            reached = self.move_to_tcp_target(self.dustbinPosition,tol= 0.22,slow=True)

            self.get_logger().info(f"  fruits unload time  {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
            if reached:
                self.get_logger().info("WE ARE AT DUSTBIN .")
                self.phase = 'DUSTBIN_WAIT'

# ------------------------------------------------------------------------------------------------------------

        elif self.phase == 'DUSTBIN_WAIT':
            if self.wait_for_timer(3.0):
                self.phase = 'CALL_DEATTACH_FRUITS'

# -------------------------------------------------------------------------------------------------------------

        elif self.phase == 'CALL_DEATTACH_FRUITS':
                self.get_logger().info("Deactivating Gripper...")
                self.set_gripper_state('detach')
                self.current_fruit_index += 1
                self.phase = 'FRUITS_TRAY_ALING'

# -----------------------------------------------------------------------------------------------------------------

        elif self.phase  == 'FRUITS_TRAY_ALING':
            if self.move_to_tcp_target(self.fruitHomePosition, tol=0.02):
                self.get_logger().info(f" hover to the fruits tray{self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.get_logger().info(" we are at the hover at the fruits tary ")
                self.phase = 'SETTEL_FRUITS_TRAY'

# =================================================================================================================================================================
# -----------------------------------------------------------------------------------------------------ferti unload logic ----------------------------------------------------
# ==================================================================================================================================================================

        elif self.phase == 'UNLOAD_BASE_ALING':

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
                self.phase = 'UNLOAD_FERTI_INTIAL_ALING'



        elif self.pahse == 'UNLOAD_FERTI_INTIAL_ALING':

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
                self.get_logger().info("Fertilizer alignment restored (FAST).")
                self.phase = 'UNLOAD_EBOT_WAIT'

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'UNLOAD_EBOT_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'UNLOAD_EBOT_TF'
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_EBOT_TF':

            if not self.ebot_docked:
                self.get_logger().info("Waiting for eBot to return with fertilizer...", throttle_duration_sec=2.0)
                return

            # Always try to look up the new position, don't check if self.ferti_pose is None
            self.ferti_unload_pose = self.lookup_tf(self.base_link_name, self.fertilizerTFname)

            # --- 2. Check Transition Condition ---
            if self.ferti_unload_pose is None:
                self.get_logger().info("Waiting for fertilizer TF  new ...", throttle_duration_sec=2.0)
                return

            self.get_logger().info("getting fertilzer here again ")
            self.phase = 'UNLOAD_FERTI_PREPOSE'

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'UNLOAD_FERTI_PREPOSE':
            if not self.phase_initialized:
                 self.ferti_unload_target = self.ferti_unload_pose.copy()
                 self.ferti_unload_target[2] += 0.05
                 self.phase_initialized = True
                 self.get_logger().info(f"unload target approach: {self.ferti_unload_target}   target : {self.ferti_unload_pose}")

            if self.move_to_tcp_target(self.ferti_unload_target, tol=0.01, slow=False):
                self.get_logger().info(f"  ferti unload pre {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'UNLOAD_FERTI_FINAL_WAIT'

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'UNLOAD_FERTI_FINAL_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'UNLOAD_FERTI_FINAL'

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'UNLOAD_FERTI_FINAL':

            if not self.phase_initialized:
                 self.ferti_unload_target_final = self.ferti_unload_pose.copy()
                 self.ferti_unload_target_final[2] += 0.001
                 self.ferti_unload_target_final[0]  -=0.01
                 self.phase_initialized = True
                 self.get_logger().info(f"unload target approach: {self.ferti_unload_target_final}   target : {self.ferti_unload_pose}")

            if self.move_to_tcp_target(self.ferti_unload_target_final, tol=0.01, slow=True):
                self.get_logger().info(f"  ferti unload  final {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'LIFT_TARGET_UNLOAD_WAIT'
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'LIFT_TARGET_UNLOAD_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'LIFT_TARGET_UNLOAD'
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'LIFT_TARGET_UNLOAD':
             if not self.phase_initialized:
                self.temp = self.joint_pos.copy()
                self.phase_initialized = True

             targets = {


                'elbow_joint': self.temp['elbow_joint'] + 0.34,
                'wrist_1_joint': self.temp['wrist_1_joint'] - 0.33,
            }

             speed_scale = {

                'elbow_joint': 0.5,
                'wrist_1_joint': 0.5,

            }

            #  if this fail the use indivdual joint
            # self.move_joint_to_angle(self.safe_lift_angle, self.joint_names_list[1], 1):

             reached = self.move_joint_group(targets, speed_scale)
             if reached :
                self.get_logger().info(f" ferti unload wala hai   {self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.ferti_pose}")
                self.get_logger().info("we are lif success fully ")
                self.temp = None
                self.phase_initialized = False
                self.phase = 'ALING_TO_TRAY_UNLOAD_WAIT'

# ------------------------------------------------------------------------------------------------------------

        elif self.phase == 'ALING_TO_TRAY_UNLOAD_WAIT':
            if self.wait_for_timer(3.0):
                self.phase = 'ALING_TO_TRAY_UNLOAD'

# --------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ALING_TO_TRAY_UNLOAD':
            if self.move_to_tcp_target(self.fruitHomePosition, tol=0.02, slow=True):
                self.get_logger().info(f" hover to the fruits tray  for unload the ferti {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.phase = 'ALING_TO_DUSTBIN_UNLOAD'

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ALING_TO_DUSTBIN_UNLOAD':
            reached = self.move_to_tcp_target(self.dustbinPosition,tol= 0.10,slow=True)

            if reached:
                self.get_logger().info(f"  ferti unload dustbin  {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.get_logger().info("WE ARE AT DUSTBIN .")
                self.phase = 'CALL_DEATTACH_FERTI_UNLOAD'

# ------------------------------------------------------------------------------------------------------------

        elif self.phase == 'CALL_DEATTACH_FERTI_UNLOAD':
                self.get_logger().info("Deactivating Gripper...")
                self.set_gripper_state('detach')
                self.phase = 'DONE'

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'DONE':
            self.get_logger().info(f"all task done bro look at the tcp pose ", throttle_duration_sec=2.0)
            pass
# -------------------------------------------------------------------------------------------------------------------


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