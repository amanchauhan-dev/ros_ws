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


class Task6(Node):
    def __init__(self):
        super().__init__('Task6')

        self.service_callback_group = ReentrantCallbackGroup()

        # ===================publishers========================

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)

        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)
        self.fertilizer_status_publisher = self.create_publisher(Bool, '/fertilizer_placement_status', 10)

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


        self.ebot_docked = False
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

        # tcp_pose variables
        self.safe_lift_angle = None
        self.current_tcp_pos = None
        self.current_tcp_orient = None
        self.initial_arm_pos = None

        self.ferti_align_joint_state = None
        self.fruits_tray_hover_pos = None
        self.ferti_unload_pose = None

        self.max_tol = np.deg2rad(3)
        self.base_kp = 1.0
        self.base_max_speed = 0.5        # Linear Max
        self.base_max_angular = 1.5
        # phases config

        self.phase = 'START'

        self.ferti_pose = None
        self.fertilizerTFname = f'{self.teamIdentifier}_fertilizer_1'
        self.ebotTransformName = f'{self.teamIdentifier}_ebot_marker'
        self.pickupFertiOrientationQuaternion = np.array([0.707, 0.028, 0.034, 0.707])

        # --------------------------------------------------------------
        self.ebotWorldPosition = None
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
        self.dustbinPosition = np.array([-0.682, 0.210, 0.316])

        # -----------------------------------------------------------------------------------
        self.phase_initialized = False
        self.current_fruits_pose = None
        self.wait_start_time = None
        self.wrist1_delta_down = 1.36  # radinal in the degree  near 90
        self.current_fruit_index = 0
        self.current_force_z = 0.0
        # -----------------------------------------------------------------------------------------

        self.safe_lift_angle = None
        self.dustbin_hover_pose = None

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
            self.get_logger().info(f"Force Attach call  ({self.current_force_z:.2f} ). Magnet ON.")
            self.magnet_client.call_async(req)
            return True
        else:
            req.data = False
            self.get_logger().info("Magnet OFF.")
            self.magnet_client.call_async(req)
            return True

# =================motion commands==========================================
    def stop_joint(self):

        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities = [0.0] * 6
        self.joint_pub.publish(msg)

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

    def stop_all(self):
        self.stop_joint()
        self.publish_twist([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

# =============================DELTA_TWIST_CMD================================

    def move_to_tcp_target_v2(self, target, tol=0.01, slow=False,
                        tol_xy=None, tol_z=None):
        """
        Moves TCP to target [x, y, z].
        
        Default behavior (tol_xy=None, tol_z=None):
            Uses single tol for all axes — same as before (backward compatible).
        
        Per-axis behavior (tol_xy and tol_z specified):
            X err < tol_xy  AND  Y err < tol_xy  AND  Z err < tol_z
            All three must be satisfied. This eliminates the sphere bug.
        
        Recommended usage for hover above fruit:
            move_to_tcp_target(hover_target, tol_xy=0.015, tol_z=0.020)
        
        Recommended usage for final approach to fruit:
            move_to_tcp_target(final_target, tol_xy=0.010, tol_z=0.015, slow=True)
        """
        if self.current_tcp_pos is None:
            return False

        err_vec = target - self.current_tcp_pos
        err_x   = abs(err_vec[0])
        err_y   = abs(err_vec[1])
        err_z   = abs(err_vec[2])
        dist    = np.linalg.norm(err_vec)   # still used for speed calc

        # ── Arrival Check (per-axis if specified, else sphere) ──────────────────
        if tol_xy is not None and tol_z is not None:
            x_ok = err_x < tol_xy
            y_ok = err_y < tol_xy
            z_ok = err_z < tol_z
            reached = x_ok and y_ok and z_ok
            self.get_logger().info(
                f"TCP err | X:{err_x*1000:.1f}mm({'✓' if x_ok else '✗'}) "
                f"Y:{err_y*1000:.1f}mm({'✓' if y_ok else '✗'}) "
                f"Z:{err_z*1000:.1f}mm({'✓' if z_ok else '✗'})",
                throttle_duration_sec=0.4
            )
        else:
            # fallback: original sphere check
            reached = dist < tol

        if reached:
            self.stop_all()
            self.get_logger().info(
                f"✓ TCP reached | X:{err_x*1000:.1f} Y:{err_y*1000:.1f} Z:{err_z*1000:.1f} mm"
            )
            return True

        # ── Speed Zones (unchanged from v2) ─────────────────────────────────────
        if slow:
            if dist > 0.10:
                kp, max_s = 1.5, 0.12
            elif dist > 0.05:
                kp, max_s = 1.0, 0.05
            else:
                kp, max_s = 1.0, 0.02
        else:
            if dist > 0.30:
                kp, max_s = 2.5, 0.50
            elif dist > 0.20:
                kp, max_s = 2.0, 0.40
            elif dist > 0.10:
                kp, max_s = 1.8, 0.20
            else:
                kp, max_s = 1.2, 0.06

        speed      = min(kp * dist, max_s)
        direction  = err_vec / dist
        linear_vel = direction * speed

        self.get_logger().info(
            f"TCP {'SLOW' if slow else 'FAST'} | dist={dist:.3f}m | spd={speed:.3f}",
            throttle_duration_sec=0.4
        )
        self.publish_twist(linear_vel, [0.0, 0.0, 0.0])
        return False

# --------------------------------------------------------------------------------------------------------
    def move_to_tcp_target(self, target, tol=0.01, slow=False):
        """
        Moves TCP to target [x,y,z].  Distance-adaptive P-controller.
        NO hard minimum speed near target → arm gracefully decelerates to stop.
        Force/wrench logic is intentionally NOT here — handle it in your phase.
        Returns True when within tol.
        """
        if self.current_tcp_pos is None:
            return False

        err_vec = target - self.current_tcp_pos
        dist    = np.linalg.norm(err_vec)

        # --- Arrival check ---
        if dist < tol:
            self.stop_all()
            self.get_logger().info(f"✓ TCP reached. dist={dist*1000:.1f}mm")
            return True

        # --- Speed zones ---

        if slow:
            if dist > 0.10:
                kp, max_s = 1.5, 0.12
            elif dist > 0.05:
                kp, max_s = 1.0, 0.05
            else:
                kp, max_s = 1.0, 0.02
        else:
            if dist > 0.30:
                kp, max_s = 2.5, 0.50
            elif dist > 0.20:
                kp, max_s = 2.0, 0.40
            elif dist > 0.10:
                kp, max_s = 1.8, 0.20
            else:
                kp, max_s = 1.2, 0.06

        speed      = min(kp * dist, max_s)
        direction  = err_vec / dist
        linear_vel = direction * speed

        self.get_logger().info(
            f"TCP move | dist={dist:.3f}m | spd={speed:.3f} | slow={slow}",
            throttle_duration_sec=0.1
        )
        self.publish_twist(linear_vel, [0.0, 0.0, 0.0])
        return False


# ---------------------------------------------------------------------------------
    def orient_to_target(self, target_quat, tol=0.05):
        """
        Rotates end-effector to target_quat using angular velocity twist commands.
        Angular velocity published as [rx, ry, rz] in base_link frame.
        tol: quaternion xyz-part magnitude tolerance (≈ half-angle in rad).
            tol=0.05 ≈ 5.7 deg  (use 0.03 for tighter ≈ 3.4 deg)
        Returns True when aligned.
        """
        if self.current_tcp_orient is None:
            return False

        # Error quaternion:  q_err = q_target * inv(q_current)
        # This gives the rotation needed IN THE WORLD/base frame.
        q_curr_inv = conjugate_quaternion(self.current_tcp_orient)
        q_err      = multiply_quaternion(target_quat, q_curr_inv)

        # Shortest path
        if q_err[3] < 0:
            q_err = -q_err

        # q_err = [qx, qy, qz, qw].  The rotation axis*sin(half_angle) = xyz part.
        xyz_err   = q_err[:3]          # shape (3,)
        error_mag = np.linalg.norm(xyz_err)    # ≈ sin(half_angle)
        deg_err   = np.degrees(2.0 * np.arcsin(np.clip(error_mag, 0, 1)))

        if error_mag < tol:
            self.stop_all()
            self.get_logger().info(f"✓ Orientation aligned. err≈{deg_err:.1f}°")
            return True

        # P-controller — NO hard floor speed
        kp_rot    = 3.0
        max_rot   = 0.8   # rad/s

        if error_mag > 0.3:        #
            kp_rot, max_rot = 3.0, 0.8
        elif error_mag > 0.1:      #
            kp_rot, max_rot = 2.5, 0.4
        else:                      #
            kp_rot, max_rot = 2.0, 0.15  # no floor, tapers naturally

        ang_speed  = min(kp_rot * error_mag, max_rot)
        # No floor on ang_speed.  Arm decelerates smoothly.

        # Direction of rotation (unit axis in base frame)
        ang_axis   = xyz_err / error_mag  if error_mag > 1e-6 else np.zeros(3)
        angular_vel = ang_axis * ang_speed   # shape (3,) → rx, ry, rz

        self.get_logger().info(
            f"Orienting | err≈{deg_err:.1f}° | spd={ang_speed:.3f} "
            f"| axis=[{ang_axis[0]:.2f},{ang_axis[1]:.2f},{ang_axis[2]:.2f}]",
            throttle_duration_sec=0.4
        )

        # Linear = 0; pure rotation
        self.publish_twist([0.0, 0.0, 0.0], angular_vel)
        return False

# ============================ JOINT DELTA CMD =====================================================


# ----------------------------------------------------------------------------------------
    def move_joint_to_angle(self, target_angle, joint_name, joint_index, tol=0.02):
        """
        Moves a single joint to target_angle [rad].
        Speed tapers naturally to zero — no hard minimum that causes overshoot.
        """
        if joint_name not in self.joint_pos:
            return False

        current = self.joint_pos[joint_name]
        err     = self.norm(target_angle - current)
        abs_err = abs(err)

        if abs_err < tol:
            self.stop_joint()
            self.get_logger().info(f"✓ Joint {joint_name} at target. err={np.degrees(err):.2f}°")
            return True

        # Speed zones — no hard floor
        if abs_err > 1.5:
            kp, max_s = 2.0, self.base_max_speed
        elif abs_err > 0.5:
            kp, max_s = 1.5, self.base_max_speed
        elif abs_err > 0.17:
            kp, max_s = 1.2, 0.3
        else:                    # close-in: natural decel
            kp, max_s = 1.0, 0.10

        speed = kp * err                        # signed
        speed = max(min(speed, max_s), -max_s)  # clip magnitude, preserve sign

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

# ------------------------------------------------------------------------------------------------------
    def move_joint_group(self, targets, speed_scale):
        """
        Moves multiple joints simultaneously.  Each joint decelerates naturally.
        """
        if not self.joint_pos:
            return False

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

            if abs_err < self.max_tol:
                cmd[idx] = 0.0
                continue

            all_reached = False

            # Speed zones — no hard floor
            if abs_err > 1.5:
                kp, local_max = 2.0, 1.0
            elif abs_err > 0.5:
                kp, local_max = 1.5, 0.5
            elif abs_err > 0.17:
                kp, local_max = 1.2, 0.20
            else:
                kp, local_max = 1.0, 0.10

            speed    = kp * err * speed_scale.get(joint, 1.0)
            speed    = max(min(speed, local_max), -local_max)
            cmd[idx] = float(speed)

        if not all_reached:
            self.get_logger().info(
                f"Group move | max_err={np.degrees(max_err_dbg):.1f}° current joint {joint}  with err {err}",
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

# =====================================extra function ==================================================

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
                self.get_logger().info(f"current fruits id {frame_name} , pos {position}")
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

                # self.initial_arm_pos = self.joint_pos.copy()

                # self.initial_cartesian_pos = self.current_tcp_pos.copy()
                # self.initial_cartesian_orient = self.current_tcp_orient.copy()

                # self.get_logger().info(f"✓ Stored Initial Pose: {self.initial_cartesian_pos}"
                
            self.get_logger().info("✓ All  topic work.")
            self.phase = 'LIFT_FRUIRTS_ATTACH'
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------PHASE_ALIGN_TO_FERTI -------------------------------------------------------------------------------

        elif self.phase == 'LIFT_FRUIRTS_ATTACH':
            if not self.phase_initialized:
                 self.lift_target = self.current_tcp_pos.copy()
                 self.lift_target[2] -= 0.05
                 self.phase_initialized = True
                 self.get_logger().info(f" target : {self.lift_target} current {self.current_tcp_pos}")

            reached = self.move_to_tcp_target(self.lift_target,tol=0.001,slow=True)
            # reached = self.move_to_tcp_target_v2(self.lift_target,tol_xy= 0.001 , tol_z= 0.02 ,slow=True)

            if reached:
                self.get_logger().info(f" moved -z 5cm  {self.current_tcp_orient} , pose current {self.current_tcp_pos}    jointState {self.joint_pos}")
                self.phase_initialized = False
                self.phase = 'DONE'

# ============================================================================================================================================================================================================================================================================================
# ========================================================================  IF WANT TO UNLOAD   the copy paste from old version =============================================================================================================================================================================================

        elif self.phase == 'DONE':
            self.get_logger().info(f"all task done bro look at the tcp pose ", throttle_duration_sec=2.0)
            pass
# --------------------------------------------------------------------------------------------------------------



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

# -------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()