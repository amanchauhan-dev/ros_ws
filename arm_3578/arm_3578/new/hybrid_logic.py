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
        self.current_euler_angel=None

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
        self.FERTI_PICKUP_QUAT = np.array([0.05367629, 0.70780447, 0.70200457, 0.05763047])

        # --------------------------------------------------------------
        self.ebotWorldPosition = None
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
            self.current_euler_angel = np.array([roll, pitch, yaw])
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


    def move_to_tcp_pose_v3(
            self,
            target_pos,
            target_quat,
            tol_xy=0.015,
            tol_z=0.020,
            tol_rot=0.05,
            slow=False,
            ang_kp=1.0,
            ang_max=0.5):
        """
        Moves TCP to target position AND orientation simultaneously.

        Args:
            target_pos  : np.array [x, y, z]
            target_quat : np.array [qx, qy, qz, qw]
            tol_xy      : XY position tolerance in metres
            tol_z       : Z  position tolerance in metres
            tol_rot     : rotation tolerance in radians (~0.05 rad ≈ 3°)
            slow        : True = slow speed profile (final approach)
            ang_kp      : angular P-gain
            ang_max     : max angular speed (rad/s)

        Returns:
            True  when BOTH position AND orientation are within tolerance.
            False while still moving.
        """
        # ── Guard ─────────────────────────────────────────────────────────
        if self.current_tcp_pos is None or self.current_tcp_orient is None:
            return False

        # ── Position error ─────────────────────────────────────────────────
        err_vec = target_pos - self.current_tcp_pos
        err_x   = abs(err_vec[0])
        err_y   = abs(err_vec[1])
        err_z   = abs(err_vec[2])
        dist    = np.linalg.norm(err_vec)

        x_ok = err_x < tol_xy
        y_ok = err_y < tol_xy
        z_ok = err_z < tol_z

        # ── Orientation error (quaternion difference) ──────────────────────
        # q_err = q_target * conjugate(q_current)
        # For unit quaternion: conjugate = inverse = [-qx, -qy, -qz, qw]
        cq = self.current_tcp_orient   # [qx, qy, qz, qw]
        tq = target_quat               # [qx, qy, qz, qw]

        cq_inv = np.array([-cq[0], -cq[1], -cq[2], cq[3]])

        # Hamilton product: q_err = tq * cq_inv
        ex = tq[3]*cq_inv[0] + tq[0]*cq_inv[3] + tq[1]*cq_inv[2] - tq[2]*cq_inv[1]
        ey = tq[3]*cq_inv[1] - tq[0]*cq_inv[2] + tq[1]*cq_inv[3] + tq[2]*cq_inv[0]
        ez = tq[3]*cq_inv[2] + tq[0]*cq_inv[1] - tq[1]*cq_inv[0] + tq[2]*cq_inv[3]
        ew = tq[3]*cq_inv[3] - tq[0]*cq_inv[0] - tq[1]*cq_inv[1] - tq[2]*cq_inv[2]

        # Shortest-path: if w < 0 flip all — takes the short arc not the long one
        if ew < 0:
            ex, ey, ez, ew = -ex, -ey, -ez, -ew

        # Rotation error magnitude in radians
        rot_err = 2.0 * np.arctan2(np.linalg.norm([ex, ey, ez]), ew)
        rot_ok  = rot_err < tol_rot

        # ── Debug log ──────────────────────────────────────────────────────
        self.get_logger().info(
            f"[V3] X:{err_x*1000:.1f}mm({'✓' if x_ok else '✗'}) "
            f"Y:{err_y*1000:.1f}mm({'✓' if y_ok else '✗'}) "
            f"Z:{err_z*1000:.1f}mm({'✓' if z_ok else '✗'}) "
            f"| rot:{np.degrees(rot_err):.1f}°({'✓' if rot_ok else '✗'})",
            throttle_duration_sec=0.4
        )

        # ── Arrival check ──────────────────────────────────────────────────
        if x_ok and y_ok and z_ok and rot_ok:
            self.stop_all()
            self.get_logger().info(
                f"✓ V3 reached | "
                f"X:{err_x*1000:.1f} Y:{err_y*1000:.1f} Z:{err_z*1000:.1f} mm "
                f"rot:{np.degrees(rot_err):.1f}°"
            )
            return True

        # ── Angular velocity ───────────────────────────────────────────────
        # [ex,ey,ez] = rotation_axis * sin(θ/2) → multiply by 2*kp for velocity
        ang_vel_raw = 2.0 * ang_kp * np.array([ex, ey, ez])
        ang_speed   = np.linalg.norm(ang_vel_raw)
        ang_vel     = (ang_vel_raw / ang_speed * min(ang_speed, ang_max)
                       if ang_speed > 1e-6 else np.zeros(3))

        # ── Linear velocity ────────────────────────────────────────────────
        if slow:
            if dist > 0.10:   kp, max_s = 1.5, 0.12
            elif dist > 0.05: kp, max_s = 1.0, 0.05
            else:             kp, max_s = 1.0, 0.01
        else:
            if dist > 0.30:   kp, max_s = 2.5, 0.50
            elif dist > 0.20: kp, max_s = 2.0, 0.40
            elif dist > 0.10: kp, max_s = 1.8, 0.20
            else:             kp, max_s = 1.2, 0.06

        lin_speed = min(kp * dist, max_s) if dist > 1e-6 else 0.0
        lin_vel   = (err_vec / dist) * lin_speed if dist > 1e-6 else np.zeros(3)

        self.publish_twist(lin_vel, ang_vel)
        return False

# ---------------------------------------------------------------------------------

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
                kp, max_s = 1.0, 0.01
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
                kp, max_s = 1.0, 0.01
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
                kp, local_max = 1.0, 0.1

            speed    = kp * err * speed_scale.get(joint, 1.0)
            speed    = max(min(speed, local_max), -local_max)

     
            cmd[idx] = float(speed)

        if not all_reached:
            self.get_logger().info(
                f"Group move | max_err={np.degrees(max_err_dbg):.1f}° current joint {joint}  with err {err:.3f} speed{speed:.3f}",
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

# ----------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'PHASE_ALIGN_TO_FERTI':
            if self.align_joint_to_pose(self.ferti_pose, 'shoulder', self.joint_names_list[0], 0):
                self.get_logger().info("Aligned shoulder to fertilizer. Transitioning to PRE_APPROACH.")
                self.phase = 'ALING_WAIT_FERTI'

# ---------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ALING_WAIT_FERTI':
            if self.wait_for_timer(0.5):
                self.phase = 'PHASE_PRE_APPROACH'
# ----------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_PRE_APPROACH':
            target_pre = self.ferti_pose.copy()
            target_pre[1] += 0.15

            reached = self.move_to_tcp_pose_v3(
                target_pos  = target_pre,
                target_quat = self.FERTI_PICKUP_QUAT,
                tol_xy      = 0.015,
                tol_z       = 0.025,
                tol_rot     = 0.05,
                slow        = False
            )

            if reached:
                self.get_logger().info(
                    f"✓ PRE_APPROACH reached with correct orientation. "
                    f"orient={self.current_tcp_orient} euler={self.current_euler_angel} "
                    f"pos={self.current_tcp_pos}"
                )
                self.phase = 'PREAPPROACH_WAIT_FERTI'

        elif self.phase == 'PREAPPROACH_WAIT_FERTI':
            if self.wait_for_timer(0.5):
                self.phase = 'PHASE_FINAL_APPROACH_WAIT'
# -------------------------------------------------------------------------------------------------------------------------------------------------------------



        elif self.phase == 'PHASE_FINAL_APPROACH':
            self.get_logger().info(f"current magnet force ({self.current_force_z:.2f})")

            if not self.phase_initialized:
                self.final_ferti_target = self.ferti_pose.copy()
                self.final_ferti_target[0] += 0.03
                self.final_ferti_target[1] -= 0.025
                self.final_ferti_target[2] -=0.01

                self.phase_initialized = True

            if self.current_force_z > 30.0:
                self.stop_all()
                self.phase_initialized = False
                self.get_logger().info(f"Force contact! {self.current_force_z:.1f}N → advancing")
                self.phase = 'ATTACH_FERTI_PRE_WAIT'
                return
            reached = self.move_to_tcp_target_v2(self.final_ferti_target, tol=0.001, slow=True)

            if reached:
                self.stop_all()
                self.phase_initialized = False
                self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.ferti_pose}")
                self.get_logger().info("Reached fertilizer hover position. Waiting before attach...")
                self.phase = 'ATTACH_FERTI_PRE_WAIT'
# =============================================================================================================================================

        elif self.phase == 'ATTACH_FERTI_PRE_WAIT':
            if self.wait_for_timer(1.5):
                self.phase = 'ATTACH_FERTI_ACTION'

        elif self.phase == 'ATTACH_FERTI_ACTION':
                self.set_gripper_state('attach')
                self.phase = 'PHASE_LIFT_FERTILIZER'


# ---------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_LIFT_FERTILIZER':
            if not self.phase_initialized:
                 self.lift_target = self.current_tcp_pos.copy()
                 self.lift_target[2] += 0.05
                 self.phase_initialized = True
                 self.get_logger().info(f"Lifting to: {self.lift_target}")

            if self.move_to_tcp_target_v2(self.lift_target, tol=0.001, slow=True):
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'PHASE_REVERSE_FROM_FERTI_WAIT'

# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_REVERSE_FROM_FERTI_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'PHASE_REVERSE_FROM_FERTI'

# -----------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_REVERSE_FROM_FERTI':
            if not self.phase_initialized:
                self.reverse_target = self.current_tcp_pos.copy()
                self.reverse_target[1] += 0.30
                self.phase_initialized = True
                self.get_logger().info("Reversing safely...")

            if self.move_to_tcp_target_v2(self.reverse_target, tol=0.02, slow=True):
                self.get_logger().info("Reverse Complete.")
                self.phase_initialized = False
                self.phase = 'REVRSE_TO_ALING_FERTI_WAIT'
# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'REVRSE_TO_ALING_FERTI_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'REVRSE_TO_ALING_FERTI'
# ------------------------------------------------------------------------------------------------------------------------------------
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
            if self.wait_for_timer(0.5):
                self.phase = 'PHASE_ALIGN_TO_INIT'
# --------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'PHASE_GRIPPER_ORIENTATION_DOWN':

            joint_name = 'wrist_1_joint'
            joint_idx = 3

            if not self.phase_initialized:
                if joint_name not in self.joint_pos:
                    return

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
                    self.get_logger().info(f"✓ Found ebot TF. ")
                    self.phase = 'MOVED_FOR_EBOT_HOVER'

# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'MOVED_FOR_EBOT_HOVER':
            # Move to HOVER position
            target = self.ebotWorldPosition.copy()
            target[2] += 0.23

            if self.move_to_tcp_target_v2(target, 0.01):
                self.get_logger().info("Hovered over eBot. Descending to DROP.")
                self.phase = 'FINAL_APPROACH_EBOT_WAIT'

# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'FINAL_APPROACH_EBOT_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'MOVED_DOWNWARD_FERTI'

# ---------------------------------------------------------------------------------------------------------------------------

# wrist orentation rotatation


        elif self.phase == 'PERFECT_PLACE_WRIST_3_ROATTION':

            joint_name = 'wrist_3_joint'
            joint_idx = 5

            if not self.phase_initialized:
                if joint_name not in self.joint_pos:
                    return

                self.target_wrist_val = self.joint_pos[joint_name] - (np.pi-0.17)

                self.phase_initialized = True
                self.get_logger().info(f"Rotating Wrist Down... Target: {self.target_wrist_val:.2f}")

            if self.move_joint_to_angle(self.target_wrist_val, joint_name, joint_idx):
                self.get_logger().info(" Wrist Oriented Down.")
                self.get_logger().info(f"gripper down phase while load ferti {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.phase_initialized = False
                self.phase = 'MOVED_DOWNWARD_FERTI'

# ------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'MOVED_DOWNWARD_FERTI':
            if not self.phase_initialized:
                self.ebotApproach_target = self.current_tcp_pos.copy()
                self.ebotApproach_target[2] -= 0.07
                self.ebotApproach_target[0] += 0.01
                self.start_currentforece = self.current_force_z
                self.get_logger().info("Reversindownard goto  safely...")
                self.get_logger().info(f" {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.phase_initialized = True


                
            if abs(self.current_force_z - self.start_currentforece) > 10.0:
                self.stop_all()
                self.phase_initialized = False
                self.get_logger().info(f"Force contact! {self.current_force_z:.1f}N → advancing")
                self.phase = 'DROP_FERTI_ON_EBOT'
                return
            
            reached = self.move_to_tcp_target_v2(self.ebotApproach_target, tol=0.005, slow=True)

            if reached:
                self.get_logger().info("Reverse Complete.")
                self.phase_initialized = False
                self.phase = 'DROP_FERTI_ON_EBOT'

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'DROP_FERTI_ON_EBOT':
            self.set_gripper_state('detach')
            self.get_logger().info(f"detach complete ferti on ebot {self.ebotWorldPosition} current arm pose {self.current_tcp_pos} ")
            self.phase = 'RETARCT_FROM_EBOT_WAIT'
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'RETARCT_FROM_EBOT_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'RETARCT_FROM_EBOT'

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'RETARCT_FROM_EBOT':
            if not self.phase_initialized:
                self.Retract_target = self.current_tcp_pos.copy()
                self.Retract_target[0] -= 0.07
                self.Retract_target[2] += 0.07
                self.phase_initialized = True
                self.get_logger().info("we are retract from intial pose")

                self.get_logger().info(f"Retarct to: {self.Retract_target}")

            if self.move_to_tcp_target_v2(self.Retract_target, tol=0.005):
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'EBOT_MOVEMENT_ALLOW'

# ------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'EBOT_MOVEMENT_ALLOW':
            self.get_logger().info(" signal Ebot to move.")

            ferti_placed_msg = Bool()
            ferti_placed_msg.data = True
            self.fertilizer_status_publisher.publish(ferti_placed_msg)

            self.ebot_docked = False

            self.get_logger().info("Dock flag reset. Moving to Initial Phase.")

            self.phase = 'FRUITS_TRAY_ALING'

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase  == 'FRUITS_TRAY_ALING':
            if self.move_to_tcp_target_v2(self.fruitHomePosition, tol=0.002, slow=False):
                self.fruits_tray_hover_pos = self.joint_pos.copy()
                self.get_logger().info(f" hover to the fruits tray{self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.get_logger().info(" we are at the hover at the fruits tary ")
                self.phase = 'SETTEL_FRUITS_TRAY'

# -----------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'SETTEL_FRUITS_TRAY':
            if self.wait_for_timer(0.5):
                self.phase = 'APPROACH_FRUITS'

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'APPROACH_FRUITS':

            if self.current_fruit_index >= len(self.badFruitTable):
                self.get_logger().info("All fruits sorted. Stopping.")
                self.stop_joint()
                self.phase = 'INTAIL_RETURN_UNLOAD_POSE_WAIT'
                return

            if not self.phase_initialized:
                fruit_record = self.badFruitTable[self.current_fruit_index]

                self.current_fruits_pose = np.array(fruit_record['pos']).copy()
                self.hover_target = self.current_fruits_pose.copy()
                self.hover_target[2] += 0.15
                self.phase_initialized = True
                self.get_logger().info(
                    f"Fruit pos: {self.current_fruits_pose} | Hover: {self.hover_target}"
                )

            reached = self.move_to_tcp_target_v2(
                self.hover_target, tol_xy=0.0015, tol_z=0.020
            )


            if reached:
                self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos} , fruits {self.current_fruits_pose},  jointState {self.joint_pos}")
                self.get_logger().info(" we are at the hover at the fruits   now go  check the orenation  of pickup ")
                self.phase_initialized = False
                self.phase = 'FRUIST_HOVER_WAIT'
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'FRUIST_HOVER_WAIT':
            if self.wait_for_timer(1.0):
                self.set_gripper_state('attach')
                self.get_logger().info("magnet start here so get attach the cane ")
                self.phase = 'CURRECT_FRUITS_POSE_FINAL_APPROACH'

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'CURRECT_FRUITS_POSE_FINAL_APPROACH':

            self.get_logger().info(f"current magnet force  ({self.current_force_z:.2f} )", throttle_duration_sec=0.3)

            if not self.phase_initialized:
                self.final_target = self.current_fruits_pose.copy()
                self.final_target[0]  -=0.05
                self.final_target[2] -= 0.01
                self.phase_initialized = True
                self.get_logger().info(
                    f"Final approach target: {self.final_target} | fruit: {self.current_fruits_pose}"
                )

            # External force guard (not inside move function)
            if self.current_force_z > 55.0:
                self.stop_all()
                self.phase_initialized = False
                self.get_logger().info(f"Force contact! {self.current_force_z:.1f}N → advancing")
                self.phase = 'ATTACH_FRUITS_PRE_WAIT'
                return

            reached = self.move_to_tcp_target_v2(
                self.final_target, tol_xy=0.001, tol_z=0.015, slow=True
            )

            if reached:
                self.stop_all()
                self.phase_initialized = False
                self.get_logger().info(f"{self.current_tcp_orient} , \n pose current {self.current_tcp_pos} , fruits {self.final_target}, fruits {self.current_fruits_pose}  \n jointState {self.joint_pos}")
                self.get_logger().info(" arm on the fruits call attach")
                self.phase = 'ATTACH_FRUITS_PRE_WAIT'

# -----------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ATTACH_FRUITS_PRE_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'ATTACH_FRUITS_ACTION'

        elif self.phase == 'ATTACH_FRUITS_ACTION':
                self.set_gripper_state('attach')
                self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos}   current fruits pose ({self.current_fruits_pose}) jointState {self.joint_pos}")
                self.phase = 'LIFT_FRUIRTS_ATTACH'


# --------------------------------------------------------------PHASE_ALIGN_TO_FERTI------------------------------------------------------

        elif self.phase == 'LIFT_FRUIRTS_ATTACH':
            if not self.phase_initialized:
                 self.lift_target = self.current_tcp_pos.copy()
                 self.lift_target[2] += 0.13
                 self.phase_initialized = True
                 self.get_logger().info(f"Lifting to: {self.lift_target}")

            reached = self.move_to_tcp_target_v2(self.lift_target,tol=0.03,slow=True)

            if reached:
                self.get_logger().info(f"lift up done {self.current_tcp_orient} , pose current {self.current_tcp_pos}    jointState {self.joint_pos}")
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'LIFT_FRUITS_WAIT'
# ---------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'LIFT_FRUITS_WAIT':
            if self.wait_for_timer(0.0):
                self.phase = 'SAFE_HOVER_POSE'


# ----------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'SAFE_HOVER_POSE':
            if not self.phase_initialized:
                self.lift_target = self.current_tcp_pos.copy()
                self.lift_target[1] -= 0.05
                self.lift_target[2] += 0.10


                self.phase_initialized = True
                self.get_logger().info(f"Lifting to: {self.lift_target}")

            reached = self.move_to_tcp_target_v2(self.lift_target,tol=0.02,slow=False)

            if reached:
                self.get_logger().info(f"lift up done {self.current_tcp_orient} , pose current {self.current_tcp_pos}    jointState {self.joint_pos}")
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'MOVED_FOR_DUSTBIN_WAIT'  


# =========================================================================================================================================================

        elif self.phase == 'MOVED_FOR_DUSTBIN_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'MOVED_FOR_DUSTBIN'
# ----------------------------------------------------------------------------------------------------------

        elif self.phase == 'MOVED_FOR_DUSTBIN':
            # 1. Calculate the remaining distance for each axis


            if not self.phase_initialized :
                self.target_dustbin = self.dustbinPosition.copy()
                self.target_dustbin[2] += 0.07
                self.get_logger().info("we are moving at the dustbin ")
                self.phase_initialized = True

            reached = self.move_to_tcp_target_v2(self.target_dustbin, tol=0.02, slow=False)
            if reached:
                self.get_logger().info("Target reached. WE ARE AT DUSTBIN.")
                self.phase_initialized = False
                self.phase = 'DUSTBIN_WAIT'
# ---------------------------------------------------------------------------------------
        elif self.phase == 'DUSTBIN_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'CALL_DEATTACH_FRUITS'
# ----------------------------------------------------------------------------------------------------

        elif self.phase == 'CALL_DEATTACH_FRUITS':
                self.get_logger().info("Deactivating Gripper...")
                self.set_gripper_state('detach')
                self.current_fruit_index += 1
                self.phase = 'RETURN_TRAY_WAIT'  
# --------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'RETURN_TRAY_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'RETURN_TRAY_POSE'
# ----------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'RETURN_TRAY_POSE':
            if not self.phase_initialized :

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
                self.phase_initialized =True

            if self.move_joint_group(self.targets, self.speed_scale):
                self.phase_initialized  = False
                self.get_logger().info("Fertilizer alignment restored (FAST).")
                self.phase = 'SETTEL_FRUITS_TRAY'

# ============================================= RETURN HOME FOR UNLOAD   ====================================================================================================================================================================================


# ====================================== new unload path ===========================================================================================================


        elif self.phase == 'INTAIL_RETURN_UNLOAD_POSE_WAIT':
            if self.wait_for_timer(0.5):
                self.get_logger().info(f"we euler roll , pitch , yam index 3,4,5 of tcp pose  {self.current_euler_angel}  cunnet pose { self.current_tcp_pos} \n  current joint pose {self.joint_pos}")
                self.phase = 'INTAIL_RETURN_UNLOAD_POSE'



# -----------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'INTAIL_RETURN_UNLOAD_POSE':

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
                self.get_logger().info(f"we euler roll , pitch , yam index 3,4,5 of tcp pose  {self.current_euler_angel}  cunnet pose { self.current_tcp_pos} \n  current joint pose {self.joint_pos}")
                self.get_logger().info("init pose again check sucess ")
                self.phase = 'WAIT_FOR_EBOT_RETURN'

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



        elif self.phase == 'WAIT_FOR_EBOT_RETURN':

            if not self.ebot_docked:
                self.get_logger().info("Waiting for eBot to return with fertilizer...", throttle_duration_sec=2.0)
                return

            self.get_logger().info("ebot reach at the return at dockstation")
            self.phase = 'FERTI_NEW_TF_WAIT'
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



        elif self.phase == 'FERTI_NEW_TF_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'FERTI_NEW_TF_UNLOAD'

#   once time is used this if the ebot delay the reach dock when ifts proper stop at pose then we can remove the above time

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'FERTI_NEW_TF_UNLOAD':

            self.get_logger().info("we are getting the tf of ferti when its on e bot ")
            if self.ferti_unload_pose is None:
                self.ferti_unload_pose = self.lookup_tf(self.base_link_name, self.fertilizerTFname)


            if self.ferti_unload_pose is None:
                self.get_logger().info("Waiting for fertilizer TF  new ...", throttle_duration_sec=2.0)
                return
            self.get_logger().info(f"new feti pose {self.ferti_unload_pose} ")
            self.get_logger().info("getting fertilzer here again ")
            self.phase = 'GRIPPER_UNLOAD_FERTI_DOWN'


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'GRIPPER_UNLOAD_FERTI_DOWN':
            joint_name = 'wrist_1_joint'
            joint_idx = 3

            if not self.phase_initialized:
                if joint_name not in self.joint_pos:
                    self.get_logger().info("we are miss the joint pose ")
                    return

                self.target_wrist_val = self.joint_pos[joint_name] - (self.wrist1_delta_down + 0.1)

                self.phase_initialized = True
                self.get_logger().info(f"Rotating Wrist Down... Target: {self.target_wrist_val:.2f}")

            if self.move_joint_to_angle(self.target_wrist_val, joint_name, joint_idx):
                self.get_logger().info(" Wrist Oriented Down.")
                self.get_logger().info(f"gripper down phase {self.current_tcp_orient}  euler {self.current_euler_angel}, pose current {self.current_tcp_pos} ,  \n jointState {self.joint_pos}")
                self.phase_initialized = False
                self.phase = 'FERTI_UNLOAD_ALING_WAIT'  
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'FERTI_UNLOAD_ALING_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'ALING_ARM_FOR_UNLOAD_FERTI'
# --------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ALING_ARM_FOR_UNLOAD_FERTI':
            # Use your existing helper function to rotate the base
            if self.align_joint_to_pose(self.ferti_unload_pose, 'shoulder', self.joint_names_list[0], 0):
                self.get_logger().info(f"gripper down phase {self.current_tcp_orient}  euler {self.current_euler_angel}, pose current {self.current_tcp_pos} ,  \n jointState {self.joint_pos}")
                self.get_logger().info("Shoulder aligned. Now   now go to hove hovering.")
                self.phase = 'ULOAD_FERTI_PRE_APPROACH_WAIT'
# ------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'ULOAD_FERTI_PRE_APPROACH_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'ULOAD_FERTI_PRE_APPROACH'
# ----------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'ULOAD_FERTI_PRE_APPROACH':
            if not self.phase_initialized:
                 self.ferti_unload_target = self.ferti_unload_pose.copy()
                 self.ferti_unload_target[2] += 0.05
                 self.phase_initialized = True
                 self.get_logger().info(f"unload target approach: {self.ferti_unload_target}   target : {self.ferti_unload_pose}")

            if self.move_to_tcp_target_v2(self.ferti_unload_target, tol=0.01, slow=False):
                self.get_logger().info(f" phase {self.current_tcp_orient}  euler {self.current_euler_angel}, pose current {self.current_tcp_pos} ,  \n jointState {self.joint_pos}")
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'UNLOAD_FERTI_FINAL_WAIT'
# ------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'UNLOAD_FERTI_FINAL_WAIT':
            if self.wait_for_timer(1.0):
                self.set_gripper_state('attach')
                self.get_logger().info(" magnet start ")
                self.phase = 'UNLOAD_FERTI_FINAL'

# ------------------------------------------------------------------------------------------------------

        elif self.phase == 'UNLOAD_FERTI_FINAL':

            if not self.phase_initialized:
                 self.ferti_unload_target_final = self.ferti_unload_pose.copy()
                 self.ferti_unload_target_final[2] -= 0.03
                 self.ferti_unload_target_final[0] -= 0.02

                 self.phase_initialized = True
                 self.get_logger().info(f"unload target approach: {self.ferti_unload_target_final} , current magnet {self.current_force_z} ")
            # -----------------------------------------------------------------------------------------------------------------------

  
            if self.current_force_z > 35.0:
                self.stop_all()
                self.phase_initialized = False
                self.get_logger().info(f"Force contact! {self.current_force_z:.1f}N → advancing")
                self.phase = 'ATTACH_UNLOAD_FERTI'
                return
            # -------------------------------------------------------------------------------------------------------------------------
            reached = self.move_to_tcp_target_v2(self.ferti_unload_target_final, tol_xy=0.001 , tol_z=0.005, slow=True)
            self.get_logger().info(f", current magnet {self.current_force_z} ")

            if reached :
                self.stop_all()
                self.get_logger().info(f"  ferti unload  final {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'ATTACH_TARGET_UNLOAD_WAIT'

            
# --------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ATTACH_TARGET_UNLOAD_WAIT':
            if self.wait_for_timer(2.0):
                self.phase = 'ATTACH_UNLOAD_FERTI'
# ---------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ATTACH_UNLOAD_FERTI':
            self.set_gripper_state('attach')
            self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos}   current fruits pose ({self.current_fruits_pose}) jointState {self.joint_pos}")
            self.phase = 'LIFT_TARGET_UNLOAD'

# -------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'LIFT_TARGET_UNLOAD':
             if not self.phase_initialized:
                self.lift_unload_target = self.current_tcp_pos.copy()
                self.lift_unload_target[2] += 0.07
                self.get_logger().info(f"curren_pose {self.current_tcp_pos} target {self.lift_unload_target}")
                self.phase_initialized = True

             reached = self.move_to_tcp_target_v2(self.lift_unload_target, tol= 0.01, slow=True)
             if reached :
                self.get_logger().info(f" ferti unload wala hai   {self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.ferti_pose}")
                self.get_logger().info("we are lif success fully ")
                self.phase_initialized = False
                self.phase = 'ALLOW_EBOT_AFTER_UNLOAD'

# --------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'ALLOW_EBOT_AFTER_UNLOAD':
            self.get_logger().info(" Signal Ebot to move.")

            ferti_placed_msg = Bool()
            ferti_placed_msg.data = True
            self.fertilizer_status_publisher.publish(ferti_placed_msg)

            self.ebot_docked = False

            self.get_logger().info("Dock flag reset. Moving to Initial Phase.")

            self.phase = 'ALING_TO_TRAY_UNLOAD_WAIT'

# ------------------------------------------------------------------------------------------------------------

        elif self.phase == 'ALING_TO_TRAY_UNLOAD_WAIT':
            if self.wait_for_timer(0.5):
                self.phase = 'ALING_TO_TRAY_UNLOAD'

# --------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ALING_TO_TRAY_UNLOAD':
            if self.move_to_tcp_target_v2(self.fruitHomePosition, tol=0.02, slow=True):
                self.fruits_tray_hover_pos = self.joint_pos.copy()
                self.get_logger().info(f" hover to the fruits tray  for unload the ferti {self.current_tcp_orient} , pose current {self.current_tcp_pos} ,   jointState {self.joint_pos}")
                self.phase = 'MOVED_FOR_DUSTBIN_WAIT'



        elif self.phase == 'MOVED_FOR_DUSTBIN_WAIT':  
            if self.wait_for_timer(1.0):
                self.phase = 'MOVED_FOR_DUSTBIN'
# ------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'MOVED_FOR_DUSTBIN':
            # 1. Calculate the remaining distance for each axis

            if not self.phase_initialized :
                self.target_dustbin = self.dustbinPosition.copy()
                self.target_dustbin[2] += 0.15
                self.get_logger().info("we are moving at the dustbin ")
                self.phase_initialized = True

            reached = self.move_to_tcp_target_v2(self.target_dustbin, tol=0.02, slow=True)
            if reached:
                self.get_logger().info("Target reached. WE ARE AT DUSTBIN.")
                self.phase_initialized = False
                self.phase = 'DUSTBIN_WAIT'
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'DUSTBIN_WAIT':
            if self.wait_for_timer(1.0):
                self.phase = 'CALL_DEATTACH_FERTI'
# ----------------------------------------------------------------------------------------------------

        elif self.phase == 'CALL_DEATTACH_FERTI':
                self.get_logger().info("Deactivating Gripper...")
                self.set_gripper_state('detach')
                self.phase = 'RETURN_TRAY_WAIT_UNLOAD'  

# ---------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'RETURN_TRAY_WAIT_UNLOAD':
            if self.wait_for_timer(1.0):
                self.phase = 'RETURN_TRAY_POSE_UNLOAD'



# -----------------------------------------------------------------------------------------------------------------------------------------------
# ========================================================new return mrthod ==================================================

        elif self.phase == 'RETURN_TRAY_POSE_RESET':
            # 1. Calculate the remaining distance for each axis

            if not self.phase_initialized :
                self.target_tray_reset = self.fruitHomePosition.copy()
             
                self.get_logger().info("we are moving at the trya  again  ")
                self.phase_initialized = True

            reached = self.move_to_tcp_target_v2(self.target_tray_reset, tol=0.02, slow=False)
            if reached:
                self.get_logger().info("Target reached. at tray.")
                self.phase_initialized = False
                self.phase = 'REVERSE_ALING_INIT_POSE_WAIT_UNLOAD'
# ----------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'RETURN_TRAY_POSE_UNLOAD':
            if not self.phase_initialized :

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
                self.phase_initialized =True

            if self.move_joint_group(self.targets, self.speed_scale):
                self.phase_initialized  = False
                self.get_logger().info("Fertilizer alignment restored (FAST).")
                self.phase = 'REVERSE_ALING_INIT_POSE_WAIT_UNLOAD'

# ============================================= RETURN HOME FORM UNLOAD   ====================================================================================================================================================================================

# ------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'REVERSE_ALING_INIT_POSE_WAIT_UNLOAD':
            if self.wait_for_timer(1.0):
                self.phase = 'REVERSE_TO_FERTI_INTIAL_ALING_UNLOAD'



# ----------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'REVERSE_TO_FERTI_INTIAL_ALING_UNLOAD':

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
                self.get_logger().info("init pose again check sucess ")
                self.phase = 'DONE'


# -----------------------------------------------------------------------------------------------------------------------
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
