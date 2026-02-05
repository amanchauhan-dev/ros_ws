#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import numpy as np
import time
# CHANGED: Interface Imports
from geometry_msgs.msg import TwistStamped
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
        self.magnet_client = self.create_client(
                SetBool, 
                '/magnet', 
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
        self.current_tcp_pos = None
        self.current_tcp_orient = None 
        self.initial_arm_pos = None
        self.ferti_align_joint_state = None
        
        self.max_tol = np.deg2rad(3)
        self.base_kp = 1.0
        self.base_max_speed = 0.5        # Linear Max
        self.base_max_angular = 1.5   
        # phases config 
        
        self.phase = 'START'
        
        self.ferti_pose = None
        self.fertilizerTFname = f'{self.teamIdentifier}_fertilizer_1'
        self.ebotTransformName = f'{self.teamIdentifier}_ebot_marker'
        self.pickupOrientationQuaternion = np.array([0.707, 0.028, 0.034, 0.707])
        self.dropOrienationQuateration = np.array([-0.684, 0.726, 0.05, 0.008])
        # --------------------------------------------------------------
        self.ebotWorldPosition = None
        self.ebotWorldPosition = np.array([0.711, 0.006, 0.145])
        # ------------------------------------------------------------
        
        self.badFruitTable = []
        self.badFruitFrameList = [
            f'{self.teamIdentifier}_bad_fruit_1',
        ]
        
        self.fruitHomePosition = np.array([-0.159, 0.501, 0.415])
        self.fruitHomeOrientation = normalize_quaternion([0.029, 0.997, 0.045, 0.033])


        # -----------------------------------------------------------------------------------
        self.dustbinPosition = np.array([-0.806, 0.010, 0.182])
        self.dustbinOrientation = normalize_quaternion([0.0, 0.0, 0.0, 1.0])
        # -----------------------------------------------------------------------------------
        self.phase_initialized = False
        self.current_fruits_pose = None 
        self.wait_start_time = None
        self.wrist1_delta_down = -1.36
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

# -------------------------------------------------------------------------------------------------------
    def servo_to_goal(self, currentPosition, currentQuaternion, targetPosition, targetQuaternion, 
                        positionTolerance, angularTolerance, maximumLinearVelocity, maximumAngularVelocity, 
                        checkOrientation=True):
            
            # 1. Calculate Linear Error
            position_error = targetPosition - currentPosition
            distance_value = float(np.linalg.norm(position_error))
            
            # We define zones based on how far we are
            
            if distance_value > 0.30:          # Far (> 10cm)
                kp = 3.0
                allowed_speed = maximumLinearVelocity  # Full speed
            elif distance_value > 0.20:        # Approaching (5-10cm)
                kp = 2.0
                allowed_speed = 0.15           # Slow down significantly
            elif distance_value > 0.05:        # Close (1-5cm)
                kp = 1.0
                allowed_speed = 0.05           # Creep speed
            else:                              # Very Close (< 1cm)
                kp = 1.0
                allowed_speed = 0.02   # Super slow precise parking speed

            # 2. Check if we are "Done" (Stop Condition)
            position_ok = (distance_value <= positionTolerance)
            
            # 3. Calculate Linear Velocity Vector
            final_speed = 0.0
            if not position_ok:
                # P-Controller: Speed = Gain * Error
                calculated_speed = kp * distance_value
                final_speed = min(calculated_speed, allowed_speed)
# Ensure we don't drop below 0.05 m/s
                if final_speed < 0.05:
                    final_speed = 0.05

                linear_velocity_vector = (position_error / distance_value) * final_speed
            else:
                linear_velocity_vector = np.zeros(3)

            orientation_ok = True
            roll_err, pitch_err, yaw_err = 0.0, 0.0, 0.0

            if checkOrientation:

                q_err = multiply_quaternion(targetQuaternion, conjugate_quaternion(currentQuaternion))
                
                # Convert to Euler (Axis/Angle speeds)
                roll_err, pitch_err, yaw_err = quaternion_to_euler(q_err)
                
                # Check if all angles are within tolerance
                if (abs(roll_err) <= angularTolerance and 
                    abs(pitch_err) <= angularTolerance and 
                    abs(yaw_err) <= angularTolerance):
                    orientation_ok = True
                    angular_velocity_vector = np.zeros(3)
                else:
                    orientation_ok = False
                    # Simple P-Controller for Rotation
                    # Note: We can also ramp this, but usually constant is fine for rotation
                    angular_velocity_vector = np.array([roll_err, pitch_err, yaw_err]) * 2.0 
                    
                    mag_ang = np.linalg.norm(angular_velocity_vector)
                    if mag_ang > maximumAngularVelocity:
                        angular_velocity_vector = (angular_velocity_vector / mag_ang) * maximumAngularVelocity
            else:
                angular_velocity_vector = np.zeros(3)
                
            if not position_ok:
                 self.get_logger().info(
                                f"DIST: {distance_value:.3f}m | "
                                f"SPD Cmd: {final_speed:.2f} m/s | "
                                f"RotErr: {np.degrees(roll_err):.1f}° {np.degrees(pitch_err):.1f}° {np.degrees(yaw_err):.1f}°",
                                throttle_duration_sec=0.5
                            )

            # 5. Publish Command
            self.publish_twist(linear_velocity_vector, angular_velocity_vector)

            # 6. Return Status
            # We only return True if BOTH Position AND Orientation are good
            if position_ok and orientation_ok:
                return True
            else:
                return False
            

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
    

# ---------------------------------------------------------------------------------
# this is alos twist  but chek only orenation only not move only aling 
    def orient_to_target(self, target_quat, tol=0.03):
        """
        Calculates error between current orientation and target quaternion
        and publishes TwistStamped to correct it.
        """
        if self.current_tcp_orient is None:
            return False
            
        # 1. Calculate Error Quaternion
        q_curr_inv = conjugate_quaternion(self.current_tcp_orient)
        q_err = multiply_quaternion(target_quat, q_curr_inv)
        
        # Ensure we take the shortest path (flips hemisphere if needed)
        if q_err[3] < 0:
            q_err = -q_err
            
        xyz_err = q_err[:3]
        error_mag = np.linalg.norm(xyz_err)
        
        # 2. Check Success
        if error_mag < tol:
            self.stop_all() 
            self.get_logger().info(f"✓ Orientation Aligned. Error: {error_mag:.4f}")
            return True
            
        # 3. Calculate Speed
        kp_rot = 2.0  
        max_rot_speed = 0.5 
        
        ang_speed = min(kp_rot * error_mag, max_rot_speed)
        
# speed lower control
        if ang_speed < 0.05:
            ang_speed = 0.05
        
        if error_mag > 0:
            ang_dir = xyz_err / error_mag
        else:
            ang_dir = np.zeros(3)
        
        # 4. Debug Logging (Throttled)
        # Convert error to roughly degrees for easy reading
        approx_deg_error = np.degrees(error_mag * 2) 
        self.get_logger().info(
            f"Orienting... | Err: {approx_deg_error:.1f}° | Speed: {ang_speed:.2f}",
            throttle_duration_sec=0.5
        )

        # 5. Publish Command
        # Linear velocity is ZERO because we only want to rotate
        angular_vel = ang_dir * ang_speed
        self.publish_twist([0.0, 0.0, 0.0], angular_vel)
        
        return False
    
# ============================ JOINT DELTA CMD =====================================================


# ----------------------------------------------------------------------------------------
    def move_joint_to_angle(self, target_angle, joint_name, joint_index, tol=0.02):
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
            
        if abs(err) > 1.5:         # Far (> 30 deg)
            kp = 2.0
            max_s = self.base_max_speed # e.g. 1.0 rad/s
        elif abs(err) > 0.17 or abs(err) > 1.0 :       # Medium
            kp = 1.5
            max_s = 0.5
        else:                      # Close (< 6 deg)
            kp = 1.0
            max_s = 0.1            # Creep speed for joints

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
            self.get_logger().info("phase machine in the real is start ")
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
            self.phase = 'PHASE_BASE_ALING_TO_TRAY'

# -----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_BASE_ALING_TO_TRAY':
            target_pose = self.fruitHomePosition.copy()

            # 2. Pass that pose to the function
            if self.align_joint_to_pose(target_pose, 'shoulder', self.joint_names_list[0], 0):
                self.get_logger().info("Base aligned to Fruit Tray. Moving Wrist Down.")
                self.phase = 'PHASE_GRIPPER_ORIENTATION_DOWN'
# -----------------------------------------------------------------------------------------------------------------

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
                self.phase = 'FRUITS_HOVER_WAIT'

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        elif self.phase == 'FRUITS_HOVER_WAIT':
            if self.wait_for_timer(3.0):
                self.phase = 'APPROACH_FRUITS_TRAY'   
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'APPROACH_FRUITS_TRAY':   
            target = self.fruitHomePosition.copy()
            
            if self.move_to_tcp_target(target, tol=0.02):
                self.get_logger().info("Initial Approach Done. Starting Sorting.")
                self.phase = 'SETTEL_FRUITS_TRAY'
# -----------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'SETTEL_FRUITS_TRAY':
            if self.wait_for_timer(3.0):
                self.phase = 'CHECK_ON_FRUIT_TRAY_ORENTATION'   
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'CHECK_ON_FRUIT_TRAY_ORENTATION':
            self.get_logger().info("Checking pickup orientation...", throttle_duration_sec=2.0)
            
            if self.orient_to_target(self.fruitHomeOrientation, tol=self.max_tol):
                self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.ferti_pose},  jointState {self.joint_pos}")
                self.get_logger().info(" Orientation Correct. Moving to Target.")
                self.phase = 'APPROACH_FRUITS'


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'APPROACH_FRUITS':
            self.get_logger().info("we are in approach fruits ")
            
            fruit_record = self.badFruitTable[self.current_fruit_index]
            original_fruit_pose = fruit_record['pos']

            self.current_fruits_pose = original_fruit_pose.copy()

            hover_target = original_fruit_pose.copy()
            hover_target[2] += 0.10

            reached = self.move_to_tcp_target(hover_target,tol=0.01)

            if reached:
                self.get_logger().info(" we are at the hover at the fruits   now go  check the orenation  of pickup ")
                self.phase = 'FRUIST_HOVER_WAIT'



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'FRUIST_HOVER_WAIT':
            if self.wait_for_timer(3.0):
                self.phase = 'CHECK_ORENATION_FRUITS_PICKUP'  

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ 
                
        elif self.phase == 'CHECK_ORENATION_FRUITS_PICKUP':
            if self.orient_to_target(self.fruitHomeOrientation, tol= self.max_tol):
                self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.ferti_pose},  jointState {self.joint_pos}")
                self.get_logger().info(" Orientation Correct.  on fruits  Moving to Target.")
                self.phase = 'CURRECT_FRUITS_POSE_FINAL_APPROACH'
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'CURRECT_FRUITS_POSE_FINAL_APPROACH':
            self.get_logger().info(f"current magnet force  ({self.current_force_z:.2f} )")
            reached = self.move_to_tcp_target(self.current_fruits_pose,tol=0.02,slow=True)

            if reached:
                self.get_logger().info(" arm on the fruits call attach  ")
                self.phase = 'ATTACH_FRUITS_PRE_WAIT'
# -------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ATTACH_FRUITS_PRE_WAIT':
            if self.wait_for_timer(3.0):
                self.phase = 'ATTACH_FRUITS_ACTION'

        elif self.phase == 'ATTACH_FRUITS_ACTION':
                self.set_gripper_state('attach')
                self.phase = 'ATTACH_FRUITS_POST_WAIT'
        
        elif self.phase == 'ATTACH_FRUITS_POST_WAIT':
            if self.wait_for_timer(3.0):
                 self.phase = 'LIFT_FRUIRTS_ATTACH'

# -----------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'LIFT_FRUIRTS_ATTACH':
            if not self.phase_initialized:
                 self.lift_target = self.current_tcp_pos.copy()
                 self.lift_target[2] += 0.15
                 self.phase_initialized = True
                 self.get_logger().info(f"Lifting to: {self.lift_target}")

            reached = self.move_to_tcp_target(self.lift_target,tol=0.02,slow=True)

            if reached:
                self.get_logger().info("Lift Complete.")
                self.phase_initialized = False
                self.phase = 'LIFT_FRUITS_WAIT'
# ---------------------------------------------------------------------------------------
        elif self.phase == 'LIFT_FRUITS_WAIT':
            if self.wait_for_timer(3.0):
                self.phase = 'MOVED_FOR_DUSTBIN'
# ----------------------------------------------------------------------------------------------------
        elif self.phase == 'MOVED_FOR_DUSTBIN':
            
            reached = self.move_to_tcp_target(self.dustbinPosition,tol= 0.02,slow=True)

            if reached:
                self.get_logger().info("WE ARE AT DUSTBIN .")
                self.phase = 'DUSTBIN_WAIT'

# ---------------------------------------------------------------------------------------
        elif self.phase == 'DUSTBIN_WAIT':
            if self.wait_for_timer(3.0):
                self.phase = 'ON_DUSTBIN_CHECK'
# ----------------------------------------------------------------------------------------------------------

        elif self.phase == 'ON_DUSTBIN_CHECK':
            target = self.dustbinPosition.copy()
            err = np.linalg.norm(target - self.current_tcp_pos)
            if err < 0.05:
                self.get_logger().info("we are at the dustbin bro now gp tp check orenataion ")
                self.phase = 'CHECK_DUSTBIN_ORENATION'
            else :
                self.get_logger().info(f"we are  very clove lets try again {err}")
                self.phase ='MOVED_FOR_DUSTBIN'
#---------------------------------------------------------------------------------------------------------------- 
        elif self.phase == 'CHECK_DUSTBIN_ORENATION':
            if self.orient_to_target(self.dustbinOrientation, tol= np.radians(5)):
                self.get_logger().info(f"{self.current_tcp_orient} , pose current {self.current_tcp_pos} , frti {self.ferti_pose},  jointState {self.joint_pos}")
                self.get_logger().info(" Orientation Correct. Moving to Target.")
                self.phase = 'CALL_DEATTACH_FRUITS'



        elif self.phase == 'CALL_DEATTACH_FRUITS':
                self.get_logger().info("Deactivating Gripper...")
                self.set_gripper_state('detach')
                self.phase = 'DONE'

# -----------------------------------------------------------------------------------------------------------------------

        else :
            self.get_logger().info(f"all task done bro look at the tcp pose ", throttle_duration_sec=2.0)
            pass
# ------------------------------------------------------------------------------------------------------------------------


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