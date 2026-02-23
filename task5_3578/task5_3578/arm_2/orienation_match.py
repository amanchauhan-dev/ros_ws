#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import numpy as np
# CHANGED: TwistStamped
from geometry_msgs.msg import TwistStamped
# CHANGED: Added Float32 for force
from std_msgs.msg import Float64MultiArray, Float32
# CHANGED: JointJog
from control_msgs.msg import JointJog
from sensor_msgs.msg import JointState
from rclpy.duration import Duration
import tf2_ros
# CHANGED: SetBool for magnet
from std_srvs.srv import SetBool



# ==================================================================
# --------------------- MATH UTILITIES -----------------------------
# ==================================================================

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



class Task4c(Node):
    def __init__(self):
        super().__init__('TASK4C')

        # ===================publishers========================
        # CHANGED: TwistStamped
        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        # CHANGED: JointJog
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)
        
        # DEFINED: Joint names list for JointJog
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
        
        # CHANGED: Float64MultiArray for TCP Pose
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

        # 2. CREATE SERVICE CLIENTS FOR GRIPPER (Magnet)
        self.magnet_client = self.create_client(SetBool, '/magnet')

        self.timer = self.create_timer(0.02, self.main_loop)

        # ===================configurations========================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.teamIdentifier = '3578'
        self.base_link_name = 'base_link'
        self.joint1 = 'shoulder_pan_joint'
        self.joint2_name = 'shoulder_lift_joint'
        
        self.joint_pos = {}
        # tcp_pose variable
        self.current_tcp_pos = None
        self.current_tcp_orient =None
        self.initial_arm_pos = None
        self.current_force_z = None  # ADDED

        self.max_tol = np.deg2rad(2)
        self.base_kp = 2.0
        self.base_max_speed = 2.5

        # phases config 
        self.safe_lift_angle = None 
        self.phase = 'PHASE_1_GETTING_TF'
        
        self.ferti_pose = None
        self.fertilizerTFname = f'{self.teamIdentifier}_fertilizer_1'
        self.pickupOrientationQuaternion = np.array([0.707, 0.028, 0.034, 0.707])
        
        self.phase_initialized = False
        self.wrist1_delta_down = -1.36
        
        self.approach_tol = 0.03   
        
        # phase of dustbin 
        self.dustbin_fixed_pose = np.array([ -0.806, 0.010, 0.182])
        self.dustbin_hover_pose = None
        
        # Wait Timer Variable
        self.wait_start_time = None

# ====================callbacks===============================================
    def joint_state_callback(self, msg):
        for n,p in zip(msg.name, msg.position):
            self.joint_pos[n] = p   

    def tcp_pose_callback(self, msg):
        p = msg.pose.position
        self.current_tcp_pos = np.array([p.x, p.y, p.z])
        o = msg.pose.orientation
        self.current_tcp_orient = np.array([o.x, o.y, o.z, o.w])

    def force_callback(self, msg):
        self.current_force_z = msg.data

# ==================Attach/Deattach=========================================
    def set_gripper_state(self, action):
        """
        Controls the electromagnet using SetBool.
        """
        req = SetBool.Request()
        
        if action == 'attach':
            req.data = True
            self.get_logger().info(f"Magnet ON (Force Z: {self.current_force_z})")
        else:
            req.data = False
            self.get_logger().info(f"Magnet OFF (Force Z: {self.current_force_z})")

        future = self.magnet_client.call_async(req)

# =================motion commands==========================================

    def stop_joint(self):
        # CHANGED: Use JointJog
        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities = [0.0] * 6
        self.joint_pub.publish(msg)
    
    def align_joint_to_pose(self, target_pose, target_label, joint_name, joint_index):
        if target_pose is None:
            self.get_logger().warn(f"Cannot align to {target_label}: Pose is None!", throttle_duration_sec=2.0)
            return False

        x = target_pose[0]
        y = target_pose[1]

        desired = self.norm(np.arctan2(y, x)+np.pi)
        
        if joint_name not in self.joint_pos:
            self.get_logger().warn(f"Waiting for {joint_name}...", throttle_duration_sec=2.0)
            return False
            
        current = self.joint_pos[joint_name]
        err = self.norm(desired - current)

        self.get_logger().info(
            f"[{target_label}] cur={current:.2f} tgt={desired:.2f} err={err:.3f}",
            throttle_duration_sec=0.5
        )

        if abs(err) < self.max_tol:
            self.stop_joint()
            self.get_logger().info(f"✓ Aligned {target_label}. Switching phase.")
            return True

        speed = max(min(self.base_kp * err, self.base_max_speed), -self.base_max_speed)
        
        # CHANGED: Use JointJog
        cmd = [0.0] * 6          
        cmd[joint_index] = float(speed)
        
        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities = cmd
        self.joint_pub.publish(msg)
        
        return False
    
    def move_to_tcp_target(self, target, tol=0.03, slow=False):
        if self.current_tcp_pos is None:
            return False

        err = target - self.current_tcp_pos
        dist = np.linalg.norm(err)

        if dist < tol:
            self.publish_twist([0.0, 0.0, 0.0], 0.0) # Stop
            return True
        
        # Calculate Speed - KEPT YOUR EXACT SPEEDS
        direction = err / dist
        max_s = 0.15 if slow else 0.4  # Adjusted slow speed slightly for better precision
        speed = min(dist * 2.0, max_s)
        
        self.publish_twist(direction, speed)
        return False

    def move_joint_to_angle(self, target_angle, joint_name, joint_index):
            if joint_name not in self.joint_pos: return False
            
            current = self.joint_pos[joint_name]
            err = self.norm(target_angle - current)
            
            self.get_logger().info(f"[{joint_name}] err={err:.3f}")

            if abs(err) < self.max_tol:
                self.stop_joint()
                return True
                
            speed = max(min(self.base_kp * err, self.base_max_speed), -self.base_max_speed)
            
            # CHANGED: Use JointJog
            cmd = [0.0] * 6
            cmd[joint_index] = float(speed)
            
            msg = JointJog()
            msg.joint_names = self.joint_names_list
            msg.velocities = cmd
            self.joint_pub.publish(msg)
            return False
    def orient_to_target(self, target_quat, tol=0.03):
            """
            Rotates the TCP to match target_quat using angular velocity.
            """
            if self.current_tcp_orient is None:
                return False
                
            # 1. Calculate Error Quaternion: q_err = q_target * q_current_inverse
            # This gives us the rotation needed to get from Current -> Target
            q_curr_inv = conjugate_quaternion(self.current_tcp_orient)
            q_err = multiply_quaternion(target_quat, q_curr_inv)
            
            # 2. Check for "shortest path" (Antipodal check)
            # If w is negative, we are taking the long way around. Flip it.
            if q_err[3] < 0:
                q_err = -q_err
                
            # 3. Calculate error magnitude based on the vector part (x, y, z)
            # For small angles, the magnitude of (x,y,z) is proportional to the angle
            xyz_err = q_err[:3]
            error_mag = np.linalg.norm(xyz_err)
            
            # 4. Check if we are close enough
            if error_mag < tol:
                self.publish_twist([0.0, 0.0, 0.0], 0.0) # Stop motion
                return True
                
            # 5. P-Controller for Angular Velocity
            kp_rot = 2.0  # Proportional gain
            max_rot_speed = 0.5 # Max angular speed limit
            
            ang_speed = min(kp_rot * error_mag, max_rot_speed)
            
            # Normalize the direction vector
            ang_dir = xyz_err / error_mag
            
            self.get_logger().info(f"Aligning Orientation... Error: {error_mag:.3f}", throttle_duration_sec=0.5)
            
            # 6. Publish Twist (Linear=0, Angular=Calculated)
            self.publish_twist([0.0, 0.0, 0.0], 0.0, angular_direction=ang_dir, angular_speed=ang_speed)
            return False

    def wrist_orientation(self, direction, next_phase, angle=None):
        if 'wrist_1_joint' not in self.joint_pos:
            return

        if not self.phase_initialized:
            current_w1 = self.joint_pos['wrist_1_joint']

            if angle is not None:
                rotation_amount = angle
            else:
                rotation_amount = self.wrist1_delta_down
            
            self.get_logger().info(f"{rotation_amount}")
            if direction == 'down':
                self.target_w1 = current_w1 + rotation_amount 
            elif direction == 'up':
                self.target_w1 = current_w1 - rotation_amount
            else:
                self.get_logger().error(f"Invalid direction: {direction}")
                return

            self.phase_initialized = True
            self.get_logger().info(f"[Wrist] Moving {direction.upper()}... Target: {self.target_w1:.2f}")
            return

        current_w1 = self.joint_pos['wrist_1_joint']
        err = self.target_w1 - current_w1

        if abs(err) < self.max_tol:
            self.stop_joint()
            self.get_logger().info(f"✓ Wrist {direction} complete. Switching to {next_phase}")
            self.phase_initialized = False 
            self.phase = next_phase
            return

        # CHANGED: Use JointJog
        cmd = [0.0] * 6 
        speed = max(min(self.base_kp * err, self.base_max_speed), -self.base_max_speed)
        cmd[3] = speed 
        
        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities = cmd
        self.joint_pub.publish(msg)

    def publish_twist(self, direction, speed, angular_direction=None, angular_speed=0.0):
            # CHANGED: Use TwistStamped
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            
            # Linear Velocity
            msg.twist.linear.x = float(direction[0] * speed)
            msg.twist.linear.y = float(direction[1] * speed)
            msg.twist.linear.z = float(direction[2] * speed)
            
            # Angular Velocity (Rotation)
            if angular_direction is not None:
                msg.twist.angular.x = float(angular_direction[0] * angular_speed)
                msg.twist.angular.y = float(angular_direction[1] * angular_speed)
                msg.twist.angular.z = float(angular_direction[2] * angular_speed)
            
            self.twist_pub.publish(msg)

    def lookup_tf(self, target, source):
        try:
            tf = self.tf_buffer.lookup_transform(
                target, source, rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )
            t = tf.transform.translation
            return np.array([t.x, t.y, t.z])
        except Exception as e:
            print(e)
            return None

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

    @staticmethod
    def norm(a):
        while a > np.pi:
            a -= 2 * np.pi
        while a < -np.pi:
            a += 2 * np.pi
        return a
 
# =============================main loop=======================================
    def main_loop(self):

        if self.phase == 'PHASE_1_GETTING_TF':
            # We check if we have a position, and if we haven't stored it yet
            if self.initial_arm_pos is None:
                if 'shoulder_pan_joint' not in self.joint_pos:
                    self.get_logger().info("Waiting for joint_states...", throttle_duration_sec=2.0)
                    return

                self.initial_arm_pos = self.joint_pos.copy()
                self.get_logger().info(f"✓ Stored initial shoulder pan: {self.initial_arm_pos['shoulder_pan_joint']:.2f}")



            self.phase = 'ATTACH_FERTI_POSE'

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ATTACH_FERTI_POSE':
                    # ==========================================
                    # CONFIGURATION: Choose your joint and value here
                    # ==========================================
                    target_joint_name = 'wrist_1_joint'  # Change this to the joint you want (e.g., 'wrist_1_joint')
                    target_joint_index = 3                    # Index in self.joint_names_list (0=Pan, 1=Lift, 2=Elbow, 3=W1, 4=W2, 5=W3)
                    delta_value = 0.1                        # Value to increase/decrease (radians)
                    # ==========================================

                    # 1. Initialize: Calculate the target angle ONLY ONCE
                    if not self.phase_initialized:
                        if target_joint_name not in self.joint_pos:
                            self.get_logger().warn(f"Waiting for {target_joint_name} data...")
                            return

                        # Calculate target (Current + Delta)
                        current_angle = self.joint_pos[target_joint_name]
                        self.target_adjust_angle = current_angle + delta_value
                        
                        self.phase_initialized = True
                        self.get_logger().info(f"Adjusting {target_joint_name} by {delta_value}. Target: {self.target_adjust_angle:.3f}")

                    # 2. Control Loop
                    current_val = self.joint_pos[target_joint_name]
                    err = self.target_adjust_angle - current_val

                    # 3. Check if reached
                    if abs(err) < self.max_tol:
                        self.stop_joint()
                        
                        # --- LOGGING DATA AS REQUESTED ---
                        self.get_logger().info("✓ Adjustment Reached. Logging Current Pose:")
                        self.get_logger().info(f"   >> Position (XYZ): {self.current_tcp_pos}")
                        self.get_logger().info(f"   >> Quaternion (XYZW): {self.current_tcp_orient}")
                        # ---------------------------------

                        self.phase_initialized = False  # Reset flag
                        self.phase = 'TASK_COMPLETE'
                        return

                    # 4. Move the Joint (P-Controller)
                    speed = max(min(self.base_kp * err, self.base_max_speed), -self.base_max_speed)
                    
                    cmd = [0.0] * 6
                    cmd[target_joint_index] = float(speed)
                    
                    msg = JointJog()
                    msg.joint_names = self.joint_names_list
                    msg.velocities = cmd
                    self.joint_pub.publish(msg)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

        elif self.phase == 'TASK_COMPLETE': 
            self.get_logger().info("All tasks completed successfully. Shutting down.", throttle_duration_sec=2.0)
            self.stop_joint()
            
def main():
    rclpy.init()
    node = Task4c()
    rclpy.spin(node)
    node.destroy_node() 
    rclpy.shutdown()

if __name__ == '__main__':
    main()   