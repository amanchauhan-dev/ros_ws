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

    def publish_twist(self, direction, speed):
        # CHANGED: Use TwistStamped
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        msg.twist.linear.x = float(direction[0] * speed)
        msg.twist.linear.y = float(direction[1] * speed)
        msg.twist.linear.z = float(direction[2] * speed)
        
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

            # --- 1. Scan for Fertilizer ---
            if self.ferti_pose is None:
                self.ferti_pose = self.lookup_tf(self.base_link_name, self.fertilizerTFname)

            # --- 2. Check Transition Condition ---
            if self.ferti_pose is None:
                self.get_logger().info("Waiting for fertilizer TF...", throttle_duration_sec=2.0)
                return 

            self.phase = 'PHASE_SAFE_LIFT_SHOULDER'
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
                self.phase = 'WRIST_ALING_TO_FERTI'
# ------------------------------------------------------------------------------------------------------------------------------------
# i added this phase  to aling with ferti
        elif self.phase == 'WRIST_ALING_TO_FERTI':
            target_ferti = self.ferti_pose.copy()
            if self.align_joint_to_pose(target_ferti, "Fertilizer", 'wrist_3_joint', 5):
                self.get_logger().info("Wrist aligned to Fertilizer. Proceeding to Final Approach.")
                self.phase = 'PHASE_4_FINAL_APPROACH'

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_4_FINAL_APPROACH':
            self.get_logger().info(f"Force met ({self.current_force_z:.2f}). Attaching Gripper...")
            target = self.ferti_pose.copy()
            target[1] += 0.1
            # 1. Move to actual fertilizer pose (slow=True)
            reached = self.move_to_tcp_target(target, tol=self.approach_tol, slow=True)
# ================================================================================================================================================================
# this line can me used when there is is collision happend 
            # if reached or self.current_force_z > 30.0:
            if reached:
                self.get_logger().info("✓ Reached fertilizer hover position. Waiting before attach...")
                self.publish_twist([0.0, 0.0, 0.0], 0.0) # Stop twist
                self.phase = 'CHECK_ORIENATION'
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'CHECK_ORIENATION':
            self.get_logger().info("we check the exist in the pickup orienation or not ")
            #  igf not then move to that pose 
            self.phase = 'MOVED_TO_TARGET'
        elif self.phase == 'MOVE_TO_TARGET':
            self.get_logger().info(f"Force met ({self.current_force_z:.2f}). Attaching Gripper...")
            reached = self.move_to_tcp_target(self.ferti_pose, tol=self.approach_tol, slow=True)
            if reached:
                self.get_logger().info("✓ Reached fertilizer hover position. Waiting before attach...")
                self.publish_twist([0.0, 0.0, 0.0], 0.0) # Stop twist
                self.phase = 'ATTACH_FERTI_PRE_WAIT'


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

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
                self.get_logger().info(f"Force met ({self.current_force_z:.2f}). Attaching Gripper...")
                self.set_gripper_state('attach')
                self.phase = 'ATTACH_FERTI_POST_WAIT'
            else:
                self.get_logger().info(f"Force too low ({self.current_force_z:.2f}). Waiting for contact...", throttle_duration_sec=1.0)
        
        elif self.phase == 'ATTACH_FERTI_POST_WAIT':
            self.get_logger().info("Waiting 3s after attaching...", throttle_duration_sec=1.0)
            if self.wait_for_timer(3.0):
                 self.phase = 'PHASE_5_LIFT_FERTILIZER'
        # ---------------------------

        elif self.phase == 'PHASE_5_LIFT_FERTILIZER':
            # 1. Lift Fertilizer
            target = self.ferti_pose.copy()
            target[2] += 0.07
            reached = self.move_to_tcp_target(target, tol=0.05, slow=True)

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
            if self.align_joint_to_pose(self.dustbin_fixed_pose, "Dustbin", self.joint1, 0):
                self.get_logger().info("Base aligned to Dustbin. Reaching Bin.")
                self.dustbin_hover_pose = self.current_tcp_pos.copy()
                self.phase = 'GRIPPER_ORIENTATION_DOWN_FOR_FERTI'

        elif self.phase == 'GRIPPER_ORIENTATION_DOWN_FOR_FERTI':
            self.wrist_orientation('down', 'PHASE_REACH_BIN', angle=-1.0)

        elif self.phase == 'PHASE_REACH_BIN':
            # 1. Move Linear TCP to the specific Dustbin XYZ
            if self.move_to_tcp_target(self.dustbin_fixed_pose, self.approach_tol):
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
            
def main():
    rclpy.init()
    node = Task4c()
    rclpy.spin(node)
    node.destroy_node() 
    rclpy.shutdown()

if __name__ == '__main__':
    main()   