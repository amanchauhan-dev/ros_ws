#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import TwistStamped 
from std_msgs.msg import Float64MultiArray,Float32
from control_msgs.msg import JointJog
from sensor_msgs.msg import JointState
from rclpy.duration import Duration
import tf2_ros
from std_srvs.srv import SetBool

class task4A(Node):
    def __init__(self):
        super().__init__('task4A')
        
        # ===================publishers========================
        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
# ---------------------------------------------------------------------------------
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

        # Add this list to your __init__ so you can use it in every function
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
        #   here i need to chnage in the future for real world of tcp_topic 
        self.tcp_pose_sub = self.create_subscription(
            Float64MultiArray, 
            '/tcp_pose_raw', 
            self.tcp_pose_callback, 
            10
        )
        
        self.force_sub = self.create_subscription(
            Float32, 
            '/net_wrench', 
            self.force_callback, 
            10
        )


        # 2. CREATE SERVICE CLIENTS FOR GRIPPER
        self.magnet_client = self.create_client(SetBool, '/magnet')

        self.timer = self.create_timer(0.05, self.main_loop)

        # ===================configurations========================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.teamIdentifier = '3578'
        self.base_link_name = 'base_link'
        self.joint1 = 'shoulder_pan_joint'
        self.wrist2 = 'wrist_2_joint'
        self.joint_pos = {}
        # tcp_pose variable
        self.current_tcp_pos = None
        self.stored_home_pose = None
        self.wait_start_time = None
        self.current_force_z = None

        self.max_tol = np.deg2rad(2)
        self.base_kp = 2.0
        self.base_max_speed = 2.5

        # phases config 
        
        self.phase = 'PHASE_1_getting_tf'
        
        self.ferti_pose = None
        self.fertilizerTFname = f'{self.teamIdentifier}_fertilizer_1'
        
        self.badFruitTable = []
        self.badFruitFrameList = [
            f'{self.teamIdentifier}_bad_fruit_1',
            f'{self.teamIdentifier}_bad_fruit_2',
            f'{self.teamIdentifier}_bad_fruit_3'
        ]
        self.slowPhase = False
        # phase arm orietation phase 
        self.phase_initialized = False
        self.wrist1_delta_down = -1.40
        # phase fruits tray approach 
        self.fruit_tray_fixed_pose = np.array([-0.159, 0.501, 0.415])
        self.approach_offset_z = 0.20     
        self.approach_tol = 0.03   
        # phase fruits sorting
        self.current_fruit_index = 0
        # phase of dustbin 
        self.dustbin_fixed_pose = np.array([ -0.806, 0.010, 0.182])
        self.dustbin_hover_pose = None
        
# ==============================function definitions========================

# ==================callbacks===============================================
    def joint_state_callback(self, msg):
        for n,p in zip(msg.name, msg.position):
            self.joint_pos[n] = p   
# --------------------------------------------------------------------------
    def tcp_pose_callback(self, msg):
        self.current_tcp_pos = np.array([msg.data[0], msg.data[1], msg.data[2]])

# ---------------------------------------------------------------------------
    def force_callback(self, msg):
        self.current_force_z = msg.data

# ==================Attach/Deattach=========================================
    def set_gripper_state(self, action):
        """
        Controls the electromagnet.
        action: 'attach' (True) or 'detach' (False)
        object_name: Not used for SetBool /magnet, but kept here 
                     so your main_loop code doesn't crash.
        """
        req = SetBool.Request()
        
        if action == 'attach':
            req.data = True
            self.get_logger().info(f"Magnet ON (Force Z: {self.current_force_z})")
        else:
            req.data = False
            self.get_logger().info(f"Magnet OFF (Force Z: {self.current_force_z})")

        # Send the command
        future = self.magnet_client.call_async(req)
# =================motion commands==========================================
    def stop_joint(self):
        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities = [0.0] * 6
        self.joint_pub.publish(msg)
# --------------------------------------------------------------------------------------
    def align_joint_to_pose(self, target_pose, target_label, joint_name, joint_index):
        """
        Generic function to align ANY joint to look at a target [x, y].
        joint_name: The string name (e.g., 'shoulder_pan_joint') for feedback.
        joint_index: The list index (0-5) to publish the speed command.
        """
        # 1. Safety Check
        if target_pose is None:
            self.get_logger().warn(f"Cannot align to {target_label}: Pose is None!", throttle_duration_sec=2.0)
            return

        # 2. Extract coordinates
        x = target_pose[0]
        y = target_pose[1]

        # 3. Calculate Desired Angle
        desired = self.norm(np.arctan2(y, x) + np.pi)
        
        # 4. Get Current Joint Position
        if joint_name not in self.joint_pos:
            self.get_logger().warn(f"Waiting for {joint_name}...", throttle_duration_sec=2.0)
            return
            
        current = self.joint_pos[joint_name]
        err = self.norm(desired - current)

        # 5. Logging
        self.get_logger().info(
            f"[{target_label}] cur={current:.2f} tgt={desired:.2f} err={err:.3f}",
            throttle_duration_sec=0.5
        )

        # 6. Check Completion
        if abs(err) < self.max_tol:
            self.stop_joint()
            self.get_logger().info(f"✓ Aligned {target_label}. Switching phase.")
            return True

        # 7. Move Robot
        speed = max(min(self.base_kp * err, self.base_max_speed), -self.base_max_speed)*3
        

        self.get_logger().info(
            f"[{target_label}] Error: {err:.3f} rad | Speed: {speed:.4f} rad/s", 
            throttle_duration_sec=0.5
        )
        # --- DYNAMIC INDEX LOGIC 
        cmd = [0.0] * 6          # Create array [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cmd[joint_index] = float(speed) # Insert speed at the specific index
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg() # Optional but good practice
        msg.joint_names = self.joint_names_list        # Ensure this list is defined in __init__
        msg.velocities = cmd
        self.joint_pub.publish(msg)
        
        return False
    
    
# ----------------------------------------------------------------------------------------
    def move_to_tcp_target(self, target, tol=0.03, slow=False):
        """
        Returns True if reached target, False otherwise.
        Moves the robot using Twist commands.
        """
        if self.current_tcp_pos is None:
            return False

        err = target - self.current_tcp_pos
        dist = np.linalg.norm(err)

        if dist < tol:
            self.publish_twist([0.0, 0.0, 0.0], 0.0) # Stop
            return True
        
        # Calculate Speed
        direction = err / dist
        max_s = 0.15 if slow else 0.4
        speed = min(dist * 2, max_s)

        mode = "SLOW" if slow else "FAST"
        self.get_logger().info(
            f"[TCP {mode}] Dist: {dist:.3f}m | Speed: {speed:.4f} m/s", 
            throttle_duration_sec=0.5
        )
        

        self.publish_twist(direction, speed)
        return False

# -------------------------------------------------------------------------------------------
    def wrist_orientation(self, direction, next_phase):
        """
        Controls Wrist 1 to move UP or DOWN.
        direction: 'up' or 'down'
        next_phase: The string name of the phase to switch to when done.
        """
        # 1. Check if we have joint data
        if 'wrist_1_joint' not in self.joint_pos:
            return

        # 2. Initialization (Run once when entering this phase)
        if not self.phase_initialized:
            current_w1 = self.joint_pos['wrist_1_joint']
            
            # Set Target based on direction
            if direction == 'down':
                # Logic: Current + Delta (e.g. -1.30) moves it DOWN
                self.target_w1 = current_w1 + self.wrist1_delta_down 
            elif direction == 'up':
                # Logic: Current - Delta reverses the movement (moves it UP)
                self.target_w1 = current_w1 - self.wrist1_delta_down
            else:
                self.get_logger().error(f"Invalid direction: {direction}")
                return

            self.phase_initialized = True
            self.get_logger().info(f"[Wrist] Moving {direction.upper()}... Target: {self.target_w1:.2f}")
            return

        # 3. Control Loop
        current_w1 = self.joint_pos['wrist_1_joint']
        err = self.target_w1 - current_w1

        # 4. Check Completion
        if abs(err) < self.max_tol:
            self.stop_joint()
            self.get_logger().info(f"✓ Wrist {direction} complete. Switching to {next_phase}")
            self.phase_initialized = False # IMPORTANT: Reset for the next phase
            self.phase = next_phase
            return

        # 5. Move Robot (Only controlling Wrist 1 at index 3)
        cmd = [0.0] * 6 
        
        speed = max(min(self.base_kp * err, self.base_max_speed), -self.base_max_speed)
        cmd[3] = speed 
        

        self.get_logger().info(
            f"[Wrist {direction}] Error: {err:.3f} | Speed: {speed:.4f} rad/s", 
            throttle_duration_sec=0.5
        )
        
        msg = JointJog()
        msg.joint_names = self.joint_names_list
        msg.velocities = cmd
        self.joint_pub.publish(msg)


# --------------------------------------------------------------------------------------------
    def wait_for_timer(self, seconds):
        """
        Non-blocking timer.
        Returns False while waiting.
        Returns True when time is up (and automatically resets).
        """
        # 1. Start the timer if it hasn't started
        if self.wait_start_time is None:
            self.wait_start_time = self.get_clock().now()
            return False # We just started, so we aren't done yet

        # 2. Check how much time has passed
        current_time = self.get_clock().now()
        time_diff = (current_time - self.wait_start_time).nanoseconds / 1e9

        # 3. If time is not up, keep waiting
        if time_diff < seconds:
            return False
        
        # 4. If time IS up, reset the timer for next time and return True
        self.wait_start_time = None
        return True

# ---------------------------------------------------------------------------------------------
    def publish_twist(self, direction, speed):
        msg = TwistStamped()
        
        # 1. Add Timestamp
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'  # Optional, but good practice
        
        # 2. Add Velocity (Notice the .twist structure)
        msg.twist.linear.x = float(direction[0] * speed)
        msg.twist.linear.y = float(direction[1] * speed)
        msg.twist.linear.z = float(direction[2] * speed)
        
        self.twist_pub.publish(msg)

# ------------------------------------------------------------------------------------------------------
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
            self.get_logger().warn(f"Could not find transform from {source} to {target}: {e}", throttle_duration_sec=1.0)
            return None

# -----------------------------------------------------------------------------------------
    def scan_for_bad_fruit_frames(self):
        found_records = []
        all_detected = True

        for frame_name in self.badFruitFrameList:
            position= self.lookup_tf(self.base_link_name, frame_name)
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
    @staticmethod
    def norm(a):
        while a > np.pi:
            a -= 2 * np.pi
        while a < -np.pi:
            a += 2 * np.pi
        return a
 
# =============================main loop=======================================
    def main_loop(self):
        if self.phase == 'PHASE_1_getting_tf':
            # --- 1. Scan for Fertilizer ---
            if self.ferti_pose is None:
                self.ferti_pose = self.lookup_tf(self.base_link_name, self.fertilizerTFname)

            # --- 2. Check Transition Condition ---
            if self.ferti_pose is None:
                self.get_logger().info("Waiting for fertilizer TF...", throttle_duration_sec=2.0)
                return 
                
            # --- 3. Scan for Fruits ---
            if not self.badFruitTable:
                fruit_records = self.scan_for_bad_fruit_frames()
                if fruit_records:
                    self.badFruitTable = fruit_records
                    self.get_logger().info(f"✓ Found all {len(fruit_records)} bad fruits")
                else:
                    self.get_logger().info("Scanning for bad fruits...", throttle_duration_sec=2.0)
                    return

            self.get_logger().info("All TFs acquired. Transitioning to PHASE_2.")
            self.phase = 'PHASE_BASE_ALING_TO_TRAY'
# -------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_BASE_ALING_TO_TRAY':
            # 1. Find the specific pose you want from your list/table
            target_pose = self.fruit_tray_fixed_pose.copy()

            # 2. Pass that pose to the function
            if self.align_joint_to_pose(target_pose, "Fruit_Tray_Center", self.joint1, 0):
                self.get_logger().info("Base aligned to Fruit Tray. Moving Wrist Down.")
                self.phase = 'PHASE_GRIPPER_ORIENTATION'
# -----------------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_GRIPPER_ORIENTATION':
            self.wrist_orientation('down', 'APPROACH_FRUITS_TRAY')
# -----------------------------------------------------------------------------------------------------------------
        elif self.phase == 'APPROACH_FRUITS_TRAY':   
            target = self.fruit_tray_fixed_pose.copy()
            
            if self.move_to_tcp_target(target, self.approach_tol):
                self.get_logger().info("Initial Approach Done. Starting Sorting.")
                self.phase = 'FRUITS_SORT_PHASE'


# =====================================================================================================================
#                         LAST-TIME-ISSUE-ARISE_FROM-HERE    
# -----------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'FRUITS_SORT_PHASE':
            
            # 1. Get the current fruit location
            fruit_record = self.badFruitTable[self.current_fruit_index]
            original_fruit_pose = fruit_record['pos']
            # 3. Create a target to HOVER ABOVE the fruit
            hover_target = original_fruit_pose.copy()
            hover_target[2] += 0.15
       
            if self.move_to_tcp_target(hover_target, self.approach_tol):
                self.publish_twist([0.0, 0.0, 0.0], 0.0) # Stop briefly
                self.get_logger().info("Hover on the current index fruits . Descending now...")
                self.phase = 'PHASE_SLOW_APPROACH_TO_FRUITS'
             
            
# =============================================================================================================================================
        elif self.phase == 'PHASE_SLOW_APPROACH_TO_FRUITS':
            fruit_record = self.badFruitTable[self.current_fruit_index]
            original_fruit_pose = fruit_record['pos']
            if self.move_to_tcp_target(original_fruit_pose, self.approach_tol, slow=True):
                self.get_logger().info("we are slow go to postion ")
                self.publish_twist([0.0, 0.0, 0.0], 0.0)
                self.phase = 'ATTACH_FRUITS_PRE_WAIT'

#----------------------------------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'ATTACH_FRUITS_PRE_WAIT':
            self.get_logger().info("Settling before grab...", throttle_duration_sec=1.0)
            if self.wait_for_timer(3.0):
                self.phase = 'ATTACH_FRUITS_ACTION'

        elif self.phase == 'ATTACH_FRUITS_ACTION':
            self.get_logger().info("Activating Gripper...")
            self.set_gripper_state('attach')
            self.phase = 'ATTACH_FRUITS_POST_WAIT'

        elif self.phase == 'ATTACH_FRUITS_POST_WAIT':
            self.get_logger().info("Waiting for gripper to lock...", throttle_duration_sec=1.0)
            if self.wait_for_timer(3.0):
                self.phase = 'LIFT_AFTER_GRAB'
# -----------------------------------------------------------------------------------------------------------
# ========================================================================================================================
# last time issue arise here too like the get collied with dustbin 
# 1. old one have increased the the height so not get collied with the dutbin 
# 2. instead of lift go to the dustbin fixed pose i commrent this for now 
# ========================================================================================================================
        elif self.phase == 'LIFT_FRUITS_TRAY_POSE':
            target = self.fruit_tray_fixed_pose.copy()
            if self.move_to_tcp_target(target,self.approach_tol):
                self.get_logger("safe to tray pose now dispose the item")
                self.phase = 'LIFT_AFTER_GRAB'
# ========================================================================================================================
        elif self.phase == 'LIFT_AFTER_GRAB':
            # 1. Get current fruit location again to calculate the "Up" position
            fruit_record = self.badFruitTable[self.current_fruit_index]
            lift_target = fruit_record['pos'].copy()  
            lift_target[2] += 0.20

            
            # 3. Move to the lifted positio
            if self.move_to_tcp_target(lift_target, self.approach_tol):
                self.get_logger().info("Lifted successfully. Moving to Dustbin.")
                self.stop_joint() 
                self.phase = 'PHASE_ALIGN_TO_BIN'
# -------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_ALIGN_TO_BIN':
            if self.align_joint_to_pose(self.dustbin_fixed_pose, "Dustbin", self.joint1, 0):
                self.get_logger().info("Base aligned to Dustbin. Reaching Bin.")
                self.dustbin_hover_pose = self.current_tcp_pos.copy()
                self.phase = 'PHASE_REACH_BIN'
# ---------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_REACH_BIN':
            # 1. Move Linear TCP to the specific Dustbin XYZ
            if self.move_to_tcp_target(self.dustbin_fixed_pose, self.approach_tol):
                self.get_logger().info("Reached Dustbin. Dropping fruit.")
                self.publish_twist([0.0, 0.0, 0.0], 0.0) # Stop
                
                # 4. Transition to Reset/Next Fruit
                self.phase = 'DETACH_FRUITS'
# ----------------------------------------------------------------------------------------------------
        elif self.phase == 'DETACH_FRUITS':
            self.get_logger().info("Settling before detach...", throttle_duration_sec=1.0)
            if self.wait_for_timer(3.0):
                
                self.get_logger().info("Deactivating Gripper...")
                self.set_gripper_state('detach')
                self.phase = 'PHASE_RESET_FOR_NEXT'
# -----------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_RESET_FOR_NEXT':
            # retuen to safe height to return to the fruits tary 
            if self.move_to_tcp_target(self.dustbin_hover_pose, self.approach_tol):
                self.get_logger().info("sucessfully return to the safe postion ")
                self.publish_twist([0.0, 0.0, 0.0], 0.0)

                self.current_fruit_index += 1
                self.get_logger().info(f"Fruit dropped. Index is now {self.current_fruit_index}")
           

                self.phase = 'RETURN_TO_TRAY_APPROACH'
# ---------------------------------------------------------------------------------------------------------
        elif self.phase == 'RETURN_TO_TRAY_APPROACH':

            if self.current_fruit_index >= len(self.badFruitTable):
                self.get_logger().info("All fruits sorted. Stopping.")
                self.stop_joint()
                self.phase = 'PHASE_GRIPPER_UP'
                return
            else:
                # 1. Find the specific pose you want from your list/table
                target_pose = self.fruit_tray_fixed_pose.copy()
                target_pose[2] += self.approach_offset_z

                # 2. Pass that pose to the function
                if self.align_joint_to_pose(target_pose, "Fruit_Tray_return", self.joint1, 0):
                    self.get_logger().info("Returned to tray. Sorting next fruit ")
                    self.phase ='FRUITS_SORT_PHASE'

                    
# -----------------------------------------------------------------------------------------------------------
        elif self.phase == 'PHASE_GRIPPER_UP':
            self.wrist_orientation('up', 'MOVED_FOR_FERTILZER')
# ----------------------------------------------------------------------------------------------------------
        elif self.phase == 'MOVED_FOR_FERTILZER':
            target_fertilzer_hover = self.ferti_pose.copy()
            target_fertilzer_hover[1] += 0.15 # Move Y by +15 cm to hover side position
            
            target_fertilzer_actual = self.ferti_pose.copy()

            
            if not self.slowPhase:
                # Step 1: Go to the side/hover position
                if self.move_to_tcp_target(target_fertilzer_hover, self.approach_tol):
                    self.publish_twist([0.0, 0.0, 0.0], 0.0)
                    self.get_logger().info("Reached Fertilizer Side. Approaching...")
                    self.slowPhase = True
            else:
                # Step 2: Go to actual fertilizer
                if self.align_joint_to_pose(target_fertilzer_actual, 'wrist2', self.wrist2, 4):
                    if self.move_to_tcp_target(target_fertilzer_actual, self.approach_tol, slow=True):
                        self.publish_twist([0.0, 0.0, 0.0], 0.0)
                        self.get_logger().info(f"Reached fertilizer. Attaching...")
                        
                        
                        self.slowPhase = False # Reset for safety
                        self.phase = 'ATTACH_FERTI_PRE_WAIT'
# -----------------------------------------------------------------------------------------------------------
        elif self.phase == 'ATTACH_FERTI_PRE_WAIT':
            self.get_logger().info("Settling before grab...", throttle_duration_sec=1.0)
            if self.wait_for_timer(3.0):
                self.phase = 'ATTACH_FERTI_ACTION'

        elif self.phase == 'ATTACH_FERTI_ACTION':
            self.get_logger().info("Activating Gripper...")
            self.set_gripper_state('attach')
            self.phase = 'ATTACH_FERTI_POST_WAIT'
        
        elif self.phase == 'ATTACH_FERTI_POST_WAIT':
            self.get_logger().info("Waiting for gripper to lock...", throttle_duration_sec=1.0)
            if self.wait_for_timer(3.0):
                 self.phase = 'LIFT_FERTILZER'
# ---------------------------------------------------------------------------------------------------------------
        elif self.phase == 'LIFT_FERTILZER':
            lift_target = self.ferti_pose.copy()
            lift_target[2] += 0.05

            if self.move_to_tcp_target(lift_target, self.approach_tol):
                self.get_logger().info("Fertilizer Lifted. Task Complete.")
                self.stop_joint()
                self.phase = 'REVERSE_FERTILZER'

# ------------------------------------------------------------------------------------------------------------------
        elif self.phase == 'REVERSE_FERTILZER':
            lift_target = self.ferti_pose.copy()
            lift_target[1] += 0.15

            if self.move_to_tcp_target(lift_target, self.approach_tol):
                self.get_logger().info("Fertilizer reverse. Task Complete.")
                self.stop_joint()
                self.phase = 'ALING_DUSTBIN_BASE'
# --------------------------------------------------------------------------------
        elif self.phase == 'ALING_DUSTBIN_BASE':
            if self.align_joint_to_pose(self.dustbin_fixed_pose, "Dustbin_final", self.joint1, 0):
                self.get_logger().info("Base aligned to Dustbin. Moving to Dustbin.")
                self.phase = 'PHASE_GRIPPER_ORIENTATION_DOWN_FERTILIZER'
# ---------------------------------------------------------------------------------
        elif self.phase == 'PHASE_GRIPPER_ORIENTATION_DOWN_FERTILIZER':
            self.wrist_orientation('down', 'PHASE_REACH_BIN_FERTILIZER')
# ---------------------------------------------------------------------------------
        elif self.phase == 'PHASE_REACH_BIN_FERTILIZER':
            # 1. Move Linear TCP to the specific Dustbin XYZ
            if self.move_to_tcp_target(self.dustbin_fixed_pose, self.approach_tol):
                self.get_logger().info("Reached Dustbin. Dropping Fertilizer.")
                self.publish_twist([0.0, 0.0, 0.0], 0.0) # Stop
                
            
                
                # 4. Transition to Reset/Next Fruit
                self.phase = 'DETACH_FERTILIZER'
# -----------------------------------------------------------------------------
        elif self.phase == 'DETACH_FERTILIZER':
            self.get_logger().info("Settling before grab...", throttle_duration_sec=1.0)
            if self.wait_for_timer(3.0):
                self.get_logger().info("Deactivating Gripper...")
                self.set_gripper_state('detach')
                self.phase = 'TASK_COMPLETE'
# ---------------------------------------------------------------------------------
        elif self.phase == 'TASK_COMPLETE': 
            self.get_logger().info("All tasks completed successfully. Shutting down.")
            self.stop_joint()


            
def main():
    rclpy.init()
    node = task4A()
    rclpy.spin(node)
    node.destroy_node() 
    rclpy.shutdown()

if __name__ == '__main__':
    main()
