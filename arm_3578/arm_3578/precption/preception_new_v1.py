#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
# Team ID:          3578
# Theme:            Krishi coBot
# Author List:      Raghav Jibachha Mandal, Ashishkumar Rajeshkumar Jha, Aman Ratanlal Chauhan, Harshil Rahulbhai Mehta
# Filename:         task4b_perception.py
'''

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
from cv2 import aruco
import tf2_ros
import tf2_geometry_msgs  # noqa: F401
import numpy as np
from geometry_msgs.msg import TransformStamped, PointStamped
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation as R

# --- Global configuration ---
SHOW_IMAGE = True
DISABLE_MULTITHREADING = False
LOG_ALL_TF = True
PUBLISH_CAMERA_TF = False

# --- Utility functions (calculate_rectangle_area, detect_aruco_markers remain unchanged) ---
def calculate_rectangle_area(coordinates):
    if coordinates is None or len(coordinates) != 4: return 0.0, 0.0
    width = np.linalg.norm(coordinates[0] - coordinates[1])
    height = np.linalg.norm(coordinates[1] - coordinates[2])
    return width * height, width

def detect_aruco_markers(image, cam_mat, dist_mat, marker_size=0.13):
    ARUCO_AREA_THRESHOLD = 1500
    center_list, distance_list, angle_list, width_list, ids_list, rvecs_list, tvecs_list = [], [], [], [], [], [], []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0: return center_list, distance_list, angle_list, width_list, ids_list, rvecs_list, tvecs_list
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    for i, marker_id in enumerate(ids):
        corner = corners[i][0]
        area, width = calculate_rectangle_area(corner)
        if area < ARUCO_AREA_THRESHOLD: continue
        cX, cY = int(np.mean(corner[:, 0])), int(np.mean(corner[:, 1]))
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, cam_mat, dist_mat)
        cv2.drawFrameAxes(image, cam_mat, dist_mat, rvecs[0], tvecs[0], 0.1)
        distance = float(np.linalg.norm(tvecs[0]))
        rot_mat, _ = cv2.Rodrigues(rvecs[0])
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        angle_aruco = (0.788 * yaw) - ((yaw ** 2) / 3160)
        center_list.append((cX, cY)); distance_list.append(distance); angle_list.append(angle_aruco)
        width_list.append(width); ids_list.append(int(marker_id)); rvecs_list.append(rvecs[0]); tvecs_list.append(tvecs[0])
    return center_list, distance_list, angle_list, width_list, ids_list, rvecs_list, tvecs_list

class FruitsAndArucoTF(Node):
    def __init__(self):
        super().__init__('fruits_aruco_tf_publisher')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        self.team_id = "3578"
        self.ARUCO_OBJECT_MAP = {3: 'fertilizer_1'}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.cb_group = MutuallyExclusiveCallbackGroup() if DISABLE_MULTITHREADING else ReentrantCallbackGroup()
        
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)

        self.latched_tf = {}
        self.LATCH_REFRESH_SEC = 20.0
        self.last_latch_time = self.get_clock().now()
        self.AVG_SAMPLE_COUNT = 15
        self.position_buffer = {}

        self.debug_image_pub = self.create_publisher(Image, '/task3a/debug_image', 10)
        self.camera_frame = None
        self.centerCamX, self.centerCamY, self.focalX, self.focalY = 640.0, 360.0, 915.3, 914.0
        self.cam_mat = np.array([[self.focalX, 0.0, self.centerCamX], [0.0, self.focalY, self.centerCamY], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_mat = np.zeros((5, 1), dtype=np.float32)
        self.R_optical_correction = np.eye(3, dtype=np.float32)
        self.create_timer(0.1, self.process_image, callback_group=self.cb_group)

        if SHOW_IMAGE: cv2.namedWindow('preception_detection', cv2.WINDOW_NORMAL)
        self.get_logger().info("preception Node Started")

    def depthimagecb(self, data):
        try:
            arr = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            self.depth_image = arr.astype(np.float32) / 1000.0 if arr.dtype == np.uint16 else arr.astype(np.float32)
        except Exception as e: self.get_logger().error(f"Depth error: {e}")

    def colorimagecb(self, data):
        try: self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e: self.get_logger().error(f"Color error: {e}")

    def depth_calculator(self, depth_img, x, y, window_size=5):
        if depth_img is None: return 0.0
        h, w = depth_img.shape
        window = depth_img[max(0, y-window_size//2):min(h, y+window_size//2+1), max(0, x-window_size//2):min(w, x+window_size//2+1)]
        valid = window[window > 0]
        if len(valid) == 0: return 0.0
        med = float(np.median(valid))
        return med / 1000.0 if med > 100.0 else med

    def convert_pixel_to_3d(self, cX, cY, distance):
        x_opt = (cX - self.centerCamX) * distance / self.focalX
        y_opt = (cY - self.centerCamY) * distance / self.focalY
        p_cam = self.R_optical_correction @ np.array([[x_opt], [y_opt], [distance]])
        return tuple(map(float, p_cam.flatten()))

    def compute_average_position(self, samples):
        mean = np.mean(np.array(samples, dtype=np.float32), axis=0)
        return float(mean[0]), float(mean[1]), float(mean[2])

    def detect_bad_fruits(self, rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        h, w, _ = hsv.shape
        green_mask = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=3)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=3)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bad_fruits, seq_id = [], 1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000 or area < 800: continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            if len(approx) > 4 and circularity > 0.6:
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                cx, cy, r = int(cx), int(cy), int(r)
                x_s, y_s, x_e, y_e = max(0, cx-r), max(0, cy), min(w, cx+r), min(h, int(cy+1.5*r))
                if x_e > x_s and y_e > y_s:
                    roi = rgb_image[y_s:y_e, x_s:x_e]
                    if roi.size == 0: continue
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    grey_px = cv2.countNonZero(cv2.inRange(hsv_roi, (0, 0, 50), (180, 50, 200)))
                    if grey_px / (roi.shape[0]*roi.shape[1]) > 0.2:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        bad_fruits.append({'id': seq_id, 'center': (cx, cy), 'box': (bx, by, bw, bh)})
                        seq_id += 1
        return bad_fruits

    def publish_tf(self, frame_id, child_frame_id, x, y, z, quat=None, stamp=None):
        if quat is None: quat = [0.0, 0.0, 0.0, 1.0]
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        tf_msg.header.frame_id, tf_msg.child_frame_id = frame_id, child_frame_id
        tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z = float(x), float(y), float(z)
        tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w = quat
        try:
            self.tf_broadcaster.sendTransform(tf_msg)
            if LOG_ALL_TF: self.get_logger().info(f"TF: {child_frame_id} at ({x:.3f}, {y:.3f}, {z:.3f})")
            return True
        except Exception as e: return False

    def publish_debug_image(self, cv_img):
        try:
            msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.camera_frame if self.camera_frame else "camera_color_optical_frame"
            self.debug_image_pub.publish(msg)
        except Exception: pass

    def process_image(self):
        if self.cv_image is None or self.depth_image is None: return
        img, depth = self.cv_image.copy(), self.depth_image.copy()
        now = self.get_clock().now()
        elapsed = (now - self.last_latch_time).nanoseconds * 1e-9

        if elapsed > self.LATCH_REFRESH_SEC:
            self.get_logger().info(f"[LATCH RESET] Refreshing all TFs")
            self.latched_tf.clear()
            self.position_buffer.clear() # FIX 2: Clear buffer on refresh
            self.last_latch_time = now

        # --- ArUco Processing ---
        c_list, d_list, a_list, _, ids, _, t_list = detect_aruco_markers(img, self.cam_mat, self.dist_mat)
        if ids:
            for mid, center, e_dist, angle, tvec in zip(ids, c_list, d_list, a_list, t_list):
                try:
                    cX, cY = int(center[0]), int(center[1])
                    dist = self.depth_calculator(depth, cX, cY)
                    if dist == 0 or np.isnan(dist) or dist > 5.0: dist = float(e_dist) if e_dist else 0.0
                    if dist == 0: continue
                    
                    x_c, y_c, z_c = self.convert_pixel_to_3d(cX, cY, dist)
                    quat = R.from_euler('z', float(angle)).as_quat()
                    obj_type = self.ARUCO_OBJECT_MAP.get(int(mid), f"obj_{mid}")
                    child_frame = f"{self.team_id}_{obj_type}" if int(mid) in self.ARUCO_OBJECT_MAP else obj_type

                    if child_frame not in self.latched_tf:
                        p_cam = PointStamped()
                        p_cam.header.frame_id, p_cam.header.stamp = "camera_color_optical_frame", self.get_clock().now().to_msg()
                        p_cam.point.x, p_cam.point.y, p_cam.point.z = x_c, y_c, z_c
                        try:
                            p_base = self.tf_buffer.transform(p_cam, "base_link", timeout=Duration(seconds=0.5))
                            pos = (float(p_base.point.x), float(p_base.point.y), float(p_base.point.z))
                            if child_frame not in self.position_buffer: self.position_buffer[child_frame] = []
                            self.position_buffer[child_frame].append(pos)
                            
                            # Safety Clamp (Fix 6)
                            if len(self.position_buffer[child_frame]) > self.AVG_SAMPLE_COUNT: self.position_buffer[child_frame].pop(0)

                            if len(self.position_buffer[child_frame]) >= self.AVG_SAMPLE_COUNT:
                                x_a, y_a, z_a = self.compute_average_position(self.position_buffer[child_frame])
                                self.latched_tf[child_frame] = (x_a, y_a, z_a, quat)
                                self.position_buffer.pop(child_frame)
                                self.get_logger().info(f"[AVG LATCHED] {child_frame}")
                            # FIX 3: Removed false [LATCHED] log from here
                        except Exception: continue

                    if child_frame in self.latched_tf: # FIX 1: Guard access
                        xb, yb, zb, qb = self.latched_tf[child_frame]
                        self.publish_tf("base_link", child_frame, xb, yb, zb, quat=qb)
                        cv2.circle(img, (cX, cY), 5, (0, 255, 255), -1)
                        cv2.putText(img, obj_type, (cX+10, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                except Exception: continue

        # --- Bad Fruit Processing ---
        fruits = self.detect_bad_fruits(img)
        for idx, fruit in enumerate(fruits, start=1):
            try:
                (cX, cY), (fx, fy, fw, fh) = fruit['center'], fruit['box']
                fruit_frame = f"{self.team_id}_bad_fruit_{idx}"
                dist = self.depth_calculator(depth, int(cX), int(cY))
                if dist == 0 or dist > 5.0: continue
                x_c, y_c, z_c = self.convert_pixel_to_3d(cX, cY, dist)

                if fruit_frame not in self.latched_tf:
                    p_cam = PointStamped()
                    p_cam.header.frame_id, p_cam.header.stamp = "camera_color_optical_frame", self.get_clock().now().to_msg()
                    p_cam.point.x, p_cam.point.y, p_cam.point.z = x_c, y_c, z_c
                    try:
                        p_base = self.tf_buffer.transform(p_cam, "base_link", timeout=Duration(seconds=0.5))
                        pos = (float(p_base.point.x), float(p_base.point.y), float(p_base.point.z))
                        if fruit_frame not in self.position_buffer: self.position_buffer[fruit_frame] = []
                        self.position_buffer[fruit_frame].append(pos)
                        
                        if len(self.position_buffer[fruit_frame]) > self.AVG_SAMPLE_COUNT: self.position_buffer[fruit_frame].pop(0)

                        if len(self.position_buffer[fruit_frame]) >= self.AVG_SAMPLE_COUNT:
                            xa, ya, za = self.compute_average_position(self.position_buffer[fruit_frame])
                            self.latched_tf[fruit_frame] = (xa, ya, za, [0.0, 0.0, 0.0, 1.0])
                            self.position_buffer.pop(fruit_frame)
                            self.get_logger().info(f"[AVG LATCHED] {fruit_frame}")
                        # FIX 3: Removed false [LATCHED] log from here
                    except Exception: continue

                if fruit_frame in self.latched_tf: # FIX 1: Guard access
                    xb, yb, zb, qb = self.latched_tf[fruit_frame]
                    self.publish_tf("base_link", fruit_frame, xb, yb, zb, quat=qb)
                    cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
                    cv2.putText(img, f"bad_{idx}", (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception: continue

        if SHOW_IMAGE:
            cv2.imshow('task4b_detection', img)
            cv2.waitKey(1)
        self.publish_debug_image(img)

def main(args=None):
    rclpy.init(args=args)
    node = FruitsAndArucoTF()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE: cv2.destroyAllWindows()

if __name__ == '__main__':
    main()