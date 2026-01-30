#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
# Team ID:          3578
# Theme:            Krishi coBot
# Author List:      Raghav Jibachha Mandal,
#                   Ashishkumar Rajeshkumar Jha,
#                   Aman Ratanlal Chauhan,
#                   Harshil Rahulbhai Mehta
# Filename:         task4b_perception.py
# Purpose:          Detect ArUco markers (fertilizer) and bad fruits,
#                   publish TFs for detected objects, and publish a
#                   debug image (/task3a/debug_image) with annotations.
#
# Coding standard:  File-level, function-level, and inline comments provided.
#                   Followed team coding template for readability and reuse.
#
# Notes:
#   - Uses OpenCV (cv2), cv_bridge, rclpy and tf2_ros.
#   - Publishes TFs for detected objects (camera and base frames).
#   - Publishes debug image on /task3a ./Y/debug_image (bgr8).
#   - Toggle SHOW_IMAGE to disable OpenCV windows if running headless.
'''

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import (
    ReentrantCallbackGroup,
    MutuallyExclusiveCallbackGroup
)
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
from cv2 import aruco
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 (kept for tf conversions)
import numpy as np
from geometry_msgs.msg import TransformStamped, PointStamped
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation as R
import numpy as np

# --------------------------------------------------------------------- #
# Global configuration
# --------------------------------------------------------------------- #
SHOW_IMAGE = True                  # Show OpenCV window (set False for headless)
DISABLE_MULTITHREADING = False     # True -> single threaded callbacks
LOG_ALL_TF = True                # Log all TFs published (can be noisy)
PUBLISH_CAMERA_TF = False          # If True -> also publish object TFs in camera frame

# --------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------- #
def calculate_rectangle_area(coordinates):
    """
    Compute area and width of a 4-corner rectangle from pixel coordinates.

    Args:
        coordinates (np.ndarray): 4x2 array of corner (x,y) coordinates.

    Returns:
        (area, width): tuple(float, float)
    """
    if coordinates is None or len(coordinates) != 4:
        return 0.0, 0.0

    width = np.linalg.norm(coordinates[0] - coordinates[1])
    height = np.linalg.norm(coordinates[1] - coordinates[2])
    area = width * height
    return area, width


def detect_aruco_markers(image, cam_mat, dist_mat, marker_size=0.13):
    """
    Detect ArUco markers and estimate pose.

    Args:
        image (np.ndarray): BGR image.
        cam_mat (np.ndarray): 3x3 camera matrix.
        dist_mat (np.ndarray): distortion coefficients.
        marker_size (float): marker side length (meters).

    Returns:
        center_list, distance_list, angle_list, width_list,
        ids_list, rvecs_list, tvecs_list
    """
    ARUCO_AREA_THRESHOLD = 1500

    center_list = []
    distance_list = []
    angle_list = []
    width_list = []
    ids_list = []
    rvecs_list = []
    tvecs_list = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return center_list, distance_list, angle_list, width_list, ids_list, rvecs_list, tvecs_list

    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    for i, marker_id in enumerate(ids):
        corner = corners[i][0]
        area, width = calculate_rectangle_area(corner)

        if area < ARUCO_AREA_THRESHOLD:
            continue

        cX = int(np.mean(corner[:, 0]))
        cY = int(np.mean(corner[:, 1]))

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[i], marker_size, cam_mat, dist_mat
        )

        # draw axes for visualization
        cv2.drawFrameAxes(image, cam_mat, dist_mat, rvecs[0], tvecs[0], 0.1)

        distance = float(np.linalg.norm(tvecs[0]))

        rot_mat, _ = cv2.Rodrigues(rvecs[0])
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        angle_aruco = (0.788 * yaw) - ((yaw ** 2) / 3160)

        center_list.append((cX, cY))
        distance_list.append(distance)
        angle_list.append(angle_aruco)
        width_list.append(width)
        ids_list.append(int(marker_id))
        rvecs_list.append(rvecs[0])
        tvecs_list.append(tvecs[0])

    return center_list, distance_list, angle_list, width_list, ids_list, rvecs_list, tvecs_list


# --------------------------------------------------------------------- #
# Main Node
# --------------------------------------------------------------------- #
class FruitsAndArucoTF(Node):


    def __init__(self):
        """
        Initialize node, subscriptions, TF infra, publishers, and timers.
        """
        super().__init__('fruits_aruco_tf_publisher')

        # cv bridge for image conversions
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None

        # Team ID (used to name TF child frames)
        self.team_id = "3578"

        # Map ArUco id -> object type (per task spec)
        self.ARUCO_OBJECT_MAP = {
            3: 'fertilizer_1',
        }

        # TF infrastructure
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Callback group
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        # Subscriptions for color, depth, and camera info
        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.colorimagecb,
            10,
            callback_group=self.cb_group
        )

        self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depthimagecb,
            10,
            callback_group=self.cb_group
        )

        # ---------------- LATCHED TF STORAGE ----------------
        # key -> child_frame_name
        # value -> (x, y, z, quat)
        self.latched_tf = {}
        # ---------------- LATCH REFRESH CONFIG ----------------
        self.LATCH_REFRESH_SEC = 20.0   # re-detect every 5 seconds
        # Last time latch was refreshed
        self.last_latch_time = self.get_clock().now()


        # ---------------- AVERAGING CONFIG ----------------
        self.AVG_SAMPLE_COUNT = 15   # number of samples per object

        # key -> list of (x, y, z)
        self.position_buffer = {}


        # Debug image publisher
        self.debug_image_pub = self.create_publisher(Image, '/task3a/debug_image', 10)

        # Camera frame id (updated by CameraInfo)
        self.camera_frame = None

        # Default intrinsics (overwritten by CameraInfo)
        self.sizeCamX = 1280
        self.sizeCamY = 720
        self.centerCamX = 640.0
        self.centerCamY = 360.0
        self.focalX = 915.3
        self.focalY = 914.0

        self.cam_mat = np.array([
            [self.focalX, 0.0, self.centerCamX],
            [0.0, self.focalY, self.centerCamY],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.dist_mat = np.zeros((5, 1), dtype=np.float32)

        # Real camera is already optical frame; correction kept for future use
        self.R_optical_correction = np.eye(3, dtype=np.float32)

        # Main processing timer (10 Hz)
        self.create_timer(0.1, self.process_image, callback_group=self.cb_group)

        # Optional OpenCV window
        if SHOW_IMAGE:
            cv2.namedWindow('preception_detection', cv2.WINDOW_NORMAL)

        self.get_logger().info("preception  Node Started")

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def depthimagecb(self, data):
        """
        Depth Image callback.

        Convert incoming ROS depth image to float32 meters and store in self.depth_image.

        Args:
            data (sensor_msgs.msg.Image): depth image message
        """
        try:
            arr = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            if arr.dtype == np.uint16:
                # sensor provides depth in mm, convert to meters
                arr = arr.astype(np.float32) / 1000.0
            else:
                arr = arr.astype(np.float32)
            self.depth_image = arr
        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")

    def colorimagecb(self, data):
        """
        Color Image callback.

        Convert incoming ROS color image to BGR OpenCV image and store in self.cv_image.

        Args:
            data (sensor_msgs.msg.Image): color image message
        """
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Color conversion error: {e}")



    # ------------------------------------------------------------------ #
    # Depth + geometry helpers
    # ------------------------------------------------------------------ #
    def depth_calculator(self, depth_img, x, y, window_size=5):
        """
        Return median depth around pixel (x, y).

        Args:
            depth_img (np.ndarray): single-channel depth in meters
            x (int): pixel x
            y (int): pixel y
            window_size (int): square window size

        Returns:
            float: median depth (meters) or 0.0 if no valid depth
        """
        if depth_img is None:
            return 0.0

        h, w = depth_img.shape
        x1 = max(0, x - window_size // 2)
        x2 = min(w, x + window_size // 2 + 1)
        y1 = max(0, y - window_size // 2)
        y2 = min(h, y + window_size // 2 + 1)

        window = depth_img[y1:y2, x1:x2]
        valid_depths = window[window > 0]

        if len(valid_depths) == 0:
            return 0.0

        median_depth = float(np.median(valid_depths))
        # handle mm vs m
        if median_depth > 100.0:
            median_depth /= 1000.0
        return median_depth

    def convert_pixel_to_3d(self, cX, cY, distance):
        """
        Convert pixel coordinates + depth -> 3D point in camera optical frame.

        Args:
            cX (float): pixel x
            cY (float): pixel y
            distance (float): depth in meters

        Returns:
            (x, y, z) in camera optical frame (meters)
        """
        x_opt = (cX - self.centerCamX) * distance / self.focalX
        y_opt = (cY - self.centerCamY) * distance / self.focalY
        z_opt = distance

        p_cam = self.R_optical_correction @ np.array([[x_opt], [y_opt], [z_opt]])
        return tuple(map(float, p_cam.flatten()))

    # ------------------------------------------------------------------ #
    # Fruit color classification / detection
    # ------------------------------------------------------------------ #
    def compute_average_position(self, samples):
        """
        Compute average (x, y, z) from samples.
        """
        arr = np.array(samples, dtype=np.float32)
        mean = np.mean(arr, axis=0)
        return float(mean[0]), float(mean[1]), float(mean[2])



    def detect_bad_fruits(self, rgb_image):
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            h, w, _ = hsv.shape

            # Green mask logic...
            green_mask = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=3)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=3)
            
            cv2.imshow('task3a1', green_mask)
            cv2.waitKey(1)
        
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bad_fruits = []
            seq_id = 1
            
            MIN_AREA = 800
            MAX_AREA = 3000 

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area > MAX_AREA or area < MIN_AREA:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue

                approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
                corner = len(approx)
                circularity = (4 * np.pi * area) / (perimeter * perimeter)

                if corner > 4 and circularity > 0.6:
                    (cx, cy), r = cv2.minEnclosingCircle(cnt)
                    cx, cy, r = int(cx), int(cy), int(r)
                    
                    
                    # Start must be smaller than End
                    x_start = int(cx - r) 
                    x_end = int(cx + r)
                    
                    y_start = int(cy)
                    y_end = int(cy + 1.5 * r)
                    
                    # Boundary checks
                    x_start = max(0, x_start)
                    y_start = max(0, y_start)
                    x_end = min(w, x_end)
                    y_end = min(h, y_end)
        
                    # Check valid dimensions before slicing
                    if x_end - x_start > 0 and y_end - y_start > 0:
                        roi = rgb_image[y_start:y_end, x_start:x_end]
                        
                        
                        if roi.size == 0:
                            continue  
                        
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                        # Check for Grey spots (Bad Fruit)
                        lower_gray = np.array([0, 0, 50])
                        upper_gray = np.array([180, 50, 200]) # Tweaked saturation slightly
                        
                        # Count how many pixels are grey
                        grey_pixels = cv2.countNonZero(cv2.inRange(hsv_roi, lower_gray, upper_gray))

                        total_pixel_roi = roi.shape[0]*roi.shape[1]
                        grey_ration = grey_pixels/total_pixel_roi

                        
                        if grey_ration > 0.2: 
                            x, y, w_box, h_box = cv2.boundingRect(cnt)
                            bad_fruits.append({
                                'id': seq_id,
                                'center': (cx, cy),
                                'box': (x, y, w_box, h_box)
                            })
                            # self.get_logger().info(f"sed_id,grey{seq_id,grey_pixels,total_pixel_roi}")
                            seq_id += 1

            
            return bad_fruits
    # ------------------------------------------------------------------ #
    # TF helpers
    # ------------------------------------------------------------------ #
    def publish_tf(self, frame_id, child_frame_id, x, y, z, quat=None, stamp=None):
        """
        Publish TF (TransformStamped) using tf2 broadcaster.

        Args:
            frame_id (str): parent frame id
            child_frame_id (str): child frame id
            x,y,z (float): translation in meters
            quat (list[4]): quaternion [x, y, z, w]
            stamp: optional ROS2 time (builtin) to use for header
        """
        if quat is None:
            quat = [0.0, 0.0, 0.0, 1.0]
        quat = [float(q) for q in quat]

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        tf_msg.header.frame_id = frame_id
        tf_msg.child_frame_id = child_frame_id
        tf_msg.transform.translation.x = float(x)
        tf_msg.transform.translation.y = float(y)
        tf_msg.transform.translation.z = float(z)
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]

        try:
            self.tf_broadcaster.sendTransform(tf_msg)

            if LOG_ALL_TF:
                self.get_logger().info(
                    f"TF: parent='{frame_id}' child='{child_frame_id}' "
                    f"pos=({x:.3f}, {y:.3f}, {z:.3f}) "
                    f"quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})"
                )

            return True
        except Exception as e:
            self.get_logger().error(f"publish_tf failed for {child_frame_id}: {e}")
            return False

    def transform_and_publish(
        self,
        x_cam,
        y_cam,
        z_cam,
        child_frame_id,
        quat=None,
        publish_camera_tf=None,     # None -> follow global flag
        target_frame='base_link',
        timeout_sec=0.5,
    ):
        """
        Publish TFs: optionally in camera frame and always in target_frame (e.g., base_link).

        Args:
            x_cam,y_cam,z_cam: coordinates in camera frame (meters)
            child_frame_id: name of child TF
            quat: quaternion in camera frame (if applicable)
            publish_camera_tf: override global flag
            target_frame: destination frame for transformation
            timeout_sec: transform buffer timeout
        """
        if quat is None:
            quat = [0.0, 0.0, 0.0, 1.0]

        if publish_camera_tf is None:
            publish_camera_tf = PUBLISH_CAMERA_TF

        try:
            x_cam = float(x_cam)
            y_cam = float(y_cam)
            z_cam = float(z_cam)
            quat = [float(q) for q in quat]
        except Exception as e:
            self.get_logger().error(f"transform_and_publish: invalid input: {e}")
            return False

        src = self.camera_frame if self.camera_frame is not None else 'camera_color_optical_frame'
        now = self.get_clock().now().to_msg()

        # Publish camera-frame TF if requested (suffix _cam)
        if publish_camera_tf:
            cam_child = f"{child_frame_id}_cam"
            ok_cam = self.publish_tf(src, cam_child, x_cam, y_cam, z_cam, quat=quat, stamp=now)
            if not ok_cam:
                self.get_logger().warn(f"Failed to publish camera TF {cam_child}")

        # Build a PointStamped in camera frame and transform to target frame
        point_in_camera = PointStamped()
        point_in_camera.header.frame_id = src
        point_in_camera.header.stamp = now
        point_in_camera.point.x = x_cam
        point_in_camera.point.y = y_cam
        point_in_camera.point.z = z_cam

        try:
            try:
                if not self.tf_buffer.can_transform(target_frame, src, rclpy.time.Time()):
                    self.get_logger().warn(
                        f"No transform available from {src} to {target_frame} yet."
                    )
                    return False
            except Exception:
                pass

            point_in_base = self.tf_buffer.transform(
                point_in_camera,
                target_frame,
                timeout=Duration(seconds=timeout_sec),
            )

            x_b = float(point_in_base.point.x)
            y_b = float(point_in_base.point.y)
            z_b = float(point_in_base.point.z)

            ok_base = self.publish_tf(
                target_frame,
                child_frame_id,
                x_b,
                y_b,
                z_b,
                quat=quat,
                stamp=now,
            )
            if not ok_base:
                self.get_logger().error(f"Failed to publish base TF for {child_frame_id}")
                return False

            return True

        except Exception as e:
            self.get_logger().error(f"Transform/Publish failed for {child_frame_id}: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Debug image publisher
    # ------------------------------------------------------------------ #
    def publish_debug_image(self, cv_img):
        """
        Publish an annotated OpenCV BGR image as a ROS2 Image on /task3a/debug_image.

        Args:
            cv_img (np.ndarray): BGR image to publish
        """
        try:
            msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.camera_frame if self.camera_frame else "camera_color_optical_frame"
            self.debug_image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def process_image(self):
        """
        Main processing: detect ArUco markers and bad fruits, publish TFs, and
        create annotated debug image for visualization.
        """
        if self.cv_image is None or self.depth_image is None:
            return

        img = self.cv_image.copy()
        depth = self.depth_image.copy()
        # ---------------- LATCH REFRESH LOGIC ----------------
        now = self.get_clock().now()
        elapsed = (now - self.last_latch_time).nanoseconds * 1e-9

        if elapsed > self.LATCH_REFRESH_SEC:
            self.get_logger().info(
                f"[LATCH RESET] Refreshing all TFs after {elapsed:.1f}s"
            )
            self.latched_tf.clear()
            self.last_latch_time = now

        # -------- ArUco detection (fertilizer) -------- #
        center_aruco_list, distance_aruco_list, angle_aruco_list, _, \
            ids_aruco, rvecs_aruco_list, tvecs_aruco_list = detect_aruco_markers(
                img, self.cam_mat, self.dist_mat, marker_size=0.13
            )

        if ids_aruco:
            for marker_id, center, est_dist, angle, tvec_cam in zip(
                ids_aruco,
                center_aruco_list,
                distance_aruco_list,
                angle_aruco_list,
                tvecs_aruco_list,
            ):
                try:
                    cX, cY = int(center[0]), int(center[1])

                    distance_depth = self.depth_calculator(depth, cX, cY, window_size=5)
                    distance = distance_depth
                    if distance == 0 or np.isnan(distance) or distance > 5.0:
                        distance = float(est_dist) if est_dist is not None else 0.0
                    if distance == 0 or np.isnan(distance) or distance > 5.0:
                        self.get_logger().warn(f"Skipping ArUco {marker_id}: no valid depth")
                        continue

                    try:
                        if tvec_cam is not None and np.linalg.norm(tvec_cam) > 0.001:
                            x_cam = float(tvec_cam[0])
                            y_cam = float(tvec_cam[1])
                            z_cam = float(tvec_cam[2])
                        else:
                            x_cam, y_cam, z_cam = self.convert_pixel_to_3d(cX, cY, distance)
                    except Exception:
                        x_cam, y_cam, z_cam = self.convert_pixel_to_3d(cX, cY, distance)

                    try:
                        quat = R.from_euler('z', float(angle), degrees=False).as_quat()
                    except Exception:
                        quat = None
                        

                    if int(marker_id) in self.ARUCO_OBJECT_MAP:
                        object_type = self.ARUCO_OBJECT_MAP[int(marker_id)]
                        child_frame = f"{self.team_id}_{object_type}"
                        label, color = object_type, (0, 255, 255)
                    else:
                        child_frame = f"obj_{int(marker_id)}"
                        label, color = f"ID:{int(marker_id)}", (255, 0, 255)

                    # -------- LATCHED ARUCO TF --------
                    if child_frame not in self.latched_tf:
                        point_cam = PointStamped()
                        point_cam.header.frame_id = "camera_color_optical_frame"
                        point_cam.header.stamp = self.get_clock().now().to_msg()
                        point_cam.point.x = x_cam
                        point_cam.point.y = y_cam
                        point_cam.point.z = z_cam

                        try:
                            point_base = self.tf_buffer.transform(
                                point_cam,
                                "base_link",
                                timeout=Duration(seconds=0.5)
                            )

                            pos = (
                                float(point_base.point.x),
                                float(point_base.point.y),
                                float(point_base.point.z),
                            )

                            # Initialize buffer
                            if child_frame not in self.position_buffer:
                                self.position_buffer[child_frame] = []

                            # Collect samples
                            self.position_buffer[child_frame].append(pos)

                            # Once enough samples → average & latch
                            if len(self.position_buffer[child_frame]) >= self.AVG_SAMPLE_COUNT:
                                x_avg, y_avg, z_avg = self.compute_average_position(
                                    self.position_buffer[child_frame]
                                )

                                self.latched_tf[child_frame] = (
                                    x_avg,
                                    y_avg,
                                    z_avg,
                                    quat
                                )

                                self.position_buffer.pop(child_frame)

                                self.get_logger().info(
                                    f"[AVG LATCHED] {child_frame} @ ({x_avg:.3f}, {y_avg:.3f}, {z_avg:.3f})"
                                )


                            self.get_logger().info(f"[LATCHED] {child_frame}")

                        except Exception as e:
                            self.get_logger().warn(f"ArUco latch failed for {child_frame}: {e}")
                            continue

                    x_b, y_b, z_b, quat_b = self.latched_tf[child_frame]

                    self.publish_tf(
                        "base_link",
                        child_frame,
                        x_b,
                        y_b,
                        z_b,
                        quat=quat_b
                    )

                    # Always draw (no ok check)
                    cv2.circle(img, (cX, cY), 5, color, -1)
                    cv2.putText(
                        img, label, (cX + 10, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

                except Exception as e:
                    self.get_logger().warn(f"Aruco handling error: {e}")
                    continue

        # -------- Bad fruit detection -------- #
        bad_fruits_list = self.detect_bad_fruits(img)

        # Renumber bad fruits each frame: 3578_bad_fruit_1, 3578_bad_fruit_2, ...
        for idx, fruit in enumerate(bad_fruits_list, start=1):
            try:
                (cX, cY) = fruit['center']
                (x, y, w, h) = fruit['box']
                seq_id = idx
                fruit_frame_name = f"{self.team_id}_bad_fruit_{seq_id}"

                if not (0 <= int(cY) < depth.shape[0] and 0 <= int(cX) < depth.shape[1]):
                    self.get_logger().warn(
                        f"Fruit {seq_id} pixel out of depth bounds, skipping"
                    )
                    continue

                distance = self.depth_calculator(depth, int(cX), int(cY), window_size=5)
                if distance == 0 or np.isnan(distance) or distance > 5.0:
                    self.get_logger().warn(f"Fruit {seq_id} has invalid depth, skipping")
                    continue

                x_cam, y_cam, z_cam = self.convert_pixel_to_3d(cX, cY, distance)

                # -------- LATCHED BAD FRUIT TF --------
                if fruit_frame_name not in self.latched_tf:
                    point_cam = PointStamped()
                    point_cam.header.frame_id = "camera_color_optical_frame"
                    point_cam.header.stamp = self.get_clock().now().to_msg()
                    point_cam.point.x = x_cam
                    point_cam.point.y = y_cam
                    point_cam.point.z = z_cam

                    try:
                        point_base = self.tf_buffer.transform(
                            point_cam,
                            "base_link",
                            timeout=Duration(seconds=0.5)
                        )

                        pos = (
                            float(point_base.point.x),
                            float(point_base.point.y),
                            float(point_base.point.z),
                        )

                        if fruit_frame_name not in self.position_buffer:
                            self.position_buffer[fruit_frame_name] = []

                        self.position_buffer[fruit_frame_name].append(pos)

                        if len(self.position_buffer[fruit_frame_name]) >= self.AVG_SAMPLE_COUNT:
                            x_avg, y_avg, z_avg = self.compute_average_position(
                                self.position_buffer[fruit_frame_name]
                            )

                            self.latched_tf[fruit_frame_name] = (
                                x_avg,
                                y_avg,
                                z_avg,
                                [0.0, 0.0, 0.0, 1.0]
                            )

                            self.position_buffer.pop(fruit_frame_name)

                            self.get_logger().info(
                                f"[AVG LATCHED] {fruit_frame_name} @ ({x_avg:.3f}, {y_avg:.3f}, {z_avg:.3f})"
                            )


                        self.get_logger().info(f"[LATCHED] {fruit_frame_name}")

                    except Exception as e:
                        self.get_logger().warn(f"Fruit latch failed for {fruit_frame_name}: {e}")
                        continue

                x_b, y_b, z_b, quat_b = self.latched_tf[fruit_frame_name]

                self.publish_tf(
                    "base_link",
                    fruit_frame_name,
                    x_b,
                    y_b,
                    z_b,
                    quat=quat_b
                )

                # Always draw
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(img, (int(cX), int(cY)), 6, (0, 255, 0), -1)
                cv2.putText(
                    img, f"bad_{seq_id}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2
                )



            except Exception as e:
                self.get_logger().warn(f"Bad fruit handling error: {e}")
                continue

        # Show OpenCV window if enabled
        if SHOW_IMAGE:
            try:
                cv2.imshow('task4b_detection', img)
                cv2.waitKey(1)
            except Exception:
                pass

        # Publish annotated debug image for RViz / rosbag inspection
        self.publish_debug_image(img)


# --------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------- #
def main(args=None):
    """
    Initialize rclpy, create the node and spin until KeyboardInterrupt.
    """
    rclpy.init(args=args)
    node = FruitsAndArucoTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down Task 3A")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()