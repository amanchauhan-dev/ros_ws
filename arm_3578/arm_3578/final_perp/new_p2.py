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
#
# MERGE RATIONALE:
#   - Detection pipeline (stages [1]–[6], [9], [11]) taken from Code-1:
#       white balance, HSV+intensity segmentation, area+circularity+solidity
#       contour filtering, IoU-based persistent tracking, darkness+grey-spot
#       classification with confidence score, EMA temporal filtering.
#   - TF / 3D conversion (stages [7], [8], [10]) taken from Code-2:
#       simple median-depth window, straight pinhole unproject
#       (convert_pixel_to_3d) WITHOUT any surface-offset split.
#       This avoids the surf_d / z_cam inconsistency that caused
#       wrong arm reach in Code-1.
#   - ArUco 3D: tvec from PnP used directly (same in both codes).
#
# Image Processing Pipeline:
#
#   Raw Frame
#       |
#  [1]  Frame Sync & Validation        - staleness check, shape check
#       |
#  [2]  Light Normalization            - gray-world white balance
#       |
#  [3]  Hybrid Segmentation            - HSV green mask + intensity gate
#       |
#  [4]  Contour Filtering              - area + circularity + solidity
#       |
#  [5]  Fruit Tracking                 - persistent IoU-based ID assignment
#       |
#  [6]  Bad Fruit Classification       - darkness + grey-spot + confidence
#       |
#  [7]  Depth Processing               - simple median window (Code-2 style)
#       |
#  [8]  3D Conversion                  - plain pinhole unproject (no surf offset)
#       |
#  [9]  Temporal Filtering             - EMA + outlier clamp per track
#       |
#  [10] TF Publishing                  - stable TransformStamped frames
#       |
#  [11] Debug Visualization            - annotated windows + HUD + ROS topic
'''

import time
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import (
    ReentrantCallbackGroup,
    MutuallyExclusiveCallbackGroup,
)
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
from collections import deque


# ======================================================================
# Global configuration
# ======================================================================
SHOW_IMAGE             = True    # Show OpenCV window (set False for headless)
DISABLE_MULTITHREADING = False   # True -> single threaded callbacks
LOG_ALL_TF             = True    # Log all TFs published (can be noisy)
PUBLISH_CAMERA_TF      = False   # If True -> also publish object TFs in camera frame

# ── [1] Frame Validation ───────────────────────────────────────────── #
FRAME_STALE_SEC        = 0.5     # reject frame pairs older than this (seconds)

# ── [2] Light Normalization ────────────────────────────────────────── #
WB_ENABLE              = True    # Set False to bypass white balance

# ── [3] Hybrid Segmentation ───────────────────────────────────────── #
FRUIT_HSV_LO           = (33, 40, 40)
FRUIT_HSV_HI           = (92, 255, 255)
SEG_MIN_MEAN_V         = 50
SEG_OPEN_K             = (3, 3)
SEG_CLOSE_K            = (7, 7)
SEG_OPEN_ITER          = 2
SEG_CLOSE_ITER         = 3

# ── [4] Contour Filtering ─────────────────────────────────────────── #
FRUIT_MIN_AREA         = 700
FRUIT_MAX_AREA         = 4000
FRUIT_CIRC_MIN         = 0.52
FRUIT_CORNER_MIN       = 4
FRUIT_SOLIDITY_MIN     = 0.75

# ── [5] Fruit Tracking ────────────────────────────────────────────── #
TRACK_IOU_THRESH       = 0.25
TRACK_MAX_LOST         = 8
TRACK_MAX_FRUITS       = 20

# ── [6] Bad Fruit Classification ─────────────────────────────────── #
DARK_V_THRESH          = 65
BAD_DARK_RATIO         = 0.25
DARK_SCORE_WEIGHT      = 0.45
GREY_SAT_HI            = 55
GREY_VAL_LO            = 45
GREY_VAL_HI            = 205
BAD_GREY_RATIO         = 0.16
GREY_SCORE_WEIGHT      = 0.55
BAD_CONF_THRESH        = 0.30

# ── [7] Depth Processing (Code-2 style: simple median window) ──────── #
DEPTH_WINDOW_SIZE      = 5       # square window radius around fruit centre
DEPTH_MIN_VALID        = 0.08    # metres – closer readings rejected
DEPTH_MAX_VALID        = 5.00    # metres – farther readings rejected

# ── [9] Temporal Filtering ───────────────────────────────────────── #
EMA_ALPHA              = 0.35
EMA_HISTORY            = 10
CLAMP_MAX_JUMP_M       = 0.15

# ── [11] Debug ───────────────────────────────────────────────────── #
DEBUG_LOG_PERIOD_S     = 2.0

# ── ArUco ─────────────────────────────────────────────────────────── #
ARUCO_AREA_THRESHOLD   = 1500


# ======================================================================
# [1]  Frame Sync & Validation
# ======================================================================

class FrameValidator:
    """
    [1] Validates that color and depth frames are recent and correctly shaped.
    """

    def __init__(self, stale_sec=FRAME_STALE_SEC):
        self.stale_sec    = stale_sec
        self._color_stamp = 0.0
        self._depth_stamp = 0.0

    def accept_color(self, img):
        if img is None or img.ndim != 3:
            return False
        self._color_stamp = time.monotonic()
        return True

    def accept_depth(self, depth):
        if depth is None or depth.ndim != 2:
            return False
        self._depth_stamp = time.monotonic()
        return True

    def pair_is_valid(self, color_img, depth_img):
        if color_img is None:
            return False, "no color frame"
        if depth_img is None:
            return False, "no depth frame"

        now = time.monotonic()
        if (now - self._color_stamp) > self.stale_sec:
            return False, f"color stale ({now - self._color_stamp:.2f}s)"
        if (now - self._depth_stamp) > self.stale_sec:
            return False, f"depth stale ({now - self._depth_stamp:.2f}s)"

        ch, cw = color_img.shape[:2]
        dh, dw = depth_img.shape[:2]
        if ch != dh or cw != dw:
            return False, f"shape mismatch color={cw}x{ch} depth={dw}x{dh}"

        return True, "ok"


# ======================================================================
# [2]  Light Normalization  --  Gray-world White Balance
# ======================================================================

def white_balance(img):
    """
    [2] Gray-world white balance. Removes color casts without inflating
    the green channel (which would cause false positives in [3]).
    """
    if not WB_ENABLE:
        return img.copy()

    f     = img.astype(np.float32)
    avg_b = np.mean(f[:, :, 0])
    avg_g = np.mean(f[:, :, 1])
    avg_r = np.mean(f[:, :, 2])
    avg   = (avg_b + avg_g + avg_r) / 3.0

    f[:, :, 0] = np.clip(f[:, :, 0] * (avg / (avg_b + 1e-6)), 0, 255)
    f[:, :, 1] = np.clip(f[:, :, 1] * (avg / (avg_g + 1e-6)), 0, 255)
    f[:, :, 2] = np.clip(f[:, :, 2] * (avg / (avg_r + 1e-6)), 0, 255)
    return f.astype(np.uint8)


# ======================================================================
# [3]  Hybrid Segmentation  --  HSV Green Mask + Intensity Gate
# ======================================================================

def segment_green_fruits(img_wb):
    """
    [3] Two-stage green fruit segmentation:
        Stage A – HSV range mask + morphological cleanup.
        Stage B – Intensity gate: discard blobs whose mean V < SEG_MIN_MEAN_V
                  (removes shadows / floor glints that fall in green hue range).
    """
    hsv  = cv2.cvtColor(img_wb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(FRUIT_HSV_LO), np.array(FRUIT_HSV_HI))

    k_open  = np.ones(SEG_OPEN_K,  np.uint8)
    k_close = np.ones(SEG_CLOSE_K, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=SEG_OPEN_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=SEG_CLOSE_ITER)

    _, _, v_chan = cv2.split(hsv)
    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for lbl in range(1, n_labels):
        comp_px = (labels == lbl)
        if comp_px.any() and float(v_chan[comp_px].mean()) >= SEG_MIN_MEAN_V:
            clean_mask[comp_px] = 255

    return clean_mask


# ======================================================================
# [4]  Contour Filtering  --  Area + Circularity + Solidity
# ======================================================================

def filter_fruit_contours(green_mask):
    """
    [4] Extract valid fruit contours with three shape criteria:
        1. Area       FRUIT_MIN_AREA <= area <= FRUIT_MAX_AREA
        2. Circularity (4*pi*A / P^2) >= FRUIT_CIRC_MIN
        3. Solidity   area / hull_area >= FRUIT_SOLIDITY_MIN
    """
    cnts, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (FRUIT_MIN_AREA <= area <= FRUIT_MAX_AREA):
            continue

        perim = cv2.arcLength(cnt, True)
        if perim < 1e-3:
            continue

        circ   = (4.0 * np.pi * area) / (perim * perim)
        approx = cv2.approxPolyDP(cnt, 0.04 * perim, True)
        if len(approx) <= FRUIT_CORNER_MIN or circ < FRUIT_CIRC_MIN:
            continue

        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0.0
        if solidity < FRUIT_SOLIDITY_MIN:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        x, y, bw, bh    = cv2.boundingRect(cnt)
        candidates.append({
            'contour':  cnt,
            'center':   (int(cx), int(cy)),
            'radius':   float(radius),
            'box':      (x, y, bw, bh),
            'area':     float(area),
            'circ':     float(circ),
            'solidity': float(solidity),
        })

    return candidates


# ======================================================================
# [5]  Fruit Tracking  --  Persistent IoU-based ID Assignment
# ======================================================================

def _box_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1);  ix2 = min(ax2, bx2)
    iy1 = max(ay1, by1);  iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FruitTracker:
    """
    [5] Persistent fruit tracker using greedy IoU matching.
    Assigns stable integer track_id values across frames.
    """

    def __init__(self):
        self._tracks  = {}
        self._next_id = 1

    def update(self, candidates):
        det_boxes = [c['box'] for c in candidates]
        track_ids = list(self._tracks.keys())

        iou_mat = np.zeros((len(track_ids), len(det_boxes)), dtype=np.float32)
        for ti, tid in enumerate(track_ids):
            for di, dbox in enumerate(det_boxes):
                iou_mat[ti, di] = _box_iou(self._tracks[tid]['box'], dbox)

        assigned_dets = set()
        assigned_trks = set()

        while iou_mat.size > 0:
            flat_best = np.argmax(iou_mat)
            ti, di    = divmod(int(flat_best), iou_mat.shape[1])
            best_iou  = iou_mat[ti, di]
            if best_iou < TRACK_IOU_THRESH:
                break
            tid = track_ids[ti]
            self._tracks[tid]['box']  = det_boxes[di]
            self._tracks[tid]['lost'] = 0
            self._tracks[tid]['age'] += 1
            assigned_dets.add(di)
            assigned_trks.add(tid)
            iou_mat[ti, :] = -1
            iou_mat[:, di] = -1

        for tid in track_ids:
            if tid not in assigned_trks:
                self._tracks[tid]['lost'] += 1

        for tid in list(self._tracks.keys()):
            if self._tracks[tid]['lost'] > TRACK_MAX_LOST:
                del self._tracks[tid]

        for di, cand in enumerate(candidates):
            if di not in assigned_dets and len(self._tracks) < TRACK_MAX_FRUITS:
                self._tracks[self._next_id] = {'box': det_boxes[di], 'lost': 0, 'age': 1}
                cand['track_id'] = self._next_id
                self._next_id   += 1
                assigned_dets.add(di)

        for di, cand in enumerate(candidates):
            if 'track_id' not in cand:
                for ti, tid in enumerate(track_ids):
                    if tid in assigned_trks and _box_iou(
                            self._tracks[tid]['box'], det_boxes[di]) > 0:
                        cand['track_id'] = tid
                        break
                if 'track_id' not in cand:
                    cand['track_id'] = -1

        return candidates

    def active_count(self):
        return len(self._tracks)


# ======================================================================
# [6]  Bad Fruit Classification  --  Darkness + Grey-spot + Confidence
# ======================================================================

def classify_fruit(img_wb, candidate):
    """
    [6] Classify a fruit as good/bad using two weighted criteria:
        A – Darkness  (full bounding box, HSV-V < DARK_V_THRESH)
        B – Grey Spots (lower 60% of enclosing circle)
    Combined confidence clamped to [0, 1].
    Bad = conf >= BAD_CONF_THRESH AND at least one criterion fires.
    """
    H, W   = img_wb.shape[:2]
    cx, cy = candidate['center']
    r      = candidate['radius']
    x, y, bw, bh = candidate['box']

    dark_ratio = 0.0
    grey_ratio = 0.0
    reasons    = []

    # Criterion A: Darkness
    x1 = max(0, x);      x2 = min(W, x + bw)
    y1 = max(0, y);      y2 = min(H, y + bh)
    if x2 > x1 and y2 > y1:
        roi_d = img_wb[y1:y2, x1:x2]
        if roi_d.size > 0:
            hsv_d      = cv2.cvtColor(roi_d, cv2.COLOR_BGR2HSV)
            _, _, v    = cv2.split(hsv_d)
            dark_ratio = float(np.count_nonzero(v < DARK_V_THRESH)) / v.size
            if dark_ratio > BAD_DARK_RATIO:
                reasons.append(f'dark:{dark_ratio:.2f}')

    # Criterion B: Grey Spots (lower 60% of enclosing circle)
    x1g = max(0, int(cx - r));         x2g = min(W, int(cx + r))
    y1g = max(0, int(cy + 0.15 * r)); y2g = min(H, int(cy + 1.6 * r))
    if x2g > x1g and y2g > y1g:
        roi_g = img_wb[y1g:y2g, x1g:x2g]
        if roi_g.size > 0:
            hsv_g  = cv2.cvtColor(roi_g, cv2.COLOR_BGR2HSV)
            g_mask = cv2.inRange(
                hsv_g,
                np.array([0,   0,          GREY_VAL_LO]),
                np.array([180, GREY_SAT_HI, GREY_VAL_HI]),
            )
            grey_ratio = float(cv2.countNonZero(g_mask)) / (
                roi_g.shape[0] * roi_g.shape[1]
            )
            if grey_ratio > BAD_GREY_RATIO:
                reasons.append(f'grey:{grey_ratio:.2f}')

    conf = min(1.0,
               (dark_ratio / (BAD_DARK_RATIO + 1e-6)) * DARK_SCORE_WEIGHT
               + (grey_ratio / (BAD_GREY_RATIO + 1e-6)) * GREY_SCORE_WEIGHT)

    is_bad = (conf >= BAD_CONF_THRESH) and (len(reasons) > 0)
    return is_bad, conf, {'dark': dark_ratio, 'grey': grey_ratio}, reasons


# ======================================================================
# [9]  Temporal Filtering  --  EMA + Outlier Clamp
# ======================================================================

class TemporalFilter:
    """
    [9] Per-track EMA with outlier clamping.
    Prevents single noisy depth spikes from shifting the TF.
    """

    def __init__(self):
        self._state = {}

    def update(self, track_id, x, y, z):
        raw = np.array([x, y, z], dtype=np.float64)

        if track_id not in self._state:
            self._state[track_id] = {
                'pos': raw.copy(),
                'raw': deque([raw.copy()], maxlen=EMA_HISTORY),
            }
        else:
            st   = self._state[track_id]
            diff = np.linalg.norm(raw - st['pos'])

            if diff > CLAMP_MAX_JUMP_M:
                direction = (raw - st['pos']) / (diff + 1e-9)
                raw = st['pos'] + direction * CLAMP_MAX_JUMP_M

            st['pos'] = EMA_ALPHA * raw + (1.0 - EMA_ALPHA) * st['pos']
            st['raw'].append(raw.copy())

        pos = self._state[track_id]['pos']
        return float(pos[0]), float(pos[1]), float(pos[2])

    def reset(self, track_id=None):
        if track_id is None:
            self._state.clear()
        elif track_id in self._state:
            del self._state[track_id]

    def active_ids(self):
        return list(self._state.keys())


# ======================================================================
# Utility  --  ArUco detection
# ======================================================================

def calculate_rectangle_area(coordinates):
    if coordinates is None or len(coordinates) != 4:
        return 0.0, 0.0
    width  = np.linalg.norm(coordinates[0] - coordinates[1])
    height = np.linalg.norm(coordinates[1] - coordinates[2])
    return width * height, width


def detect_aruco_markers(image, cam_mat, dist_mat, marker_size=0.13):
    """
    Detect ArUco markers and estimate pose using PnP (tvec).
    Runs on the full white-balanced frame.
    """
    center_list   = []
    distance_list = []
    angle_list    = []
    width_list    = []
    ids_list      = []
    rvecs_list    = []
    tvecs_list    = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_params = aruco.DetectorParameters()
    detector     = aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return (center_list, distance_list, angle_list, width_list,
                ids_list, rvecs_list, tvecs_list)

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

    return (center_list, distance_list, angle_list, width_list,
            ids_list, rvecs_list, tvecs_list)


# ======================================================================
# Main Node
# ======================================================================

class FruitsAndArucoTF(Node):

    def __init__(self):
        super().__init__('fruits_aruco_tf_publisher')

        self.bridge      = CvBridge()
        self.cv_image    = None
        self.depth_image = None

        self.team_id = "3578"

        self.ARUCO_OBJECT_MAP = {
            3: 'fertilizer_1',
            6: 'ebot_marker',
        }

        # Pipeline stage objects (Code-1 detection)
        self.validator   = FrameValidator()
        self.tracker     = FruitTracker()
        self.temp_filter = TemporalFilter()

        # Monitoring state
        self._last_log_t  = time.monotonic()
        self._frame_count = 0
        self._last_fps_t  = time.monotonic()
        self._fps         = 0.0

        # TF infrastructure
        self.tf_buffer      = tf2_ros.Buffer()
        self.tf_listener    = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.colorimagecb,
            10,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depthimagecb,
            10,
            callback_group=self.cb_group,
        )

        self.debug_image_pub = self.create_publisher(Image, '/task3a/debug_image', 10)

        self.camera_frame = None

        # Default intrinsics
        self.sizeCamX   = 1280
        self.sizeCamY   = 720
        self.centerCamX = 640.0
        self.centerCamY = 360.0
        self.focalX     = 915.3
        self.focalY     = 914.0

        self.cam_mat = np.array([
            [self.focalX, 0.0,         self.centerCamX],
            [0.0,         self.focalY, self.centerCamY],
            [0.0,         0.0,         1.0            ],
        ], dtype=np.float32)
        self.dist_mat = np.zeros((5, 1), dtype=np.float32)

        self.R_optical_correction = np.eye(3, dtype=np.float32)

        self.create_timer(0.1, self.process_image, callback_group=self.cb_group)

        if SHOW_IMAGE:
            cv2.namedWindow('task4b_detection', cv2.WINDOW_NORMAL)
            cv2.namedWindow('green_mask',       cv2.WINDOW_NORMAL)

        self.get_logger().info("Perception Node Started")

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def depthimagecb(self, data):
        """
        Convert ROS depth image to float32 meters and validate with [1].
        Handles both uint16 (mm) and float32 (m) encodings.
        """
        try:
            arr = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            if arr.dtype == np.uint16:
                arr = arr.astype(np.float32) / 1000.0
            else:
                arr = arr.astype(np.float32)
            if self.validator.accept_depth(arr):
                self.depth_image = arr
        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")

    def colorimagecb(self, data):
        """
        Convert ROS color image to BGR OpenCV image and validate with [1].
        """
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            if self.validator.accept_color(img):
                self.cv_image = img
                if self.camera_frame is None:
                    self.camera_frame = (data.header.frame_id
                                         or 'camera_color_optical_frame')
        except Exception as e:
            self.get_logger().error(f"Color conversion error: {e}")

    # ------------------------------------------------------------------ #
    # [7]  Depth helper  (Code-2 style: simple median window)
    # ------------------------------------------------------------------ #
    def depth_calculator(self, depth_img, x, y, window_size=DEPTH_WINDOW_SIZE):
        """
        [7] Return median depth in a square window around pixel (x, y).

        This is the simpler Code-2 approach: no multi-ring logic, no
        outlier z-score rejection.  It produces correct depth values that
        feed directly into the pinhole unproject in [8] without any
        surface-offset correction, giving accurate arm-reach positions.

        Args:
            depth_img   (np.ndarray): float32 depth map in metres.
            x, y        (int):        pixel coordinates.
            window_size (int):        side length of square sample window.

        Returns:
            float: median depth in metres, or 0.0 if no valid readings.
        """
        if depth_img is None:
            return 0.0

        h, w = depth_img.shape
        x1 = max(0, x - window_size // 2)
        x2 = min(w, x + window_size // 2 + 1)
        y1 = max(0, y - window_size // 2)
        y2 = min(h, y + window_size // 2 + 1)

        window       = depth_img[y1:y2, x1:x2]
        valid_depths = window[(window > DEPTH_MIN_VALID) & (window < DEPTH_MAX_VALID)]

        if len(valid_depths) == 0:
            return 0.0

        median_depth = float(np.median(valid_depths))
        if median_depth > 100.0:
            median_depth /= 1000.0
        return median_depth

    # ------------------------------------------------------------------ #
    # [8]  3D Conversion  (Code-2 style: plain pinhole, no surf offset)
    # ------------------------------------------------------------------ #
    def convert_pixel_to_3d(self, cX, cY, distance):
        """
        [8] Back-project pixel (cX, cY) + depth to 3D camera optical frame.

        Plain pinhole model – no surface-offset correction applied.
        Using the raw median depth for ALL three axes (X, Y, Z) gives
        consistent, accurate TF positions that the robot arm can reach.

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth

        Args:
            cX, cY   (float): pixel coordinates.
            distance (float): depth in metres.

        Returns:
            (x, y, z) in camera optical frame (metres).
        """
        x_opt = (cX - self.centerCamX) * distance / self.focalX
        y_opt = (cY - self.centerCamY) * distance / self.focalY
        z_opt = distance

        p_cam = self.R_optical_correction @ np.array([[x_opt], [y_opt], [z_opt]])
        return tuple(map(float, p_cam.flatten()))

    # ------------------------------------------------------------------ #
    # TF helpers
    # ------------------------------------------------------------------ #
    def publish_tf(self, frame_id, child_frame_id, x, y, z, quat=None, stamp=None):
        """
        Publish a TransformStamped via tf2 broadcaster.
        """
        if quat is None:
            quat = [0.0, 0.0, 0.0, 1.0]
        quat = [float(q) for q in quat]

        tf_msg = TransformStamped()
        tf_msg.header.stamp             = stamp if stamp is not None else self.get_clock().now().to_msg()
        tf_msg.header.frame_id          = frame_id
        tf_msg.child_frame_id           = child_frame_id
        tf_msg.transform.translation.x  = float(x)
        tf_msg.transform.translation.y  = float(y)
        tf_msg.transform.translation.z  = float(z)
        tf_msg.transform.rotation.x     = quat[0]
        tf_msg.transform.rotation.y     = quat[1]
        tf_msg.transform.rotation.z     = quat[2]
        tf_msg.transform.rotation.w     = quat[3]

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
        publish_camera_tf=None,
        target_frame='base_link',
        timeout_sec=0.5,
    ):
        """
        Transform camera-frame coordinates to target_frame and publish TF.
        Optionally also publishes the camera-frame TF (suffix _cam).
        """
        if quat is None:
            quat = [0.0, 0.0, 0.0, 1.0]
        if publish_camera_tf is None:
            publish_camera_tf = PUBLISH_CAMERA_TF

        try:
            x_cam = float(x_cam)
            y_cam = float(y_cam)
            z_cam = float(z_cam)
            quat  = [float(q) for q in quat]
        except Exception as e:
            self.get_logger().error(f"transform_and_publish: invalid input: {e}")
            return False

        src = self.camera_frame if self.camera_frame is not None else 'camera_color_optical_frame'
        now = self.get_clock().now().to_msg()

        if publish_camera_tf:
            cam_child = f"{child_frame_id}_cam"
            ok_cam = self.publish_tf(src, cam_child, x_cam, y_cam, z_cam,
                                     quat=quat, stamp=now)
            if not ok_cam:
                self.get_logger().warn(f"Failed to publish camera TF {cam_child}")

        point_in_camera = PointStamped()
        point_in_camera.header.frame_id = src
        point_in_camera.header.stamp    = now
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
                x_b, y_b, z_b,
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
        Publish annotated BGR image on /task3a/debug_image.
        """
        try:
            msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = self.camera_frame if self.camera_frame else "camera_color_optical_frame"
            self.debug_image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def process_image(self):
        """
        Main processing loop (10 Hz).

        Executes the merged pipeline:
            [1]  Frame sync & validation          (Code-1)
            [2]  White balance                    (Code-1)
            [3]  Hybrid HSV + intensity segmentation (Code-1)
            [4]  Contour filtering                (Code-1)
            [5]  Persistent IoU tracking          (Code-1)
            [6]  Darkness + grey-spot confidence  (Code-1)
            [7]  Simple median depth window       (Code-2)
            [8]  Plain pinhole unproject, no surf offset (Code-2)
            [9]  EMA temporal filtering           (Code-1)
            [10] TF publishing                    (both)
            [11] Debug visualization + HUD        (Code-1)
        """

        # ============================================================== #
        # [1]  Frame Sync & Validation
        # ============================================================== #
        valid, reason = self.validator.pair_is_valid(self.cv_image, self.depth_image)
        if not valid:
            self.get_logger().debug(f"[1] Frame invalid: {reason}")
            return

        img   = self.cv_image.copy()
        depth = self.depth_image.copy()

        # FPS tracking
        self._frame_count += 1
        now_t = time.monotonic()
        dt    = now_t - self._last_fps_t
        if dt >= 1.0:
            self._fps         = self._frame_count / dt
            self._frame_count = 0
            self._last_fps_t  = now_t

        # ============================================================== #
        # [2]  Light Normalization  --  White Balance
        # ============================================================== #
        img_wb = white_balance(img)
        canvas = img_wb.copy()

        # ============================================================== #
        # ArUco detection on full white-balanced frame
        # [7] depth: simple window  [8] 3D: tvec from PnP (most accurate)
        # [9] temporal filter  [10] TF publish
        # ============================================================== #
        center_aruco_list, distance_aruco_list, angle_aruco_list, _, \
            ids_aruco, rvecs_aruco_list, tvecs_aruco_list = detect_aruco_markers(
                canvas, self.cam_mat, self.dist_mat, marker_size=0.13
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

                    # [7] Fallback depth (PnP tvec is primary)
                    distance_depth = self.depth_calculator(depth, cX, cY)
                    distance = distance_depth
                    if distance == 0 or np.isnan(distance) or distance > 5.0:
                        distance = float(est_dist) if est_dist is not None else 0.0
                    if distance == 0 or np.isnan(distance) or distance > 5.0:
                        self.get_logger().warn(f"Skipping ArUco {marker_id}: no valid depth")
                        continue

                    # [8] 3D: prefer PnP tvec; fall back to pixel unproject
                    try:
                        if tvec_cam is not None and np.linalg.norm(tvec_cam) > 0.001:
                            x_cam = float(tvec_cam[0])
                            y_cam = float(tvec_cam[1])
                            z_cam = float(tvec_cam[2])
                        else:
                            x_cam, y_cam, z_cam = self.convert_pixel_to_3d(cX, cY, distance)
                    except Exception:
                        x_cam, y_cam, z_cam = self.convert_pixel_to_3d(cX, cY, distance)

                    # [9] Temporal filtering
                    aruco_frame = (
                        f"{self.team_id}_{self.ARUCO_OBJECT_MAP[int(marker_id)]}"
                        if int(marker_id) in self.ARUCO_OBJECT_MAP
                        else f"obj_{int(marker_id)}"
                    )
                    x_cam, y_cam, z_cam = self.temp_filter.update(
                        aruco_frame, x_cam, y_cam, z_cam
                    )

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

                    # [10] TF publish
                    ok = self.transform_and_publish(x_cam, y_cam, z_cam, child_frame, quat=quat)
                    if ok:
                        cv2.circle(canvas, (cX, cY), 5, color, -1)
                        cv2.putText(canvas, label, (cX + 10, cY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception as e:
                    self.get_logger().warn(f"Aruco handling error: {e}")
                    continue

        # ============================================================== #
        # [3]  Hybrid Segmentation
        # ============================================================== #
        green_mask = segment_green_fruits(img_wb)

        if SHOW_IMAGE:
            try:
                cv2.imshow('green_mask', green_mask)
                cv2.waitKey(1)
            except Exception:
                pass

        # ============================================================== #
        # [4]  Contour Filtering
        # ============================================================== #
        candidates = filter_fruit_contours(green_mask)

        # ============================================================== #
        # [5]  Fruit Tracking
        # ============================================================== #
        candidates = self.tracker.update(candidates)

        n_bad = 0

        for cand in candidates:
            try:
                cx, cy       = cand['center']
                x, y, bw, bh = cand['box']
                tid          = cand.get('track_id', -1)

                # ====================================================== #
                # [6]  Bad Fruit Classification
                # ====================================================== #
                is_bad, conf, score, reasons = classify_fruit(img_wb, cand)

                # Good fruits are silently skipped — no annotation drawn.
                if not is_bad:
                    continue

                # ====================================================== #
                # [7]  Depth  --  Simple median window (Code-2 approach)
                # ====================================================== #
                if not (0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]):
                    self.get_logger().warn(f"Fruit T{tid}: pixel out of depth bounds")
                    continue

                distance = self.depth_calculator(depth, cx, cy)
                if distance == 0 or np.isnan(distance) or distance > DEPTH_MAX_VALID:
                    self.get_logger().warn(f"Fruit T{tid}: invalid depth, skipping")
                    continue

                # ====================================================== #
                # [8]  3D Conversion  --  Plain pinhole (Code-2 approach)
                # ====================================================== #
                x_cam, y_cam, z_cam = self.convert_pixel_to_3d(cx, cy, distance)

                # ====================================================== #
                # [9]  Temporal Filtering  --  EMA + Outlier Clamp
                # ====================================================== #
                fruit_frame_name = f"{self.team_id}_bad_fruit_{tid}"
                x_cam, y_cam, z_cam = self.temp_filter.update(
                    fruit_frame_name, x_cam, y_cam, z_cam
                )

                self.get_logger().info(
                    f"BadFruit T{tid}: conf={conf:.2f} "
                    f"dark={score['dark']:.2f} grey={score['grey']:.2f} "
                    f"depth={distance:.3f}m "
                    f"3D=({x_cam:.3f},{y_cam:.3f},{z_cam:.3f})"
                )

                # ====================================================== #
                # [10]  TF Publishing
                # ====================================================== #
                ok = self.transform_and_publish(
                    x_cam, y_cam, z_cam, fruit_frame_name, quat=None
                )

                # ====================================================== #
                # [11]  Debug Annotation  (bad fruits only)
                # ====================================================== #
                if ok:
                    reason_str = '+'.join(reasons) if reasons else 'bad'
                    label      = f"BAD T{tid} [{reason_str}] c={conf:.2f}"
                    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                    cv2.circle(canvas, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(canvas, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 2)
                    cv2.putText(
                        canvas,
                        f"({x_cam:.2f},{y_cam:.2f},{z_cam:.2f})m",
                        (x, y + bh + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 200, 60), 1,
                    )
                    n_bad += 1

            except Exception as e:
                self.get_logger().warn(f"Bad fruit handling error: {e}")
                continue

        # ============================================================== #
        # [11]  HUD + windows + ROS debug image
        # ============================================================== #
        hud_lines = [
            f"FPS: {self._fps:.1f}",
            f"Candidates: {len(candidates)}",
            f"Tracks: {self.tracker.active_count()}",
            f"Bad TFs: {n_bad}",
        ]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (185, 8 + len(hud_lines) * 18), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
        for i, line in enumerate(hud_lines):
            cv2.putText(canvas, line, (5, 14 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1, cv2.LINE_AA)

        now_t = time.monotonic()
        if (now_t - self._last_log_t) >= DEBUG_LOG_PERIOD_S:
            self.get_logger().info(
                f"Pipeline: fps={self._fps:.1f} "
                f"candidates={len(candidates)} "
                f"tracks={self.tracker.active_count()} bad={n_bad} "
                f"ema_ids={self.temp_filter.active_ids()}"
            )
            self._last_log_t = now_t

        if SHOW_IMAGE:
            try:
                cv2.imshow('task4b_detection', canvas)
                cv2.waitKey(1)
            except Exception:
                pass

        self.publish_debug_image(canvas)


# ======================================================================
# main()
# ======================================================================

def main(args=None):
    rclpy.init(args=args)
    node = FruitsAndArucoTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down Task 4B")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()