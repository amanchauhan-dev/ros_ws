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
# Image Processing Pipeline:
#
#   Raw Frame
#       |
#  [1]  Frame Sync & Validation        - staleness check, shape check
#       |
#  [2]  Light Normalization            - gray-world white balance ONLY
#       |
#  [3]  Hybrid Segmentation            - HSV green mask + intensity gate
#       |
#  [4]  Contour Filtering              - area + circularity + solidity
#       |
#  [5]  Fruit Tracking                 - persistent IoU-based ID assignment
#       |
#  [6]  Bad Fruit Classification       - darkness + grey-spot + confidence
#       |
#  [7]  Depth Processing               - multi-ring median + outlier reject
#       |
#  [8]  3D Conversion                  - pinhole unproject + surface offset
#       |
#  [9]  Temporal Filtering             - EMA + outlier clamp per track
#       |
#  [10] TF Publishing                  - stable TransformStamped frames
#                                        (per-frame sequential IDs: bad_fruit_1, _2 ...)
#       |
#  [11] Debug Visualization            - annotated windows + HUD + ROS topic
#
# Notes:
#   - Uses OpenCV (cv2), cv_bridge, rclpy and tf2_ros.
#   - Publishes TFs for detected objects (camera and base frames).
#   - Publishes debug image on /task3a/debug_image (bgr8).
#   - Toggle SHOW_IMAGE to disable OpenCV windows if running headless.
#   - ArUco detection runs on the full white-balanced frame.
#   - NO ROI / tray-mask logic - full image is processed at every stage.
#   - Good fruits are NOT labeled or annotated (only bad fruits are shown).
#   - Bad fruit TF child frames are renumbered 1..N each frame
#     (3578_bad_fruit_1, 3578_bad_fruit_2, ...) matching task spec.
#   - Tracker IDs are used internally ONLY for EMA temporal filtering;
#     they are never exposed in the published TF child frame names.
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
import tf2_geometry_msgs  # noqa: F401  (kept for tf conversions)
import numpy as np
from geometry_msgs.msg import TransformStamped, PointStamped
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation as R
from collections import deque


# ======================================================================
# Global configuration
# ======================================================================
SHOW_IMAGE             = True    # Show OpenCV window (set False for headless)
DISABLE_MULTITHREADING = False   # True  -> single threaded callbacks
LOG_ALL_TF             = True    # Log all TFs published (can be noisy)
PUBLISH_CAMERA_TF      = False   # If True -> also publish object TFs in camera frame

# ── [1] Frame Validation ───────────────────────────────────────────── #
FRAME_STALE_SEC        = 0.5     # Reject frame pairs older than this (seconds)

# ── [2] Light Normalization ────────────────────────────────────────── #
WB_ENABLE              = True    # Set False to bypass white balance

# ── [3] Hybrid Segmentation ───────────────────────────────────────── #
FRUIT_HSV_LO           = (33, 40, 40)    # HSV lower bound – green fruit
FRUIT_HSV_HI           = (92, 255, 255)  # HSV upper bound – green fruit
SEG_MIN_MEAN_V         = 50      # Blobs darker than this are shadow/floor
SEG_OPEN_K             = (3, 3)  # Morphological open kernel
SEG_CLOSE_K            = (7, 7)  # Morphological close kernel
SEG_OPEN_ITER          = 2
SEG_CLOSE_ITER         = 3

# ── [4] Contour Filtering ─────────────────────────────────────────── #
FRUIT_MIN_AREA         = 700     # px²  – smaller  -> noise
FRUIT_MAX_AREA         = 4000    # px²  – larger   -> merged blobs
FRUIT_CIRC_MIN         = 0.52    # Circularity (4π·A / P²)
FRUIT_CORNER_MIN       = 4       # approxPolyDP corners > this -> round
FRUIT_SOLIDITY_MIN     = 0.75    # area / convex-hull area

# ── [5] Fruit Tracking ────────────────────────────────────────────── #
TRACK_IOU_THRESH       = 0.25    # IoU to match detection to existing track
TRACK_MAX_LOST         = 8       # Frames without match before track deleted
TRACK_MAX_FRUITS       = 20      # Maximum simultaneous fruit tracks

# ── [6] Bad Fruit Classification ─────────────────────────────────── #
DARK_V_THRESH          = 65      # HSV-V below this = dark pixel
BAD_DARK_RATIO         = 0.25    # Dark-pixel fraction to flag bad
DARK_SCORE_WEIGHT      = 0.45    # Weight in combined confidence
GREY_SAT_HI            = 55      # Saturation upper bound for grey pixel
GREY_VAL_LO            = 45      # Value lower bound for grey (not black)
GREY_VAL_HI            = 205     # Value upper bound for grey (not specular)
BAD_GREY_RATIO         = 0.16    # Grey-spot fraction to flag bad
GREY_SCORE_WEIGHT      = 0.55    # Weight in combined confidence
BAD_CONF_THRESH        = 0.30    # Minimum combined confidence to publish TF

# ── [7] Depth Processing ──────────────────────────────────────────── #
DEPTH_WIN_INNER        = 5       # Inner ring radius (pixels)
DEPTH_WIN_OUTER        = 11      # Outer ring – used if inner is sparse
DEPTH_MIN_SAMPLES      = 4       # Minimum valid pixels required
DEPTH_MIN_VALID        = 0.08    # Metres – closer readings rejected
DEPTH_MAX_VALID        = 5.00    # Metres – farther readings rejected
DEPTH_OUTLIER_Z        = 2.0     # Z-score threshold for outlier rejection

# ── [8] 3D Conversion ────────────────────────────────────────────── #
SURFACE_OFFSET_ENABLE  = True    # Subtract estimated sphere radius from depth

# ── [9] Temporal Filtering ───────────────────────────────────────── #
EMA_ALPHA              = 0.35    # EMA weight for new measurement
EMA_HISTORY            = 10      # Raw positions kept per track
CLAMP_MAX_JUMP_M       = 0.15    # Metres – outlier clamp threshold

# ── [11] Debug ───────────────────────────────────────────────────── #
DEBUG_LOG_PERIOD_S     = 2.0     # Periodic diagnostics interval (seconds)

# ── ArUco ─────────────────────────────────────────────────────────── #
ARUCO_AREA_THRESHOLD   = 1500    # px² – reject tiny/distant detections


# ======================================================================
# [1]  Frame Sync & Validation
# ======================================================================

class FrameValidator:
    """
    [1] Validates that color and depth frames are recent and correctly shaped.

    Rejects:
      - None frames
      - Incorrect number of dimensions (color must be 3-D, depth must be 2-D)
      - Frames older than FRAME_STALE_SEC
      - Mismatched spatial dimensions between color and depth
    """

    def __init__(self, stale_sec=FRAME_STALE_SEC):
        self.stale_sec    = stale_sec
        self._color_stamp = 0.0
        self._depth_stamp = 0.0

    def accept_color(self, img):
        """
        Record arrival of a new color frame.

        Args:
            img (np.ndarray): BGR image

        Returns:
            bool: True if structurally valid
        """
        if img is None or img.ndim != 3:
            return False
        self._color_stamp = time.monotonic()
        return True

    def accept_depth(self, depth):
        """
        Record arrival of a new depth frame.

        Args:
            depth (np.ndarray): single-channel depth array

        Returns:
            bool: True if structurally valid
        """
        if depth is None or depth.ndim != 2:
            return False
        self._depth_stamp = time.monotonic()
        return True

    def pair_is_valid(self, color_img, depth_img):
        """
        Check that both frames exist, are fresh, and have matching shapes.

        Args:
            color_img (np.ndarray): BGR image
            depth_img (np.ndarray): depth image

        Returns:
            (bool, str): (is_valid, reason_string)
        """
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
# [2]  Light Normalization  --  Gray-world White Balance ONLY
# ======================================================================

def white_balance(img):
    """
    [2] Apply gray-world white balance to normalise illumination.

    Scales each BGR channel so that the per-channel mean equals the overall
    mean of the three channels.  If WB_ENABLE is False the input is returned
    unchanged (copy).

    Args:
        img (np.ndarray): BGR image (uint8)

    Returns:
        np.ndarray: white-balanced BGR image (uint8)
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
    [3] Two-stage green fruit segmentation on the full white-balanced image.

    Stage 1 – HSV range mask isolates green hues.
    Stage 2 – Connected-component brightness gate removes dark shadow blobs.

    Args:
        img_wb (np.ndarray): white-balanced BGR image

    Returns:
        np.ndarray: binary mask (uint8, 0/255) of green fruit regions
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
    [4] Extract valid fruit contours using three shape criteria:
        - area within [FRUIT_MIN_AREA, FRUIT_MAX_AREA]
        - circularity  >= FRUIT_CIRC_MIN
        - solidity     >= FRUIT_SOLIDITY_MIN

    Args:
        green_mask (np.ndarray): binary mask from segment_green_fruits()

    Returns:
        list[dict]: each dict contains:
            'contour', 'center'(cx,cy), 'radius', 'box'(x,y,w,h),
            'area', 'circ', 'solidity'
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
    """
    Compute Intersection-over-Union of two bounding boxes.

    Args:
        a, b: (x, y, w, h) tuples

    Returns:
        float: IoU in [0, 1]
    """
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

    Assigns a stable internal track_id to each fruit across frames.
    This ID is used ONLY by the TemporalFilter for position smoothing;
    it is NOT exposed in published TF child frame names (those are
    renumbered 1..N each frame by process_image).
    """

    def __init__(self):
        self._tracks  = {}   # track_id -> {'box', 'lost', 'age'}
        self._next_id = 1

    def update(self, candidates):
        """
        Match current-frame detections to existing tracks via greedy IoU.
        Increments lost counter for unmatched tracks, creates new tracks
        for unmatched detections, and prunes stale tracks.

        Args:
            candidates (list[dict]): output from filter_fruit_contours()

        Returns:
            list[dict]: same list with 'track_id' added to each entry
                        (-1 if no track could be assigned)
        """
        det_boxes = [c['box'] for c in candidates]
        track_ids = list(self._tracks.keys())

        # Build IoU matrix  [n_tracks x n_detections]
        iou_mat = np.zeros((len(track_ids), len(det_boxes)), dtype=np.float32)
        for ti, tid in enumerate(track_ids):
            for di, dbox in enumerate(det_boxes):
                iou_mat[ti, di] = _box_iou(self._tracks[tid]['box'], dbox)

        assigned_dets = set()
        assigned_trks = set()

        # Greedy matching: repeatedly take the best remaining pair
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
            iou_mat[ti, :] = -1   # suppress row
            iou_mat[:, di] = -1   # suppress col

        # Increment lost counter for unmatched tracks
        for tid in track_ids:
            if tid not in assigned_trks:
                self._tracks[tid]['lost'] += 1

        # Prune stale tracks
        for tid in list(self._tracks.keys()):
            if self._tracks[tid]['lost'] > TRACK_MAX_LOST:
                del self._tracks[tid]

        # Spawn new tracks for unmatched detections
        for di, cand in enumerate(candidates):
            if di not in assigned_dets and len(self._tracks) < TRACK_MAX_FRUITS:
                self._tracks[self._next_id] = {'box': det_boxes[di], 'lost': 0, 'age': 1}
                cand['track_id'] = self._next_id
                self._next_id   += 1
                assigned_dets.add(di)

        # Assign track_id to every candidate (matched or new)
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
        """Return the number of currently live tracks."""
        return len(self._tracks)


# ======================================================================
# [6]  Bad Fruit Classification  --  Darkness + Grey-spot + Confidence
# ======================================================================

def classify_fruit(img_wb, candidate):
    """
    [6] Classify a single fruit candidate as good or bad.

    Two weighted criteria are evaluated:
      A) Darkness  – fraction of pixels in bounding-box ROI with HSV-V < DARK_V_THRESH
      B) Grey-spot – fraction of pixels in lower-hemisphere ROI within grey HSV range

    Combined confidence:
        conf = clamp(dark_ratio/BAD_DARK_RATIO * DARK_SCORE_WEIGHT
                   + grey_ratio/BAD_GREY_RATIO * GREY_SCORE_WEIGHT, 0, 1)

    A fruit is flagged bad when conf >= BAD_CONF_THRESH AND at least
    one criterion independently exceeds its threshold.

    Args:
        img_wb (np.ndarray): white-balanced BGR image
        candidate (dict):    output entry from filter_fruit_contours()

    Returns:
        (is_bad, conf, score_dict, reasons):
            is_bad  (bool)   – True if fruit is classified bad
            conf    (float)  – combined confidence in [0, 1]
            score_dict (dict)– {'dark': float, 'grey': float}
            reasons (list)   – human-readable strings for active criteria
    """
    H, W   = img_wb.shape[:2]
    cx, cy = candidate['center']
    r      = candidate['radius']
    x, y, bw, bh = candidate['box']

    dark_ratio = 0.0
    grey_ratio = 0.0
    reasons    = []

    # ── Criterion A: Darkness ──────────────────────────────────────── #
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

    # ── Criterion B: Grey Spots ────────────────────────────────────── #
    x1g = max(0, int(cx - r));           x2g = min(W, int(cx + r))
    y1g = max(0, int(cy + 0.15 * r));   y2g = min(H, int(cy + 1.6 * r))
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
# [7]  Depth Processing  --  Multi-ring Median + Outlier Rejection
# ======================================================================

def get_robust_depth(depth_img, px, py):
    """
    [7] Return a robust depth estimate at pixel (px, py).

    Tries the inner window first; falls back to the outer window when
    too few valid samples are found.  Applies z-score outlier rejection
    before computing the median.

    Args:
        depth_img (np.ndarray): single-channel depth in metres (float32)
        px (int): pixel x coordinate
        py (int): pixel y coordinate

    Returns:
        float: robust depth estimate in metres, or 0.0 if unavailable
    """
    if depth_img is None:
        return 0.0

    H, W = depth_img.shape

    def _sample(radius):
        half = radius // 2
        x1 = max(0, px - half);  x2 = min(W, px + half + 1)
        y1 = max(0, py - half);  y2 = min(H, py + half + 1)
        patch = depth_img[y1:y2, x1:x2].copy()
        if patch.max() > 100.0:
            patch = patch / 1000.0
        return patch[(patch > DEPTH_MIN_VALID) & (patch < DEPTH_MAX_VALID)]

    samples = _sample(DEPTH_WIN_INNER)
    if samples.size < DEPTH_MIN_SAMPLES:
        samples = _sample(DEPTH_WIN_OUTER)
    if samples.size < DEPTH_MIN_SAMPLES:
        return 0.0

    std = np.std(samples)
    if std > 0:
        samples = samples[np.abs(samples - np.mean(samples)) < DEPTH_OUTLIER_Z * std]
    if samples.size == 0:
        return 0.0

    return float(np.median(samples))


# ======================================================================
# [8]  3D Conversion  --  Pinhole Unproject + Surface Offset
# ======================================================================

def pixel_to_3d(px, py, depth, fx, fy, cx_cam, cy_cam, R_corr):
    """
    [8] Back-project pixel (px, py) + depth to 3D in camera optical frame.

    Args:
        px, py   (float): pixel coordinates
        depth    (float): depth in metres
        fx, fy   (float): focal lengths (pixels)
        cx_cam   (float): principal point x
        cy_cam   (float): principal point y
        R_corr   (np.ndarray): 3x3 optional optical correction matrix

    Returns:
        (x, y, z) in camera optical frame (metres)
    """
    xo = (px - cx_cam) * depth / fx
    yo = (py - cy_cam) * depth / fy
    zo = depth
    p  = R_corr @ np.array([[xo], [yo], [zo]], dtype=np.float64)
    return float(p[0]), float(p[1]), float(p[2])


def compute_surface_depth(raw_depth, box_width_px, focal_x):
    """
    [8-helper] Subtract estimated fruit radius from centre depth.

    Converts bounding-box half-width from pixels to metres using the
    pinhole model, then subtracts it so the TF lands on the fruit
    surface rather than its centre.

    Args:
        raw_depth    (float): depth at fruit centre (metres)
        box_width_px (float): bounding-box width in pixels
        focal_x      (float): horizontal focal length (pixels)

    Returns:
        float: adjusted depth (metres), clamped to DEPTH_MIN_VALID
    """
    if not SURFACE_OFFSET_ENABLE:
        return raw_depth
    r_px = box_width_px / 2.0
    r_m  = (r_px * raw_depth) / focal_x
    return max(raw_depth - r_m, DEPTH_MIN_VALID)


# ======================================================================
# [9]  Temporal Filtering  --  EMA + Outlier Clamp
# ======================================================================

class TemporalFilter:
    """
    [9] Per-track Exponential Moving Average (EMA) with outlier clamping.

    Smooths 3D positions over time using EMA.  Large position jumps
    (> CLAMP_MAX_JUMP_M) are clamped to prevent sudden discontinuities
    from corrupting the filter state.

    Used for BOTH ArUco marker positions AND bad fruit positions.
    """

    def __init__(self):
        self._state = {}   # track_key -> {'pos': np.ndarray, 'raw': deque}

    def update(self, track_key, x, y, z):
        """
        Feed a new 3D measurement and return the smoothed position.

        Args:
            track_key (str or int): unique key identifying the track
            x, y, z  (float):      new raw 3D measurement (metres)

        Returns:
            (x_smooth, y_smooth, z_smooth): EMA-smoothed position
        """
        raw = np.array([x, y, z], dtype=np.float64)

        if track_key not in self._state:
            # First observation – initialise directly
            self._state[track_key] = {
                'pos': raw.copy(),
                'raw': deque([raw.copy()], maxlen=EMA_HISTORY),
            }
        else:
            st   = self._state[track_key]
            diff = np.linalg.norm(raw - st['pos'])

            # Clamp outlier jumps to avoid filter divergence
            if diff > CLAMP_MAX_JUMP_M:
                direction = (raw - st['pos']) / (diff + 1e-9)
                raw = st['pos'] + direction * CLAMP_MAX_JUMP_M

            st['pos'] = EMA_ALPHA * raw + (1.0 - EMA_ALPHA) * st['pos']
            st['raw'].append(raw.copy())

        pos = self._state[track_key]['pos']
        return float(pos[0]), float(pos[1]), float(pos[2])

    def reset(self, track_key=None):
        """
        Clear filter state.

        Args:
            track_key: if given, reset only that track; otherwise reset all
        """
        if track_key is None:
            self._state.clear()
        elif track_key in self._state:
            del self._state[track_key]

    def active_keys(self):
        """Return list of all active filter keys."""
        return list(self._state.keys())


# ======================================================================
# Utility  --  ArUco helpers
# ======================================================================

def calculate_rectangle_area(coordinates):
    """
    Compute area and width of a 4-corner rectangle from pixel coordinates.

    Args:
        coordinates (np.ndarray): 4x2 array of corner (x, y) coordinates

    Returns:
        (area, width): tuple(float, float)
    """
    if coordinates is None or len(coordinates) != 4:
        return 0.0, 0.0
    width  = np.linalg.norm(coordinates[0] - coordinates[1])
    height = np.linalg.norm(coordinates[1] - coordinates[2])
    return width * height, width


def detect_aruco_markers(image, cam_mat, dist_mat, marker_size=0.13):
    """
    Detect ArUco markers and estimate their 6-DoF pose.

    Markers smaller than ARUCO_AREA_THRESHOLD pixels are ignored.
    Draws detected marker outlines and frame axes onto `image` in-place.

    Args:
        image       (np.ndarray): BGR image (modified in-place for visualisation)
        cam_mat     (np.ndarray): 3x3 camera intrinsic matrix
        dist_mat    (np.ndarray): distortion coefficients
        marker_size (float):      physical marker side length (metres)

    Returns:
        center_list   (list[tuple]):      pixel centres (x, y)
        distance_list (list[float]):      Euclidean distances to markers (m)
        angle_list    (list[float]):      corrected yaw angles (rad)
        width_list    (list[float]):      marker widths in pixels
        ids_list      (list[int]):        detected marker IDs
        rvecs_list    (list[np.ndarray]): rotation vectors (one per marker)
        tvecs_list    (list[np.ndarray]): translation vectors (one per marker)
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

        # Draw coordinate axes for visualisation
        cv2.drawFrameAxes(image, cam_mat, dist_mat, rvecs[0], tvecs[0], 0.1)

        distance = float(np.linalg.norm(tvecs[0]))

        rot_mat, _ = cv2.Rodrigues(rvecs[0])
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        # Empirical angle correction (matches task spec calibration)
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
    """
    ROS2 node that:
      - Detects ArUco markers (fertilizer, ebot) and publishes their TFs
      - Detects bad fruits and publishes their TFs
      - Publishes an annotated debug image on /task3a/debug_image

    TF naming (per task spec):
      ArUco (known) : 3578_fertilizer_1,  3578_ebot_marker
      ArUco (other) : obj_<id>
      Bad fruits    : 3578_bad_fruit_1,  3578_bad_fruit_2, ...  (renumbered each frame)
    """

    def __init__(self):
        """
        Initialise node, subscriptions, TF infrastructure, publishers, and timers.
        """
        super().__init__('fruits_aruco_tf_publisher')

        # cv_bridge for ROS <-> OpenCV conversions
        self.bridge      = CvBridge()
        self.cv_image    = None
        self.depth_image = None

        # Team ID used as prefix in TF child frame names
        self.team_id = "3578"

        # Map ArUco ID -> semantic object name (per task specification)
        self.ARUCO_OBJECT_MAP = {
            3: 'fertilizer_1',
            6: 'ebot_marker',
        }

        # ── Pipeline helpers ──────────────────────────────────────── #
        self.validator   = FrameValidator()          # [1]
        self.tracker     = FruitTracker()            # [5]
        self.temp_filter = TemporalFilter()          # [9]

        # ── FPS / diagnostics state ──────────────────────────────── #
        self._last_log_t  = time.monotonic()
        self._frame_count = 0
        self._last_fps_t  = time.monotonic()
        self._fps         = 0.0

        # ── TF infrastructure ─────────────────────────────────────── #
        self.tf_buffer      = tf2_ros.Buffer()
        self.tf_listener    = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── Callback group ────────────────────────────────────────── #
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        # ── Subscriptions ─────────────────────────────────────────── #
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

        # ── Debug image publisher ─────────────────────────────────── #
        self.debug_image_pub = self.create_publisher(Image, '/task3a/debug_image', 10)

        # ── Camera intrinsics (overwritten by live CameraInfo) ──────── #
        self.camera_frame = None
        self.sizeCamX     = 1280
        self.sizeCamY     = 720
        self.centerCamX   = 640.0
        self.centerCamY   = 360.0
        self.focalX       = 915.3
        self.focalY       = 914.0

        self.cam_mat = np.array([
            [self.focalX, 0.0,         self.centerCamX],
            [0.0,         self.focalY, self.centerCamY],
            [0.0,         0.0,         1.0            ],
        ], dtype=np.float32)
        self.dist_mat = np.zeros((5, 1), dtype=np.float32)

        # Identity correction (camera is already in optical frame)
        self.R_optical_correction = np.eye(3, dtype=np.float64)

        # ── Main processing timer (10 Hz) ─────────────────────────── #
        self.create_timer(0.1, self.process_image, callback_group=self.cb_group)

        # ── Optional OpenCV windows ───────────────────────────────── #
        if SHOW_IMAGE:
            cv2.namedWindow('preception_detection', cv2.WINDOW_NORMAL)
            cv2.namedWindow('green_mask',           cv2.WINDOW_NORMAL)

        self.get_logger().info("Perception Node Started")

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def depthimagecb(self, data):
        """
        Depth image callback.

        Converts incoming ROS depth image to float32 metres (divides uint16
        millimetre data by 1000) and stores in self.depth_image after
        passing the frame validator.

        Args:
            data (sensor_msgs.msg.Image): incoming depth image message
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
        Color image callback.

        Converts incoming ROS BGR image and stores in self.cv_image after
        passing the frame validator.  Also captures the camera frame ID
        on the first valid message.

        Args:
            data (sensor_msgs.msg.Image): incoming color image message
        """
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            if self.validator.accept_color(img):
                self.cv_image = img
                if self.camera_frame is None:
                    self.camera_frame = (
                        data.header.frame_id or 'camera_color_optical_frame'
                    )
        except Exception as e:
            self.get_logger().error(f"Color conversion error: {e}")

    # ------------------------------------------------------------------ #
    # Depth + geometry helpers
    # ------------------------------------------------------------------ #

    def depth_calculator(self, depth_img, x, y, window_size=5):
        """
        Return median depth around pixel (x, y) using a square window.

        Used exclusively for ArUco marker depth estimation.
        (Bad fruits use the more robust get_robust_depth() pipeline function.)

        Args:
            depth_img   (np.ndarray): single-channel depth in metres
            x           (int):        pixel x
            y           (int):        pixel y
            window_size (int):        square window side length

        Returns:
            float: median depth (metres) or 0.0 if no valid depth
        """
        if depth_img is None:
            return 0.0

        h, w = depth_img.shape
        x1 = max(0, x - window_size // 2)
        x2 = min(w, x + window_size // 2 + 1)
        y1 = max(0, y - window_size // 2)
        y2 = min(h, y + window_size // 2 + 1)

        window       = depth_img[y1:y2, x1:x2]
        valid_depths = window[window > 0]

        if len(valid_depths) == 0:
            return 0.0

        median_depth = float(np.median(valid_depths))
        if median_depth > 100.0:    # handle accidental mm values
            median_depth /= 1000.0
        return median_depth

    def convert_pixel_to_3d(self, cX, cY, distance):
        """
        Convert pixel coordinates + depth to 3D point in camera optical frame.

        Used as fallback for ArUco markers when tvec is unavailable.

        Args:
            cX       (float): pixel x
            cY       (float): pixel y
            distance (float): depth in metres

        Returns:
            (x, y, z): 3D point in camera optical frame (metres)
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
        Publish a TransformStamped via the tf2 broadcaster.

        Args:
            frame_id       (str):      parent frame id
            child_frame_id (str):      child frame id
            x, y, z        (float):    translation in metres
            quat           (list[4]):  quaternion [x, y, z, w]
                                       (identity used when None)
            stamp:                     optional ROS2 time; uses now() when None

        Returns:
            bool: True on success, False on error
        """
        if quat is None:
            quat = [0.0, 0.0, 0.0, 1.0]
        quat = [float(q) for q in quat]

        tf_msg = TransformStamped()
        tf_msg.header.stamp            = stamp if stamp is not None else self.get_clock().now().to_msg()
        tf_msg.header.frame_id         = frame_id
        tf_msg.child_frame_id          = child_frame_id
        tf_msg.transform.translation.x = float(x)
        tf_msg.transform.translation.y = float(y)
        tf_msg.transform.translation.z = float(z)
        tf_msg.transform.rotation.x    = quat[0]
        tf_msg.transform.rotation.y    = quat[1]
        tf_msg.transform.rotation.z    = quat[2]
        tf_msg.transform.rotation.w    = quat[3]

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
        Transform a point from camera frame to target_frame, then publish TF.

        Optionally also publishes a camera-frame TF (with _cam suffix) when
        publish_camera_tf is True.

        Args:
            x_cam, y_cam, z_cam (float): coordinates in camera frame (metres)
            child_frame_id      (str):   name of the child TF frame to publish
            quat                (list):  quaternion [x,y,z,w] in camera frame
            publish_camera_tf   (bool):  override PUBLISH_CAMERA_TF global flag
                                         (None -> use global flag)
            target_frame        (str):   destination frame (default 'base_link')
            timeout_sec         (float): transform buffer lookup timeout

        Returns:
            bool: True on success, False on any error
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

        # Optional camera-frame TF (useful for debugging)
        if publish_camera_tf:
            cam_child = f"{child_frame_id}_cam"
            ok_cam = self.publish_tf(src, cam_child, x_cam, y_cam, z_cam,
                                     quat=quat, stamp=now)
            if not ok_cam:
                self.get_logger().warn(f"Failed to publish camera TF {cam_child}")

        # Build PointStamped in camera frame
        point_in_camera             = PointStamped()
        point_in_camera.header.frame_id = src
        point_in_camera.header.stamp    = now
        point_in_camera.point.x = x_cam
        point_in_camera.point.y = y_cam
        point_in_camera.point.z = z_cam

        try:
            # Check transform availability before blocking lookup
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
        Publish an annotated OpenCV BGR image as a ROS2 Image on /task3a/debug_image.

        Args:
            cv_img (np.ndarray): BGR image to publish
        """
        try:
            msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = (
                self.camera_frame if self.camera_frame else 'camera_color_optical_frame'
            )
            self.debug_image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def process_image(self):
        """
        Main processing loop running at 10 Hz.

        Pipeline stages:
          [1]  Validate frame pair
          [2]  White balance
               ArUco detection & TF publishing
          [3]  Green segmentation
          [4]  Contour filtering
          [5]  Fruit tracking (internal IDs for EMA only)
          [6]  Bad fruit classification
          [7]  Robust depth
          [8]  3D back-projection + surface offset
          [9]  Temporal (EMA) filtering
          [10] TF publishing  (child frame: 3578_bad_fruit_<seq_id>)
          [11] HUD overlay + debug image publish

        Bad fruit TF child frames are renumbered 1..N each frame,
        matching the task specification.  The tracker IDs are used
        internally only for the EMA filter.
        """

        # ── [1] Frame Sync & Validation ─────────────────────────── #
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

        # ── [2] White Balance ────────────────────────────────────── #
        img_wb = white_balance(img)
        canvas = img_wb.copy()

        # ── ArUco detection ──────────────────────────────────────── #
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

                    # Prefer depth-sensor reading; fall back to ArUco PnP distance
                    distance_depth = self.depth_calculator(depth, cX, cY, window_size=5)
                    distance = distance_depth
                    if distance == 0 or np.isnan(distance) or distance > 5.0:
                        distance = float(est_dist) if est_dist is not None else 0.0
                    if distance == 0 or np.isnan(distance) or distance > 5.0:
                        self.get_logger().warn(f"Skipping ArUco {marker_id}: no valid depth")
                        continue

                    # Prefer tvec from PnP; fall back to pixel back-projection
                    try:
                        if tvec_cam is not None and np.linalg.norm(tvec_cam) > 0.001:
                            x_cam = float(tvec_cam[0])
                            y_cam = float(tvec_cam[1])
                            z_cam = float(tvec_cam[2])
                        else:
                            x_cam, y_cam, z_cam = self.convert_pixel_to_3d(cX, cY, distance)
                    except Exception:
                        x_cam, y_cam, z_cam = self.convert_pixel_to_3d(cX, cY, distance)

                    # Build TF child frame name
                    if int(marker_id) in self.ARUCO_OBJECT_MAP:
                        object_type = self.ARUCO_OBJECT_MAP[int(marker_id)]
                        child_frame = f"{self.team_id}_{object_type}"
                        label, color = object_type, (0, 255, 255)
                    else:
                        child_frame = f"obj_{int(marker_id)}"
                        label, color = f"ID:{int(marker_id)}", (255, 0, 255)

                    # [9] Temporal filtering for ArUco positions
                    x_cam, y_cam, z_cam = self.temp_filter.update(
                        child_frame, x_cam, y_cam, z_cam
                    )

                    try:
                        quat = R.from_euler('z', float(angle), degrees=False).as_quat()
                    except Exception:
                        quat = None

                    # [10] Publish TF
                    ok = self.transform_and_publish(
                        x_cam, y_cam, z_cam, child_frame, quat=quat
                    )
                    if ok:
                        cv2.circle(canvas, (cX, cY), 5, color, -1)
                        cv2.putText(
                            canvas, label, (cX + 10, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                        )
                except Exception as e:
                    self.get_logger().warn(f"Aruco handling error: {e}")
                    continue

        # ── [3] Hybrid Segmentation ──────────────────────────────── #
        green_mask = segment_green_fruits(img_wb)

        if SHOW_IMAGE:
            try:
                cv2.imshow('green_mask', green_mask)
                cv2.waitKey(1)
            except Exception:
                pass

        # ── [4] Contour Filtering ────────────────────────────────── #
        candidates = filter_fruit_contours(green_mask)

        # ── [5] Fruit Tracking ───────────────────────────────────── #
        candidates = self.tracker.update(candidates)

        n_bad  = 0     # count of bad fruits successfully published this frame
        seq_id = 1     # per-frame sequential counter for TF child frame names

        for cand in candidates:
            try:
                cx, cy       = cand['center']
                x, y, bw, bh = cand['box']
                tid          = cand.get('track_id', -1)   # internal tracker ID

                # ── [6] Bad Fruit Classification ──────────────── #
                is_bad, conf, score, reasons = classify_fruit(img_wb, cand)

                # Good fruits are silently skipped – no annotation, no TF
                if not is_bad:
                    continue

                # ── [7] Depth Processing ──────────────────────── #
                if not (0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]):
                    self.get_logger().warn(
                        f"Fruit T{tid}: pixel ({cx},{cy}) out of depth bounds, skipping"
                    )
                    continue

                distance = get_robust_depth(depth, cx, cy)
                if distance == 0 or np.isnan(distance) or distance > DEPTH_MAX_VALID:
                    self.get_logger().warn(
                        f"Fruit T{tid}: invalid depth ({distance:.3f}m), skipping"
                    )
                    continue

                # ── [8] 3D Conversion ─────────────────────────── #
                surf_d = compute_surface_depth(distance, bw, self.focalX)
                x_cam, y_cam, _ = pixel_to_3d(
                    cx, cy, surf_d,
                    self.focalX, self.focalY,
                    self.centerCamX, self.centerCamY,
                    self.R_optical_correction,
                )
                z_cam = float(distance)

                # ── [9] Temporal Filtering ────────────────────── #
                # Key on the tracker ID so EMA is stable across frames even
                # though the published frame name changes each frame.
                ema_key = f"bad_fruit_track_{tid}"
                x_cam, y_cam, z_cam = self.temp_filter.update(
                    ema_key, x_cam, y_cam, z_cam
                )

                # ── [10] TF Publishing ────────────────────────── #
                # Published child frame uses per-frame sequential ID (task spec)
                fruit_frame_name = f"{self.team_id}_bad_fruit_{seq_id}"

                self.get_logger().info(
                    f"BadFruit seq={seq_id} T{tid}: conf={conf:.2f} "
                    f"dark={score['dark']:.2f} grey={score['grey']:.2f} "
                    f"depth={distance:.3f}m surf_d={surf_d:.3f}m "
                    f"3D=({x_cam:.3f},{y_cam:.3f},{z_cam:.3f}) "
                    f"frame='{fruit_frame_name}'"
                )

                ok = self.transform_and_publish(
                    x_cam, y_cam, z_cam, fruit_frame_name, quat=None
                )

                # ── [11] Debug Annotation (bad fruits only) ───── #
                if ok:
                    reason_str = '+'.join(reasons) if reasons else 'bad'
                    label      = f"bad_{seq_id} [{reason_str}] c={conf:.2f}"
                    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.circle(canvas, (cx, cy), 6, (0, 255, 0), -1)
                    cv2.putText(
                        canvas, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                    )
                    cv2.putText(
                        canvas,
                        f"({x_cam:.2f},{y_cam:.2f},{z_cam:.2f})m",
                        (x, y + bh + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 200, 60), 1,
                    )
                    n_bad  += 1
                    seq_id += 1     # only increment after a successful TF publish

            except Exception as e:
                self.get_logger().warn(f"Bad fruit handling error: {e}")
                continue

        # ── [11] HUD overlay ─────────────────────────────────────── #
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
            cv2.putText(
                canvas, line, (5, 14 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1, cv2.LINE_AA,
            )

        # Periodic diagnostics log
        now_t = time.monotonic()
        if (now_t - self._last_log_t) >= DEBUG_LOG_PERIOD_S:
            self.get_logger().info(
                f"Pipeline: fps={self._fps:.1f} "
                f"candidates={len(candidates)} "
                f"tracks={self.tracker.active_count()} bad={n_bad} "
                f"ema_keys={self.temp_filter.active_keys()}"
            )
            self._last_log_t = now_t

        # Show OpenCV windows if enabled
        if SHOW_IMAGE:
            try:
                cv2.imshow('preception_detection', canvas)
                cv2.waitKey(1)
            except Exception:
                pass

        # Publish annotated debug image for RViz / rosbag inspection
        self.publish_debug_image(canvas)


# ======================================================================
# main()
# ======================================================================

def main(args=None):
    """
    Initialise rclpy, spin the node, and clean up on KeyboardInterrupt.
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