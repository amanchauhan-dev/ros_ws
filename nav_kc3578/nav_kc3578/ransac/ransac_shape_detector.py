import math
import time
import numpy as np
from itertools import combinations

# ---------------- LiDAR sector constants ---------------- #
FRONT_ANGLE = 0.0
LEFT_ANGLE  = math.pi / 2
RIGHT_ANGLE = -math.pi / 2
SECTOR_WIDTH = math.radians(15)
# -------------------------------------------------------- #


class RansacShapeDetector:
    """
    Stateless ROS-independent shape detector.
    Feed LiDAR global points + robot pose.
    """

    def __init__(self):
        # ---------------- parameters ---------------- #
        self.buffer_length = 3
        self.global_buffer_distance_difference = 0.2
        self.side_distance_tolerance = 1.5

        self.iterations = 200
        self.threshold = 0.01
        self.min_inliers = 5
        self.max_lines = 6

        self.side_length = 0.25
        self.len_tol = 0.08
        self.ang90_tol = 15
        self.ang60_tol = 20

        self.plants_tracks = [
            [0.26, -1.95],
            [-0.50, -4.0941],
            [-0.50, -2.7702],
            [-0.50, -1.4044],
            [-0.50, -0.0461],
            [-2.40, -4.0941],
            [-2.40, -2.7702],
            [-2.40, -1.4044],
            [-2.40, -0.0461],
        ]

        # ---------------- buffers ---------------- #
        self.right_buffer = []
        self.left_buffer = []
        self.last_pose = None

        self.right_shape_votes = [0, 0]
        self.left_shape_votes = [0, 0]
        self.right_plant_votes = [0] * 9
        self.left_plant_votes = [0] * 9

    # ======================================================
    # PUBLIC API
    # ======================================================

    def update(self, robot_pose, lidar_global_points):
        """
        robot_pose: (x, y, yaw)
        lidar_global_points: Nx2 numpy array
        """
        rx, ry, yaw = robot_pose

        if self.last_pose is None:
            self.last_pose = (rx, ry)

        dist = math.hypot(rx - self.last_pose[0], ry - self.last_pose[1])
        if dist < self.global_buffer_distance_difference:
            return None

        self.last_pose = (rx, ry)

        right_pts, left_pts = self._split_sides(
            robot_pose, lidar_global_points
        )

        self.right_buffer.append(self._filter_side(right_pts, rx, ry))
        self.left_buffer.append(self._filter_side(left_pts, rx, ry))

        self.right_buffer = self.right_buffer[-self.buffer_length:]
        self.left_buffer = self.left_buffer[-self.buffer_length:]

        result = (
            self._process_side("right", self.right_buffer, rx, ry) or
            self._process_side("left", self.left_buffer, rx, ry)
        )

        return result

    # ======================================================
    # INTERNAL PIPELINE
    # ======================================================

    def _split_sides(self, pose, pts):
        rx, ry, yaw = pose
        right, left = [], []

        for gx, gy in pts:
            dx, dy = gx - rx, gy - ry
            ang = math.atan2(dy, dx) - yaw
            ang = math.atan2(math.sin(ang), math.cos(ang))

            if abs(ang - RIGHT_ANGLE) <= SECTOR_WIDTH:
                right.append([gx, gy])
            elif abs(ang - LEFT_ANGLE) <= SECTOR_WIDTH:
                left.append([gx, gy])

        return np.array(right), np.array(left)

    def _filter_side(self, pts, rx, ry):
        if len(pts) == 0:
            return pts

        dists = np.linalg.norm(pts - np.array([rx, ry]), axis=1)
        mask = dists < self.side_distance_tolerance
        pts = pts[mask]
        dists = dists[mask]

        if len(pts) == 0:
            return pts

        med = np.median(dists)
        mad = np.median(np.abs(dists - med)) + 1e-6
        return pts[np.abs(dists - med) < 2.5 * mad]

    def _process_side(self, side, buffers, rx, ry):
        lines = self.extract_multiple_lines(buffers)
        shape = self.detect_shape(lines)

        if not shape:
            return None

        shape_type, group = shape
        cx, cy, dist = self.compute_shape_position(group, rx, ry)

        for idx, (px, py) in enumerate(self.plants_tracks):
            if abs(cx - px) <= 1.0 and abs(cy - py) <= 0.5:
                return {
                    "side": side,
                    "shape": shape_type,
                    "plant_id": idx,
                    "center": (cx, cy),
                    "distance": dist,
                }

        return None

    # ======================================================
    # RANSAC + GEOMETRY (UNCHANGED LOGIC)
    # ======================================================

    def ransac_line(self, points):
        pts = np.array(points)
        best_model, best_inliers = None, None

        if len(pts) < self.min_inliers:
            return None, None

        for _ in range(self.iterations):
            p1, p2 = pts[np.random.choice(len(pts), 2, replace=False)]
            a, b = p2[1] - p1[1], -(p2[0] - p1[0])
            c = p2[0]*p1[1] - p2[1]*p1[0]

            norm = math.hypot(a, b)
            if norm < 1e-6:
                continue

            dist = np.abs(a*pts[:,0] + b*pts[:,1] + c) / norm
            inliers = dist < self.threshold

            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best_model, best_inliers = (a, b, c), inliers

        return best_model, best_inliers

    def extract_multiple_lines(self, buffers):
        if not buffers:
            return []

        pts = np.vstack(buffers)
        lines = []

        while len(pts) >= self.min_inliers:
            model, inliers = self.ransac_line(pts)
            if model is None:
                break

            line_pts = pts[inliers]
            p1, p2 = line_pts[0], line_pts[-1]
            lines.append({
                "model": model,
                "p1": p1,
                "p2": p2,
                "length": np.linalg.norm(p2 - p1)
            })

            pts = pts[~inliers]
            if len(lines) >= self.max_lines:
                break

        return lines

    def detect_shape(self, lines):
        if len(lines) < 3:
            return None

        for group in combinations(lines, 3):
            lengths = [L["length"] for L in group]
            near = sum(abs(l - self.side_length) < self.len_tol for l in lengths)

            angles = [
                self.angle_between_lines(group[i]["model"], group[j]["model"])
                for i, j in [(0,1),(1,2),(2,0)]
            ]

            if near >= 2 and sum(abs(a - 90) < self.ang90_tol for a in angles) == 2:
                return "square", group

            if near >= 2 and sum(abs(a - 60) < self.ang60_tol for a in angles) >= 2:
                return "triangle", group

        return None

    def angle_between_lines(self, m1, m2):
        d1 = np.array([m1[1], -m1[0]])
        d2 = np.array([m2[1], -m2[0]])
        d1 /= np.linalg.norm(d1)
        d2 /= np.linalg.norm(d2)

        ang = math.degrees(math.acos(np.clip(np.dot(d1, d2), -1, 1)))
        return min(ang, 180 - ang)

    def compute_shape_position(self, group, rx, ry):
        pts = np.vstack([L["p1"] for L in group] + [L["p2"] for L in group])
        cx, cy = np.mean(pts, axis=0)
        return cx, cy, math.hypot(cx - rx, cy - ry)
