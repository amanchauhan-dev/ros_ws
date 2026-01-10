#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# ========================= USER INPUT =========================
BAG_PATH = "/home/aman/ros_ws/src/1_not_pkg/aman_test/data/eBot full run teleoped"   # folder, not .db3
SCAN_TOPIC = "/scan"
ODOM_TOPIC = "/odom"
# =============================================================


def quaternion_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def main():
    # -------- Open bag --------
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    scan_type = get_message(topic_types[SCAN_TOPIC])
    odom_type = get_message(topic_types[ODOM_TOPIC])

    # -------- State --------
    current_x = None
    current_y = None
    current_yaw = None

    global_x = []
    global_y = []

    print("[INFO] Reading bag...")

    # -------- Read messages --------
    while reader.has_next():
        topic, data, _ = reader.read_next()

        if topic == ODOM_TOPIC:
            msg = deserialize_message(data, odom_type)
            current_x = msg.pose.pose.position.x
            current_y = msg.pose.pose.position.y
            current_yaw = quaternion_to_yaw(msg.pose.pose.orientation)

        elif topic == SCAN_TOPIC:
            if current_x is None:
                continue

            msg = deserialize_message(data, scan_type)

            ranges = np.array(msg.ranges)
            angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

            valid = np.isfinite(ranges)
            ranges = ranges[valid]
            angles = angles[valid]

            x_local = ranges * np.cos(angles)
            y_local = ranges * np.sin(angles)

            cy = math.cos(current_yaw)
            sy = math.sin(current_yaw)

            xg = current_x + (x_local * cy - y_local * sy)
            yg = current_y + (x_local * sy + y_local * cy)

            global_x.extend(xg.tolist())
            global_y.extend(yg.tolist())

    print(f"[INFO] Total points: {len(global_x)}")

    # -------- Plot --------
    fig, ax = plt.subplots(figsize=(9, 9))
    sc = ax.scatter(
    global_x[1000000:1300000:50],
    global_y[1000000:1300000:50],
    s=1,
    c="red"
    )

    ax.set_title("Global LiDAR Map (Hover for Coordinates)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True)

    # -------- Hover interaction --------
    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        sel.annotation.set_text(f"x={x:.3f}\ny={y:.3f}")

    plt.show()


if __name__ == "__main__":
    main()
