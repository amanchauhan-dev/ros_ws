#!/usr/bin/env python3

import math
import time
import rosbag2_py
import matplotlib.pyplot as plt

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from tf_transformations import euler_from_quaternion
from matplotlib.widgets import Button

# ================= USER CONFIG =================
bag_path = "/home/aman/ros_ws/src/1_not_pkg/aman_test/data/bag_file/rosbag2_2025_12_26-17_47_12_0.db3"

ODOM_TOPIC = "/odom"
SCAN_TOPIC = "/scan"

PLAYBACK_SLEEP = 0.05     # seconds between frames
PREV_STEP = 20
NEXT_STEP = 10
# ===============================================


# ---------- open bag ----------
reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
    rosbag2_py.ConverterOptions("cdr", "cdr")
)

topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
odom_type = get_message(topic_types[ODOM_TOPIC])
scan_type = get_message(topic_types[SCAN_TOPIC])


# ---------- storage per frame ----------
frames = []     # each frame = (xs, ys, lidar_x, lidar_y)
latest_odom = None

traj_x, traj_y = [], []


# ---------- read bag ----------
while reader.has_next():
    topic, data, _ = reader.read_next()

    if topic == ODOM_TOPIC:
        msg = deserialize_message(data, odom_type)

        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(
            [ori.x, ori.y, ori.z, ori.w]
        )

        latest_odom = (pos.x, pos.y, yaw)
        traj_x.append(pos.x)
        traj_y.append(pos.y)

    elif topic == SCAN_TOPIC and latest_odom is not None:
        msg = deserialize_message(data, scan_type)

        x_r, y_r, yaw = latest_odom
        gx, gy = [], []

        angle = msg.angle_min
        for r in msg.ranges:
            if math.isinf(r) or math.isnan(r):
                angle += msg.angle_increment
                continue

            # local
            x_l = r * math.cos(angle)
            y_l = r * math.sin(angle)

            # world
            x_w = x_r + (x_l * math.cos(yaw) - y_l * math.sin(yaw))
            y_w = y_r + (x_l * math.sin(yaw) + y_l * math.cos(yaw))

            gx.append(x_w)
            gy.append(y_w)

            angle += msg.angle_increment

        frames.append((traj_x.copy(), traj_y.copy(), gx, gy))


print(f"Total frames loaded: {len(frames)}")


# =================== PLOT ===================
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.25)

traj_plot, = ax.plot([], [], lw=2, label="Trajectory")
lidar_plot = ax.scatter([], [], s=2, c="red", label="LiDAR")

ax.set_aspect("equal")
ax.grid(True)
ax.legend()


frame_idx = 0
paused = False


def draw_frame(idx):
    ax.cla()
    tx, ty, lx, ly = frames[idx]

    ax.plot(tx, ty, lw=2, label="Trajectory")
    ax.scatter(lx, ly, s=2, c="red", label="LiDAR")

    ax.set_title(f"Frame {idx}/{len(frames)-1}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    plt.draw()


# ---------- buttons ----------
ax_pause = plt.axes([0.1, 0.1, 0.15, 0.08])
ax_prev = plt.axes([0.3, 0.1, 0.2, 0.08])
ax_next = plt.axes([0.55, 0.1, 0.2, 0.08])

btn_pause = Button(ax_pause, "Pause / Play")
btn_prev = Button(ax_prev, "Prev 20")
btn_next = Button(ax_next, "Next 10")


def toggle_pause(event):
    global paused
    paused = not paused


def prev_frames(event):
    global frame_idx
    frame_idx = max(0, frame_idx - PREV_STEP)
    draw_frame(frame_idx)


def next_frames(event):
    global frame_idx
    frame_idx = min(len(frames) - 1, frame_idx + NEXT_STEP)
    draw_frame(frame_idx)


btn_pause.on_clicked(toggle_pause)
btn_prev.on_clicked(prev_frames)
btn_next.on_clicked(next_frames)


# ---------- playback loop ----------
plt.ion()
draw_frame(frame_idx)

while plt.fignum_exists(fig.number):
    if not paused:
        frame_idx += 1
        if frame_idx >= len(frames):
            frame_idx = len(frames) - 1
            paused = True
        draw_frame(frame_idx)

    plt.pause(0.001)
    time.sleep(PLAYBACK_SLEEP)

plt.ioff()
plt.show()
