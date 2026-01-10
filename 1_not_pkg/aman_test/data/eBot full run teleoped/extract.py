import csv
import math
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from tf_transformations import euler_from_quaternion

bag_path = "rosbag2_2025_12_26-17_47_12_0.db3"
csv_path = "output.csv"

reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(
    uri=bag_path,
    storage_id="sqlite3"
)
converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr"
)
reader.open(storage_options, converter_options)

topic_types = {
    t.name: t.type for t in reader.get_all_topics_and_types()
}

odom_type = get_message(topic_types["/odom"])
scan_type = get_message(topic_types["/scan"])

latest_odom = None

with open(csv_path, "w", newline="") as f:
    writer = None

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        # ---------- ODOM ----------
        if topic == "/odom":
            msg = deserialize_message(data, odom_type)

            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation

            _, _, yaw = euler_from_quaternion(
                [ori.x, ori.y, ori.z, ori.w]
            )

            latest_odom = {
                "x": pos.x,
                "y": pos.y,
                "yaw": yaw
            }

        # ---------- SCAN ----------
        elif topic == "/scan" and latest_odom is not None:
            msg = deserialize_message(data, scan_type)

            row = [
                timestamp,
                latest_odom["x"],
                latest_odom["y"],
                latest_odom["yaw"],
                *msg.ranges
            ]

            # write header once
            if writer is None:
                header = (
                    ["timestamp", "x", "y", "yaw"] +
                    [f"d{i+1}" for i in range(len(msg.ranges))]
                )
                writer = csv.writer(f)
                writer.writerow(header)

            writer.writerow(row)

print("CSV saved to:", csv_path)
