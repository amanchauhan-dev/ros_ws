#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
# Team ID:          3578
# Theme:            Krishi coBot
# Script:           Mock TF Publisher
# Description:      Publishes hardcoded TFs from previous detections 
#                   without running any image processing or camera nodes.
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros

class MockFruitsAndArucoTF(Node):
    def __init__(self):
        super().__init__('mock_fruits_aruco_tf_publisher')

        # Create a Transform Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Extracted static poses from the provided logs
        self.target_poses = [
            {
                'child_frame_id': '3578_ebot_marker',
                'pos': (0.831, -0.023, 0.226),
                'quat': (0.000, 0.000, 0.940, 0.341)
            },
            {
                'child_frame_id': '3578_fertilizer_1',
                'pos': (-0.284, -0.458, 0.639),
                'quat': (0.707, 0.028, 0.034, 0.707)
            },
            #                         {
            #     'child_frame_id': '3578_fertilizer_2',
            #     'pos': (-0.284, -0.458, 0.639),
            #     'quat': (0.05367629 ,0.70780447 ,0.70200457 ,0.05763047)
            # },
            {
                'child_frame_id': '3578_bad_fruit_1',
                'pos': (-0.166, 0.551, 0.016),
                'quat': (0.029,0.997,0.045,0.033)
            },
            #        {
            #     'child_frame_id': '3578_bad_fruit_1O',
            #     'pos': (-0.012, 0.551, 0.016),
            #     'quat': (0.000, 0.000, 0.000, 1.000)
            # },
            # {
            #     'child_frame_id': '3578_bad_fruit_2',
            #     'pos': (-0.012, 0.689, 0.059) ,
            #     'quat': (0.000, 0.000, 0.000, 1.000)
            # },
            #        {
            #     'child_frame_id': '3578_bad_fruit_2O',
            #     'pos': (-0.166, 0.689, 0.059) ,
            #     'quat': (0.000, 0.000, 0.000, 1.000)
            # },
            # {
            #     'child_frame_id': '3578_bad_fruit_3',
            #     'pos': (-0.013, 0.448, -0.041),
            #     'quat': (0.000, 0.000, 0.000, 1.000)
            # },
            #           {
            #     'child_frame_id': '3578_bad_fruit_3O',
            #     'pos': (-0.166, 0.448, -0.041),
            #     'quat': (0.000, 0.000, 0.000, 1.000)
            # },
            
            {
                'child_frame_id': 'Dustbin',
                'pos': (-0.706, 0.10, 0.182),
                'quat': (0.000, 0.000, 0.000, 1.000)
            },
               
            {
                'child_frame_id': 'Dustbin_Want',
                'pos': (-0.-0.682, 0.210, 0.316),
                'quat': (0.000, 0.000, 0.000, 1.000)
            }    
        ]

        # Timer to continuously publish the TFs at 10 Hz (every 0.1 seconds)
        self.timer = self.create_timer(0.1, self.publish_hardcoded_tfs)

        self.get_logger().info("=" * 60)
        self.get_logger().info("Mock TF Publisher Started (NO DETECTION)")
        self.get_logger().info("Continuously publishing hardcoded TFs for testing.")
        self.get_logger().info("=" * 60)

    def publish_hardcoded_tfs(self):
        """
        Iterates through the hardcoded poses and broadcasts them.
        """
        now = self.get_clock().now().to_msg()

        for pose_data in self.target_poses:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = 'base_link'
            t.child_frame_id = pose_data['child_frame_id']

            # Set Translation
            t.transform.translation.x = pose_data['pos'][0]
            t.transform.translation.y = pose_data['pos'][1]
            t.transform.translation.z = pose_data['pos'][2]

            # Set Rotation (Quaternion)
            t.transform.rotation.x = pose_data['quat'][0]
            t.transform.rotation.y = pose_data['quat'][1]
            t.transform.rotation.z = pose_data['quat'][2]
            t.transform.rotation.w = pose_data['quat'][3]

            # Publish the transform
            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = MockFruitsAndArucoTF()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down Mock TF publisher")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
