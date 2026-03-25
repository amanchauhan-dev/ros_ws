import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import math
import heapq
import threading
import numpy as np

# --- MATPLOTLIB COMMENTED OUT FOR HARDWARE COMPATIBILITY ---
# import matplotlib
# matplotlib.use('TkAgg')        # Live interactive window
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.lines import Line2D
# -----------------------------------------------------------

from scipy.ndimage import distance_transform_edt


# ============================================================
# DEBUG LOGGER
# ============================================================
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(name)s] %(message)s"
)
dbg = logging.getLogger("Navigation")


# ============================================================
# LIVE MAP VISUALIZER (PLOTTING COMMENTED OUT)
# ============================================================

class LiveMapVisualizer:
    def __init__(self, raw_map: np.ndarray, path_pixels: list,
                 start_pixel: tuple, goal_pixel: tuple):
        self._map        = raw_map.copy().astype(float)
        self._path       = path_pixels          
        self._start      = start_pixel          
        self._goal       = goal_pixel           

        self._robot_pos  = start_pixel          
        self._reached    = 0                    
        self._lock       = threading.Lock()
        self._redraw     = threading.Event()    

        self._fig        = None
        self._ax         = None
        self._im         = None
        self._robot_dot  = None
        self._path_line  = None
        self._done_line  = None

    def start(self):
        # t = threading.Thread(target=self._run, daemon=True)
        # t.start()
        # dbg.info("LiveMapVisualizer: window thread started")
        pass

    def update_robot(self, row: float, col: float):
        with self._lock:
            self._robot_pos = (row, col)
        self._redraw.set()

    def mark_waypoint_reached(self, index: int):
        with self._lock:
            self._reached = index
        self._redraw.set()

    def update_map_pixel(self, row: int, col: int, val: float = 1.0):
        with self._lock:
            self._map[row, col] = val
        self._redraw.set()

    def update_path(self, new_path_pixels: list):
        with self._lock:
            self._path = new_path_pixels
            self._reached = 0
        self._redraw.set()

    def _run(self):
        pass
        # plt.ion()
        # self._fig, self._ax = plt.subplots(figsize=(9, 9))
        # self._fig.canvas.manager.set_window_title("Navigation — Live Map")

        # display_map = np.where(self._map == 0, 0.9,   
        #               np.where(self._map  > 0, 0.1,   
        #                        0.5))                   
        # self._im = self._ax.imshow(display_map, cmap='gray', vmin=0, vmax=1, origin='upper')

        # self._path_line, = self._ax.plot([], [], color='cyan', linewidth=1.2, alpha=0.7, label='Planned path')
        # self._done_line, = self._ax.plot([], [], color='lime', linewidth=2.5, label='Completed')

        # if self._path:
        #     rows = [p[0] for p in self._path]
        #     cols = [p[1] for p in self._path]
        #     self._path_line.set_data(cols, rows)

        # self._ax.plot(self._start[1], self._start[0], 'gs', markersize=10, label='Start')
        # self._ax.plot(self._goal[1],  self._goal[0], 'r*', markersize=14, label='Goal')

        # self._robot_dot, = self._ax.plot(
        #     self._start[1], self._start[0], 'o', color='yellow', markersize=10,
        #     markeredgecolor='black', markeredgewidth=1.5, label='Robot', zorder=5
        # )

        # self._ax.legend(loc='upper right', fontsize=8,
        #                 handles=[
        #                     Line2D([0],[0], color='cyan',  lw=1.5, label='Planned path'),
        #                     Line2D([0],[0], color='lime',  lw=2.5, label='Completed'),
        #                     mpatches.Patch(color='green', label='Start'),
        #                     mpatches.Patch(color='red',   label='Goal'),
        #                     Line2D([0],[0], marker='o', color='yellow', markeredgecolor='black', lw=0, label='Robot'),
        #                 ])
        # self._ax.set_title("A* Path — Live Robot Tracker", fontsize=11)
        # self._fig.tight_layout()
        # plt.draw()
        # plt.pause(0.05)

        # while plt.fignum_exists(self._fig.number):
        #     if self._redraw.wait(timeout=0.1):
        #         self._redraw.clear()
        #         self._refresh()

    def _refresh(self):
        pass
        # with self._lock:
        #     rr, rc   = self._robot_pos
        #     reached  = self._reached
        #     current_map = self._map.copy()
        #     current_path = list(self._path)

        # # Update Map Image (Dynamic Obstacles)
        # display_map = np.where(current_map == 0, 0.9, np.where(current_map > 0, 0.1, 0.5))
        # self._im.set_data(display_map)

        # # Move robot dot
        # self._robot_dot.set_data([rc], [rr])

        # # Update Paths
        # if current_path:
        #     self._path_line.set_data([p[1] for p in current_path], [p[0] for p in current_path])
        #     if reached > 0:
        #         done = current_path[:reached + 1]
        #         self._done_line.set_data([p[1] for p in done], [p[0] for p in done])
        #     else:
        #         self._done_line.set_data([], [])

        # self._ax.set_title(f"A* Path — WP {reached}/{max(0, len(current_path)-1)}", fontsize=11)
        # self._fig.canvas.draw_idle()
        # self._fig.canvas.flush_events()


# ============================================================
# COORDINATE HELPERS
# ============================================================

def odom_to_pixel(odom_x, odom_y, resolution, origin_x, origin_y, map_height=None):
    pixel_x = int((odom_x - origin_x) / resolution)
    pixel_y = int((odom_y - origin_y) / resolution)
    if map_height is not None:
        pixel_y = map_height - pixel_y
    return pixel_x, pixel_y


def pixel_to_odom(pixel_x, pixel_y, resolution, origin_x, origin_y, map_height=None):
    if map_height is not None:
        pixel_y = map_height - pixel_y
    odom_x = pixel_x * resolution + origin_x
    odom_y = pixel_y * resolution + origin_y
    return odom_x, odom_y


# ============================================================
# A* PATH PLANNER
# ============================================================

def path_planner(start_y, start_x, goal_y, goal_x, input_map=None):
    """
    Returns (final_path, raw_map, distance_map)
    Takes an optional 'input_map' for dynamic replanning.
    """
    dbg.info("=== path_planner() CALLED ===")

    # ---------- Load map ----------
    if input_map is None:
        try:
            mini_map_2d = np.load('map.npy').copy().astype(int)
        except FileNotFoundError:
            dbg.error("map.npy NOT FOUND!")
            return [], None, None
    else:
        mini_map_2d = input_map.copy()

    raw_map = mini_map_2d.copy()  
    h, w = mini_map_2d.shape

    # ---------- Sanity checks ----------
    if not (0 <= start_x < h and 0 <= start_y < w) or not (0 <= goal_x < h and 0 <= goal_y < w):
        dbg.error("Start or Goal is OUT OF MAP BOUNDS! Aborting.")
        return [], raw_map, None

    # ---------- Distance / costmap ----------
    binary_grid  = (mini_map_2d == 0).astype(int)
    distance_map = distance_transform_edt(binary_grid)

    ROBOT_RADIUS   = 5
    SAFE_DISTANCE  = 15
    PENALTY_WEIGHT = 50

    # ---------- A* ----------
    max_row = h - 1
    max_column = w - 1
    for_x = [-1,  0, 1,  0, -1,  1, -1, 1]
    for_y = [ 0, -1, 0,  1, -1,  1,  1, -1]

    open_list = []
    heapq.heappush(open_list, (0, 0, start_x, start_y))
    visited = set()
    came_from = {}
    goal_found = False

    while open_list:
        f, current_g, x, y = heapq.heappop(open_list)

        if x == goal_x and y == goal_y:
            goal_found = True
            break

        if (x, y) in visited:
            continue
        visited.add((x, y))

        for k in range(len(for_x)):
            ni = x + for_x[k]
            nj = y + for_y[k]

            if 0 <= ni <= max_row and 0 <= nj <= max_column:
                if mini_map_2d[ni, nj] == 0 or (ni == goal_x and nj == goal_y):
                    dist_to_wall = distance_map[ni, nj]
                    if dist_to_wall < ROBOT_RADIUS and not (ni == goal_x and nj == goal_y):
                        continue
                    if (ni, nj) not in visited:
                        move_cost = 25 if k > 3 else 10
                        penalty = 0
                        if dist_to_wall < SAFE_DISTANCE:
                            normalized_closeness = (SAFE_DISTANCE - dist_to_wall) / (SAFE_DISTANCE - ROBOT_RADIUS)
                            penalty = int(PENALTY_WEIGHT * normalized_closeness)

                        g = current_g + move_cost + penalty
                        dx = abs(ni - goal_x)
                        dy = abs(nj - goal_y)
                        h_val = int(math.hypot(dx, dy) * 10)
                        f_new = h_val + g

                        heapq.heappush(open_list, (f_new, g, ni, nj))
                        if (ni, nj) not in came_from:
                            came_from[(ni, nj)] = (x, y)

    if not goal_found:
        dbg.error("A* FAILED — goal not reached.")
        return [], raw_map, distance_map

    # ---------- Reconstruct path ----------
    current = (goal_x, goal_y)
    final_path = []
    while current in came_from:
        final_path.append(current)
        current = came_from[current]

    dbg.info(f"Path reconstructed: {len(final_path)} waypoints")
    return final_path, raw_map, distance_map


# ============================================================
# ROS2 NODE
# ============================================================

class Navigation(Node):
    def __init__(self):
        super().__init__("Navigation")
        self.get_logger().info("Navigation node STARTING …")

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        # Control tuning
        self.distance_tolerance = 0.15
        self.k_linear = 4.5
        self.k_angular = 2.0

        # Map metadata
        self.res = 0.025
        self.ox  = -25.612499
        self.oy  = -25.612499

        # Global Goals
        self.start_odom = (0, 0)
        self.goal_odom  = (2.268, -1.162)

        # ---- Dynamic Obstacle Tracking Setup ----
        self.current_x = 0
        self.current_y = 0
        self.current_yaw = None
        self.dynamic_hits = {}       # {(row, col): count}
        self.hit_threshold = 5       # Scans required to mark obstacle
        self.is_replanning = False
        
        # Load map for dynamic modifications
        self.dynamic_map = np.load('map.npy').copy().astype(int)
        self.map_h, self.map_w = self.dynamic_map.shape

        # ---- Initial Path Planning ----
        self.path = []
        self.current_wp_index = 0
        self.viz = None

        pixel_path, raw_map = self.plan_and_set_path(self.start_odom, self.goal_odom)

        if pixel_path:
            start_px = odom_to_pixel(*self.start_odom, self.res, self.ox, self.oy)
            goal_px  = odom_to_pixel(*self.goal_odom, self.res, self.ox, self.oy)
            # viz expects (row, col)
            
            # --- VISUALIZER INSTANTIATION COMMENTED OUT FOR HARDWARE ---
            # self.viz = LiveMapVisualizer(raw_map, pixel_path, (start_px[1], start_px[0]), (goal_px[1], goal_px[0]))
            # self.viz.start()

        self.get_logger().info("Navigation node READY!")

    def plan_and_set_path(self, start_odom, goal_odom):
        """Helper to generate path and convert to odom coordinates."""
        start_px = odom_to_pixel(*start_odom, self.res, self.ox, self.oy)
        goal_px  = odom_to_pixel(*goal_odom, self.res, self.ox, self.oy)

        pixel_path, raw_map, _ = path_planner(
            start_px[0], start_px[1], goal_px[0], goal_px[1], input_map=self.dynamic_map
        )

        if not pixel_path:
            self.get_logger().error("path_planner returned EMPTY path!")
            return [], raw_map

        pixel_path.reverse() # Start -> Goal

        # Interval pruning
        pruned_pixel_path = []
        if len(pixel_path) > 0:
            pruned_pixel_path.append(pixel_path[0])
            for i in range(1, len(pixel_path) - 1, 8): 
                pruned_pixel_path.append(pixel_path[i])
            pruned_pixel_path.append(pixel_path[-1])

        self.path = []
        for (row, col) in pruned_pixel_path:
            mx, my = pixel_to_odom(col, row, self.res, self.ox, self.oy)
            self.path.append((mx, my))

        self.current_wp_index = 0
        return pixel_path, raw_map

    # --------------------------------------------------------

    def lidar_callback(self, msg):
        if self.current_x is None or self.is_replanning:
            return

        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        new_obstacle_confirmed = False

        for i, r in enumerate(msg.ranges):
            # Ignore bad data or obstacles too far away (e.g., > 2.0 meters)
            if r < msg.range_min or r > msg.range_max or math.isinf(r) or math.isnan(r) or r > 2.0:
                continue

            global_angle = self.current_yaw + (angle_min + i * angle_inc)
            hit_x = self.current_x + r * math.cos(global_angle)
            hit_y = self.current_y + r * math.sin(global_angle)

            pixel_x, pixel_y = odom_to_pixel(hit_x, hit_y, self.res, self.ox, self.oy)
            col, row = pixel_x, pixel_y 

            if 0 <= row < self.map_h and 0 <= col < self.map_w:
                if self.dynamic_map[row, col] == 0:
                    # Mark immediate area to create a safer padding
                    neighbors = [(row, col), (row+1, col), (row-1, col), (row, col+1), (row, col-1)]
                    
                    for ny, nx in neighbors:
                        if 0 <= ny < self.map_h and 0 <= nx < self.map_w:
                            pixel = (ny, nx)
                            self.dynamic_hits[pixel] = self.dynamic_hits.get(pixel, 0) + 1

                            if self.dynamic_hits[pixel] == self.hit_threshold:
                                self.dynamic_map[ny, nx] = 1 # Update Map to Wall
                                new_obstacle_confirmed = True
                                
                                if self.viz:
                                    self.viz.update_map_pixel(ny, nx, val=1)

        if new_obstacle_confirmed:
            self.check_path_and_replan()

    def check_path_and_replan(self):
        if not self.path or self.is_replanning:
            return

        path_blocked = False
        
        # Check all upcoming waypoints
        for wp_x, wp_y in self.path[self.current_wp_index:]:
            pixel_x, pixel_y = odom_to_pixel(wp_x, wp_y, self.res, self.ox, self.oy)
            col, row = pixel_x, pixel_y
            
            if self.dynamic_map[row, col] != 0:
                path_blocked = True
                break

        if path_blocked:
            self.get_logger().warn("⚠️ PATH BLOCKED BY DYNAMIC OBSTACLE! Replanning...")
            self.is_replanning = True
            self.stop_robot()
            
            # Replan from current exact position to global goal
            current_odom = (self.current_x, self.current_y)
            pixel_path, _ = self.plan_and_set_path(current_odom, self.goal_odom)

            if pixel_path and self.viz:
                # Update visualizer with new path
                self.viz.update_path(pixel_path)
            elif not pixel_path:
                self.get_logger().error("🛑 NO VALID PATH AROUND OBSTACLE.")

            self.is_replanning = False


    def euler_from_quaternion(self, x, y, z, w):
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)

        if not self.path or self.is_replanning:
            return

        if self.current_wp_index >= len(self.path):
            self.get_logger().info("All waypoints reached! Stopping robot.", throttle_duration_sec=5.0)
            self.stop_robot()
            return

        if self.viz:
            robot_px = odom_to_pixel(self.current_x, self.current_y, self.res, self.ox, self.oy)
            self.viz.update_robot(robot_px[1], robot_px[0])

        target_x, target_y = self.path[self.current_wp_index]
        diff_x = target_x - self.current_x
        diff_y = target_y - self.current_y
        distance_error = math.hypot(diff_x, diff_y)
        target_yaw = math.atan2(diff_y, diff_x)
        yaw_error = math.atan2(math.sin(target_yaw - self.current_yaw), math.cos(target_yaw - self.current_yaw))

        if distance_error < self.distance_tolerance:
            self.current_wp_index += 1
            if self.viz:
                self.viz.mark_waypoint_reached(self.current_wp_index)
            return

        cmd = Twist()
        if abs(yaw_error) > 0.5:
            cmd.angular.z = self.k_angular * yaw_error
            cmd.linear.x  = 0.0
        else:
            cmd.linear.x  = min(self.k_linear * distance_error, 1.0)
            cmd.angular.z = self.k_angular * yaw_error

        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)


# ============================================================
# ENTRY POINT
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = Navigation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()