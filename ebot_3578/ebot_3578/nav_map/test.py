import math 
import heapq
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt # NEW: Required for generating the costmap


def odom_to_pixel(odom_x, odom_y, resolution, origin_x, origin_y, map_height=None):
    pixel_x = int((odom_x - origin_x) / resolution)
    pixel_y = int((odom_y - origin_y) / resolution)

    # Optional: fix Y-axis inversion
    if map_height is not None:
        pixel_y = map_height - pixel_y

    return pixel_x, pixel_y


def pixel_to_odom(pixel_x, pixel_y, resolution, origin_x, origin_y, map_height=None):
    
    # Step 1: Undo Y-axis flip (if applied earlier)
    if map_height is not None:
        pixel_y = map_height - pixel_y

    # Step 2: Convert pixels → meters
    odom_x = pixel_x * resolution + origin_x
    odom_y = pixel_y * resolution + origin_y

    return odom_x, odom_y


# Load the .npy file and make a working copy
mini_map_2d = np.load('map.npy').copy().astype(int)

# ========================================================
# NEW: Generate Distance Map & Define Robot Parameters
# ========================================================
# distance_transform_edt calculates distance to the nearest 0.
# We want distance FROM free space (0) TO obstacles (!= 0).
# So we invert our grid: Free space = 1, Obstacles = 0.

binary_grid = (mini_map_2d == 0).astype(int) 
distance_map = distance_transform_edt(binary_grid)

# --- TUNE THESE PARAMETERS FOR YOUR ROBOT ---
ROBOT_RADIUS = 5    # Hard limit: Minimum pixels from a wall (Collision threshold)
SAFE_DISTANCE = 20  # Soft limit: Distance at which we start penalizing the path
PENALTY_WEIGHT = 50 # How aggressively to avoid walls (Higher = stays more centered)

# --- NEW: Interactive Start and Goal Selection ---
print("Opening map... Please CLICK TWO POINTS on the map.")
print("1st click: Start Position")
print("2nd click: Goal Position")

plt.imshow(mini_map_2d, cmap='viridis')
plt.title("Click 1st: Start | Click 2nd: Goal")
pts = plt.ginput(1, timeout=0) # Wait infinitely for 2 clicks
plt.close() # Close window after clicking so A* can run

# ginput returns (column, row) so we must map them correctly to your numpy array
# start_y, start_x = int(pts[0][0]), int(pts[0][1])

pixel = odom_to_pixel(0,0, 0.025, -25.612499, -25.612499)
pixel_goal = odom_to_pixel(2.268, -1.662, 0.025, -25.612499, -25.612499)

start_y, start_x = pixel[0], pixel[1]
goal_y, goal_x   = pixel_goal[0], pixel_goal[1]

print(f"Selected Start: ({start_x}, {start_y})")
print(f"Selected Goal:  ({goal_x}, {goal_y})")

# Sanity check: Ensure you clicked on Free Space (0)
if mini_map_2d[start_x, start_y] != 0:
    print("WARNING: Your start point is inside a wall/unknown space! A* might fail.")
if mini_map_2d[goal_x, goal_y] != 0:
    print("WARNING: Your goal point is inside a wall/unknown space! A* might fail.")

# ========================================================
# Your exact A* logic below
# ========================================================

max_row = mini_map_2d.shape[0] - 1
max_column = mini_map_2d.shape[1] - 1

for_x = [-1, 0, 1, 0, -1, 1, -1, 1]
for_y = [ 0,-1, 0, 1, -1, 1,  1,-1]

open_list = []
heapq.heappush(open_list, (0, 0, start_x, start_y)) 

visited = set()
came_from = {} 

while len(open_list) > 0:
    
    f, current_g, x, y = heapq.heappop(open_list)

    if x == goal_x and y == goal_y:
        print("Goal Found!\n")
        break

    if (x, y) in visited:
        continue
        
    visited.add((x, y))

    for k in range(len(for_x)):
        ni = x + for_x[k]
        nj = y + for_y[k]

        if 0 <= ni <= max_row and 0 <= nj <= max_column:
            
            # Explicitly check for Free Space (0)
            if mini_map_2d[ni, nj] == 0 or (ni == goal_x and nj == goal_y):

                # --- NEW: Check distance map limits ---
                dist_to_wall = distance_map[ni, nj]
                
                # 1. HARD LIMIT: Prevent clipping into walls
                # We skip this check if (ni, nj) is the exact goal, just in case the user clicked near a wall
                if dist_to_wall < ROBOT_RADIUS and not (ni == goal_x and nj == goal_y):
                    continue
                
                if (ni, nj) not in visited: 
                    
                    if k > 3:
                        move_cost =  14
                    else:
                        move_cost = 10

                    # 3. SOFT LIMIT: Add penalty cost to push path to the center
                    penalty = 0
                    if dist_to_wall < SAFE_DISTANCE:
                        # Calculates a sliding scale: Highest penalty at ROBOT_RADIUS, 0 penalty at SAFE_DISTANCE
                        normalized_closeness = (SAFE_DISTANCE - dist_to_wall) / (SAFE_DISTANCE - ROBOT_RADIUS)
                        penalty = int(PENALTY_WEIGHT * normalized_closeness)

                    # Add the penalty to the actual g-cost
                    g = current_g + move_cost + penalty


                    dx = abs(ni - goal_x)
                    dy = abs(nj - goal_y)
                    h = int(math.hypot(dx, dy) * 10)

                    f = h + g
                    
                    heapq.heappush(open_list, (f, g, ni, nj))
                    
                    if (ni, nj) not in came_from:
                        came_from[(ni, nj)] = (x, y)

current = (goal_x, goal_y)

counter = 0
while current in came_from:
    mini_map_2d[current] = 2
    current = came_from[current]
    counter = counter +1 
    print(current)

print("Counter", counter)
# print(len(came_from))


mini_map_2d[start_x, start_y] = 3
mini_map_2d[goal_x, goal_y] = 3

plt.imshow(mini_map_2d, cmap='viridis')
plt.title("A* Path Result")
plt.colorbar()
plt.show()

