import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Conversion Functions (same as yours)
# ==============================

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


# ==============================
# PARAMETERS (same as yours)
# ==============================

RESOLUTION = 0.025
ORIGIN_X = -25.612499
ORIGIN_Y = -25.612499

# ==============================
# Load Map
# ==============================

mini_map_2d = np.load('map.npy')
map_height = mini_map_2d.shape[0]

# ==============================
# Click Event Function
# ==============================

def onclick(event):
    if event.xdata is None or event.ydata is None:
        return

    pixel_x = int(event.xdata)
    pixel_y = int(event.ydata)

    odom_x, odom_y = pixel_to_odom(
        pixel_x, pixel_y,
        RESOLUTION,
        ORIGIN_X,
        ORIGIN_Y,
        map_height
    )

    print("\nClicked Point:")
    print(f"Pixel  -> (x={pixel_x}, y={pixel_y})")
    print(f"Odom   -> (x={odom_x:.3f}, y={odom_y:.3f})")


# ==============================
# Plot Map
# ==============================

fig, ax = plt.subplots()
ax.imshow(mini_map_2d, cmap='viridis')
ax.set_title("Click anywhere to get Pixel + Odom coordinates")

# Connect click event
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()