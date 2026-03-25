#!/usr/bin/env python3
"""
pgm_to_occupancy_grid.py
------------------------
Converts a ROS-standard .pgm map file into a binary occupancy grid numpy array.

Usage (standalone):
    python3 pgm_to_occupancy_grid.py map.pgm
    python3 pgm_to_occupancy_grid.py map.pgm --yaml map.yaml   # also load origin/resolution
    python3 pgm_to_occupancy_grid.py map.pgm --save            # save grid as .npy file

Usage (as a module):
    from pgm_to_occupancy_grid import load_occupancy_grid

    grid, info = load_occupancy_grid("map.pgm", yaml_path="map.yaml")
    # grid[row][col]:  0 = free,  1 = obstacle,  -1 = unknown
    # info dict: resolution, origin_x, origin_y, width, height

    # Grid cell -> world coordinates:
    world_x = info["origin_x"] + col * info["resolution"]
    world_y = info["origin_y"] + row * info["resolution"]

    # World coordinates -> grid cell:
    col = int((world_x - info["origin_x"]) / info["resolution"])
    row = int((world_y - info["origin_y"]) / info["resolution"])
"""

import argparse
import os
import sys

import numpy as np
import cv2


# ROS PGM pixel thresholds (from map_server defaults)
#   254  = free  |  0 = obstacle  |  205 = unknown   (ros map_saver standard)
FREE_THRESH_PX = 240    # pixels >= this  -> free
OCC_THRESH_PX  = 10     # pixels <= this  -> obstacle
# everything in between  -> unknown (-1)


def load_pgm(pgm_path: str) -> np.ndarray:
    """Load a .pgm file and return a grayscale numpy array (uint8)."""
    if not os.path.isfile(pgm_path):
        raise FileNotFoundError(f"PGM file not found: {pgm_path}")
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {pgm_path}")
    return img


def load_yaml(yaml_path: str) -> dict:
    """Parse a ROS map .yaml file -> dict with resolution, origin, negate."""
    import re
    info = {"resolution": 0.05, "origin_x": 0.0, "origin_y": 0.0, "negate": 0}

    if not os.path.isfile(yaml_path):
        print(f"[WARN] YAML not found: {yaml_path}. Using defaults.")
        return info

    with open(yaml_path, "r") as f:
        content = f.read()

    m = re.search(r"resolution\s*:\s*([\d.eE+\-]+)", content)
    if m:
        info["resolution"] = float(m.group(1))

    m = re.search(r"origin\s*:\s*\[\s*([\d.eE+\-]+)\s*,\s*([\d.eE+\-]+)", content)
    if m:
        info["origin_x"] = float(m.group(1))
        info["origin_y"] = float(m.group(2))

    m = re.search(r"negate\s*:\s*(\d+)", content)
    if m:
        info["negate"] = int(m.group(1))

    return info


def pgm_to_binary_grid(img: np.ndarray, negate: int = 0) -> np.ndarray:
    """
    Convert grayscale PGM image to binary occupancy grid.

    Returns
    -------
    grid : np.ndarray (int8)
        -1 = unknown
         0 = free
         1 = obstacle

    Row-0 = y_min (bottom of world) after vertical flip.
    """
    pixels = img.astype(np.int16)

    if negate:
        pixels = 255 - pixels

    # PGM row-0 = top of image = y_max in world
    # Flip so that row-0 = y_min (standard ROS / path-planner convention)
    pixels = np.flipud(pixels)

    grid = np.full(pixels.shape, -1, dtype=np.int8)
    grid[pixels >= FREE_THRESH_PX] = 0   # free
    grid[pixels <= OCC_THRESH_PX]  = 1   # obstacle

    return grid


def load_occupancy_grid(pgm_path: str, yaml_path: str = None):
    """
    Load a .pgm (+ optional .yaml) and return a binary occupancy grid.

    Parameters
    ----------
    pgm_path  : str  path to the .pgm file
    yaml_path : str  path to the .yaml file (auto-detected if omitted)

    Returns
    -------
    grid : np.ndarray (int8, shape [height, width])
        0  = free
        1  = obstacle
       -1  = unknown

    info : dict
        resolution  - metres per cell
        origin_x    - world X of cell [row=0, col=0]
        origin_y    - world Y of cell [row=0, col=0]
        width       - number of columns
        height      - number of rows
    """
    # Auto-detect yaml
    if yaml_path is None:
        candidate = os.path.splitext(pgm_path)[0] + ".yaml"
        if os.path.isfile(candidate):
            yaml_path = candidate

    map_info = load_yaml(yaml_path) if yaml_path else {
        "resolution": 0.05, "origin_x": 0.0, "origin_y": 0.0, "negate": 0
    }

    img  = load_pgm(pgm_path)
    grid = pgm_to_binary_grid(img, negate=map_info.get("negate", 0))

    h, w = grid.shape
    info = {
        "resolution": map_info["resolution"],
        "origin_x":   map_info["origin_x"],
        "origin_y":   map_info["origin_y"],
        "width":       w,
        "height":      h,
    }

    return grid, info


def print_grid_stats(grid: np.ndarray, info: dict):
    h, w   = grid.shape
    n_free = int(np.sum(grid == 0))
    n_obs  = int(np.sum(grid == 1))
    n_unk  = int(np.sum(grid == -1))
    total  = h * w
    res    = info["resolution"]

    print("\n" + "=" * 54)
    print("  Occupancy Grid Summary")
    print("=" * 54)
    print(f"  Shape          : {w} cols x {h} rows  ({total:,} cells)")
    print(f"  Resolution     : {res} m/cell")
    print(f"  World size     : {w*res:.2f} m  x  {h*res:.2f} m")
    print(f"  Origin (world) : x={info['origin_x']:.3f}  y={info['origin_y']:.3f}")
    print(f"  Free     (0)   : {n_free:>9,}  ({100*n_free/total:.1f}%)")
    print(f"  Obstacle (1)   : {n_obs:>9,}  ({100*n_obs/total:.1f}%)")
    print(f"  Unknown  (-1)  : {n_unk:>9,}  ({100*n_unk/total:.1f}%)")
    print("=" * 54)
    print("\n  Coordinate helpers:")
    print("    world_x = origin_x + col * resolution")
    print("    world_y = origin_y + row * resolution")
    print("    col     = int((world_x - origin_x) / resolution)")
    print("    row     = int((world_y - origin_y) / resolution)")
    print("=" * 54 + "\n")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    global FREE_THRESH_PX, OCC_THRESH_PX

    parser = argparse.ArgumentParser(
        description="Convert a ROS .pgm map to a binary occupancy grid numpy array."
    )
    parser.add_argument("pgm",  help="Path to the .pgm map file")
    parser.add_argument("--yaml",  help="Path to matching .yaml (auto-detected if omitted)")
    parser.add_argument("--save",  action="store_true",
                        help="Save the grid as <name>_occupancy_grid.npy")
    parser.add_argument("--free-thresh", type=int, default=FREE_THRESH_PX,
                        help=f"Pixel >= this -> free (default {FREE_THRESH_PX})")
    parser.add_argument("--occ-thresh",  type=int, default=OCC_THRESH_PX,
                        help=f"Pixel <= this -> obstacle (default {OCC_THRESH_PX})")
    args = parser.parse_args()

    FREE_THRESH_PX = args.free_thresh
    OCC_THRESH_PX  = args.occ_thresh

    grid, info = load_occupancy_grid(args.pgm, yaml_path=args.yaml)
    print_grid_stats(grid, info)

    if args.save:
        out = os.path.splitext(args.pgm)[0] + "_occupancy_grid.npy"
        np.save(out, grid)
        print(f"  Saved : {out}")
        print(f"  Load  : grid = np.load('{out}')\n")

    return grid, info


if __name__ == "__main__":
    grid, info = main()