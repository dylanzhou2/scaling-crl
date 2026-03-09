import os
import xml.etree.ElementTree as ET
import random
from jax import numpy as jnp
import numpy as np

def generate_procedural_grid(width, height):
    """Generates a maze grid using DFS to ensure a guaranteed path."""
    grid = np.ones((height, width), dtype=int)
    
    def walk(x, y):
        grid[y, x] = 0
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] == 1:
                grid[y + dy // 2, x + dx // 2] = 0
                walk(nx, ny)

    walk(1, 1)
    
    # Define Start and Goal at opposite corners
    grid[1, 1] = 2  # Robot Start (R)
    grid[height - 2, width - 2] = 3  # Goal (G)
    return grid

def make_tidybot_maze(grid, size_scaling=4.0, include_clutter=True):
    # Get the absolute path to the assets folder
    current_dir = os.path.dirname(__file__)
    # Adjust this path to point exactly to where your .stl files live
    assets_dir = os.path.abspath(os.path.join(current_dir, "..", "assets", "stanford_tidybot2", "assets"))

    asset_path = os.path.join(current_dir, "..", "assets", "stanford_tidybot2", "tidybot.xml")
    tree = ET.parse(asset_path)
    root = tree.getroot()

    # Update the compiler's meshdir to use the absolute path
    compiler = root.find(".//compiler")
    if compiler is not None:
        compiler.set("meshdir", assets_dir)
    worldbody = tree.find(".//worldbody")
    
    # 1. Add Maze Walls
    height, width = grid.shape
    for i in range(height):
        for j in range(width):
            if grid[i, j] == 1:
                ET.SubElement(
                    worldbody, "geom",
                    name=f"wall_{i}_{j}",
                    pos=f"{i * size_scaling} {j * size_scaling} 0.5",
                    size=f"{size_scaling/2} {size_scaling/2} 0.5",
                    type="box",
                    rgba="0.5 0.5 0.5 1"
                )
            elif grid[i, j] == 0 and include_clutter and random.random() < 0.1:
                # 2. Add Static Clutter (ensure it doesn't block critical paths)
                ET.SubElement(
                    worldbody, "geom",
                    name=f"clutter_{i}_{j}",
                    pos=f"{i * size_scaling} {j * size_scaling} 0.2",
                    size="0.15 0.15 0.2",
                    type="box",
                    rgba="0.8 0.2 0.2 1"
                )

    # 3. Add Goal Visualization (Ghost Object)
    goal_pos = np.argwhere(grid == 3)[0] * size_scaling
    ET.SubElement(
        worldbody, "geom",
        name="goal_site",
        pos=f"{goal_pos[0]} {goal_pos[1]} 0.01",
        size=f"{size_scaling/4} 0.01",
        type="cylinder",
        rgba="0 1 0 0.3",
        contype="0", conaffinity="0"
    )

    return ET.tostring(tree.getroot(), encoding='unicode'), goal_pos

def make_tidybot_mocap_maze(width, height, size_scaling=4.0):
    import os
    import xml.etree.ElementTree as ET
    
    current_dir = os.path.dirname(__file__)
    assets_dir = os.path.abspath(os.path.join(current_dir, "..", "assets", "stanford_tidybot2", "assets"))
    asset_path = os.path.join(current_dir, "..", "assets", "stanford_tidybot2", "tidybot.xml")
    
    tree = ET.parse(asset_path)
    root = tree.getroot()
    compiler = root.find(".//compiler")
    if compiler is not None:
        compiler.set("meshdir", assets_dir)
    worldbody = tree.find(".//worldbody")

    # Create a movable MOCAP body for EVERY possible grid cell
    for i in range(height):
        for j in range(width):
            body = ET.SubElement(
                worldbody, "body",
                name=f"wall_{i}_{j}",
                pos=f"{i * size_scaling} {j * size_scaling} -10.0",  # Hidden underground by default
                mocap="true"
            )
            ET.SubElement(
                body, "geom",
                type="box",
                size=f"{size_scaling/2} {size_scaling/2} 0.5",
                rgba="0.5 0.5 0.5 1"
            )

    # Create a MOCAP body for the goal visualization
    goal_body = ET.SubElement(
        worldbody, "body",
        name="goal_site",
        pos="0 0 -10.0",
        mocap="true"
    )
    ET.SubElement(
        goal_body, "geom",
        type="cylinder",
        size=f"{size_scaling/4} 0.01",
        rgba="0 1 0 0.3",
        contype="0", conaffinity="0"
    )

    return ET.tostring(tree.getroot(), encoding='unicode')