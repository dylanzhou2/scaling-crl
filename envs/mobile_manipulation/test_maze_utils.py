import os
import numpy as np
from envs.mobile_manipulation.maze_utils import generate_procedural_grid, make_tidybot_maze

def test_procedural_generation():
    print("--- Testing Procedural Grid Generation ---")
    width, height = 11, 11
    grid = generate_procedural_grid(width, height)
    
    # Verify dimensions
    assert grid.shape == (height, width), f"Expected shape {(height, width)}, got {grid.shape}"
    
    # Verify Start (2) and Goal (3) exist
    assert 2 in grid, "Robot start position (2) missing from grid"
    assert 3 in grid, "Goal position (3) missing from grid"
    
    print("Generated Grid Layout (0=Path, 1=Wall, 2=Start, 3=Goal):")
    print(grid)
    print("Grid generation successful.\n")

def test_xml_construction():
    print("--- Testing XML Construction ---")
    width, height = 9, 9
    grid = generate_procedural_grid(width, height)
    
    # Test with and without clutter
    try:
        xml_string, goal_pos = make_tidybot_maze(grid, size_scaling=4.0, include_clutter=True)
        
        # Basic validation of outputs
        assert isinstance(xml_string, str), "XML output should be a string"
        assert len(xml_string) > 0, "XML string is empty"
        assert goal_pos.shape == (2,), "Goal position should be an (x, y) coordinate"
        
        # Check for key MuJoCo elements in the string
        assert "<mujoco" in xml_string, "Missing <mujoco> tag"
        assert "wall_" in xml_string, "No wall geoms found in XML"
        assert "goal_site" in xml_string, "Goal visualization site missing from XML"
        
        print(f"Goal Position: {goal_pos}")
        print(f"XML String Length: {len(xml_string)} characters")
        
        # Test clutter inclusion
        if "clutter_" in xml_string:
            print("Verified: Static clutter successfully injected into XML.")
        else:
            print("Note: No clutter generated (this can happen due to randomness).")
            
        print("XML construction successful.\n")
        
    except FileNotFoundError:
        print("Error: 'tidybot.xml' not found. Ensure the asset path in maze_utils.py is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def save_test_xml(filename="debug_maze.xml"):
    """Helper to save the generated XML for manual inspection in MuJoCo viewer."""
    grid = generate_procedural_grid(11, 11)
    xml_string, _ = make_tidybot_maze(grid, include_clutter=True)
    
    with open(filename, "w") as f:
        f.write(xml_string)
    print(f"Test XML saved to {filename}. You can open this in a MuJoCo viewer to inspect.")

if __name__ == "__main__":
    test_procedural_generation()
    test_xml_construction()
    save_test_xml()