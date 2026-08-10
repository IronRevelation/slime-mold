import numpy as np

'''
This file contains a method to generate a binary mask (numpy matrix) of specified size
that represents the position of the walls in the simulation. 
It returns thw 'walls' matrix, whose values can be easily interpreted as follows:
    1.0 = wall present
    0.0 = wall not present
'''

def create_walls(height, width, thickness=8):
    wall_thickness = 8    
    walls = np.zeros((height, width), dtype=np.float32)

    # Horizontal wall in the middle of the frame
    center_y = height // 2
    half_thickness = thickness // 2
    walls[center_y - half_thickness : center_y + half_thickness, :] = 1.0

    # Verical wall in the middle of the frame
    center_x = width // 2
    walls[:, center_x - half_thickness : center_x + half_thickness] = 1.0
    return walls