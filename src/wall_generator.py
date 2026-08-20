import numpy as np

'''
This file contains a method to generate a binary mask (numpy matrix) of specified size
that represents the position of the walls in the simulation. 
It returns thw 'walls' matrix, whose values can be easily interpreted as follows:
    1.0 = wall present
    0.0 = wall not present
'''

def create_walls(height, width, thickness):
    walls = np.zeros((height, width), dtype=np.float32)

    walls[:,0:0+thickness] = 1
    walls[:,-1-thickness:] = 1

    walls[0:0+thickness,:] = 1
    walls[-1-thickness:,:] = 1

    walls[height//2-thickness:height//2+thickness,:] = 1
    walls[:,width//2-thickness:width//2+thickness] = 1

    return walls