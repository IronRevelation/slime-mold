import numpy as np

def get_spawn_structure(height, width):
    matrix = np.zeros((height, width, 2), dtype=np.float32) # 3rd index = [agent species, angle]

    for i in range(height//2):
        for j in range(width//2):
            matrix[i,j,0] = 1
            matrix[height//2+i,width//2+j,0] = 2
            matrix[i,j,1] = np.random.uniform()
            matrix[height//2+i,width//2+j,1] = np.random.uniform()

    return np.asarray(matrix_to_agent_vector(matrix))

def matrix_to_agent_vector(matrix):
    # Extract coordinates (R,C) of non-zero pixels
    rows, cols = np.nonzero(matrix[:,:,0])
    species_values = matrix[rows, cols, 0] - 1.0
    angle_values = matrix[rows, cols, 1]

    # Creates the initialization structure with the format used by update_agents (x, y, species)
    return list(zip(cols.astype(np.float32), rows.astype(np.float32), angle_values, species_values))