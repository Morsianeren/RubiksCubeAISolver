# In this class i will use quaternions to keep track of the orientation of the pieces!

# " Quaternions came from Hamilton after his really good work had been done; 
# and, though beautifully ingenious, 
# have been an unmixed evil to those who have touched them in any way, including Clerk Maxwell"
# - Lord Kelvin

# %% 
# Automatically reload changed modules
#%load_ext autoreload 
#%autoreload 2
from .Piece import Piece
from .setup_cube_script import setup_cube, POINT_TO_INDEX
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal

class RubiksCube():
    def __init__(self):
        self.pieces = setup_cube()
        
    def plot(self, style:Literal['square', 'arrows']='square', **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scale = 4

        for piece in self.pieces:
            if isinstance(piece, Piece):
                piece.plot(style, **kwargs)

        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_zlim(-scale, scale)
        plt.show()

    def rotate_side(self, axis:Literal['x', 'y', 'z'], row:int, k:int):
        lower_position = -1
        middle_position = 0
        upper_position = 1

        pieces = self.pieces

        if row not in [lower_position, upper_position, middle_position]:
            raise ValueError(f"Row must be one of {lower_position}, {middle_position}, or {upper_position}, got {row}.")
        # Filter out the pieces that should be rotated
        idx = 'xyz'.index(axis)
        pieces_to_rotate = [piece for piece in pieces if piece.position[idx] == row]

        # Rotate pieces
        rotate_pieces(pieces_to_rotate, axis, k)
    
    def scramble(self, iterations = 64):
        for _ in range(iterations):
            # Get a random axis
            axis = np.random.choice(['x', 'y', 'z'])
            # Get a random row
            row = np.random.choice([-1, 0, 1])
            # Get a random number of rotations
            k = np.random.choice([1, 2, 3])
            # Rotate it
            self.rotate_side(axis, row, k)

    def reset(self):
        for piece in self.pieces:
            piece.reset()

    def array(self):
        """This function returns a flattened list with the colors of the rubiks cube
        The order is depicted by the POINT_TO_INDEX lookup
        """

        cube_color_matrix = np.empty((6,9), dtype='U')

        # Iterate over all pieces
        for piece in self.pieces:
            position = piece.position
            v_x, v_y, v_z = piece.orientation.get_xyz_vectors()
            vectors = [v_x, v_y, v_z]
            colors = piece.colors.values()
            for color, vector in zip(colors, vectors):
                point = position + vector
                point = tuple(point)
                if point not in POINT_TO_INDEX.keys():
                    #print(f"Position {position} + vector {vector} not found in POINT_TO_INDEX")
                    continue
                index = POINT_TO_INDEX[point]
                cube_color_matrix[index] = color

        return cube_color_matrix

def rotate_pieces(pieces: list, axis:Literal['x', 'y', 'z'], k:int):
    """Rotates all given pieces k*90 degrees around the given axis.

    Args:
        pieces (list): All pieces to rotate
        axis (Literal['x', 'y', 'z']): Which axis to rotate around
        k (int): Number of 90 degrees rotations to rotate

    Returns:
        list: The rotated pieces
    """
    # Create a copy of the pieces
    pieces_copy = pieces.copy()
    for piece in pieces_copy:
        piece.rotate(axis, k)
        #piece.rotate(axis)
    return pieces_copy

# %% Test code
#cube = RubiksCube()

#cube.plot(exploded=True)
#array_view = cube.array()

#cube.scramble(2)

#cube.plot()

#cube.reset()

#cube.plot()