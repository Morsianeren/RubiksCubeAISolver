# In this class i will use quaternions to keep track of the orientation of the pieces!

# " Quaternions came from Hamilton after his really good work had been done; 
# and, though beautifully ingenious, 
# have been an unmixed evil to those who have touched them in any way, including Clerk Maxwell"
# - Lord Kelvin

# The order is front, back, right, left, up, down

POINT_TO_INDEX = {
    ( 1, -1,  1):   (0, 0), # Front
    ( 1,  0,  1):   (0, 1),
    ( 1,  1,  1):   (0, 2),
    ( 1, -1,  0):   (0, 3),
    ( 1,  0,  0):   (0, 4),
    ( 1,  1,  0):   (0, 5),
    ( 1, -1, -1):   (0, 6),
    ( 1,  0, -1):   (0, 7),
    ( 1,  1, -1):   (0, 8),

    (-1,  1,  1):   (1, 0), # Back
    (-1,  0,  1):   (1, 1),
    (-1, -1,  1):   (1, 2),
    (-1,  1,  0):   (1, 3),
    (-1,  0,  0):   (1, 4),
    (-1, -1,  0):   (1, 5),
    (-1,  1, -1):   (1, 6),
    (-1,  0, -1):   (1, 7),
    (-1, -1, -1):   (1, 8),

    ( 1,  1,  1):   (2, 0), # Right
    ( 0,  1,  1):   (2, 1),
    (-1,  1,  1):   (2, 2),
    ( 1,  1,  0):   (2, 3),
    ( 0,  1,  0):   (2, 4),
    (-1,  1,  0):   (2, 5),
    ( 1,  1, -1):   (2, 6),
    ( 0,  1, -1):   (2, 7),
    (-1,  1, -1):   (2, 8),

    (-1, -1,  1):   (3, 0), # Left
    ( 0, -1,  1):   (3, 1),
    ( 1, -1,  1):   (3, 2),
    (-1, -1,  0):   (3, 3),
    ( 0, -1,  0):   (3, 4),
    ( 1, -1,  0):   (3, 5),
    (-1, -1, -1):   (3, 6),
    ( 0, -1, -1):   (3, 7),
    ( 1, -1, -1):   (3, 8),

    (-1, -1,  1):   (4, 0), # Up
    (-1,  0,  1):   (4, 1),
    (-1,  1,  1):   (4, 2),
    ( 0, -1,  1):   (4, 3),
    ( 0,  0,  1):   (4, 4),
    ( 0,  1,  1):   (4, 5),
    ( 1, -1,  1):   (4, 6),
    ( 1,  0,  1):   (4, 7),
    ( 1,  1,  1):   (4, 8),

    ( 1, -1, -1):   (5, 0), # Down
    ( 1,  0, -1):   (5, 1),
    ( 1,  1, -1):   (5, 2),
    ( 0, -1, -1):   (5, 3),
    ( 0,  0, -1):   (5, 4),
    ( 0,  1, -1):   (5, 5),
    (-1, -1, -1):   (5, 6),
    (-1,  0, -1):   (5, 7),
    (-1,  1, -1):   (5, 8),
}
# %% 
# Automatically reload changed modules
#%load_ext autoreload 
#%autoreload 2
from Piece import Piece
from setup_cube_script import setup_cube
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

    def flatten(self):
        """This function returns a flattened list with the colors of the rubiks cube
        The order is front, right, back, left, up, down
        0 1 2
        3 4 5
        6 7 8
        """

        cube_color_matrix = np.zeros((18,3))


        # Iterate over all pieces
        for piece in self.pieces:
            # Get the position of the piece
            position = piece.position
            # Get the colors of the piece
            colors = list(piece.colors.values())
            
            x_vector, y_vector, z_vector = piece.orientation.get_xyz_vectors()

            p1 = tuple(position + x_vector)
            p2 = tuple(position + y_vector)
            p3 = tuple(position + z_vector)

            for i, point in enumerate([p1, p2, p3]):
                try:
                    idx = POINT_TO_INDEX[point]
                except KeyError:
                    # No more valid colors
                    continue

                color = colors[i]

                if color is None:
                    raise TypeError("Color is None, this is a fault in the code!")

                cube_color_matrix[idx] = color

                    

        return cube_color_matrix.flatten()

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
cube = RubiksCube()

cube.plot(exploded=True)

cube.scramble(2)

cube.plot()

#cube.reset()

#cube.plot()

#face_list = cube.flatten()