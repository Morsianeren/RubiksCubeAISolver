# In this class i will use quaternions to keep track of the orientation of the pieces!

# " Quaternions came from Hamilton after his really good work had been done; 
# and, though beautifully ingenious, 
# have been an unmixed evil to those who have touched them in any way, including Clerk Maxwell"
# - Lord Kelvin

# %% 
# Automatically reload changed modules
#%load_ext autoreload 
#%autoreload 2
from Piece import Piece
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal

class RubiksCube():
    def __init__(self):

        self.pieces = set()

        # Intialize the pieces and orientation
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                #for z in [-1]:
                    if z == -1:
                        piece = Piece(x, y, 1)
                    else:
                        piece = Piece(x, y, z)
                    k = 0
                    if y == x == -1:
                        k = 2
                    elif y == -1:
                        k = -1
                    elif x == -1:
                        k = 1
                    piece.rotate('z', k, rotate_position=False)
                    if z == -1:
                        piece.rotate('x', 2, rotate_position=True)
                    self.pieces.add(piece)

        for piece in self.pieces:
            piece.reset_initial_state()
            
            # Remove colors inside the cube
            pos = piece.position
            x = pos[0]
            y = pos[1]
            z = pos[2]
            x_vector, y_vector, z_vector = piece.orientation.get_xyz_vectors()

            mask = [x, y, z]

            for i, vector in enumerate([x_vector, y_vector, z_vector]):
                result = any([a == b and abs(a) == 1 for a, b in zip(vector, mask)])
                
                if not result:
                    idx = 'xyz'[i]
                    piece.colors[idx] = None

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
        # Initialize a list to store all the colors
        face_list = np.zeros(9*6)

        # Iterate over all pieces
        for piece in self.pieces:
            # Get the position of the piece
            position = piece.position
            # Get the colors of the piece
            colors = piece.colors
            
            # If piece is front
            if position[2] == -1:
                face_list[0] = 0 # TODO

        return face_list

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

cube.plot()

cube.scramble(2)

cube.plot()

#cube.reset()

#cube.plot()

#face_list = cube.flatten()