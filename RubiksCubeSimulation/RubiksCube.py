# In this class i will use quaternions to keep track of the orientation of the pieces!

# " Quaternions came from Hamilton after his really good work had been done; 
# and, though beautifully ingenious, 
# have been an unmixed evil to those who have touched them in any way, including Clerk Maxwell"
# - Lord Kelvin

# %% 
from Piece import Piece
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal

class RubiksCube():
    def __init__(self):
        pieces = np.zeros((3, 3, 3), dtype=Piece)

        # Initialize the pieces
        # The cube will have center in origo
        for x, y, z in np.ndindex(3, 3, 3):
            pieces[x, y, z] = Piece((x-1),
                                    (y-1),
                                    (z-1))

        # Assign colors to the pieces
        for row in pieces[0,:,:]:
            for piece in row:
                piece.colors['-x'] = "yellow"

        for row in pieces[2,:,:]:
            for piece in row:
                piece.colors['x'] = "white"

        for row in pieces[:,0,:]:
            for piece in row:
                piece.colors['-y'] = "orange"

        for row in pieces[:,2,:]:
            for piece in row:
                piece.colors['y'] = "red"

        for row in pieces[:,:,0]:
            for piece in row:
                piece.colors['-z'] = "green"

        for row in pieces[:,:,2]:
            for piece in row:
                piece.colors['z'] = "blue"

        # After this there is no need for the array to be 3x3x3
        # since rotation will mess up the indexing
        self.pieces = pieces.flatten()

    def plot(self, style:Literal['square', 'arrows']='square', **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scale = 2
        #s = scale*2

        # Plot lines in x y and z direction
        #ax.plot([-s, s], [0, 0], [0, 0], color='red')
        #ax.plot([0, 0], [-s, s], [0, 0], color='green')
        #ax.plot([0, 0], [0, 0], [-s, s], color='blue')

        for piece in self.pieces:
           piece.plot(ax, style, **kwargs)

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
            # Get the index of the piece
            idx = np.ravel_multi_index(position+1, (3,3,3))
            # Add the colors to the list
            face_list[idx*6:idx*6+6] = list(colors.values())

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

#cube.plot()

cube.scramble()

cube.plot()

#cube.reset()

#cube.plot()

#face_list = cube.flatten()