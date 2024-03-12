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

POSITION_SCALE = 4

pieces = np.zeros((3, 3, 3), dtype=Piece)

# Initialize the pieces
# The cube will have center in origo
for x, y, z in np.ndindex(3, 3, 3):
    pieces[x, y, z] = Piece((x-1)*POSITION_SCALE, 
                            (y-1)*POSITION_SCALE, 
                            (z-1)*POSITION_SCALE)

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

# Plot the pieces
style='square'
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for piece in pieces.flatten():
    piece.plot(ax, style)

ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_zlim(-8, 8)
plt.show()

# After this there is no need for the array to be 3x3x3
# since rotation will mess up the indexing
pieces = pieces.flatten()

def rotate_side(pieces:list, axis:Literal['x', 'y', 'z'], row:int, k:int):
    lower_position = -POSITION_SCALE
    middle_position = 0
    upper_position = POSITION_SCALE

    if row not in [lower_position, upper_position, middle_position]:
        raise ValueError(f"Row must be one of {lower_position}, {middle_position}, or {upper_position}, got {row}.")
    # Filter out the pieces that should be rotated
    idx = 'xyz'.index(axis)
    pieces_to_rotate = [piece for piece in pieces if piece.position[idx] == row]

    # Rotate pieces
    rotated_pieces = rotate_pieces(pieces_to_rotate, axis, k)

    # Assemble the rotated pieces back the original list
    for i, piece in enumerate(pieces):
        if piece in pieces_to_rotate:
            pieces[i] = rotated_pieces.pop(0)

    return pieces

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

# %% Rotate the cube
# Rotate the cube 90 degrees around the x-axis
pieces = rotate_side(pieces, 'y', 0, 1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for piece in pieces:
    piece.plot(ax, style)

ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_zlim(-8, 8)
plt.show()