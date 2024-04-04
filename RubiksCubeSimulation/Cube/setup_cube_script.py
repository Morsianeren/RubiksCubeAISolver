from .Piece import Piece

# This script manually initializes the pieces' position, 
# orientation and color of the cube

def setup_cube():
    pieces = set()

    # Intialize the pieces and orientation
    # Rules: 
    # For center pieces the x direction must have a color
    # Next, for egde pieces, the y direction must have a color
    # Last for corner pieces, the z direction must have a color

    x = 1
    piece = Piece(x, -1,1) # Front 0
    piece.rotate('x', 1, rotate_position=False)
    piece.colors['x'] = 'white'
    piece.colors['y'] = 'blue'
    piece.colors['z'] = 'red'
    pieces.add(piece)
    piece = Piece(x, 0, 1) # Front 1
    piece.rotate('x', 1, rotate_position=False)
    piece.colors['x'] = 'white'
    piece.colors['y'] = 'blue'
    pieces.add(piece)
    piece = Piece(x, 1, 1) # Front 2
    piece.colors['x'] = 'white'
    piece.colors['y'] = 'orange'
    piece.colors['z'] = 'blue'
    pieces.add(piece)
    piece = Piece(x, -1, 0) # Front 3
    piece.rotate('x', 2, rotate_position=False)
    piece.colors['x'] = 'white'
    piece.colors['y'] = 'red'
    pieces.add(piece)
    piece = Piece(x, 0, 0) # Front 4
    piece.colors['x'] = 'white'
    pieces.add(piece)
    piece = Piece(x, 1, 0) # Front 5
    piece.colors['x'] = 'white'
    piece.colors['y'] = 'orange'
    pieces.add(piece)
    piece = Piece(x, -1, -1) # Front 6
    piece.rotate('x', 2, rotate_position=False)
    piece.colors['x'] = 'white'
    piece.colors['y'] = 'red'
    piece.colors['z'] = 'green'
    pieces.add(piece)
    piece = Piece(x, 0, -1) # Front 7
    piece.rotate('x', -1, rotate_position=False)
    piece.colors['x'] = 'white'
    piece.colors['y'] = 'green'
    pieces.add(piece)
    piece = Piece(x, 1, -1) # Front 8
    piece.rotate('x', -1, rotate_position=False)
    piece.colors['x'] = 'white'
    piece.colors['y'] = 'green'
    piece.colors['z'] = 'orange'
    pieces.add(piece)

    x = 0
    piece = Piece(x, -1, 1) # Middle 0
    piece.rotate('z', -1, rotate_position=False)
    piece.rotate('y', -1, rotate_position=False)
    piece.colors['x'] = 'red'
    piece.colors['y'] = 'blue'
    pieces.add(piece)
    piece = Piece(x, 0, 1) # Middle 1
    piece.rotate('y', -1, rotate_position=False)
    piece.colors['x'] = 'blue'
    pieces.add(piece)
    piece = Piece(x, 1, 1) # Middle 2
    piece.rotate('y', -1, rotate_position=False)
    piece.colors['x'] = 'blue'
    piece.colors['y'] = 'orange'
    pieces.add(piece)
    piece = Piece(x, -1, 0) # Middle 3
    piece.rotate('z', -1, rotate_position=False)
    piece.colors['x'] = 'red'
    pieces.add(piece)
    piece = Piece(x, 0, 0) # Middle 4
    pieces.add(piece)
    piece = Piece(x, 1, 0) # Middle 5
    piece.rotate('z', 1, rotate_position=False)
    piece.colors['x'] = 'orange'
    pieces.add(piece)
    piece = Piece(x, -1, -1) # Middle 6
    piece.rotate('z', -1, rotate_position=False)
    piece.rotate('y', 1, rotate_position=False)
    piece.colors['x'] = 'red'
    piece.colors['y'] = 'green'
    pieces.add(piece)
    piece = Piece(x, 0, -1) # Middle 7
    piece.rotate('y', 1, rotate_position=False)
    piece.colors['x'] = 'green'
    pieces.add(piece)
    piece = Piece(x, 1, -1) # Middle 8
    piece.rotate('y', 1, rotate_position=False)
    piece.colors['x'] = 'green'
    piece.colors['y'] = 'orange'
    pieces.add(piece)

    x = -1
    piece = Piece(x, -1, 1) # Back 0
    piece.rotate('z', 2, rotate_position=False)
    piece.colors['x'] = 'yellow'
    piece.colors['y'] = 'red'
    piece.colors['z'] = 'blue'
    pieces.add(piece)
    piece = Piece(x, 0, 1) # Back 1
    piece.rotate('z', 2, rotate_position=False)
    piece.rotate('x', -1, rotate_position=False)
    piece.colors['x'] = 'yellow'
    piece.colors['y'] = 'blue'
    pieces.add(piece)
    piece = Piece(x, 1, 1) # Back 2
    piece.rotate('z', 2, rotate_position=False)
    piece.rotate('x', -1, rotate_position=False)
    piece.colors['x'] = 'yellow'
    piece.colors['y'] = 'blue'
    piece.colors['z'] = 'orange'
    pieces.add(piece)
    piece = Piece(x, -1, 0) # Back 3
    piece.rotate('z', 2, rotate_position=False)
    piece.colors['x'] = 'yellow'
    piece.colors['y'] = 'red'
    pieces.add(piece)
    piece = Piece(x, 0, 0) # Back 4
    piece.rotate('z', 2, rotate_position=False)
    piece.colors['x'] = 'yellow'
    pieces.add(piece)
    piece = Piece(x, 1, 0) # Back 5
    piece.rotate('z', 1, rotate_position=False)
    piece.colors['x'] = 'orange'
    piece.colors['y'] = 'yellow'
    pieces.add(piece)
    piece = Piece(x, -1, -1) # Back 6
    piece.rotate('z', 2, rotate_position=False)
    piece.rotate('x', 1, rotate_position=False)
    piece.colors['x'] = 'yellow'
    piece.colors['y'] = 'green'
    piece.colors['z'] = 'red'
    pieces.add(piece)
    piece = Piece(x, 0, -1) # Back 7
    piece.rotate('z', 1, rotate_position=False)
    piece.rotate('x', -1, rotate_position=False)
    piece.colors['x'] = 'green'
    piece.colors['y'] = 'yellow'
    pieces.add(piece)
    piece = Piece(x, 1, -1) # Back 8
    piece.rotate('z', 1, rotate_position=False)
    piece.rotate('x', -1, rotate_position=False)
    piece.colors['x'] = 'green'
    piece.colors['y'] = 'yellow'
    piece.colors['z'] = 'orange'
    pieces.add(piece)

    for piece in pieces:
        # Reset initial orientation
        piece.reset_initial_state()

    return pieces

# The following dictionary is a lookup table used to 
# transpose the pieces position + color vector into
# an array.
#
# The order is front (white), back (yellow), right (orange), 
# left (red), up (blue), down (green)
# Looking at the front of the cube (white with blue top)
# the numbering of the colors will be the following:
# 0 1 2
# 3 4 5
# 6 7 8
# Note that looking at the blue face the top will be yellow
# and looking at the bottom face the top will be white 

POINT_TO_INDEX = {
    ( 2, -1,  1):   (0, 0), # Front
    ( 2,  0,  1):   (0, 1),
    ( 2,  1,  1):   (0, 2),
    ( 2, -1,  0):   (0, 3),
    ( 2,  0,  0):   (0, 4),
    ( 2,  1,  0):   (0, 5),
    ( 2, -1, -1):   (0, 6),
    ( 2,  0, -1):   (0, 7),
    ( 2,  1, -1):   (0, 8),

    (-2,  1,  1):   (1, 0), # Back
    (-2,  0,  1):   (1, 1),
    (-2, -1,  1):   (1, 2),
    (-2,  1,  0):   (1, 3),
    (-2,  0,  0):   (1, 4),
    (-2, -1,  0):   (1, 5),
    (-2,  1, -1):   (1, 6),
    (-2,  0, -1):   (1, 7),
    (-2, -1, -1):   (1, 8),

    ( 1,  2,  1):   (2, 0), # Right
    ( 0,  2,  1):   (2, 1),
    (-1,  2,  1):   (2, 2),
    ( 1,  2,  0):   (2, 3),
    ( 0,  2,  0):   (2, 4),
    (-1,  2,  0):   (2, 5),
    ( 1,  2, -1):   (2, 6),
    ( 0,  2, -1):   (2, 7),
    (-1,  2, -1):   (2, 8),

    (-1, -2,  1):   (3, 0), # Left
    ( 0, -2,  1):   (3, 1),
    ( 1, -2,  1):   (3, 2),
    (-1, -2,  0):   (3, 3),
    ( 0, -2,  0):   (3, 4),
    ( 1, -2,  0):   (3, 5),
    (-1, -2, -1):   (3, 6),
    ( 0, -2, -1):   (3, 7),
    ( 1, -2, -1):   (3, 8),

    (-1, -1,  2):   (4, 0), # Up
    (-1,  0,  2):   (4, 1),
    (-1,  1,  2):   (4, 2),
    ( 0, -1,  2):   (4, 3),
    ( 0,  0,  2):   (4, 4),
    ( 0,  1,  2):   (4, 5),
    ( 1, -1,  2):   (4, 6),
    ( 1,  0,  2):   (4, 7),
    ( 1,  1,  2):   (4, 8),

    ( 1, -1, -2):   (5, 0), # Down
    ( 1,  0, -2):   (5, 1),
    ( 1,  1, -2):   (5, 2),
    ( 0, -1, -2):   (5, 3),
    ( 0,  0, -2):   (5, 4),
    ( 0,  1, -2):   (5, 5),
    (-1, -1, -2):   (5, 6),
    (-1,  0, -2):   (5, 7),
    (-1,  1, -2):   (5, 8),
}