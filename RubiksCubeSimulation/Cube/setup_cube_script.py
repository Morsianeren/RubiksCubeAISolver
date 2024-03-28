from Piece import Piece

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