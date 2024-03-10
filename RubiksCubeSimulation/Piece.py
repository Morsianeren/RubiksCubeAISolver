# %%
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from Quaternion import Quaternion
try:
    from Quaternions.Quaternion import Quaternion
except ModuleNotFoundError:
    from Quaternion import Quaternion

# Constants to rotate 90 degrees
COS_ROTATION = np.cos(np.pi / 4)
SIN_ROTATION = np.sin(np.pi / 4)

class Piece():
    """This class represents a piece of the Rubik's Cube. 
    It has a position and orientation."""
    def __init__(self, x_pos:int, y_pos:int, z_pos:int) -> None:
        # The 
        self.orientation = Quaternion([1, 0, 0, 0])
        self.position = np.array([x_pos, y_pos, z_pos], dtype=int)
        # Used to make sure the piece is in the correct orientation
        self._INITIAL_ORIENTATION = Quaternion([1, 0, 0, 0]) 
        # Used to make sure the piece is in the correct position
        self._INITIAL_POSITION = np.array([x_pos, y_pos, z_pos], dtype=int)

        # The colors of the piece
        self.colors = {'x': None, 'y': None, 'z': None}

    def rotate(self, axis: Literal['x', 'y', 'z']) -> None:
        """Rotate the piece 90 degrees around the given axis."""
        sqrt2 = np.sqrt(2)
        if axis == 'x':
            rotation_quaternion = Quaternion([COS_ROTATION, SIN_ROTATION, 0, 0])
            #self.orientation = Quaternion([0, 1, 0, 0]) * self.orientation
        elif axis == 'y':
            self.orientation = Quaternion([0, 0, 1, 0]) * self.orientation
        elif axis == 'z':
            self.orientation = Quaternion([0, 0, 0, 1]) * self.orientation
        else:
            raise ValueError("Invalid axis. It should be 'x', 'y' or 'z'.")
        self.orientation = rotation_quaternion * self.orientation
        
    def reset(self) -> None:
        """Reset the piece to its initial position and orientation."""
        self.orientation = self._INITIAL_ORIENTATION
        self.position = self._INITIAL_POSITION

    def plot(self, ax) -> None:
        """Plot the piece in 3D space."""
        x_vector, y_vector, z_vector = self.orientation.get_xyz_vectors()
        vectors = [x_vector, y_vector, z_vector]
        colors = self.colors.values()

        iterations = 0

        for vector, color in zip(vectors, colors):
            if color is None:
                continue
            iterations += 1
            center = self.position + vector
            vertices = get_translated_vertices(vector, center)
            ax.add_collection3d(Poly3DCollection([vertices], facecolors=color, edgecolors="black"))

        if iterations == 0:
            print("Warning: No colors were set for the piece and therefore it will not be plotted.")

def get_translated_vertices(direction_vector, new_center):
    """This function generates the vertices of a square with a specified center and orientation.

    Args:
        direction_vector (np.array): The vector that the square will be orthogonal to
        new_center (np.array): The center of the square in a 3D coordinate system

    Returns:
        np.array: An array with 4 points, each defining the edge of the square
    """
    # First get the vertices
    vertices = get_vertices(direction_vector)

    # Next we find a translation vector to move the square to the correct position
    # First we find the desired center of the square
    # This can be found by adding the face vector to the coordinate
    desired_center = [x + y for x, y in zip(direction_vector, new_center)]

    # Then we find the original midpoint of the square
    vertex_sum = np.sum(vertices, axis=0)
    original_midpoint = vertex_sum / 4

    # Subtract these two and you have the translation vector
    translation_vector = desired_center - original_midpoint

    # Move the square to the correct position
    vertices += translation_vector

    return vertices

def get_vertices(direction_vector):
    """This function calculates the vertices for a square
    with respect to the direction vector, where the first vertix has center in origo.

    Args:
        direction_vector (list): The square will be orthogonal to this vector

    Returns:
        np.array([list]): An array with 4 lists, 
        each list containing the x, y, z coordinates of a vertex
    """
    side_length = 4
    # First create a surface with respect to the face vector.
    # This will give us a square with the correct orientation
    # Later we will transpose the surface to the correct position

    # To create the square, we need two orthogonal vectors
    # Since our face vector is always 1 in one of the coordinates,
    # we can use the other two coordinates as the orthogonal vectors

    u = None
    w = None

    for i, val in enumerate(direction_vector):
        o_vector = [0, 0, 0]
        if int(val) == 0:
            o_vector[i] = 1
            if u is None:
                u = o_vector
            else:
                w = o_vector
                break
    
    # Get the unit vectors
    u_unit = u / np.linalg.norm(u)
    w_unit = w / np.linalg.norm(w)
    # NOTE: The vectors are already unit vectors

    # Now we have the orthogonal vectors

    # Now we can create the vertices
    P0 = np.array([0., 0., 0.])  # This can be adjusted as needed
    P1 = P0 + side_length * u_unit
    P2 = P1 + side_length * w_unit
    P3 = P0 + side_length * w_unit

    vertices = np.array([P0, P1, P2, P3])

    return vertices


# Test code

# Create a piece
# piece = Piece(0, 0, 0)
# piece.rotate('x')
# 
# # Set the colors
# piece.colors['x'] = "red"
# piece.colors['y'] = "green"
# piece.colors['z'] = "blue"
# 
# # Plot the piece
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# piece.plot(ax)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
# ax.set_zlim(-3, 3)
# plt.show()