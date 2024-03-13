# %%
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
try:
    from Quaternions.Quaternion import Quaternion
except ModuleNotFoundError:
    from Quaternion import Quaternion

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
        self.colors = {'x': None, 'y': None, 'z': None,
                       '-x': None, '-y': None, '-z': None}

    def rotate(self, axis: Literal['x', 'y', 'z'], k: int) -> None:
        """Rotate the piece k*90 degrees around the given axis."""
        cos_part = np.cos(k * np.pi / 4)
        sin_part = np.sin(k * np.pi / 4)
        if axis == 'x':
            rotation_quaternion = Quaternion([cos_part, sin_part, 0, 0])
            #self.orientation = Quaternion([0, 1, 0, 0]) * self.orientation
        elif axis == 'y':
            rotation_quaternion = Quaternion([cos_part, 0, sin_part, 0])
            #self.orientation = Quaternion([0, 0, 1, 0]) * self.orientation
        elif axis == 'z':
            rotation_quaternion = Quaternion([cos_part, 0, 0, sin_part])
            #self.orientation = Quaternion([0, 0, 0, 1]) * self.orientation
        else:
            raise ValueError("Invalid axis. It should be 'x', 'y' or 'z'.")
        self.orientation = rotation_quaternion * self.orientation
        self.position = rotate_coordinates_90_degrees(self.position, axis, k)
        
    def reset(self) -> None:
        """Reset the piece to its initial position and orientation."""
        self.orientation = self._INITIAL_ORIENTATION
        self.position = self._INITIAL_POSITION

    def plot(self, ax, style:Literal['square', 'arrows'] = 'square', **kwargs) -> None:
        """Plot the piece in 3D space."""
        x_vector, y_vector, z_vector = self.orientation.get_xyz_vectors()
        vectors = [x_vector, y_vector, z_vector, -x_vector, -y_vector, -z_vector]
        colors = self.colors.values()

        iterations = 0

        if style == 'square':
            for vector, color in zip(vectors, colors):
                if color is None:
                    continue
                iterations += 1
                center = self.position - vector * 0.5
                vertices = get_translated_vertices(vector, center)
                ax.add_collection3d(Poly3DCollection([vertices], facecolors=color, edgecolors="black", **kwargs))
        elif style == 'arrows':
            for vector, color in zip(vectors, colors):
                if color is None:
                    continue    
                iterations += 1
                center = self.position + vector
                ax.quiver(center[0], center[1], center[2], vector[0], vector[1], vector[2], color=color, **kwargs)

        if iterations == 0:
            print(f"Warning: No colors were set for piece {self.position} and therefore it will not be plotted.")

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
    side_length = 1
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

# Function from ChatGPT
def rotate_coordinates_90_degrees(coordinates, axis:Literal['x', 'y', 'z'], k:int):
    """
    Rotate XYZ coordinates k*90 degrees around a given axis.

    Parameters:
        coordinates: numpy.ndarray
            The input coordinates in the shape (3, N), where N is the number of points.
        axis: int
            The axis around which to rotate the coordinates (0 for x, 1 for y, 2 for z).

    Returns:
        numpy.ndarray
            The rotated coordinates.
    """

    # Define rotation matrices for 90-degree rotations around each axis
    if axis == 'x':  # Rotate around X axis
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, 0, -1],
                                    [0, 1, 0]])
    elif axis == 'y':  # Rotate around Y axis
        rotation_matrix = np.array([[0, 0, 1],
                                    [0, 1, 0],
                                    [-1, 0, 0]])
    elif axis == 'z':  # Rotate around Z axis
        rotation_matrix = np.array([[0, -1, 0],
                                    [1, 0, 0],
                                    [0, 0, 1]])
    else:
        raise ValueError(f"{axis} is invalid axis. It should be 'x', 'y' or 'z'.")

    # Reduce k to effective number rotations
    # (e.g., 4 rotations is equivalent to 0 rotation, -5 to 3 rotations, etc.)
    k %= 4

    # Apply rotation matrix to coordinates
    rotated_coordinates = coordinates.copy()
    for _ in range(k):
        rotated_coordinates = np.dot(rotation_matrix, rotated_coordinates)

    return rotated_coordinates


# Test code

# Create a piece
#piece = Piece(4, 4, 0)
#
#print("Position before rotation:", piece.position)
#piece.rotate('x')
#print("Position after rotation:", piece.position)
#
## Set the colors
#piece.colors['x'] = "red"
#piece.colors['y'] = "green"
#piece.colors['z'] = "blue"
#
## Plot the piece
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#piece.plot(ax)
#ax.set_xlim(-10, 10)
#ax.set_ylim(-10, 10)
#ax.set_zlim(-10, 10)
#plt.show()