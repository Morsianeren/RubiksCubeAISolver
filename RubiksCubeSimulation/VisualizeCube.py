# %%
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Cube import Cube, EdgePiece, CornerPiece, MiddlePiece, CorePiece

cube = Cube()

figure = plt.figure()
ax = figure.add_subplot(111, projection='3d')


def get_vertices(face_vector, coordinate):
    side_length = 1
    # First create a surface with respect to the face vector.
    # This will give us a square with the correct orientation
    # Later we will transpose the surface to the correct position

    # To create the square, we need two orthogonal vectors
    # Since our face vector is always 1 in one of the coordinates,
    # we can use the other two coordinates as the orthogonal vectors

    u = None
    w = None

    for i, val in enumerate(face_vector):
        o_vector = [0, 0, 0]
        if val == 0:
            o_vector[i] = 1
            if u is None:
                u = o_vector
            else:
                w = o_vector
                break
    
    # Get the unit vectors
    u_unit = u / np.linalg.norm(u)
    w_unit = w / np.linalg.norm(w)

    # Now we have the orthogonal vectors

    # Now we can create the vertices
    P0 = np.array([0, 0, 0])  # This can be adjusted as needed
    P1 = P0 + side_length * u_unit
    P2 = P1 + side_length * w_unit
    P3 = P0 + side_length * w_unit

    # Next we find a translation vector to move the square to the correct position
    # First we find the desired center of the square
    # This can be found by adding the face vector to the coordinate
    desired_center = [x + y for x, y in zip(face_vector, coordinate)]

    # Then we find the original midpoint of the square
    original_midpoint = (P0 + P1 + P2 + P3) / 4

    # Subtract these two and you have the translation vector
    translation_vector = desired_center - original_midpoint

    # Move the square to the correct position
    vertices = np.array([P0, P1, P2, P3])
    vertices += translation_vector

    return vertices

def get_color(face_vector) -> str:
    up_vector = np.array([0, 0, 1])
    down_vector = np.array([0, 0, -1])
    right_vector = np.array([1, 0, 0])
    left_vector = np.array([-1, 0, 0])
    front_vector = np.array([0, 1, 0])
    back_vector = np.array([0, -1, 0])
    # Figure out face color
    unit_vector = face_vector / np.linalg.norm(face_vector)

    if np.allclose(unit_vector, up_vector):
        return "white"
    elif np.allclose(unit_vector, down_vector):
        return "yellow"
    elif np.allclose(unit_vector, right_vector):
        return "green"
    elif np.allclose(unit_vector, left_vector):
        return "blue"
    elif np.allclose(unit_vector, front_vector):
        return "red"
    elif np.allclose(unit_vector, back_vector):
        return "orange"
    else:
        raise ValueError("Invalid face vector: " + str(face_vector))

def rotation_matrix(axis, theta):
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    return np.array([[a*a + b*b - c*c - d*d, 2 * (b*c - a*d), 2 * (b*d + a*c)],
                     [2 * (b*c + a*d), a*a + c*c - b*b - d*d, 2 * (c*d - a*b)],
                     [2 * (b*d - a*c), 2 * (c*d + a*b), a*a + d*d - b*b - c*c]])


# Draw the cube
for x, y, z in np.ndindex(cube.cube.shape):
    # First we get the current vector/coordinate and the corresponding piece
    v_current = (x, y, z)
    
    piece = cube.cube[v_current]
    
    # Get the original vector of the piece
    # This is used to identify colors
    v_initial = list(x - 1 for x in piece.matrix_coordinate)

    # Split each 1 into individual vectors
    # Example: A corner piece will have 3 vectors, one for each face/color
    face_vectors = list()
    for i, val in enumerate(v_initial):
        face = [0, 0, 0]
        if abs(val) == 1:
            face[i] = val
            face_vectors.append(face)
            continue

    # If the piece is a core piece, it will have no face vectors
    if len(face_vectors) == 0:
        continue

    # Loop for each face vector
    for face_vector in face_vectors:
        # First identify the colors of the faces
        color = get_color(face_vector)

        # Next we get the vertices of the square
        vertices = get_vertices(face_vector, v_initial) # TODO - currently uses the initial vector, should use the current vector

        # Add the square to the plot
        print("Plotting square: " + str(vertices) + " with color: " + color)
        ax.add_collection3d(Poly3DCollection([vertices], facecolors=color, edgecolors="black"))


# %%





# Calculate the rotation axis
axis = np.cross(v_initial, v_current) / np.linalg.norm(np.cross(v_initial, v_current))

# Calculate the rotation angle
angle = np.arccos(np.dot(v_initial, v_current) / (np.linalg.norm(v_initial) * np.linalg.norm(v_current)))

global_vector = list(x - 1 for x in current_coordinate)

v_unit = global_vector / np.linalg.norm(global_vector)

# From the global coordinate, we want to create a plane
# The plane must have a surface area of 1
# The plane must be perpendicular to the global vector

# Find a vector u that is orthogonal to v
# A simple method is to use the cross product with an arbitrary different vector
# If v is not parallel to the z-axis, we can use the z-axis for simplicity
if (np.allclose(v_unit, [0, 0, 1]) or np.allclose(v_unit, [0, 0, -1])) == False:
    arbitrary_vector = np.array([0, 0, 1])
else:
    # If v is parallel to the z-axis, use the x-axis instead
    arbitrary_vector = np.array([-1, 0, 0])

# First orthogonal vector u
u = np.cross(v_unit, arbitrary_vector)
u_unit = u / np.linalg.norm(u)

# Second orthogonal vector w, perpendicular to both v and u
w = np.cross(v_unit, u_unit)
w_unit = w / np.linalg.norm(w)

# Length of the square's side to have an area of 1 square cm
side_length = 1 ** 0.5  # Since area = 1 cm^2, side = sqrt(1)

# Assuming the origin (0, 0, 0) is on the plane, calculate the coordinates of the square's vertices
P0 = np.array([0, 0, 0])  # This can be adjusted as needed
P1 = P0 + side_length * u_unit
P2 = P1 + side_length * w_unit
P3 = P0 + side_length * w_unit

# Now we calculate the translation vector to move the square to the center
# of the global vector
desired_center = [x * 1.5 for x in global_vector]  # 1.5 is the distance from the center of the cube to the center of a face

original_midpoint = (P0 + P1 + P2 + P3) / 4

translation_vector = desired_center - original_midpoint

vertices = np.array([P0, P1, P2, P3])

vertices += translation_vector

# Get color
face_vector = piece.matrix_coordinate
color = get_color(face_vector)

ax.add_collection3d(Poly3DCollection([vertices], facecolors=color, edgecolors="black"))

#print("Global vector")
#print(global_vector)
#print("Coordinates of the square's vertices")
#print(vertices)
    

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
# %%
