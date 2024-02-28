# %%
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Cube import Cube, EdgePiece, CornerPiece, MiddlePiece, CorePiece

cube = Cube()
print(cube)
cube.rotate_center("z", "cw")
#print(cube)
# %%

figure = plt.figure()
ax = figure.add_subplot(111, projection='3d')

def rotate_vector(face_vector: list, original_coordinate: list, new_coordinate: list):
    # TODO: Make this function so it can take a list of rotation angles
    # and rotate a face vector accordingly

    # The difference between the org_vector and the new_vector is the rotation
    # You can multiply the difference with 45 degrees to get the rotation angles
    # E.g. if the difference is [0, 0, 2], 
    # then the rotation is 90 degrees around the z-axis

    x_delta = new_coordinate[0] - original_coordinate[0]
    y_delta = new_coordinate[1] - original_coordinate[1]
    z_delta = new_coordinate[2] - original_coordinate[2]

    theta_x = 45 * x_delta
    theta_y = 45 * y_delta
    theta_z = 45 * z_delta

    Rx = np.array([[1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]])

    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]])
        
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1]])

    R_combined = np.dot(Rx, Ry, Rz)
    #print(R_combined)

    #face_vector = np.array([-1, 0, 0])
    point_a = original_coordinate
    point_b = [x + y for x, y in zip(original_coordinate, face_vector)]

    a_rotated = np.dot(R_combined, point_a)
    b_rotated = np.dot(R_combined, point_b)

    new_face_vector = b_rotated - a_rotated

    return new_face_vector


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
        if val == 0:
            o_vector[i] = 1
            if u is None:
                u = o_vector
            else:
                w = o_vector
                break
    
    # Get the unit vectors
    u_unit = u
    w_unit = w
    # NOTE: The vectors are already unit vectors

    # Now we have the orthogonal vectors

    # Now we can create the vertices
    P0 = np.array([0., 0., 0.])  # This can be adjusted as needed
    P1 = P0 + side_length * u_unit
    P2 = P1 + side_length * w_unit
    P3 = P0 + side_length * w_unit

    vertices = np.array([P0, P1, P2, P3])

    return vertices

def get_translated_vertices(direction_vector, new_center):
    
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
    # First we get the corresponding piece
    xyz = (x, y, z)
    piece = cube.cube[xyz]

    # Get the original vector of the piece and normalize it so the corepiece is at (0, 0, 0)
    # This is used to identify colors
    v_initial = list(x - 1 for x in piece.matrix_coordinate)
    v_current = [x - 1 for x in xyz]

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

    print("Face vectors:")
    print(face_vectors)

    # Loop for each face vector
    for face_vector in face_vectors:
        # First identify the colors of the faces
        color = get_color(face_vector)

        # Next we rotate the face vector to the correct position
        rotated_face_vector = rotate_vector(face_vector, v_initial, v_current)
        # rotated_face_vector = face_vector # TODO: Replace this with a function that rotates the face vector

        print("Orginial face vector: " + str(face_vector))
        print("Rotated face vector: " + str(rotated_face_vector))
        print("")

        # Next we get the vertices of the square
        try:
            vertices = get_translated_vertices(rotated_face_vector, v_initial)
        except:
            print("ERROR")
            break

        # Add the square to the plot
        #print("Plotting square: " + str(vertices) + " with color: " + color)
        ax.add_collection3d(Poly3DCollection([vertices], facecolors=color, edgecolors="black"))

ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
# %%

