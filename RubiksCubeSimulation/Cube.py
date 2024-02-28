#%%
import numpy as np
from typing import Literal
from abc import ABC, abstractmethod


class Piece(ABC):
    def __init__(self, letter, name = None):
        self.name = name
        self.letter = letter

    def __str__(self):
        return str(self.name) if self.name is not None else self.letter
         

class EdgePiece(Piece):
        def __init__(self, name = None):
            super().__init__("E", name)

class CornerPiece(Piece):
    def __init__(self, name = None):
            super().__init__("C", name)

class MiddlePiece(Piece):
    def __init__(self, name = None):
            super().__init__("M", name)
    
class CorePiece(Piece):
    def __init__(self, name = None):
            super().__init__("X", name)


class Cube():
    def __init__(self):

        # Create a 3D cube
        self.cube = np.full((3, 3, 3), [1], dtype=object)

        # Iterate over the cube
        for x, y, z in np.ndindex(self.cube.shape):
            # Classify the piece
            self.cube[x, y, z] = self._classify_piece(x, y, z)
            # Make the values of the cube unique (for testing purposes)
            self.cube[x, y, z].matrix_coordinate = (x, y, z)

        self.org_cube = self.cube.copy() # Save the original state of the cube

    def __str__(self):
        counter = 1
        string = ""
        for x, y, z in np.ndindex(self.cube.shape):
            string += str(self.cube[x, y, z]) + " "
            if counter % 9 == 0:
                string += "\n\n"
                counter = 0
            elif counter % 3 == 0:
                string += "\n"
            counter += 1

        return string
    
    def rotate_center(self, axis: Literal["x", "y", "z"], direction: Literal["cw", "ccw"]):
        """Function to rotate the center of the cube like a rubiks cube

        Args:
            axis (Literal["x", "y", "z"]): Which axis to rotate
            direction (Literal["cw", "ccw"]): Direction of the rotation. Clockwise (cw) or counter clockwise (ccw)
        """
        axis_dict = {
            "x": (1, 2),
            "y": (0, 2),
            "z": (0, 1)
        }
        self.cube = np.rot90(self.cube, axes=axis_dict[axis], k=1 if direction == "cw" else -1)


    def rotate_edge(self):
        pass

    def _classify_piece(self, x, y, z):
        ones_count = sum(x == 1 for x in (x, y, z))

        if ones_count == 0:
            return CornerPiece()
        elif ones_count == 1:
            return EdgePiece()
        elif ones_count == 2:
            return MiddlePiece()
        elif ones_count == 3:
            return CorePiece()
        else:
            raise ValueError("Invalid piece")
        

 
cube = Cube()
print(cube)



