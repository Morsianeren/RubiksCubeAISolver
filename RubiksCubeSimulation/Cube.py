#%%
import numpy as np
from typing import Literal

class EdgePiece():
        def __init__(self):
            pass

        def __str__(self):
            return "E"

class CornerPiece():
    def __init__(self):
        pass
    def __str__(self):
        return "C"

class MiddlePiece():
    def __init__(self):
        pass
    def __str__(self):
        return "M"
    
class CorePiece():
    def __init__(self) -> None:
        pass
    def __str__(self):
        return "X"

class Cube():
    def __init__(self):

        # Create a 3D cube
        self.cube = np.full((3, 3, 3), [1], dtype=object)

        # Iterate over the cube
        for x, y, z in np.ndindex(self.cube.shape):
            # Make the values of the cube unique
            #self.cube[x, y, z] = (x + 1) * 1 + (y + 1) * 10  + (z + 1) * 100

            # Classify the piece
            self.cube[x, y, z] = self._classify_piece(x, y, z)

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
        

 
cube = Cube()
print(cube)



