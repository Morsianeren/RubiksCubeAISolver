# In this class i will use quaternions to keep track of the orientation of the pieces!

# " Quaternions came from Hamilton after his really good work had been done; 
# and, though beautifully ingenious, 
# have been an unmixed evil to those who have touched them in any way, including Clerk Maxwell"
# - Lord Kelvin

# %% 
from Piece import Piece
import matplotlib.pyplot as plt

# Create a array with coordinates for all the pieces
pieces = []
for x in range(-1, 2):
    for y in range(-1, 2):
        for z in range(-1, 2):
            pieces.append(Piece(x, y, z))

# Plot the pieces
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for piece in pieces:
    piece.plot(ax)