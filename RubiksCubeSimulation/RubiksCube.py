# In this class i will use quaternions to keep track of the orientation of the pieces!

# " Quaternions came from Hamilton after his really good work had been done; 
# and, though beautifully ingenious, 
# have been an unmixed evil to those who have touched them in any way, including Clerk Maxwell"
# - Lord Kelvin

# %% 
from Piece import Piece
import matplotlib.pyplot as plt
import numpy as np

# Create a empty array to store the pieces
pieces = np.zeros((3, 3, 3), dtype=Piece)

# Initialize the pieces
for x, y, z in np.ndindex(3, 3, 3):
    pieces[x, y, z] = Piece((x-1)*4, (y-1)*4, (z-1)*4)

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