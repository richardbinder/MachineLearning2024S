import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

WALL = 0
TRACK = 1
START = 2
FINISH = 3
 
# Define an array of 30 columns and 30 rows
track = np.full((30, 30), WALL)

# Define the circle-like track
for i in range(6, 24):
    track[i, 3:9] = TRACK
    track[i, 21:27] = TRACK

# Top Left Corner
track[4:6, 4] = TRACK
track[3:6, 5] = TRACK
track[2:6, 6] = TRACK
track[1:6, 7:9] = TRACK
track[1:8, 9] = TRACK
track[1:7, 10] = TRACK
track[1:6, 11:15] = TRACK

# Top Right Corner
track[4:6, 25] = TRACK
track[3:6, 24] = TRACK
track[2:6, 23] = TRACK
track[1:6, 21:23] = TRACK
track[1:8, 20] = TRACK
track[1:7, 19] = TRACK
track[1:6, 14:19] = TRACK

# Bottom Left Corner
track[23:25, 4] = TRACK
track[23:26, 5] = TRACK
track[23:27, 6] = TRACK
track[23:28, 7:9] = TRACK
track[21:28, 9] = TRACK
track[22:28, 10] = TRACK
track[23:28, 11:15] = TRACK

# Bottom Right Corner
track[23:25, 25] = TRACK
track[23:26, 24] = TRACK
track[23:27, 23] = TRACK
track[23:28, 21:23] = TRACK
track[21:28, 20] = TRACK
track[22:28, 19] = TRACK
track[23:28, 14:19] = TRACK

# Define starting line (a horizontal line on the left)
track[16, 3:9] = START

# Define wall below the starting line
track[15, 3:9] = WALL

# Define finishing line below the wall
track[14, 3:9] = FINISH

print(track)

# Save track in npy format
with open('track_b.npy', 'wb') as f:
    np.save(f, track)
    
# Personalized color map
cmap = ListedColormap(['#FFFFFF', '#CECFCE', '#FA9884', '#93C695'])

# Create the plot
plt.imshow(track, cmap=cmap)

# Set a title
plt.title('Map B')

# Set the ticks
plt.tight_layout()

# Save the plot
plt.savefig('map_b.png')

# Show the plot
plt.show()