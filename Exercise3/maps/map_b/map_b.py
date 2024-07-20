import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

WALL = 0
TRACK = 1
START = 2
FINISH = 3
 
# Define an array of 30 columns and 30 rows
track = np.full((30, 30), WALL)

track[20:28, 3:9] = TRACK
track[18:20, 4] = TRACK
track[17:20, 5] = TRACK
track[16:20, 6] = TRACK
track[15:20, 7:9] = TRACK
track[15:22, 9] = TRACK
track[15:21, 10] = TRACK
track[15:20, 11:19] = TRACK
track[2:13, 21:27] = TRACK
track[13:17, 26] = TRACK
track[13:18, 25] = TRACK
track[13:19, 24] = TRACK
track[13:20, 20:24] = TRACK
track[14:20, 19] = TRACK


# Define starting line (a horizontal line on the left)
track[28, 3:9] = START

# Define finishing line below the wall
track[1, 21:27] = FINISH

print(track)

# Save track in npy format
with open('./maps/saved_tracks/track_b.npy', 'wb') as f:
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