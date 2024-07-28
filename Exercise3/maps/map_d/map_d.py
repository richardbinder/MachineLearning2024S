import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

WALL = 0
TRACK = 1
START = 2
FINISH = 3
 
# Define an array of 30 columns and 30 rows
track = np.full((15, 20), WALL)

track[12:16, 3:16] = TRACK
track[10:12, 3:9] = TRACK
track[8:10, 4] = TRACK
track[7:10, 5] = TRACK
track[6:10, 6] = TRACK
track[5:10, 7:9] = TRACK
track[5:12, 9] = TRACK
track[5:11, 10] = TRACK
track[0:10, 11:16] = TRACK


# Define starting line (a horizontal line on the left)
track[12:16, 15] = START

# Define finishing line below the wall
track[0, 11:16] = FINISH

print(track)

# Save track in npy format
with open('../saved_tracks/track_d.npy', 'wb') as f:
    np.save(f, track)
    
# Personalized color map
cmap = ListedColormap(['#FFFFFF', '#CECFCE', '#FA9884', '#93C695'])

# Create the plot
plt.imshow(track, cmap=cmap)

# Set a title
plt.title('Map D')

# Set the ticks
plt.tight_layout()

# Save the plot
plt.savefig('map_d.png')

# Show the plot
plt.show()