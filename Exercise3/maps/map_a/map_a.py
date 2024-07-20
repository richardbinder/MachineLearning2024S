import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

WALL = 0
TRACK = 1
START = 2
FINISH = 3
 
# Define an array of 17 columns and 30 rows
track = np.full((30, 17), WALL)

# Define starting line from colum 3 to 8
track[-1, 3:9] = START

# Define finishing line last column, last 6 rows
track[0:6, 16] = FINISH

# Define the track
track[4:14,0] = TRACK
track[3:22,1] = TRACK
track[1:27,2] = TRACK
track[0:29, 3:9] = TRACK
track[0:7, 9] = TRACK
track[0:6, 10:16] = TRACK

print(track)

# Save track in npy format
with open('./saved_tracks/track_a.npy', 'wb') as f:
    np.save(f, track)
    
# Personalized color map
cmap = ListedColormap(['#FFFFFF', '#CECFCE', '#FA9884', '#93C695'])

# Create the plot
plt.imshow(track, cmap=cmap)

# Set a title
plt.title('Map A')

# Set the ticks
plt.tight_layout()

# Save the plot
plt.savefig('map_a.png')

# Show the plot
plt.show()