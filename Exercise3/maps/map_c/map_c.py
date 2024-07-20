import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

WALL = 0
TRACK = 1
START = 2
FINISH = 3
 
# Define an array of 30 rows and 60 columns for a wide racetrack
track = np.full((100, 30), WALL)

# Define the cosine curve track
x = np.linspace(0, 4 * np.pi, 100)  # x values from 0 to 4*pi
y = (np.cos(x) * 10 + 15).astype(int)  # map cosine values to rows 0 to 29

for i in range(98):
    track[i+1, y[i+1] - 2:y[i+1] + 4] = TRACK  # make the track wider

# Define starting line
track[99, 22:28] = START

# Define finishing line
track[0, 22:28] = FINISH

print(track)

# Save track in npy format
with open('./saved_tracks/track_c.npy', 'wb') as f:
    np.save(f, track)
    
# Personalized color map
cmap = ListedColormap(['#FFFFFF', '#CECFCE', '#FA9884', '#93C695'])

# Create the plot
plt.imshow(track, cmap=cmap)

# Set a title
plt.title('Map C')

# Set the ticks
plt.tight_layout()

# Save the plot
plt.savefig('map_c.png')

# Show the plot
plt.show()