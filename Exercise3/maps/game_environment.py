from matplotlib.pylab import randint
import numpy as np
from gymnasium import Env
import pygame

WALL = 0
TRACK = 1
START = 2
FINISH = 3

COLOR_WALL = '#FFFFFF'
COLOR_TRACK = '#CECFCE'
COLOR_START = '#FA9884'
COLOR_END = '#93C695'

COLOR_CAR = '#000000'

color_mapping = {
    WALL: COLOR_WALL,
    TRACK: COLOR_TRACK,
    START: COLOR_START,
    FINISH: COLOR_END
}

class RaceTrack(Env):
    def __init__(self, track_dir: str, track: str, size: int = 20):
        self.size = size

        track_name = 'track_' + track + '.npy'

        with open(track_dir + track_name, 'rb') as f:
            self.track_map = np.load(f)

        self.window_size = (self.track_map.shape[1] * self.size, self.track_map.shape[0] * self.size)
        self.window = None
        self.clock = None
        
        self.start_positions = np.dstack(np.where(self.track_map == START))[0]
        self.finish_positions = np.where(self.track_map == FINISH)

        # Observation space is the shape of the track map, # of speed in y direction (-4, 0), # of speed in x direction (-4, 4)
        self.observation_space = (*self.track_map.shape, 5, 9)

        # Action space is 9, as we have 9 possible actions
        self.action_space = 9

        self.car_position = None
        self.speed = (0, 0)
        
        # Define all possible speed acceleration events, as a tuple of (speed_y, speed_x)
        # 0: (-1, -1) means decrease speed in x and y directions
        self.speed_acceleration_events = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 0),
            5: (0, 1),
            6: (1, -1),
            7: (1, 0),
            8: (1, 1)
        }

    def render(self):
        """
        Render the environment
        """
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption('RL Race Track Game')

        if self.clock is None:
            self.clock = pygame.time.Clock()

        rows, cols = self.track_map.shape
        self.window.fill((255, 255, 255))

        for row in range(rows):
            for col in range(cols):
                
                # Fill the cell based on its value
                fill = color_mapping.get(self.track_map[row, col])

                # Draw the fill color
                pygame.draw.rect(self.window, fill, (col * self.size, row * self.size, self.size, self.size), 0)

                # Draw white border around all cells
                pygame.draw.rect(self.window, COLOR_WALL, (col * self.size, row * self.size, self.size, self.size), 1)

        if self.car_position is not None:
            # Draw the car
            pygame.draw.rect(self.window, COLOR_CAR, (self.car_position[1] * self.size, self.car_position[0] * self.size, self.size, self.size), 0)

        pygame.display.update()


    # Get position and speed of the car
    def _car_state(self):
       return (*self.car_position, *self.speed)


    # Check if the car has reached the finish line
    def _check_finish(self):
        rows = self.finish_positions[0]
        col = self.finish_positions[1][0]
        if self.car_position[0] in rows and self.car_position[1] >= col:
            return True
        return False


    # Check if the car has hit a wall in order to reset the game
    def _check_wall_hit(self, next_position):
        row, col = next_position
        map_height, map_width = self.track_map.shape

        # Check if the car is out of bounds
        if not (0 <= row < map_height and 0 <= col < map_width):
            return True

        # Check if the car has hit a wall
        if self.track_map[row, col] == WALL:
            return True
            
        # Check if the path from current position to next position hits a wall
        if self._path_hits_wall(self.car_position, next_position):
            return True

        return False


    def _path_hits_wall(self, current_position, next_position):
        current_row, current_col = current_position
        next_row, next_col = next_position

        # Check vertical path
        # Step is 1 if the next row is bigger than the current row, otherwise -1. This is used to determine the direction of the loop
        step = 1 if next_row > current_row else -1
        for row_step in range(current_row, next_row, step):
            if self.track_map[row_step, current_col] == WALL:
                return True

        # Check horizontal path
        # Step is 1 if the next row is bigger than the current row, otherwise -1. This is used to determine the direction of the loop
        step = 1 if next_col > current_col else -1
        for col_step in range(current_col, next_col, step):
            if self.track_map[next_row, col_step] == WALL:
                return True

        return False


    # Execute an action (increase or decrease speed in x and y directions) and update the car position
    def step(self, action):
            # Get current car position
            new_car_position = np.copy(self.car_position)
            
            # Get the speed acceleration based on the action
            y_speed, x_speed = self.speed_acceleration_events[action]
            
            # With a probability of 0.1, the speed is set to 0 in both directions
            if np.random.rand() < 0.1:
                y_speed = 0
                x_speed = 0
            
            # Calculate the new speed
            temp_y_speed = self.speed[0] + y_speed
            temp_x_speed = self.speed[1] + x_speed
            
            # Y speed is limited between -4 and 0
            # Positive values mean movement to the bottom, negative values mean movement to the top
            # No backward movement is allowed so Y cannot be bigger than 0
            temp_y_speed = max(-4, min(temp_y_speed, 0))  # Y speed is limited between -4 and 0 (no backward movement)
            
            # X speed is limited between -4 and 4
            # Positive values mean movement to the right, negative values mean movement to the left
            temp_x_speed = max(-4, min(temp_x_speed, 4))
            
            # Calculate the new car position based on the new speed in x and y directions
            new_car_position[0] += temp_y_speed
            new_car_position[1] += temp_x_speed
            
            goal_reached = False
            
            # Check if the car has reached the finish line
            if self._check_finish():
                goal_reached = True 

            # Check if the car has hit a wall
            elif self._check_wall_hit(new_car_position):
                self.reset()

            # If the car has not hit a wall or reached the finish line, update the car position and speed
            else:
                self.car_position = new_car_position
                self.speed = (temp_y_speed, temp_x_speed)

            # Return the observation, reward, done, and info
            return self._car_state(), -1, goal_reached


    def reset(self, start_index=-1):
        # Get a random starting position from the list of starting positions
        if start_index == -1:
            # If start_index is -1, get a random starting position
            start_idx = np.random.choice(self.start_positions.shape[0])
        else:
            # If start_index is not -1, use the provided start_index
            start_idx = start_index
        
        # Set the car position to the starting position
        self.car_position = self.start_positions[start_idx]
        
        # Set the speed to 0 in both directions
        self.speed = (0, 0)
        
        # Return the car state (position and speed)
        return self._car_state()


# For testing, execute the environment with random actions until the goal is reached
if __name__ == "__main__":
    race_track = RaceTrack(track_dir='./saved_tracks/', track='a', size=20)
    #race_track = RaceTrack(track_dir='./saved_tracks/', track='c', size=10)
    
    # Initialize state
    race_track.reset()

    # Initialize pygame
    pygame.init()
    
    # Set up the display window
    race_track.window = pygame.display.set_mode(race_track.window_size)
    
    # Initialize the clock
    race_track.clock = pygame.time.Clock()

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Select a random action
        action = np.random.choice(race_track.action_space)
        
        # Get the observation, reward, and done values after taking the action
        observation, reward, done = race_track.step(action)
        print(observation)

        if done:
            print("Goal reached!")
            running = False

        # Render the environment
        race_track.render()
        
        # Control the frame rate (frames per second)
        race_track.clock.tick(960)

    pygame.quit()