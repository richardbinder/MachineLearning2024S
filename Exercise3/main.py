import pickle
from matplotlib.colors import ListedColormap
import numpy as np

from matplotlib import pyplot as plt
from maps.game_environment import RaceTrack
from policies.off_policy_motecarlo import off_policy_monte_carlo
from scipy.ndimage import uniform_filter


# Plot the result
def plot_result(reward_hist, track_name, total_episodes) -> None:
    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(c='lightgray')

    x = np.arange(total_episodes)

    plt.plot(x, uniform_filter(reward_hist, size=20), 
                label=track_name,
                c="#2675CF",
                alpha=0.95)

    plt.title(track_name + ' training')
    plt.xlabel('Episodes)')
    plt.ylabel('Reward')    
    plt.legend()
    plt.savefig(f'./results/{"_".join(track_name.lower().split())}.png')
    plt.show()


if __name__ == "__main__":

    train = False # Switch between train and evaluation
    track_dir = "./maps/saved_tracks/" # Directory of the track files
    track = 'a' # Track name
    total_episodes = 1000000 # Total number of episodes
  
    if train:
        track_name = f'Track {track.capitalize()}'
        reward_hist, Q = off_policy_monte_carlo(total_episodes, track_dir, track)
        
        plot_result(reward_hist, track_name, total_episodes)

        with open(f'./results/training/track_{track}.pkl', 'wb') as f:
            pickle.dump(Q, f)

    else: # Evaluate the Q values and plot sample paths

        with open(f'./results/training/track_{track}.pkl', 'rb') as f:
            Q = pickle.load(f)
        
        # Get the greedy policy
        policy = np.argmax(Q, axis=-1)
        
        env = RaceTrack(track_dir=track_dir, track=track, size=20)
        fig = plt.figure(figsize=(12, 5), dpi=150)
        fig.suptitle(f'Sample trajectories - Track {track}', size=12, weight='bold')
        
        start_positions = env.start_positions
        
        for i, start_position in enumerate(start_positions):
            track_map = np.copy(env.track_map)
            position = env.reset(start_index=i)
            goal_reached = False
            previous_position = None
            previous__previous_position = None
            
            while not goal_reached:
                track_map[position[0], position[1]] = 4 
                action = policy[position]
                
                previous_position = position
                previous__previous_position = previous_position
    
                next_position, reward, goal_reached = env.step(action)
                position = next_position
            
                # Check if the car is stuck in the same position with action 4
                if np.array_equal(position, previous_position) and action == 4:
                # If stuck, select a different action randomly (except action 4)
                    goal_reached = True 
                               
                if np.array_equal(position, previous_position, previous__previous_position) and action == 7:
                # If stuck, select a different action randomly (except action 4)
                    goal_reached = True 
            print("goal_reached")
            
            
            # Personalized color map
            cmap = ListedColormap(['#FFFFFF', '#CECFCE', '#FA9884', '#93C695', '#000000'])

            ax = plt.subplot(2, 5, i + 1)
            ax.axis('off')
            ax.imshow(track_map, cmap=cmap)
            ax.set_title(start_position[1])
           
        plt.tight_layout()
        plt.savefig(f'./results/track_{track}_optimal_trajectories.png')
        plt.show()
