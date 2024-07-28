import argparse
import pickle
from matplotlib.colors import ListedColormap
import numpy as np
import pygame

from matplotlib import pyplot as plt
from maps.game_environment import RaceTrack
from policies.off_policy_motecarlo import off_policy_monte_carlo
from policies.on_policy_motecarlo import on_policy_monte_carlo
from scipy.ndimage import uniform_filter


results_folder = "results"


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
    plt.savefig(f'./{results_folder}/{"_".join(track_name.lower().split())}.png')
    plt.show()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--plot', action='store_true', help='Plot the results')
    parser.add_argument('--sim', action='store_true', help='Run simulations in the environment')
    parser.add_argument('--track', type=str, default='a', help='Track name (a, b or c)')
    parser.add_argument('--policy', type=str, default='off', help='Monte Carlo policy (on, off)')

    args = parser.parse_args()

    train = args.train # Train the model
    plot = args.plot # Save plots of the simulation
    sim = args.sim # Run simulations 
    track_dir = "./maps/saved_tracks/" # Directory of the track files
    track = args.track # Track name (a, b, c or d)
    policy = args.policy # MC Policy
    total_episodes = 100000 # Total number of episodes

    if policy == "on":
        results_folder = "results_on"

    if train:
        track_name = f'Track {track.capitalize()}'
        if policy == "on":
            reward_hist, Q = on_policy_monte_carlo(total_episodes, track_dir, track)
        else:
            reward_hist, Q = off_policy_monte_carlo(total_episodes, track_dir, track)
        
        plot_result(reward_hist, track_name, total_episodes)

        with open(f'./{results_folder}/training/track_{track}.pkl', 'wb') as f:
            pickle.dump(Q, f)


    if plot or sim: # Evaluate the Q values and plot sample paths
        with open(f'./{results_folder}/training/track_{track}.pkl', 'rb') as f:
            Q = pickle.load(f)
        
        # Get the greedy policy
        policy = np.argmax(Q, axis=-1)
        
        size = 10 if track == 'c' else 20
        env = RaceTrack(track_dir=track_dir, track=track, size=size)
        
        start_positions = env.start_positions
        routes = []
        
        for i, start_position in enumerate(start_positions):
            track_map = np.copy(env.track_map)
            position = env.reset(start_index=i)
            goal_reached = False
            previous_position = None
            previous__previous_position = None
            route = []
            
            while not goal_reached:
                route.append(position)
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
            
            routes.append(route)
    
    if plot:
        # Personalized color map
        cmap = ListedColormap(['#FFFFFF', '#CECFCE', '#FA9884', '#93C695', '#000000'])

        # Calculate number of rows needed for 5 columns per row
        num_positions = len(start_positions)
        num_cols = 6
        num_rows = (num_positions + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3), dpi=150)
        fig.suptitle(f'Sample trajectories - Track {track}', size=12, weight='bold')

        for i, (ax, start_position) in enumerate(zip(axes.flat, start_positions)):
            track_map = np.copy(env.track_map)
            for position in routes[i]:
                track_map[position[0], position[1]] = 4
            
            ax.axis('off')
            ax.imshow(track_map, cmap=cmap)
            ax.set_title(f"Start {start_position[1]}")

        # Turn off any unused subplots
        for j in range(i + 1, len(axes.flat)):
            axes.flat[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(f'./{results_folder}/track_{track}_optimal_trajectories.png')
        plt.show()

    if sim:
        print("Running simulation")
        pygame.init()
        env.window = pygame.display.set_mode(env.window_size)
        env.clock = pygame.time.Clock()

        running = True
        
        while running:
            for route in routes:
                for position in route:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False

                    env.car_position = position
                    env.render()
                    env.clock.tick(5)

                    if not running:
                        break

                if not running:
                    break

        # Stop by pressing the close button or Ctrl+C
        pygame.quit()
