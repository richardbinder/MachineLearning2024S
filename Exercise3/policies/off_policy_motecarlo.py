import numpy as np
from maps.game_environment import RaceTrack
from policies.soft_behavior_policy import soft_policy

def off_policy_monte_carlo(
    total_episodes: int,
    track_dir: str = "../maps/saved_tracks/",
    track: str = "a"
):
    # Probability of selecting a random action (exploration)
    epsilon = 0.2

    # Discount factor for future rewards
    gamma = 0.9
    
    # Initialize the game environment
    env = RaceTrack(track_dir, track, size=20)
    
    # Get action and observation spaces from the environment
    action_space = env.action_space
    observation_space = env.observation_space
    
    # Q: state-action value function
    # This matrix holds the estimated values for each state-action pair.
    Q = np.random.normal(size=(*observation_space, action_space))

    # To encourage exploration, the initial Q values are set to an optimistically high value.
    # Here, we subtract 500 from the randomly initialized values to simulate this optimistic initialization.
    # This means that all actions are initially seen as highly rewarding until proven otherwise through exploration.
    Q -= 500
    
    # C: Cumulative weights for each state-action pair
    # This matrix is used to keep track of the total weight (or importance) of each state-action pair over time.
    # Initialize C as a zero matrix with the same shape as Q
    C = np.zeros_like(Q)
    
    # Target policy initialized to the max Q value
    target_policy = np.argmax(Q, axis=-1)

    # Reward history, to store the total reward for each episode
    reward_hist = np.zeros(shape=(total_episodes), dtype=np.float32) 

    for i in range(total_episodes):
        # Array to store the whole trajectory of the episode
        T = []
        
        # Flag to check if the goal is reached
        goal_reached = False
        
        # Reset the environment and get the initial position, after each episode
        position = env.reset()

        # Get the first action and its probability
        (action, act_prob) = soft_policy(position, env.action_space, target_policy, epsilon)
        
        # Initialize the total reward for the episode
        total_reward = 0

        # Generate a trajectory using the behavior policy
        while not goal_reached:
            
            # With a probability of 0.1, select action #4 (no increase in acceleration in any direction)
            if np.random.rand() <= 0.1:
                # Stablish action as 4 (non-acceleration action)
                non_acc_act = 4
                observation, reward, goal_reached = env.step(non_acc_act)
            else:
                observation, reward, goal_reached = env.step(action)

            # Accumulate reward
            total_reward += reward
            
            # Append the trajectory
            T.append((position, action, reward, act_prob))
            
            # Update position of the car
            position = observation
            
             # Get new action and probability
            (action, act_prob) = soft_policy(position, env.action_space, target_policy, epsilon)
        
        G = 0  # Initialize the return value of the episode
        W = 1  # Initialize sampling weight

        # Loop inversely to update G and Q values
        while T:
            (position, action, reward, act_prob) = T.pop()
            G = gamma * G + reward  # Update the return G
            C[position][action] = C[position][action] + W  # Update the sum of weights
            Q[position][action] = Q[position][action] + (W / C[position][action]) * (G - Q[position][action])  # Update the Q value

            target_policy[position] = np.argmax(Q[position])  # Update the target policy
            if action != target_policy[position]:
                break  # Exit if the action is not the best action in the target policy
            W = W * (1 / act_prob)  # Update the weight
        
        reward_hist[i] = total_reward  # Save the total reward in the history

        # Print the reward every 1000 episodes
        if i % 10 == 0:
            print(f'Episode: {i}, reward: {total_reward}, epsilon: {epsilon}')
        
    return reward_hist, Q  # Return the reward history and the Q value function
