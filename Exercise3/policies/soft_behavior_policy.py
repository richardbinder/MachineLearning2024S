import numpy as np


def soft_policy(position:tuple, action_space: int, target_pi: np.ndarray, epsilon:float):
    """
        Soft behavior policy that returns the probability of that action
        and the selected action. e-greedy policy.
        
        Parameters:
        - position: The current position of the environment.
        - action_space: The total number of possible actions (9 in this case).
        - target_pi: The target policy array, where target_pi[position] is the greedy action for current position.
        - epsilon: The probability of choosing a random action (exploration rate).
        
        Returns:
        - action: The selected action.
        - probability: The probability of the selected action.
    """
    
    # Get the greedy action for the current position
    greedy_act = target_pi[position]
    
    # If the random number is greater than epsilon, select the greedy action
    if np.random.rand() > epsilon:
        action = greedy_act
        
        # Probability of selecting the greedy action
        probability = 1 - epsilon + epsilon / action_space

    else:
        # If the random number is less than epsilon, select a random action
        action = np.random.choice(action_space)
        
        if action == greedy_act:
            #  If the random action is the greedy action, the probability the same as above
            probability = 1 - epsilon + epsilon / action_space
        else:
            # If the random action is not the greedy action, the probability is epsilon / action_space
            probability = epsilon / action_space
    
    return action, probability