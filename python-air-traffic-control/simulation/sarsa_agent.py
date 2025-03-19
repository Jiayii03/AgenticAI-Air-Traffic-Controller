# simulation/sarsa_agent.py
import numpy as np
import random
import pickle
import os

class SARSAAgent:
    """
    SARSA (State-Action-Reward-State-Action) reinforcement learning agent.
    
    This implementation uses linear function approximation with discrete
    state buckets for the state space.
    """
    
    def __init__(self, action_space, bucket_sizes, value_ranges, alpha=0.5, gamma=0.9, epsilon=0.5):
        """
        Initialize the SARSA agent.
        
        Args:
            action_space: Number of possible actions
            bucket_sizes: Number of buckets for each state dimension
            value_ranges: Min and max values for each state dimension
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.action_space = action_space
        self.bucket_sizes = bucket_sizes
        self.value_ranges = value_ranges
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = {}
        
        # Statistics for evaluation
        self.rewards = []
        self.exploration_rate = []
    
    def get_discretized_state(self, state):
        """Convert continuous state to discretized state index tuple."""
        discretized = []
        for i, val in enumerate(state):
            # Clip value to be within the range
            val = max(self.value_ranges[i][0], min(val, self.value_ranges[i][1]))
            
            # Scale to bucket index
            bucket_width = (self.value_ranges[i][1] - self.value_ranges[i][0]) / self.bucket_sizes[i]
            bucket_index = int((val - self.value_ranges[i][0]) / bucket_width)
            
            # Handle edge case
            if bucket_index == self.bucket_sizes[i]:
                bucket_index = self.bucket_sizes[i] - 1
                
            discretized.append(bucket_index)
        
        return tuple(discretized)
    
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair."""
        discretized_state = self.get_discretized_state(state)
        return self.q_table.get((discretized_state, action), 0.0)
    
    def update_q_value(self, state, action, reward, next_state, next_action):
        """Update Q-value using SARSA update rule."""
        discretized_state = self.get_discretized_state(state)
        discretized_next_state = self.get_discretized_state(next_state)
        
        # Get current Q-value
        q_value = self.q_table.get((discretized_state, action), 0.0)
        
        # Get next Q-value
        next_q_value = self.q_table.get((discretized_next_state, next_action), 0.0)
        
        # SARSA update
        new_q_value = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value)
        
        # Update Q-table
        self.q_table[(discretized_state, action)] = new_q_value
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Exploration: choose random action
            return random.randrange(self.action_space)
        else:
            # Exploitation: choose best action
            discretized_state = self.get_discretized_state(state)
            q_values = [self.q_table.get((discretized_state, a), 0.0) for a in range(self.action_space)]
            return np.argmax(q_values)
    
    def decay_epsilon(self, episode, total_episodes):
        """Decay epsilon over time."""
        self.epsilon = max(0.1, 0.5 - 0.4 * (episode / total_episodes))
        self.exploration_rate.append(self.epsilon)
    
    def save(self, filepath):
        """Save the agent's Q-table to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filepath):
        """Load the agent's Q-table from a file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)