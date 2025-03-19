# simulation/evaluate.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.atc_env import ATCEnv
from simulation.sarsa_agent import SARSAAgent

def evaluate_agent(episodes=10, render=True):
    """Evaluate a trained SARSA agent on the ATC environment."""
    # Create the environment
    env = ATCEnv(num_planes=5, num_obstacles=3)
    
    # Define state buckets and ranges
    bucket_sizes = [50, 36, 36]  # 50 for distance, 36 for angles (10 degree increments)
    value_ranges = [[0, 50], [0, 360], [0, 360]]  # Ranges for each state variable
    
    # Create the agent
    agent = SARSAAgent(
        action_space=env.action_space.n,
        bucket_sizes=bucket_sizes,
        value_ranges=value_ranges
    )
    
    # Load trained model
    model_path = 'models/sarsa_atc_model.pkl'
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        return
    
    print(f"Loading trained model from {model_path}")
    agent.load(model_path)
    
    # Disable exploration for evaluation
    agent.epsilon = 0.0
    
    # Evaluation loop
    episode_rewards = []
    collision_count = 0
    landing_count = 0
    
    for episode in range(episodes):
        # Reset environment
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Render if needed
            if render:
                env.render()
            
            # Select action (no exploration)
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Move to next state
            state = next_state
        
        # Track metrics
        episode_rewards.append(total_reward)
        
        print(f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}")
    
    # Print evaluation results
    avg_reward = np.mean(episode_rewards)
    print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")
    
    return avg_reward

if __name__ == "__main__":
    evaluate_agent()