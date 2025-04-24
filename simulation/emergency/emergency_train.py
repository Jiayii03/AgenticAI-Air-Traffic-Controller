import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import random
import sys
import os

# Add the parent directory to the system path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'simulation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core import conf
from emergency_atc_env import EmergencyATCEnv
from emergency_dqn_agent import EmergencyDQNAgent
from common.logger import Logger

def train_emergency_agent(episodes=400, render_every=50, seed=43):
    """Train an emergency rerouting DQN agent on the EmergencyATCEnv."""
    # Initialize logger
    logger = Logger(log_dir='../logs', prefix='emergency_training', debug=False).start()
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Configuration parameters from conf
    num_planes = conf.get()["game"]["n_aircraft"]
    num_obstacles = conf.get()["game"]["n_obstacles"]
    screen_size = (800, 800)
    
    # Create the emergency environment
    print("Creating Emergency ATC Environment...")
    env = EmergencyATCEnv(logger=logger, num_planes=num_planes, num_obstacles=num_obstacles, screen_size=screen_size)
    
    # Get state and action dimensions from the environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create the emergency DQN agent
    print("Creating Emergency DQN Agent...")
    agent = EmergencyDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.0001,
        gamma=0.99,
        epsilon=0.7,  # Start with lower exploration
        epsilon_min=0.05,
        epsilon_decay=0.9999,  # Slower decay
        buffer_size=50000,
        batch_size=64,
        update_frequency=10,
        target_update=500
    )
    
    # Optionally load an existing model if available
    model_path = '../../models/emergency_dqn_refactor.pth'
    if os.path.exists(model_path):
        print(f"Loading existing emergency model from {model_path}")
        agent.load(model_path)
    
    # Training metrics
    episode_rewards = []
    successful_reroutes = []
    training_start_time = time.time()
    
    print("Starting training loop for emergency agent...")
    for episode in range(episodes):
        print(f"Episode {episode+1}/{episodes}")
        obs = env.reset(seed=seed + episode)
        
        total_reward = 0
        done = False
        step_count = 0
        reroute_count = 0
        
        while not done:
            step_count += 1
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Reward adjustments
            if info.get("rerouted", False):
                reward += 1000
                reroute_count += 1
            if info.get("collision", False):
                reward -= 20
            if info.get("distance_to_destination", 0) > 0:
                reward -= info["distance_to_destination"] * 0.05

            # total_reward += reward
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            
            if step_count >= 4000:
                print("Episode aborted after 4000 steps")
                break
        
        episode_rewards.append(total_reward)
        successful_reroutes.append(reroute_count)
        print(f"Episode {episode+1} complete: Total Reward = {total_reward:.2f}, Safe Reroutes = {reroute_count}")
    
    # Save the final emergency model after training
    agent.save(model_path)
    print(f"Final emergency model saved to {model_path}")
    
    # Ensure the results directory exists
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Plot training results
    plt.figure(figsize=(15, 10))

    # Plot 1: Total Reward per Episode with Regression Line
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards, label="Raw Rewards", alpha=0.6)

    # Smoothed rewards
    smoothed_rewards = np.convolve(episode_rewards, np.ones(20)/20, mode='valid')
    plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label="Smoothed Rewards", color='blue')

    # Regression line
    x = np.arange(len(episode_rewards))
    y = np.array(episode_rewards)
    regression_coeffs = np.polyfit(x, y, 1)  # Linear regression (degree=1)
    regression_line = np.poly1d(regression_coeffs)
    plt.plot(x, regression_line(x), label="Regression Line", color='red', linestyle='--')

    # Labels and title
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Emergency DQN Learning Curve (Reward Trend)')
    plt.legend()
    plt.grid(True)

    plt.show()
    plt.savefig(os.path.join(results_dir, 'emergency_training_rewards.png'))
    plt.close()

    return agent

if __name__ == "__main__":
    train_emergency_agent(episodes=400)


