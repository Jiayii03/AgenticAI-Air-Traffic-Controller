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

# Add the simulation directory to the system path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'simulation')))

# Add the project root directory to the system path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core import conf
from emergency_atc_env import EmergencyATCEnv
from emergency_dqn_agent import EmergencyDQNAgent
from common.logger import Logger

def train_emergency_agent(episodes=2, render_every=50, seed=43):
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
    state_dim = env.observation_space.shape[0]  # Expected to be 9 features
    action_dim = env.action_space.n            # Expected to be 3 safe destination choices
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create the emergency DQN agent
    print("Creating Emergency DQN Agent...")
    agent = EmergencyDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.0001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
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
    episode_steps = []
    safe_reroutes = []  # Count safe reroutes per episode
    training_start_time = time.time()
    
    print("Starting training loop for emergency agent...")
    for episode in range(episodes):
        print(f"Episode {episode+1}/{episodes}")
        # Use a unique seed per episode for reproducibility
        episode_seed = seed + episode
        obs = env.reset(seed=episode_seed)
        
        episode_start_time = time.time()
        total_reward = 0
        done = False
        step_count = 0
        reroute_count = 0
        
        while not done:
            step_count += 1
            action = agent.select_action(obs)
            logger.debug_print(f"Selected emergency action: {action}")

            next_obs, reward, done, truncated, info = env.step(action)
            logger.debug_print(f"Step {step_count}: Action {action}, Reward {reward:.4f}")

            if info.get("rerouted", False):
                reroute_count += 1  # ✅ Count actual reroutes

            total_reward += reward
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs

            if step_count >= 4000:
                print("Episode aborted after 4000 steps")
                break
        
        episode_rewards.append(total_reward)
        episode_steps.append(step_count)
        safe_reroutes.append(reroute_count)
        episode_duration = time.time() - episode_start_time
        print(f"Episode {episode+1} complete: Steps = {step_count}, Total Reward = {total_reward:.2f}, Safe Reroutes = {reroute_count}, Duration = {episode_duration:.2f}s")
        
        # Render the environment every render_every episodes
        if (episode + 1) % render_every == 0:
            env.render(mode='human')
    
    # Save the final emergency model after training
    agent.save(model_path)
    print(f"Final emergency model saved to {model_path}")
        # Run test episodes with greedy policy (epsilon = 0)
    print("Running 10 test episodes (exploitation only)...")
    agent.epsilon = 0.0
    test_reroutes = 0
    test_rewards = []
    
    for test_ep in range(10):
        obs = env.reset()
        done = False
        rerouted_this_ep = False
        ep_reward = 0

        while not done:
            action = agent.select_action(obs)
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward
            if info.get("rerouted", False):
                rerouted_this_ep = True

        if rerouted_this_ep:
            test_reroutes += 1
        test_rewards.append(ep_reward)
    
    print(f"\n[Evaluation]")
    print(f"Successful reroutes in test episodes: {test_reroutes}/10")
    print(f"Average reward in test episodes: {np.mean(test_rewards):.2f}")
    
    # Training summary
    avg_steps = sum(episode_steps) / len(episode_steps)
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_reroutes = sum(safe_reroutes) / len(safe_reroutes)
    training_duration = time.time() - training_start_time
    print("Emergency Training Summary:")
    print(f"Average steps per episode: {avg_steps:.1f}")
    print(f"Average reward per episode: {avg_reward:.2f}")
    print(f"Average safe reroutes per episode: {avg_reroutes:.2f}")
    print(f"Total training duration: {training_duration:.2f} seconds")
    print("Running 10 test episodes (exploitation only)...")
    
    # Plot training results
    plt.figure(figsize=(15, 10))
    
    # Plot Total Reward per Episode
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Emergency DQN: Total Reward per Episode')
    plt.grid(True)
    
    # Plot Steps per Episode
    plt.subplot(2, 2, 2)
    plt.plot(episode_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Emergency DQN: Steps per Episode')
    plt.grid(True)
    
    # Plot Safe Reroutes per Episode
    plt.subplot(2, 2, 3)
    plt.plot(safe_reroutes)
    plt.xlabel('Episode')
    plt.ylabel('Safe Reroutes')
    plt.title('Emergency DQN: Safe Reroutes per Episode')
    plt.grid(True)
    
    plt.tight_layout()
    results_filename = f"../results/emergency_training_results_{logger.timestamp}.png"
    plt.savefig(results_filename)
    plt.show()
    
    return agent

if __name__ == "__main__":
    train_emergency_agent()
