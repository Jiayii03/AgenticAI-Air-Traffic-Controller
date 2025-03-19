# simulation/train.py
from dqn_agent import DQNAgent
from atc_env import ATCEnv
from logger import Logger
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_dqn_agent(episodes=500, render_every=50):
    """Train a DQN agent on the ATC environment."""
    # Initialize logger
    logger = Logger(log_dir='logs', prefix='dqn_training').start()

    # Create the environment
    print("Creates an ATC environment...")
    env = ATCEnv(num_planes=5, num_obstacles=3)
    
    # Get state and action dimensions from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create the agent
    print("Creating a DQN agent...")
    agent = DQNAgent(
        state_dim=state_dim,  # State space dimension
        action_dim=action_dim, # Action space dimension
        lr=0.001,              # Learning rate
        gamma=0.99,            # Discount factor
        epsilon=1.0,           # Initial exploration rate
        epsilon_min=0.1,       # Minimum exploration rate
        epsilon_decay=0.995,   # Exploration decay rate
        buffer_size=10000,     # Replay buffer size
        batch_size=64          # Training batch size
    )
    
    # Load existing model if available
    model_path = 'models/dqn_atc_model.pth'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        agent.load(model_path)
    
    # Training loop
    episode_rewards = []
    
    print("Entering training loop...")
    for episode in range(episodes):
        print(f"Episode {episode+1}/{episodes}")
        episode_start_time = time.time()
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        # Render the environment at specified intervals
        render = episode % render_every == 0
        
        while not done:
            # Render if needed
            if render:
                env.render()
                
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step {step_count}: Action={action}, Reward={reward:.4f}, Total Reward={total_reward:.4f}")
            
            # Update the agent
            loss = agent.update(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            step_count += 1
        
        # Log results
        episode_rewards.append(total_reward)
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        print(f"Episode {episode+1} complete, Steps: {step_count}, Total Episode Reward: {total_reward:.2f}")
        print(f"Episode {episode+1} took {episode_duration:.2f} seconds")
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # Save model periodically
        if (episode + 1) % 50 == 0:
            agent.save(model_path)
            print(f"Model saved to {model_path}")
            
        break
    
    # Save final model
    agent.save(model_path)
    print(f"Final model saved to {model_path}")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Learning Curve')
    plt.savefig('dqn_learning_curve.png')
    plt.show()
    
    return agent

if __name__ == "__main__":
    train_dqn_agent()
