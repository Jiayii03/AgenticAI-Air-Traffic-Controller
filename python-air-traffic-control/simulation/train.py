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

def train_dqn_agent(episodes=3, render_every=50):
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
    # if os.path.exists(model_path):
    #     print(f"Loading existing model from {model_path}")
    #     agent.load(model_path)
    
    # Training loop
    episode_rewards = []
    episode_steps = []
    successful_landings_per_episode = []
    collision_rate = []
    
    print("Entering training loop...")
    for episode in range(episodes):
        print(f"Episode {episode+1}/{episodes}")
        episode_start_time = time.time()
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        initial_aircraft_count = len(env.simulation.aircraft)
        landed_aircraft_count = 0
        had_collision = False
        
        # Render the environment at specified intervals
        # render = episode % render_every == 0
        render = False
        
        while not done:
            # Render if needed
            if render:
                env.render()
                
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            # Extract landing and collision info
            if "num_planes_landed" in info:
                landed_aircraft_count += info["num_planes_landed"]
                
            if "had_collision" in info and info["had_collision"]:
                had_collision = True
            print(f"Step {step_count}: Action={action}, Reward={reward:.4f}, Total Reward={total_reward:.4f}")
            
            # Update the agent
            loss = agent.update(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            step_count += 1
            
            if step_count >= 3000:
                print("Episode aborted after 3000 steps")
                break
        
        # Log results
        episode_rewards.append(total_reward)
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        episode_steps.append(step_count)
        successful_landings_per_episode.append(landed_aircraft_count)
        collision_rate.append(1 if had_collision else 0)
        print(f"Episode {episode+1} complete, Steps: {step_count}, Total Episode Reward: {total_reward:.2f}")
        print(f"Episode {episode+1} took {episode_duration:.2f} seconds")
        print(f"Aircraft landed in episode {episode+1}: {landed_aircraft_count}/{initial_aircraft_count}")
        if had_collision:
            print(f"Episode {episode+1} ended in a collision")
        else:
            if landed_aircraft_count > 0:
                print(f"Average steps per landing for episode {episode+1}: {step_count / landed_aircraft_count:.1f}")
            else:
                print(f"No aircraft landed in episode {episode+1}")
        
        # Save model periodically
        # if (episode + 1) % 50 == 0:
        #     agent.save(model_path)
        #     print(f"Model saved to {model_path}")
    
    #------------- Save final model after all episodes -------------
    agent.save(model_path)
    print(f"Final model saved to {model_path}")
    
    # --------------- Print training results ---------------
    avg_steps = sum(episode_steps) / len(episode_steps)
    avg_landings = sum(successful_landings_per_episode) / len(successful_landings_per_episode)
    collision_percentage = (sum(collision_rate) / len(collision_rate)) * 100

    print(f"Training results across {len(episode_steps)} episodes:")
    print(f"Average steps per episode: {avg_steps:.1f}")
    print(f"Average aircraft landed: {avg_landings:.2f}/{initial_aircraft_count}")
    print(f"Collision rate: {collision_percentage:.1f}%")
    
    # --------------- Plotting ---------------
    plt.figure(figsize=(15, 10))

    # Plot 1: Total Reward per Episode
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.xticks(range(0, len(episode_rewards)))
    plt.ylabel('Total Reward')
    plt.title('DQN Learning Curve (Total Reward)')
    plt.grid(True)

    # Plot 2: Steps per Landing (for successful episodes)
    plt.subplot(2, 2, 2)
    successful_episodes = [i for i in range(len(collision_rate)) if not collision_rate[i]]
    if successful_episodes:
        # Calculate steps per landing for each successful episode
        steps_per_landing = [episode_steps[i] / successful_landings_per_episode[i] 
                            if successful_landings_per_episode[i] > 0 else 0 
                            for i in successful_episodes]
        plt.plot(successful_episodes, steps_per_landing)
        plt.xlabel('Episode')
        plt.xticks(range(0, len(successful_episodes)))
        plt.ylabel('Steps per Landing')
        plt.title('Efficiency (Steps per Landing for Successful Episodes)')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No successful episodes yet', ha='center', va='center')

    # Plot 3: Collision Rate (running average)
    plt.subplot(2, 2, 3)
    window_size = min(10, len(collision_rate))  # Use smaller window if not enough episodes
    if window_size > 0:
        running_collision_rate = [sum(collision_rate[max(0, i-window_size+1):i+1]) / window_size * 100 
                                for i in range(len(collision_rate))]
        plt.plot(running_collision_rate)
        plt.xlabel('Episode')
        plt.ylabel('Collision Rate (%)')
        plt.xticks(range(0, len(collision_rate)))
        plt.title(f'Safety (Collision Rate - {window_size}-episode average)')
        plt.ylim(0, 100)
        plt.grid(True)

    # Plot 4: Successful Landings per Episode
    plt.subplot(2, 2, 4)
    plt.plot(successful_landings_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Aircraft Landed')
    plt.xticks(range(0, len(episode_rewards)))
    plt.title('Success (Aircraft Landed per Episode)')
    plt.ylim(0, initial_aircraft_count + 0.5)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"results/dqn_training_results_{logger.timestamp}.png")
    plt.show()
    
    return agent

if __name__ == "__main__":
    train_dqn_agent()
