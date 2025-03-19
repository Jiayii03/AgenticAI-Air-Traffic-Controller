# simulation/train.py
from sarsa_agent import SARSAAgent
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


def train_sarsa_agent(episodes=500, render_every=50):
    """Train a SARSA agent on the ATC environment."""
    # Initialize logger
    logger = Logger(log_dir='logs', prefix='sarsa_training').start()

    # Create the environment
    print("Creates an ATC environment...")
    env = ATCEnv(num_planes=5, num_obstacles=3)

    # Define state buckets and ranges
    # 50 for distance, 36 for angles (10 degree increments)
    bucket_sizes = [50, 36, 36, 50, 36]
    # Ranges for each state variable
    value_ranges = [[0, 50], [0, 360], [0, 360], [0, 50], [0, 360]]

    # Create the agent
    print("Creating a SARSA agent...")
    agent = SARSAAgent(
        action_space=env.action_space.n,
        bucket_sizes=bucket_sizes,
        value_ranges=value_ranges,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.5
    )

    # Load existing model if available
    model_path = 'models/sarsa_atc_model.pkl'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        agent.load(model_path)

    # Training loop
    episode_rewards = []

    print("Entering training loop...")
    for episode in range(episodes):
        print(f"Episode {episode+1}/{episodes}")
        episode_start_time = time.time()
        # Reset environment
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        max_steps = 1000  # Limit episode to 1000 steps to prevent infinite loops

        # Select first action
        print("Selecting action...")
        action = agent.select_action(state)

        # Render the environment at specified intervals
        render = False

        print("Entering episode loop...")
        while not done and step_count < max_steps:
            # Log step time occasionally
            if step_count % 10 == 0:
                elapsed = time.time() - episode_start_time
                print(f"Step {step_count}, Time elapsed: {elapsed:.2f}s")

            # Render if needed
            if render:
                env.render()

            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            print(
                f"Step {step_count}: Action={action}, Reward={reward:.4f}, Total Reward={total_reward:.4f}")

            # Print state information occasionally to see what's changing
            if step_count % 50 == 0:
                print(f"State: {state}, Next State: {next_state}")

            # Select next action
            next_action = agent.select_action(next_state)

            # Update Q-value
            agent.update_q_value(state, action, reward,
                                 next_state, next_action)

            # Move to next state-action pair
            state = next_state
            action = next_action
            step_count += 1

            # Force early termination check
            if step_count >= max_steps:
                print(f"Episode terminated early after {max_steps} steps")
                done = True

        print(
            f"Episode {episode+1} complete, Steps: {step_count}, Total Reward: {total_reward}")
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        print(f"Episode {episode+1} took {episode_duration:.2f} seconds")
    
        # Track metrics
        episode_rewards.append(total_reward)

        # Decay exploration rate
        agent.decay_epsilon(episode, episodes)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(
                f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        # Save model every 50 episodes
        if (episode + 1) % 50 == 0:
            agent.save(model_path)
            print(f"Model saved to {model_path}")

    # Save final model
    agent.save(model_path)
    print(f"Final model saved to {model_path}")

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('SARSA Learning Curve')
    plt.savefig('results/sarsa_learning_curve.png')
    plt.show()

    print(f"Training completed. Log saved to: {logger.get_filename()}")

    return agent


if __name__ == "__main__":
    train_sarsa_agent()
