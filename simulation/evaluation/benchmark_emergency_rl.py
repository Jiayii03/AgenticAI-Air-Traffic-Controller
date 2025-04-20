import numpy as np
import time
import sys
import os
from tqdm import tqdm  

# Add parent directory to path so we can import game components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulation.emergency.emergency_atc_env import EmergencyATCEnv
from simulation.emergency.emergency_rl_controller import EmergencyRLController
from simulation.common.logger import Logger
from core import conf

def simulate_random_assignment(env, num_simulations=100):
    """
    Simulate emergency scenarios where aircraft are randomly assigned to safe destinations.
    """
    total_rerouted_distance = 0
    total_time_saved = 0
    rerouted_count = 0
    for _ in tqdm(range(num_simulations), desc="Simulating Random Assignment"):
        obs = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            
            # Track rerouted distance
            if info.get("rerouted", False):
                total_rerouted_distance += info.get("rerouted_distance", 0)
                rerouted_count += 1
                
                # Calculate time saved for this reroute
                speed = info.get("aircraft_speed", 1)  # Default to 1 if speed is not provided
                total_time_saved += info.get("rerouted_distance", 0) / speed
    
    # Calculate average rerouted distance and time saved
    avg_rerouted_distance = total_rerouted_distance / rerouted_count if rerouted_count > 0 else 0
    avg_time_saved = total_time_saved / rerouted_count if rerouted_count > 0 else 0
    return avg_rerouted_distance, avg_time_saved

def simulate_rl_agent(env, rl_controller, num_simulations=100):
    """
    Simulate emergency scenarios using the Emergency RL Agent.
    """
    total_rerouted_distance = 0
    total_time_saved = 0
    rerouted_count = 0
    for _ in tqdm(range(num_simulations), desc="Simulating RL Agent"):
        obs = env.reset()
        done = False
        
        while not done:
            action = rl_controller.select_action(obs)
            obs, reward, done, _, info = env.step(action)
            
            # Track rerouted distance
            if info.get("rerouted", False):
                total_rerouted_distance += info.get("rerouted_distance", 0)
                rerouted_count += 1
                
                # Calculate time saved for this reroute
                speed = info.get("aircraft_speed", 1)  # Default to 1 if speed is not provided
                total_time_saved += info.get("rerouted_distance", 0) / speed
    
    # Calculate average rerouted distance and time saved
    avg_rerouted_distance = total_rerouted_distance / rerouted_count if rerouted_count > 0 else 0
    avg_time_saved = total_time_saved / rerouted_count if rerouted_count > 0 else 0
    return avg_rerouted_distance, avg_time_saved

def main():
    # Initialize logger
    logger = Logger(log_dir='../logs', prefix='evaluation', debug=False).start()
    
    # Configuration parameters
    num_planes = conf.get()["game"]["n_aircraft"]
    num_obstacles = conf.get()["game"]["n_obstacles"]
    screen_size = (800, 800)
    num_simulations = 100  # Number of simulations to run for each scenario
    
    # Initialize the environment
    print("Initializing Emergency ATC Environment...")
    env = EmergencyATCEnv(logger=logger, num_planes=num_planes, num_obstacles=num_obstacles, screen_size=screen_size)
    
    # Initialize the Emergency RL Controller
    print("Initializing Emergency RL Controller...")
    emergency_model_path = '../../models/emergency_dqn_model.pth'
    rl_controller = EmergencyRLController(model_path=emergency_model_path, debug=False)
    
    # Simulate random assignment
    print("\nRunning simulations with random assignment...")
    avg_rerouted_distance_random, avg_time_saved_random = simulate_random_assignment(env, num_simulations=num_simulations)
    print(f"Average rerouted distance with random assignment: {avg_rerouted_distance_random:.2f}")
    print(f"Average time saved with random assignment: {avg_time_saved_random:.2f} seconds")
    
    # Simulate RL agent
    print("\nRunning simulations with RL agent...")
    avg_rerouted_distance_rl, avg_time_saved_rl = simulate_rl_agent(env, rl_controller, num_simulations=num_simulations)
    print(f"Average rerouted distance with RL agent: {avg_rerouted_distance_rl:.2f}")
    print(f"Average time saved with RL agent: {avg_time_saved_rl:.2f} seconds")
    
    # Compare results
    print("\nComparison Results:")
    print(f"Random Assignment: {avg_rerouted_distance_random:.2f} units, {avg_time_saved_random:.2f} seconds")
    print(f"RL Agent: {avg_rerouted_distance_rl:.2f} units, {avg_time_saved_rl:.2f} seconds")
    print(f"Improvement with RL Agent: {avg_rerouted_distance_random - avg_rerouted_distance_rl:.2f} units, {avg_time_saved_random - avg_time_saved_rl:.2f} seconds")

if __name__ == "__main__":
    main()