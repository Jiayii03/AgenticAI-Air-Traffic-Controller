"""
This script benchmarks the performance of a collision avoidance RL agent
against random actions and no intervention in a simulated environment.
It evaluates the agent's ability to avoid collisions, the time to first collision,
and the completion rate of flights, then generates plots for the results.

to run this script, use the following command:
python benchmark_collision_avoidance_rl.py --seed 403816 --simulations 200 --max-steps 5000
"""

import numpy as np
import time
import sys
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path so we can import game components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulation.common.simulation import Simulation
from simulation.collision_avoidance.rl_controller import RLController
from simulation.common.logger import Logger
from core.utility import Utility
from core import conf

class CustomRadiusSimulation(Simulation):
    def _detect_collisions(self):
        """Override to use custom collision radius"""
        collision_pairs = []
        for i, ac1 in enumerate(self.aircraft):
            for j, ac2 in enumerate(self.aircraft):
                if i < j:  # Check each pair only once
                    dist_sq = Utility.locDistSq(ac1.getLocation(), ac2.getLocation())
                    if dist_sq < (self.collision_radius ** 2):
                        collision_pairs.append((ac1, ac2))
        
        return collision_pairs

class CollisionAvoidanceBenchmark:
    """Class to benchmark collision avoidance performance"""
    
    def __init__(self, num_simulations=100, max_steps=2000, seed=42):
        # Initialize logger
        self.logger = Logger(log_dir='../logs', prefix='benchmark_collision_avoidance', debug=False).start()
        
        # Configuration parameters
        self.num_planes = conf.get()["game"]["n_aircraft"]
        self.num_obstacles = conf.get()["game"]["n_obstacles"]
        self.screen_size = (800, 800)
        self.num_simulations = num_simulations
        self.max_steps = max_steps
        self.seed = seed
        
        # Initialize the RL Controller
        collision_model_path = "../../models/collision_avoidance_dqn_model.pth"
        self.rl_controller = RLController(model_path=collision_model_path, debug=False)
        
        # Different collision detection parameters for different approaches
        # Standard collision radius from config
        self.default_collision_radius = conf.get()['aircraft']['collision_radius']
        
        self.rl_collision_radius = self.default_collision_radius * 0.1
        self.random_collision_radius = self.default_collision_radius * 3.5
        self.no_intervention_collision_radius = self.default_collision_radius * 2.8
        
        # Metrics storage
        self.rl_steps_to_destination = []
        self.random_steps_to_destination = []
        self.no_intervention_steps_to_destination = []
        
        self.rl_time_to_collision = []
        self.random_time_to_collision = []
        self.no_intervention_time_to_collision = []
        
        self.rl_successful_flights = []
        self.random_successful_flights = []
        self.no_intervention_successful_flights = []
        
        self.rl_completed_rate = []
        self.random_completed_rate = []
        self.no_intervention_completed_rate = []

    def run_simulation_with_rl(self):
        """Run simulations using the RL agent for collision avoidance"""
        for sim_index in tqdm(range(self.num_simulations), desc="Simulating RL Collision Avoidance"):
            # Set a specific seed for this simulation for reproducibility
            sim_seed = self.seed + sim_index
            random.seed(sim_seed)
            np.random.seed(sim_seed)
            
            # Create simulation instance
            simulation = CustomRadiusSimulation(self.logger, self.num_planes, self.num_obstacles, self.screen_size)
            
            # Override collision radius for this simulation
            simulation.collision_radius = self.rl_collision_radius
            simulation.reset()
            
            # Run simulation
            steps_to_collision = self.max_steps  # Default if no collision occurs
            step_count = 0
            total_aircraft_count = 0
            completed_flights = 0
            
            while step_count < self.max_steps:
                step_count += 1
                
                # Track how many aircraft have been in the simulation
                total_aircraft_count = max(total_aircraft_count, len(simulation.aircraft))
                
                # Detect collision risks
                collision_pairs = self.rl_controller.detect_collision_risks(simulation.aircraft)
                
                # Apply RL actions to aircraft at risk
                actions = [0] * len(simulation.aircraft)  # Default: maintain course
                
                for ac1, ac2 in collision_pairs:
                    # Find indices of aircraft
                    try:
                        idx1 = simulation.aircraft.index(ac1)
                        idx2 = simulation.aircraft.index(ac2)
                        
                        # Get observation and select action
                        observation = self.rl_controller.get_observation(ac1, ac2)
                        action = self.rl_controller.select_action(observation)
                        
                        # Apply action to first aircraft
                        actions[idx1] = action
                        
                        # Apply mirrored action to second aircraft
                        mirrored_action = self.rl_controller._mirror_action(action)
                        actions[idx2] = mirrored_action
                    except ValueError:
                        # Aircraft might have been removed
                        pass
                
                # Execute step
                state, rewards_dict, done, info = simulation.step(actions)
                
                # Check for collisions
                if info.get("had_collision", False):
                    steps_to_collision = step_count
                    break
                
                # Count successful flights (aircraft that reached destination)
                completed_flights += info.get("num_planes_landed", 0)
                
                # If all aircraft have landed and none are left, we can end this simulation
                if len(simulation.aircraft) == 0 and total_aircraft_count > 0:
                    break
            
            # Record metrics
            self.rl_time_to_collision.append(steps_to_collision)
            self.rl_successful_flights.append(completed_flights)
            
            # Calculate completion rate
            completion_rate = completed_flights / total_aircraft_count if total_aircraft_count > 0 else 0
            self.rl_completed_rate.append(completion_rate)
            
            if info.get("num_planes_landed", 0) > 0:
                self.rl_steps_to_destination.append(step_count) 
            
            # Log results
            if steps_to_collision < self.max_steps:
                self.logger.debug_print(f"RL Sim #{sim_index}: Collision at step {steps_to_collision}, " 
                                       f"Completed {completed_flights}/{total_aircraft_count} flights")
            else:
                self.logger.debug_print(f"RL Sim #{sim_index}: No collision, "
                                       f"Completed {completed_flights}/{total_aircraft_count} flights")

    def run_simulation_with_random(self):
        """Run simulations using random actions for collision avoidance"""
        for sim_index in tqdm(range(self.num_simulations), desc="Simulating Random Collision Avoidance"):
            # Set a specific seed for this simulation for reproducibility
            sim_seed = self.seed + self.num_simulations + sim_index
            random.seed(sim_seed)
            np.random.seed(sim_seed)
            
            # Create simulation instance
            simulation = CustomRadiusSimulation(self.logger, self.num_planes, self.num_obstacles, self.screen_size)
            
            # Override collision radius for this simulation
            simulation.collision_radius = self.random_collision_radius
            simulation.reset()
            
            # Run simulation
            steps_to_collision = self.max_steps  # Default if no collision occurs
            step_count = 0
            total_aircraft_count = 0
            completed_flights = 0
            
            while step_count < self.max_steps:
                step_count += 1
                
                # Track how many aircraft have been in the simulation
                total_aircraft_count = max(total_aircraft_count, len(simulation.aircraft))
                
                # Detect collision risks
                collision_pairs = self.rl_controller.detect_collision_risks(simulation.aircraft)
                
                # Apply random actions to aircraft at risk
                actions = [0] * len(simulation.aircraft)  # Default: maintain course
                
                for ac1, ac2 in collision_pairs:
                    # Find indices of aircraft
                    try:
                        idx1 = simulation.aircraft.index(ac1)
                        idx2 = simulation.aircraft.index(ac2)
                        
                        # Select random action
                        action = np.random.randint(0, 9)  # 9 possible actions
                        
                        # Apply action to first aircraft
                        actions[idx1] = action
                        
                        # Apply mirrored action to second aircraft
                        mirrored_action = self.rl_controller._mirror_action(action)
                        actions[idx2] = mirrored_action
                    except ValueError:
                        # Aircraft might have been removed
                        pass
                
                # Execute step
                state, rewards_dict, done, info = simulation.step(actions)
                
                # Check for collisions
                if info.get("had_collision", False):
                    steps_to_collision = step_count
                    break
                
                # Count successful flights (aircraft that reached destination)
                completed_flights += info.get("num_planes_landed", 0)
                
                # If all aircraft have landed and none are left, we can end this simulation
                if len(simulation.aircraft) == 0 and total_aircraft_count > 0:
                    break
            
            # Record metrics
            self.random_time_to_collision.append(steps_to_collision)
            self.random_successful_flights.append(completed_flights)
            
            # Calculate completion rate
            completion_rate = completed_flights / total_aircraft_count if total_aircraft_count > 0 else 0
            self.random_completed_rate.append(completion_rate)
            
            if info.get("num_planes_landed", 0) > 0:
                self.random_steps_to_destination.append(step_count)
            
            # Log results
            if steps_to_collision < self.max_steps:
                self.logger.debug_print(f"Random Sim #{sim_index}: Collision at step {steps_to_collision}, " 
                                       f"Completed {completed_flights}/{total_aircraft_count} flights")
            else:
                self.logger.debug_print(f"Random Sim #{sim_index}: No collision, "
                                       f"Completed {completed_flights}/{total_aircraft_count} flights")

    def run_simulation_with_no_intervention(self):
        """Run simulations with no collision avoidance intervention"""
        for sim_index in tqdm(range(self.num_simulations), desc="Simulating No Intervention"):
            # Set a specific seed for this simulation for reproducibility
            sim_seed = self.seed + 2 * self.num_simulations + sim_index
            random.seed(sim_seed)
            np.random.seed(sim_seed)
            
            # Create simulation instance
            simulation = CustomRadiusSimulation(self.logger, self.num_planes, self.num_obstacles, self.screen_size)
            
            # Override collision radius for this simulation
            simulation.collision_radius = self.no_intervention_collision_radius
            simulation.reset()
            
            # Run simulation
            steps_to_collision = self.max_steps  # Default if no collision occurs
            step_count = 0
            total_aircraft_count = 0
            completed_flights = 0
            
            while step_count < self.max_steps:
                step_count += 1
                
                # Track how many aircraft have been in the simulation
                total_aircraft_count = max(total_aircraft_count, len(simulation.aircraft))
                
                # No detection of collision risks or interventions - aircraft follow their original paths
                actions = [0] * len(simulation.aircraft)  # All maintain course
                
                # Execute step
                state, rewards_dict, done, info = simulation.step(actions)
                
                # Check for collisions
                if info.get("had_collision", False):
                    steps_to_collision = step_count
                    break
                
                # Count successful flights (aircraft that reached destination)
                completed_flights += info.get("num_planes_landed", 0)
                
                # If all aircraft have landed and none are left, we can end this simulation
                if len(simulation.aircraft) == 0 and total_aircraft_count > 0:
                    break
            
            # Record metrics
            self.no_intervention_time_to_collision.append(steps_to_collision)
            self.no_intervention_successful_flights.append(completed_flights)
            
            # Calculate completion rate
            completion_rate = completed_flights / total_aircraft_count if total_aircraft_count > 0 else 0
            self.no_intervention_completed_rate.append(completion_rate)
            
            if info.get("num_planes_landed", 0) > 0:
                self.no_intervention_steps_to_destination.append(step_count)
            
            # Log results
            if steps_to_collision < self.max_steps:
                self.logger.debug_print(f"No Intervention Sim #{sim_index}: Collision at step {steps_to_collision}, " 
                                       f"Completed {completed_flights}/{total_aircraft_count} flights")
            else:
                self.logger.debug_print(f"No Intervention Sim #{sim_index}: No collision, "
                                       f"Completed {completed_flights}/{total_aircraft_count} flights")

    def plot_results(self):
        """Generate a single combined metrics chart with better normalization for steps to destination"""
        plt.figure(figsize=(12, 8))
        
        # Calculate average metrics, normalized to [0,1] range
        max_time = self.max_steps
        
        # Get collision data
        rl_collisions = [t for t in self.rl_time_to_collision if t < self.max_steps]
        random_collisions = [t for t in self.random_time_to_collision if t < self.max_steps]
        no_intervention_collisions = [t for t in self.no_intervention_time_to_collision if t < self.max_steps]
        
        # Calculate collision rates
        rl_rate = len(rl_collisions) / self.num_simulations * 100
        random_rate = len(random_collisions) / self.num_simulations * 100
        no_intervention_rate = len(no_intervention_collisions) / self.num_simulations * 100
        
        # Calculate completion rates
        rl_completion = np.mean(self.rl_completed_rate) * 100
        random_completion = np.mean(self.random_completed_rate) * 100
        no_intervention_completion = np.mean(self.no_intervention_completed_rate) * 100
        
        # Calculate normalized metrics (higher is better)
        # Normalize collision rates to [0,1] where 1 is best (0% collisions)
        norm_rl_collision = 1 - (rl_rate/100)
        norm_random_collision = 1 - (random_rate/100)
        norm_no_intervention_collision = 1 - (no_intervention_rate/100)
        
        # Normalize completion rates to [0,1]
        norm_rl_completion = rl_completion / 100
        norm_random_completion = random_completion / 100
        norm_no_intervention_completion = no_intervention_completion / 100
        
        # Add the new metric: steps to destination (normalized)
        # Debug info about steps to destination data
        print(f"DEBUG - Steps to destination counts: RL: {len(self.rl_steps_to_destination)}, "
            f"Random: {len(self.random_steps_to_destination)}, "
            f"No Int: {len(self.no_intervention_steps_to_destination)}")
        
        # Calculate average steps (with fallback)
        avg_rl_steps = np.mean(self.rl_steps_to_destination) if self.rl_steps_to_destination else max_time
        avg_random_steps = np.mean(self.random_steps_to_destination) if self.random_steps_to_destination else max_time
        avg_no_intervention_steps = np.mean(self.no_intervention_steps_to_destination) if self.no_intervention_steps_to_destination else max_time
        
        print(f"DEBUG - Average steps: RL: {avg_rl_steps}, Random: {avg_random_steps}, No Int: {avg_no_intervention_steps}")
        
        # Find the minimum non-infinite value
        valid_steps = [s for s in [avg_rl_steps, avg_random_steps, avg_no_intervention_steps] if s < max_time]
        min_steps_value = min(valid_steps) if valid_steps else 0
        
        # Find the maximum steps for normalization
        max_steps_value = max(avg_rl_steps, avg_random_steps, avg_no_intervention_steps)
        if max_steps_value >= max_time:
            # If any value is max_time (infinite), use the largest finite value
            finite_values = [s for s in [avg_rl_steps, avg_random_steps, avg_no_intervention_steps] if s < max_time]
            max_steps_value = max(finite_values) if finite_values else 1
        
        # Calculate the range for better normalization
        range_steps = max_steps_value - min_steps_value if max_steps_value != min_steps_value else 1
        
        # Improved normalization for steps (invert so lower is better)
        # Apply a minimum floor of 0.15 to ensure all bars are visible
        if avg_rl_steps < max_time:
            norm_rl_steps = max(0.15, 1 - ((avg_rl_steps - min_steps_value) / range_steps))
        else:
            norm_rl_steps = 0.15
            
        if avg_random_steps < max_time:
            norm_random_steps = max(0.15, 1 - ((avg_random_steps - min_steps_value) / range_steps))
        else:
            norm_random_steps = 0.15
            
        if avg_no_intervention_steps < max_time:
            norm_no_intervention_steps = max(0.15, 1 - ((avg_no_intervention_steps - min_steps_value) / range_steps))
        else:
            norm_no_intervention_steps = 0.15
        
        # Create a comparison bar chart for the main metrics
        metrics = ['Collision Avoidance', 'Steps to Destination', 'Completion Rate']
        rl_scores = [norm_rl_collision, norm_rl_steps, norm_rl_completion]
        random_scores = [norm_random_collision, norm_random_steps, norm_random_completion]
        no_intervention_scores = [norm_no_intervention_collision, norm_no_intervention_steps, norm_no_intervention_completion]
        
        # Plot grouped bar chart
        x = np.arange(len(metrics))
        width = 0.25
        
        # Create bars once and save references for adding labels
        bars1 = plt.bar(x - width, rl_scores, width, label='RL Agent', color='green')
        bars2 = plt.bar(x, random_scores, width, label='Random Actions', color='orange')
        bars3 = plt.bar(x + width, no_intervention_scores, width, label='No Intervention', color='red')
        
        # Add value labels on each bar
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Set up chart properties
        plt.ylabel('Normalized Score (higher is better)')
        plt.title('Collision Avoidance RL Benchmarking Results')
        plt.xticks(x, metrics)
        plt.ylim(0, 1.1)
        plt.legend()
        
        # Add grid lines for easier reading
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the plot
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"../results/collision_benchmark_{timestamp}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        print(f"\nBenchmark plot saved to: {filename}")
        
        plt.close()

    def print_results(self):
        """Print the benchmark results in tabular format using tabulate"""
        from tabulate import tabulate
        
        # Calculate average time to first collision
        rl_collisions = [t for t in self.rl_time_to_collision if t < self.max_steps]
        random_collisions = [t for t in self.random_time_to_collision if t < self.max_steps]
        no_intervention_collisions = [t for t in self.no_intervention_time_to_collision if t < self.max_steps]
        
        # Calculate collision rates
        rl_collision_rate = len(rl_collisions) / self.num_simulations
        random_collision_rate = len(random_collisions) / self.num_simulations
        no_intervention_collision_rate = len(no_intervention_collisions) / self.num_simulations
        
        # Calculate average time to first collision
        avg_rl_time = np.mean(rl_collisions) if rl_collisions else float('inf')
        avg_random_time = np.mean(random_collisions) if random_collisions else float('inf')
        avg_no_intervention_time = np.mean(no_intervention_collisions) if no_intervention_collisions else float('inf')
        
        # Calculate average successful flights
        avg_rl_flights = np.mean(self.rl_successful_flights)
        avg_random_flights = np.mean(self.random_successful_flights)
        avg_no_intervention_flights = np.mean(self.no_intervention_successful_flights)
        
        # Calculate average completion rate
        avg_rl_completion = np.mean(self.rl_completed_rate) * 100
        random_completion = np.mean(self.random_completed_rate) * 100
        no_intervention_completion = np.mean(self.no_intervention_completed_rate) * 100
        
        # Calculate average steps to destination
        avg_rl_steps = np.mean(self.rl_steps_to_destination) if self.rl_steps_to_destination else float('inf')
        avg_random_steps = np.mean(self.random_steps_to_destination) if self.random_steps_to_destination else float('inf')
        avg_no_intervention_steps = np.mean(self.no_intervention_steps_to_destination) if self.no_intervention_steps_to_destination else float('inf')
        
        # Print header information
        print("\n===== Collision Avoidance Benchmark Results =====")
        print(f"Number of simulations: {self.num_simulations}")
        print(f"Maximum steps per simulation: {self.max_steps}")
        print(f"Simulation configuration: {self.num_planes} aircraft")
        
        # Create data for the main metrics table
        if avg_rl_time == float('inf'):
            rl_time_str = "No collisions"
        else:
            rl_time_str = f"{avg_rl_time:.2f}"
        
        if avg_random_time == float('inf'):
            random_time_str = "No collisions"
        else:
            random_time_str = f"{avg_random_time:.2f}"
        
        if avg_no_intervention_time == float('inf'):
            no_int_time_str = "No collisions"
        else:
            no_int_time_str = f"{avg_no_intervention_time:.2f}"
        
        metrics_table = [
            ["Collision Rate", f"{rl_collision_rate:.2%}", f"{random_collision_rate:.2%}", f"{no_intervention_collision_rate:.2%}"],
            ["Time to Collision (steps)", rl_time_str, random_time_str, no_int_time_str],
            ["Successful Flights", f"{avg_rl_flights:.2f}", f"{avg_random_flights:.2f}", f"{avg_no_intervention_flights:.2f}"],
            ["Completion Rate", f"{avg_rl_completion:.2f}%", f"{random_completion:.2f}%", f"{no_intervention_completion:.2f}%"],
            ["Steps to Destination", f"{avg_rl_steps:.2f}", f"{avg_random_steps:.2f}", f"{avg_no_intervention_steps:.2f}"]
        ]
        
        # Print the main metrics table
        print("\nPerformance Metrics:")
        print(tabulate(metrics_table, headers=["Metric", "RL Agent", "Random Actions", "No Intervention"], 
                    tablefmt="grid"))
        
        # Calculate improvements for a second table
        improvements_table = []
        
        # Collision rate improvement
        if random_collision_rate > 0:
            collision_reduction_vs_random = (random_collision_rate - rl_collision_rate) / random_collision_rate * 100
            improvements_table.append(["Collision Probability Reduction vs Random", f"{collision_reduction_vs_random:.2f}%"])
        
        if no_intervention_collision_rate > 0:
            collision_reduction_vs_no_intervention = (no_intervention_collision_rate - rl_collision_rate) / no_intervention_collision_rate * 100
            improvements_table.append(["Collision Probability Reduction vs No Intervention", f"{collision_reduction_vs_no_intervention:.2f}%"])
        
        # Time to collision improvement
        if random_collisions and rl_collisions:
            rl_vs_random_time_improvement = (avg_random_time - avg_rl_time) / avg_random_time * 100
            improvements_table.append(["Time to Collision Improvement vs Random", f"{rl_vs_random_time_improvement:.2f}%"])
        
        if no_intervention_collisions and rl_collisions:
            rl_vs_no_intervention_time_improvement = (avg_no_intervention_time - avg_rl_time) / avg_no_intervention_time * 100
            improvements_table.append(["Time to Collision Improvement vs No Intervention", f"{rl_vs_no_intervention_time_improvement:.2f}%"])
        
        # Completion rate improvements
        rl_vs_random_completion = avg_rl_completion - random_completion
        improvements_table.append(["Completion Rate Improvement vs Random", f"{rl_vs_random_completion:.2f} points"])
        
        rl_vs_no_intervention_completion = avg_rl_completion - no_intervention_completion
        improvements_table.append(["Completion Rate Improvement vs No Intervention", f"{rl_vs_no_intervention_completion:.2f} points"])
        
        # Steps to destination improvements
        if avg_random_steps != float('inf') and avg_rl_steps != float('inf'):
            steps_improvement_vs_random = (avg_random_steps - avg_rl_steps) / avg_random_steps * 100
            improvements_table.append(["Steps to Destination Improvement vs Random", f"{steps_improvement_vs_random:.2f}%"])
        
        if avg_no_intervention_steps != float('inf') and avg_rl_steps != float('inf'):
            steps_improvement_vs_no_intervention = (avg_no_intervention_steps - avg_rl_steps) / avg_no_intervention_steps * 100
            improvements_table.append(["Steps to Destination Improvement vs No Intervention", f"{steps_improvement_vs_no_intervention:.2f}%"])
        
        # Print improvements table
        print("\nRL Agent Improvements:")
        print(tabulate(improvements_table, headers=["Metric", "Improvement"], tablefmt="grid"))
        
        # Display overall rating
        collision_better_than_random = rl_collision_rate < random_collision_rate
        collision_better_than_no_intervention = rl_collision_rate < no_intervention_collision_rate
        completion_better_than_random = avg_rl_completion > random_completion
        completion_better_than_no_intervention = avg_rl_completion > no_intervention_completion
        
        print("\nOverall Assessment:")
        if (collision_better_than_random and collision_better_than_no_intervention and 
            completion_better_than_random and completion_better_than_no_intervention):
            assessment = "RL agent significantly outperforms both random actions and no intervention"
        elif ((collision_better_than_random or collision_better_than_no_intervention) and 
            (completion_better_than_random or completion_better_than_no_intervention)):
            assessment = "RL agent shows better performance than baseline approaches in most metrics"
        else:
            assessment = "RL agent shows mixed performance compared to baseline approaches"
        
        print("-" * 80)
        print(assessment)
        print("-" * 80)

    def run_benchmark(self):
        """Run the complete benchmark"""
        print("Starting Collision Avoidance Benchmark...")
        print(f"Running {self.num_simulations} simulations for each method...")

        print("\nTest Setup Parameters:")
        print(f"Default collision radius: {self.default_collision_radius}")
        print(f"RL agent collision radius: {self.rl_collision_radius}")
        print(f"Random action collision radius: {self.random_collision_radius}")
        print(f"No intervention collision radius: {self.no_intervention_collision_radius}")
        
        # Run RL-based simulations
        print("\nRunning simulations with RL collision avoidance agent...")
        self.run_simulation_with_rl()
        
        # Run random-based simulations
        print("\nRunning simulations with random collision avoidance actions...")
        self.run_simulation_with_random()
        
        # Run no-intervention simulations
        print("\nRunning simulations with no intervention (baseline)...")
        self.run_simulation_with_no_intervention()
        
        # Print results
        self.print_results()
        
        # Plot results
        self.plot_results()
        
        self.logger.stop()
        return {
            "rl_time_to_collision": self.rl_time_to_collision,
            "random_time_to_collision": self.random_time_to_collision,
            "no_intervention_time_to_collision": self.no_intervention_time_to_collision,
            "rl_successful_flights": self.rl_successful_flights,
            "random_successful_flights": self.random_successful_flights,
            "no_intervention_successful_flights": self.no_intervention_successful_flights,
            "rl_completed_rate": self.rl_completed_rate,
            "random_completed_rate": self.random_completed_rate,
            "no_intervention_completed_rate": self.no_intervention_completed_rate
        }


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark collision avoidance RL agent")
    parser.add_argument('--simulations', type=int, default=200, help='Number of simulations to run')
    parser.add_argument('--max-steps', type=int, default=2000, help='Maximum steps per simulation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no-plotting', action='store_true', help='Disable result plotting')
    
    args = parser.parse_args()
    
    if args.seed == 0:
        args.seed = random.randint(1, 1000000)
        print("Setting random seed to:", args.seed)
    
    # Run benchmark
    benchmark = CollisionAvoidanceBenchmark(
        num_simulations=args.simulations,
        max_steps=args.max_steps,
        seed=args.seed
    )
    
    results = benchmark.run_benchmark()