# simulation/atc_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import Simulation
from utility import Utility
import conf

class ATCEnv(gym.Env):
    """
    Air Traffic Control environment for OpenAI Gym.
    
    State Space:
    - Distance to intruder aircraft (d): [0, 50] pixels, discretized into 50 buckets
    - Angle to intruder (ρ): [0, 360) degrees, discretized into 36 buckets of 10 degrees
    - Relative heading of intruder (θ): [0, 360) degrees, discretized into 36 buckets of 10 degrees
    
    Action Space:
    - 0: Maintain course (N)
    - 1: Hard left (HL)
    - 2: Medium left (ML)
    - 3: Medium right (MR)
    - 4: Hard right (HR)
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, num_planes=5, num_obstacles=3, screen_size=(800, 800)):
        super(ATCEnv, self).__init__()
        
        # Initialize the simulation
        self.simulation = Simulation(num_planes, num_obstacles, screen_size)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # 5 possible actions
        
        # Observation space based on state variables
        # For the SARSA agent focusing on collision avoidance:
        # - distance to nearest aircraft: [0, 50] pixels
        # - angle to nearest aircraft: [0, 360) degrees
        # - relative heading of nearest aircraft: [0, 360) degrees
        # - distance to nearest obstacle: [0, 50] pixels
        # - angle to nearest obstacle: [0, 360) degrees
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),    # Added two dimensions for obstacle info
            high=np.array([50, 360, 360, 50, 360]),
            dtype=np.float32
        )
        
        # Collision detection threshold
        self.collision_radius = conf.get()['aircraft']['collision_radius']
        
        # Target aircraft pair for learning (will be updated in reset)
        self.pair_index = 0
        self.aircraft_pairs = []
    
    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        
        # Reset the simulation
        state = self.simulation.reset()
        print(f"Reset state: {state}")
        print(f"Number of aircraft: {len(state['aircraft'])}")
        print(f"Aircraft IDs: {[ac['id'] for ac in state['aircraft']]}")
        print(f"Aircraft positions: {[ac['location'] for ac in state['aircraft']]}")

        # Identify pairs of aircraft that are within threshold distance
        self._update_aircraft_pairs(state)
        
        # If no aircraft pairs in potential collision course, return a default observation
        if not self.aircraft_pairs:
            return np.zeros(3, dtype=np.float32), {}
        
        # Get observation for the first pair
        self.pair_index = 0
        observation = self._get_observation(self.aircraft_pairs[self.pair_index])
        
        return observation, {}
    
    def step(self, action):
        """Execute action and advance the environment by one timestep."""
        # Get the current aircraft pair
        if not self.aircraft_pairs:
            # No pairs to control, just step the simulation with action 0, i.e. maintain course
            state, rewards_dict, done, info = self.simulation.step([0] * len(self.simulation.aircraft))
            observation = np.zeros(3, dtype=np.float32)
            # +ve rewards for successful landing, -ve for aircraft and obstacle collision 
            reward = sum(rewards_dict.values())
            self._update_aircraft_pairs(state)
            return observation, reward, done, False, info
            
        # Get the pair of aircraft we're controlling
        ac1_id, ac2_id = self.aircraft_pairs[self.pair_index]
        
        # Prepare actions for all aircraft
        all_actions = []
        for ac in self.simulation.aircraft:
            if ac.getIdent() == ac1_id:
                all_actions.append(action)
            elif ac.getIdent() == ac2_id:
                # For the intruder aircraft, mirror the action to create symmetric behavior
                mirrored_action = self._mirror_action(action)
                all_actions.append(mirrored_action)
            else:
                # Other aircraft maintain course
                all_actions.append(0)
        
        # Execute actions in simulation
        state, rewards_dict, done, info = self.simulation.step(all_actions)
        
        # Update aircraft pairs if there are any changes
        self._update_aircraft_pairs(state)
        
        # Get simulation reward for target aircraft
        sim_reward = 0
        if ac1_id in rewards_dict:
            sim_reward += rewards_dict[ac1_id]
        if ac2_id in rewards_dict:
            sim_reward += rewards_dict[ac2_id]
            
        # Calculate additional reward for the target aircraft pair
        additional_reward = self._calculate_reward(state, ac1_id, ac2_id)
        
        # Total reward is the combination
        reward = sim_reward + additional_reward
        
        # Get new observation
        if not self.aircraft_pairs:
            # No more pairs, return a default observation
            observation = np.zeros(3, dtype=np.float32)
        elif self.pair_index >= len(self.aircraft_pairs):
            # Reset to the first pair if current index is out of bounds
            self.pair_index = 0
            observation = self._get_observation(self.aircraft_pairs[self.pair_index])
        else:
            # Get observation for the current pair
            observation = self._get_observation(self.aircraft_pairs[self.pair_index])
        
        return observation, reward, done, False, info
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.simulation.render(mode)
    
    def close(self):
        """Clean up resources."""
        pygame.quit()
    
    def _update_aircraft_pairs(self, state):
        """Update the list of aircraft pairs that are in potential collision course."""
        self.aircraft_pairs = []
            
        for i, ac1 in enumerate(state['aircraft']):
            for j, ac2 in enumerate(state['aircraft']):
                if i < j:  # Check each pair only once
                    dist = Utility.locDist(ac1['location'], ac2['location'])
                    if dist < 50:  # Threshold distance for potential collision
                        self.aircraft_pairs.append((ac1['id'], ac2['id']))
        
        print(f"Found {len(self.aircraft_pairs)} aircraft pairs in potential collision")
    
    def _get_observation(self, aircraft_pair):
        # Get original aircraft observations
        ac1_id, ac2_id = aircraft_pair
        ac1 = self._find_aircraft_by_id(ac1_id)
        ac2 = self._find_aircraft_by_id(ac2_id)
        
        if ac1 is None or ac2 is None:
            return np.zeros(5, dtype=np.float32)
        
        # Calculate aircraft-related observations
        distance = Utility.locDist(ac1.getLocation(), ac2.getLocation())
        angle = self._calculate_angle(ac1.getLocation(), ac2.getLocation())
        rel_heading = (angle - ac1.getHeading()) % 360
        
        # Find nearest obstacle
        nearest_obstacle = None
        min_obstacle_dist = float('inf')
        
        for obs in self.simulation.obstacles:
            dist = Utility.locDist(ac1.getLocation(), obs.location)
            if dist < min_obstacle_dist:
                min_obstacle_dist = dist
                nearest_obstacle = obs
        
        # Default obstacle values
        obstacle_dist = 50.0  # Maximum value if no obstacle
        obstacle_angle = 0.0
        
        # Calculate obstacle-related observations
        if nearest_obstacle:
            obstacle_dist = min(Utility.locDist(ac1.getLocation(), nearest_obstacle.location), 50.0)
            obstacle_angle = self._calculate_angle(ac1.getLocation(), nearest_obstacle.location)
        
        return np.array([distance, angle, rel_heading, obstacle_dist, obstacle_angle], dtype=np.float32)
    
    def _calculate_reward(self, state, ac1_id, ac2_id):
        """Calculate reward based on aircraft states."""
        # Find the aircraft states
        ac1_state = None
        ac2_state = None
        for ac_state in state['aircraft']:
            if ac_state['id'] == ac1_id:
                ac1_state = ac_state
            elif ac_state['id'] == ac2_id:
                ac2_state = ac_state
        
        if ac1_state is None or ac2_state is None:
            # One of the aircraft no longer exists
            return 0
        
        # Calculate distance
        distance = Utility.locDist(ac1_state['location'], ac2_state['location'])
        
        # Reward components
        # 1. Distance reward: penalize getting too close
        radius = self.collision_radius
        # For a proper penalty that's 0 when far apart and negative when close:
        distance_reward = 0
        if distance < radius:
            distance_reward = -500 * (1 - (distance / radius)**2)
        
        # 2. Destination reward: reward progress toward destination
        destination_reward = 0
        if len(ac1_state['waypoints']) > 0:
            dest = ac1_state['waypoints'][-1]
            distance_to_go = Utility.locDist(ac1_state['location'], dest)
            # Normalize by screen size or maximum possible distance
            max_distance = 1000  # Maximum possible distance estimate
            destination_reward = 50 * (1 - min(distance_to_go / max_distance, 1))
        
        # Total reward
        reward = distance_reward + 0.1 * destination_reward
        
        return max(min(reward, 100), -100)  # Limit between -100 and 100 per step
    
    def _mirror_action(self, action):
        """Mirror an action for the intruder aircraft. Actions are defined in simulation > _apply_action()."""
        if action == 1:  # HL -> HR
            return 4
        elif action == 2:  # ML -> MR
            return 3
        elif action == 3:  # MR -> ML
            return 2
        elif action == 4:  # HR -> HL
            return 1
        else:
            return action  # N remains N
        
    def _find_closest_aircraft_pair(self):
        """Find the closest pair of aircraft even if not in collision risk."""
        min_dist = float('inf')
        closest_pair = None
        
        aircraft_list = self.simulation.aircraft
        for i, ac1 in enumerate(aircraft_list):
            for j, ac2 in enumerate(aircraft_list):
                if i < j:  # Check each pair only once
                    dist = Utility.locDist(ac1.getLocation(), ac2.getLocation())
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (ac1.getIdent(), ac2.getIdent())
        
        print(f"Closest pair distance: {min_dist}")
        return closest_pair
    
    def _make_serializable(self, state):
        """Convert state to JSON-serializable format."""
        if isinstance(state, dict):
            return {k: self._make_serializable(v) for k, v in state.items()}
        elif isinstance(state, list):
            return [self._make_serializable(item) for item in state]
        elif isinstance(state, tuple):
            return list(self._make_serializable(item) for item in state)
        elif hasattr(state, '__dict__'):
            # For objects like obstacles
            return str(state)
        else:
            return state