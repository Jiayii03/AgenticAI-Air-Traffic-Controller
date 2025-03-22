# File: rl_controller.py
# This file should be placed in the main game directory

import os
import sys
import numpy as np
import torch
from simulation.dqn_agent import QNetwork
from utility import Utility
from waypoint import Waypoint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conf

class RLController:
    """Controls aircraft using a trained DQN model when collision risks are detected"""
    
    def __init__(self, model_path='models/dqn_atc_model.pth'):
        """Initialize the RL controller with trained model"""
        # Set up model parameters matching your training configuration
        self.state_dim = 7  # Match your training state dimensions
        self.action_dim = 5  # Number of actions: maintain, hard left/right, medium left/right
        self.hidden_dim = 128  # Match your training hidden layer size
        
        # Initialize the Q network with the same architecture as in training
        self.q_network = QNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            
            # Check if this is a full agent checkpoint or just network weights
            if "q_network" in checkpoint:
                # This is a full agent checkpoint, load just the q_network part
                self.q_network.load_state_dict(checkpoint["q_network"])
                print(f"RL agent loaded q_network from full checkpoint: {model_path}")
            else:
                # This is a direct network state dict
                self.q_network.load_state_dict(checkpoint)
                print(f"RL agent loaded network weights from: {model_path}")
            
            self.q_network.eval()  # Set to evaluation mode
        else:
            print(f"Warning: Model file {model_path} not found. RL agent will use untrained model.")
        
        # Initialize collision detection parameters
        self.collision_risk_radius = conf.get()["rl_agent"]["collision_risk_radius"]
        self.cooldown_frames = conf.get()["rl_agent"]["cooldown_frames"]  # Number of frames to wait before another RL action
        self.aircraft_cooldowns = {}  
        self.current_frame = 0        # Frame counter
        
    def detect_collision_risks(self, aircraft_list):
        """Detect pairs of aircraft that are at risk of collision"""
        collision_pairs = []
        self.current_frame += 1  # Increment frame counter
        
        for i, ac1 in enumerate(aircraft_list):
            for j, ac2 in enumerate(aircraft_list):
                if i < j:  # Check each pair only once
                    # Skip if either aircraft is on cooldown
                    if (ac1.getIdent() in self.aircraft_cooldowns and 
                        self.current_frame - self.aircraft_cooldowns[ac1.getIdent()] < self.cooldown_frames):
                        continue
                    if (ac2.getIdent() in self.aircraft_cooldowns and 
                        self.current_frame - self.aircraft_cooldowns[ac2.getIdent()] < self.cooldown_frames):
                        continue
                    
                    dist = Utility.locDist(ac1.getLocation(), ac2.getLocation())
                    if dist < self.collision_risk_radius:
                        collision_pairs.append((ac1, ac2))
        
        return collision_pairs
    
    def get_observation(self, ac1, ac2, obstacles, destinations):
        """Create an observation vector for the given aircraft pair"""
        # Calculate distance between aircraft
        distance = min(Utility.locDist(ac1.getLocation(), ac2.getLocation()), 50.0)
        
        # Calculate angle from ac1 to ac2
        angle = self._calculate_angle(ac1.getLocation(), ac2.getLocation())
        
        # Calculate relative heading
        rel_heading = (angle - ac1.getHeading()) % 360
        
        # Find nearest obstacle
        nearest_obstacle = None
        min_obstacle_dist = float('inf')
        
        for obs in obstacles:
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
        
        # Calculate destination-related observations
        destination_dist = 1000.0  # Default maximum
        destination_angle = 0.0
        
        if len(ac1.waypoints) > 0:
            dest = ac1.waypoints[-1].getLocation()  # Assuming destination is last waypoint
            destination_dist = Utility.locDist(ac1.getLocation(), dest)
            destination_angle = self._calculate_angle(ac1.getLocation(), dest)
        
        # Create observation vector (same format as in training)
        observation = np.array([
            distance, 
            angle, 
            rel_heading, 
            obstacle_dist, 
            obstacle_angle,
            destination_dist,
            destination_angle
        ], dtype=np.float32)
        
        return observation
    
    def select_action(self, observation):
        """Select the best action based on current observation"""
        # Convert observation to tensor
        state_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        # Get action with highest Q-value
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        
        return action
    
    def apply_action(self, aircraft, action):
        """Apply the selected action to aircraft by modifying its waypoints"""
        # Get current location and heading
        curr_loc = aircraft.getLocation()
        curr_heading = aircraft.getHeading()
        
        # Store original destination (last waypoint)
        destination = None
        if len(aircraft.waypoints) > 0:
            destination = aircraft.waypoints[-1]
        
        # Clear existing waypoints except destination
        aircraft.waypoints = []
        
        # Add destination back if it exists
        if destination:
            aircraft.addWaypoint(destination)
        else:
            return  # No destination, can't add waypoints
        
        # Only add tactical waypoint if changing course (action != 0)
        if action != 0:
            # Apply heading change based on action
            if action == 1:  # Hard left
                new_heading = (curr_heading - 90) % 360
            elif action == 2:  # Medium left
                new_heading = (curr_heading - 45) % 360
            elif action == 3:  # Medium right
                new_heading = (curr_heading + 45) % 360
            elif action == 4:  # Hard right
                new_heading = (curr_heading + 90) % 360
            else:
                new_heading = curr_heading
            
            # Calculate new waypoint location
            distance =  conf.get()["rl_agent"]["tactical_waypoint_distance"] # Distance for tactical waypoint
            rad_heading = np.radians(new_heading)
            dx = distance * np.sin(rad_heading)
            dy = -distance * np.cos(rad_heading)
            new_waypoint_loc = (int(curr_loc[0] + dx), int(curr_loc[1] + dy))
            
            # Create and insert tactical waypoint
            tactical_waypoint = Waypoint(new_waypoint_loc)
            aircraft.waypoints.insert(0, tactical_waypoint)  # Insert at beginning
    
    def _calculate_angle(self, point1, point2):
        """Calculate angle from point1 to point2 in degrees (0-360)"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        angle = np.degrees(np.arctan2(dy, dx))
        angle = (angle + 360) % 360
        
        return angle