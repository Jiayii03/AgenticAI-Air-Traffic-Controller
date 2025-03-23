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
    
    def __init__(self, model_path='models/dqn_atc_model.pth', debug=False):
        """Initialize the RL controller with trained model"""
        # Set up model parameters matching your training configuration
        self.state_dim = 5  # Updated to match your training state dimensions
        self.action_dim = 5  # Number of actions: maintain course, medium/slight left/right
        self.hidden_dim = 128  # Match your training hidden layer size
        self.debug = debug
        
        # Initialize the Q network with the same architecture as in training
        self.q_network = QNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        
        if os.path.exists(model_path):
            try:
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
            except Exception as e:
                print(f"Error loading model: {e}")
                print("RL agent will use untrained model.")
        else:
            print(f"Warning: Model file {model_path} not found. RL agent will use untrained model.")
        
        # Initialize collision detection parameters
        self.collision_risk_radius = conf.get()["rl_agent"]["collision_risk_radius"]
        self.cooldown_frames = conf.get()["rl_agent"]["cooldown_frames"]
        self.tactical_waypoint_distance = conf.get()["rl_agent"]["tactical_waypoint_distance"]
        self.aircraft_cooldowns = {}  
        self.current_frame = 0        # Frame counter
        
    def detect_collision_risks(self, aircraft_list):
        """Detect pairs of aircraft that are at risk of collision"""
        aircraft_pairs = []
        self.current_frame += 1  # Increment frame counter
        
        for i, ac1 in enumerate(aircraft_list):
            for j, ac2 in enumerate(aircraft_list):
                if i < j:  # Check each pair only once
                    dist = Utility.locDist(ac1.getLocation(), ac2.getLocation())
                    if dist < self.collision_risk_radius:
                        # Skip if either aircraft is on cooldown
                        if (ac1.getIdent() in self.aircraft_cooldowns and 
                            self.current_frame - self.aircraft_cooldowns[ac1.getIdent()] < self.cooldown_frames):
                            if self.debug:
                                print(f"Aircraft {ac1.getIdent()} is in collision risk, but on cooldown, skipping")
                            continue
                        if (ac2.getIdent() in self.aircraft_cooldowns and 
                            self.current_frame - self.aircraft_cooldowns[ac2.getIdent()] < self.cooldown_frames):
                            if self.debug:
                                print(f"Aircraft {ac2.getIdent()} is in collision risk, but on cooldown, skipping")
                            continue
                        
                        # if not on cooldown, add to aircraft pairs
                        aircraft_pairs.append((ac1, ac2))
                        print(f"Detected collision risk between {ac1.getIdent()} and {ac2.getIdent()}, distance: {dist:.2f}")
        
        return aircraft_pairs
    
    def get_observation(self, ac1, ac2, obstacles=None, destinations=None):
        """Create an observation vector for the given aircraft pair"""
        # Calculate distance between aircraft
        distance = Utility.locDist(ac1.getLocation(), ac2.getLocation())
        
        # Calculate angle from ac1 to ac2
        angle = self._calculate_angle(ac1.getLocation(), ac2.getLocation())
        
        # Calculate relative heading
        rel_heading = (angle - ac1.getHeading()) % 360
        
        # Calculate destination-related observations
        destination_angle = 0.0
        angular_diff_to_dest = 0.0
        
        if len(ac1.waypoints) > 0:
            dest = ac1.waypoints[-1].getLocation()  # Assuming destination is last waypoint
            destination_angle = self._calculate_angle(ac1.getLocation(), dest)
            
            # Calculate angular difference between current heading and destination
            angular_diff_to_dest = abs((ac1.getHeading() - destination_angle) % 360)
            if angular_diff_to_dest > 180:
                angular_diff_to_dest = 360 - angular_diff_to_dest
        
        # Create observation vector matching training format
        observation = np.array([
            distance,
            angle,
            rel_heading,
            destination_angle,
            angular_diff_to_dest
        ], dtype=np.float32)
        
        if self.debug:
            print(f"Observation: {observation}")
        
        return observation
    
    def select_action(self, observation):
        """Select the best action based on current observation"""
        # Convert observation to tensor
        state_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        # Get action with highest Q-value
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
            
            if self.debug:
                print(f"Q-values: {q_values.numpy()}")
                print(f"Selected action: {action}")
        
        return action

    def apply_action(self, aircraft, action):
        """Apply the selected action to aircraft by modifying its waypoints"""
        # Get current location
        curr_loc = aircraft.getLocation()
        curr_heading = aircraft.getHeading()
        
        # Store original destination (last waypoint)
        destination = None
        had_tactical_waypoint = False
        
        # Store all waypoints and check if there's more than just destination
        if len(aircraft.waypoints) > 0:
            destination = aircraft.waypoints[-1]  # Last waypoint is destination
            had_tactical_waypoint = len(aircraft.waypoints) > 1
        
        # If aircraft already has tactical waypoints and action is to maintain course,
        # don't modify the waypoints at all
        if had_tactical_waypoint and action == 0:
            if self.debug:
                print(f"Aircraft {aircraft.getIdent()}: Maintaining existing waypoints")
            return
        
        # Clear existing waypoints only if we're going to modify them
        aircraft.waypoints = []
        
        # Add destination back if it exists
        if destination:
            aircraft.addWaypoint(destination)
        else:
            # If no destination exists, just maintain current heading
            if self.debug:
                print(f"Aircraft {aircraft.getIdent()}: No destination found!")
            return
        
        # Apply tactical waypoint if action is not "maintain course"
        if action != 0:
            # Apply heading change based on action
            if action == 1:  # ML - Medium left (90° turn)
                new_heading = (curr_heading - 90) % 360
            elif action == 2:  # SL - Slight left (45° turn)
                new_heading = (curr_heading - 45) % 360
            elif action == 3:  # MR - Medium right (90° turn)
                new_heading = (curr_heading + 90) % 360
            elif action == 4:  # SR - Slight right (45° turn)
                new_heading = (curr_heading + 45) % 360
            else:  # N - maintain course
                new_heading = curr_heading
        
            # Calculate new waypoint location
            distance = self.tactical_waypoint_distance
            rad_heading = np.radians(new_heading)
            dx = distance * np.sin(rad_heading)
            dy = -distance * np.cos(rad_heading)
            new_waypoint = (int(curr_loc[0] + dx), int(curr_loc[1] + dy))
            
            # Create and insert tactical waypoint
            tactical_waypoint = Waypoint(new_waypoint)
            aircraft.waypoints.insert(0, tactical_waypoint)  # Insert at beginning
            
            if self.debug:
                print(f"Aircraft {aircraft.getIdent()}: Action={action}, Applied tactical waypoint at {new_waypoint}")
        else:
            if self.debug:
                print(f"Aircraft {aircraft.getIdent()}: Direct routing to destination")
    
    def _mirror_action(self, action):
        """Mirror an action for the intruder aircraft.
        
        Action mapping:
        - 0: Maintain course (N) -> 0: Maintain course (N)
        - 1: Medium left (ML) -> 3: Medium right (MR)
        - 2: Slight left (SL) -> 4: Slight right (SR)
        - 3: Medium right (MR) -> 1: Medium left (ML)
        - 4: Slight right (SR) -> 2: Slight left (SL)
        """
        if action == 1:  # ML -> MR
            return 3
        elif action == 2:  # SL -> SR
            return 4
        elif action == 3:  # MR -> ML
            return 1
        elif action == 4:  # SR -> SL
            return 2
        else:
            return action  # N remains N
    
    def _calculate_angle(self, point1, point2):
        """Calculate angle from point1 to point2 in degrees (0-360)"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        angle = np.degrees(np.arctan2(dy, dx))
        angle = (angle + 360) % 360
        
        return angle