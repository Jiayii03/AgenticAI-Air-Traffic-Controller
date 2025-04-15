# File: rl_controller.py
# This file should be placed in the main game directory

import os
import sys
import numpy as np
import math
import torch
from simulation.collision_avoidance.dqn_agent import QNetwork
from core.utility import Utility
from core.waypoint import Waypoint
from core import conf

class RLController:
    """Controls aircraft using a trained DQN model when collision risks are detected"""
    
    def __init__(self, model_path='models/dqn_atc_model.pth', debug=True):
        """Initialize the RL controller with trained model"""
        # Set up model parameters matching your training configuration
        self.state_dim = 7  # Updated to include relative speed and separation rate
        self.action_dim = 9  # Updated for additional speed-related actions
        self.hidden_dim = 128  # Match your training hidden layer size
        self.debug = debug
        
        # Initialize the Q network with the same architecture as in training
        self.q_network = QNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        
        # Resolve the path relative to the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        resolved_path = os.path.join(base_dir, 'models', os.path.basename(model_path))
    
        if self.debug:
            print(f"Looking for model at: {resolved_path}")
        
        if os.path.exists(resolved_path):
            try:
                checkpoint = torch.load(resolved_path, weights_only=True)
                
                # Check if this is a full agent checkpoint or just network weights
                if "q_network" in checkpoint:
                    # This is a full agent checkpoint, load just the q_network part
                    self.q_network.load_state_dict(checkpoint["q_network"])
                    print(f"RL agent loaded q_network from full checkpoint: {resolved_path}")
                else:
                    # This is a direct network state dict
                    self.q_network.load_state_dict(checkpoint)
                    print(f"RL agent loaded network weights from: {resolved_path}")
                
                self.q_network.eval()  # Set to evaluation mode
            except Exception as e:
                print(f"Error loading model: {e}")
                print("RL agent will use untrained model.")
        else:
            print(f"Warning: Model file {resolved_path} not found. RL agent will use untrained model.")
        
        # Initialize collision detection parameters
        self.collision_risk_radius = conf.get()["rl_agent"]["collision_risk_radius"]
        self.cooldown_frames = conf.get()["rl_agent"]["cooldown_frames"]
        self.tactical_waypoint_distance = conf.get()["rl_agent"]["tactical_waypoint_distance"]
        self.aircraft_cooldowns = {}  
        self.current_frame = 0        # Frame counter
        self.encounter_history = {}   # Track aircraft pair encounters
        self.speed_reset_timers = {}
        
    def detect_collision_risks(self, aircraft_list):
        """Detect pairs of aircraft that are at risk of collision"""
        aircraft_pairs = []
        self.current_frame += 1  # Increment frame counter
        
        # Track which aircraft are currently in collision risk
        aircraft_in_pairs = set()
        
        for i, ac1 in enumerate(aircraft_list):
            for j, ac2 in enumerate(aircraft_list):
                if i < j:  # Check each pair only once
                    dist = Utility.locDist(ac1.getLocation(), ac2.getLocation())
                    if dist < self.collision_risk_radius:
                        # Skip if either aircraft is on cooldown
                        if (ac1.getIdent() in self.aircraft_cooldowns and 
                            self.current_frame - self.aircraft_cooldowns[ac1.getIdent()] < self.cooldown_frames):
                            continue
                        if (ac2.getIdent() in self.aircraft_cooldowns and 
                            self.current_frame - self.aircraft_cooldowns[ac2.getIdent()] < self.cooldown_frames):
                            continue
                        
                        # If not on cooldown, add to aircraft pairs
                        aircraft_pairs.append((ac1, ac2))
                        aircraft_in_pairs.add(ac1.getIdent())
                        aircraft_in_pairs.add(ac2.getIdent())
                        
                        # Add to encounter history
                        pair_id = f"{min(ac1.getIdent(), ac2.getIdent())}_{max(ac1.getIdent(), ac2.getIdent())}"
                        self.encounter_history[pair_id] = self.current_frame
                        
                        # print(f"Detected collision risk between {ac1.getIdent()} and {ac2.getIdent()}, distance: {dist:.2f}")
        
        # Check for aircraft that need to return to default speed
        self._check_speed_reset_timers(aircraft_list, aircraft_in_pairs)
    
        return aircraft_pairs

    def _check_speed_reset_timers(self, aircraft_list, aircraft_in_pairs):
        """Check if any aircraft needs to return to default speed based on timer"""
        default_speed = conf.get()['aircraft']['speed_default']
        speed_reset_delay = conf.get()['rl_agent']['speed_reset_delay']
        
        for ac in aircraft_list:
            ac_id = ac.getIdent()
            
            # If aircraft is in collision risk, track when its speed was last modified
            if ac_id in aircraft_in_pairs:
                if ac.getSpeed() != default_speed:
                    # Aircraft has modified speed and is in collision risk
                    # Mark it for timed reset when risk clears
                    self.speed_reset_timers[ac_id] = self.current_frame
            
            # If aircraft is NOT in collision risk and has non-default speed
            elif ac.getSpeed() != default_speed:
                # Check if it has a reset timer
                if ac_id in self.speed_reset_timers:
                    # Calculate how long it's been since collision risk cleared
                    time_since_risk = self.current_frame - self.speed_reset_timers[ac_id]
                    
                    # If enough time has passed, reset speed to default
                    if time_since_risk >= speed_reset_delay:
                        ac.setSpeed(default_speed)
                        if self.debug:
                            print(f"Timed speed reset for aircraft {ac_id} after {time_since_risk} frames")
                        # Remove from reset timers
                        del self.speed_reset_timers[ac_id]
                else:
                    # No timer but speed is modified - immediate reset
                    # This handles edge cases where timers weren't properly set
                    ac.setSpeed(default_speed)
                    if self.debug:
                        print(f"Immediate speed reset for aircraft {ac_id} (no timer found)")
    
    def get_observation(self, ac1, ac2, obstacles=None, destinations=None):
        """Create an observation vector for the given aircraft pair with speed information"""
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
        
        # NEW: Calculate speed-related observations
        default_speed = conf.get()['aircraft']['speed_default']
        
        # 1. Relative speed of aircraft 2 compared to aircraft 1
        relative_speed = ac2.getSpeed() / max(ac1.getSpeed(), 0.1)  # Avoid division by zero
        
        # 2. Calculate separation rate (are they moving toward or away from each other?)
        # First, get velocity components of both aircraft
        hdg1_rad = math.radians(ac1.getHeading())
        hdg2_rad = math.radians(ac2.getHeading())
        
        vx1 = ac1.getSpeed() * math.sin(hdg1_rad)
        vy1 = -ac1.getSpeed() * math.cos(hdg1_rad)
        vx2 = ac2.getSpeed() * math.sin(hdg2_rad)
        vy2 = -ac2.getSpeed() * math.cos(hdg2_rad)
        
        # Calculate relative position vector (from ac1 to ac2)
        dx = ac2.getLocation()[0] - ac1.getLocation()[0]
        dy = ac2.getLocation()[1] - ac1.getLocation()[1]
        
        # Calculate relative velocity vector
        rel_vx = vx2 - vx1
        rel_vy = vy2 - vy1
        
        # Calculate dot product (negative means separating, positive means approaching)
        dot_product = dx * rel_vx + dy * rel_vy
        
        # Normalize by distance and speeds to get a rate between -1 and 1
        # where -1 is rapidly separating and +1 is rapidly approaching
        distance_speed_product = distance * (ac1.getSpeed() + ac2.getSpeed())
        if distance_speed_product > 0.1:  # Avoid division by very small numbers
            separation_rate = dot_product / distance_speed_product
            # Clamp to range [-1, 1]
            separation_rate = max(-1.0, min(1.0, separation_rate))
        else:
            separation_rate = 0.0
        
        # Create observation vector with new components
        observation = np.array([
            distance,
            angle,
            rel_heading,
            destination_angle,
            angular_diff_to_dest,
            relative_speed,    # NEW: Relative speed of ac2 to ac1
            separation_rate    # NEW: Rate of separation/approach
        ], dtype=np.float32)
        
        # if self.debug:
        #     print(f"Enhanced observation: {observation}")
        
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
        """Apply the selected action to aircraft by modifying its waypoints and speed"""
        # Define a distance threshold below which we won't apply collision avoidance
        proximity_threshold = conf.get()['rl_agent']['proximity_omit_threshold']
        
        # Get current location
        curr_loc = aircraft.getLocation()
        curr_heading = aircraft.getHeading()
        
        # Check if aircraft is close to destination
        if len(aircraft.waypoints) > 0:
            dest_loc = aircraft.waypoints[-1].getLocation()
            distance_to_dest = Utility.locDist(curr_loc, dest_loc)
            
            # Skip collision avoidance if too close to destination
            if distance_to_dest <= proximity_threshold:
                if self.debug:
                    print(f"Aircraft {aircraft.getIdent()}: Too close to destination ({distance_to_dest:.1f} px) - skipping collision avoidance")
                return
        
        # Get default speed
        default_speed = conf.get()['aircraft']['speed_default']
        reduced_speed = default_speed * 0.6
        
        # Store original destination (last waypoint)
        destination = None
        if len(aircraft.waypoints) > 0:
            destination = aircraft.waypoints[-1]  # Last waypoint is destination
        
        # Clear existing waypoints
        aircraft.waypoints = []
        
        # Add destination back if it exists
        if destination:
            aircraft.addWaypoint(destination)
        else:
            # If no destination exists, just maintain current heading
            if self.debug:
                print(f"Aircraft {aircraft.getIdent()}: No destination found!")
            return
        
        # Determine if this action includes a speed adjustment
        speed_adjustment = False
        if action >= 5:  # Actions 5-8 include speed reduction
            speed_adjustment = True
            aircraft.setSpeed(reduced_speed)
            if self.debug:
                print(f"Aircraft {aircraft.getIdent()}: Reducing speed to {reduced_speed}")
                self.speed_reset_timers[aircraft.getIdent()] = self.current_frame  # Set speed reset timer
        else:
            aircraft.setSpeed(default_speed)
            if self.debug and aircraft.getSpeed() != default_speed:
                print(f"Aircraft {aircraft.getIdent()}: Setting speed to default {default_speed}")
        
        # Determine the turn action (mapping actions 5-8 back to their turn equivalents)
        turn_action = action
        if action >= 5:
            turn_action = action - 4  # Convert to base turn action (1-4)
        
        # Apply tactical waypoint if action is not "maintain course"
        if turn_action != 0:
            # Apply heading change based on turn action
            if turn_action == 1:  # ML - Medium left (90° turn)
                new_heading = (curr_heading - 90) % 360
            elif turn_action == 2:  # SL - Slight left (45° turn)
                new_heading = (curr_heading - 45) % 360
            elif turn_action == 3:  # MR - Medium right (90° turn)
                new_heading = (curr_heading + 90) % 360
            elif turn_action == 4:  # SR - Slight right (45° turn)
                new_heading = (curr_heading + 45) % 360
            else:  # N - maintain course
                new_heading = curr_heading
        
            # Calculate new waypoint location
            distance = self.tactical_waypoint_distance
            rad_heading = math.radians(new_heading)
            dx = distance * math.sin(rad_heading)
            dy = -distance * math.cos(rad_heading)
            new_waypoint = (int(curr_loc[0] + dx), int(curr_loc[1] + dy))
            
            # Create and insert tactical waypoint
            tactical_waypoint = Waypoint(new_waypoint)
            aircraft.waypoints.insert(0, tactical_waypoint)  # Insert at beginning
            
            if self.debug:
                speed_str = " with reduced speed" if speed_adjustment else ""
                print(f"Aircraft {aircraft.getIdent()}: Action={action}, Applied tactical waypoint{speed_str}")
        else:
            if self.debug:
                speed_str = " with reduced speed" if speed_adjustment else ""
                print(f"Aircraft {aircraft.getIdent()}: Direct routing to destination{speed_str}")
    
    def _mirror_action(self, action):
        """Mirror an action for the intruder aircraft with asymmetric speed adjustments.
        
        Extended action space:
        - 0: Maintain course and speed (N)
        - 1: Medium left turn (90°) with normal speed (ML)
        - 2: Slight left turn (45°) with normal speed (SL)
        - 3: Medium right turn (90°) with normal speed (MR)
        - 4: Slight right turn (45°) with normal speed (SR)
        - 5: Medium left turn (90°) with reduced speed (ML-S)
        - 6: Slight left turn (45°) with reduced speed (SL-S)
        - 7: Medium right turn (90°) with reduced speed (MR-S)
        - 8: Slight right turn (45°) with reduced speed (SR-S)
        
        Mirror mapping:
        - When first aircraft turns left, second turns right (and vice versa)
        - When first aircraft reduces speed, second maintains normal speed
        """
        if action == 0:  # Maintain course (N) -> Maintain course (N)
            return 0
        elif action == 1:  # Medium left (ML) -> Medium right (MR)
            return 3
        elif action == 2:  # Slight left (SL) -> Slight right (SR)
            return 4
        elif action == 3:  # Medium right (MR) -> Medium left (ML)
            return 1
        elif action == 4:  # Slight right (SR) -> Slight left (SL)
            return 2
        elif action == 5:  # Medium left with reduced speed -> Medium right with normal speed
            return 3
        elif action == 6:  # Slight left with reduced speed -> Slight right with normal speed
            return 4
        elif action == 7:  # Medium right with reduced speed -> Medium left with normal speed
            return 1
        elif action == 8:  # Slight right with reduced speed -> Slight left with normal speed
            return 2
        else:
            return 0  # Default to maintain course for unknown actions
    
    def _calculate_angle(self, point1, point2):
        """Calculate angle from point1 to point2 in degrees (0-360)"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        angle = math.degrees(math.atan2(dy, dx))
        angle = (angle + 360) % 360
        
        return angle