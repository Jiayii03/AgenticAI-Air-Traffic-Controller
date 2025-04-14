# simulation/atc_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

from simulation.common.simulation import Simulation
from core.utility import Utility
from core import conf

class ATCEnv(gym.Env):
    """
    Air Traffic Control environment for OpenAI Gym.

    State Space:
    - Distance to intruder aircraft (d): [0, 150] pixels
    - Angle to intruder (ρ): [0, 360) degrees
    - Relative heading of intruder (θ): [0, 360) degrees
    - Angle to destination: [0, 360) degrees
    - Angular difference to destination: [0, 180] degrees
    - Relative speed of intruder (rs): [0, 2.0] ratio
    - Separation rate (sr): [-1.0, 1.0] normalized value

    Action Space:
    - 0: Maintain course and speed (N)
    - 1: Medium left turn (ML): 90° with normal speed
    - 2: Slight left turn (SL): 45° with normal speed
    - 3: Medium right turn (MR): 90° with normal speed
    - 4: Slight right turn (SR): 45° with normal speed
    - 5: Medium left turn (ML-S): 90° with reduced speed (-20%)
    - 6: Slight left turn (SL-S): 45° with reduced speed (-20%)
    - 7: Medium right turn (MR-S): 90° with reduced speed (-20%)
    - 8: Slight right turn (SR-S): 45° with reduced speed (-20%)
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, logger, num_planes=3, num_obstacles=2, screen_size=(800, 800)):
        super(ATCEnv, self).__init__()
        
        # Initialize the simulation
        self.simulation = Simulation(logger, num_planes, num_obstacles, screen_size)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Extended to 9 possible actions with speed options
        
        # Observation space based on state variables with new speed and separation components
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, -1]),
            high=np.array([150, 360, 360, 360, 180, 2, 1]),  # Bounds for all components
            dtype=np.float32
        )
        """
        [
            distance_between_pair,      # Distance between the two aircraft in the pair
            angle_to_intruder,          # Angle from aircraft 1 to aircraft 2
            relative_heading,           # Relative heading of aircraft 2 compared to aircraft 1
            angle_to_destination,       # Angle from aircraft 1 to its destination
            angular_diff_to_dest,       # Angular difference between current heading and destination angle
            relative_speed,             # Speed of aircraft 2 relative to aircraft 1
            separation_rate             # Rate of separation/approach (-1 to 1)
        ]
        """
        
        # Collision detection threshold
        self.collision_risk_radius = conf.get()['rl_agent']['collision_risk_radius']
        self.cooldown_frames = conf.get()['rl_agent']['cooldown_frames']
        
        # Add cooldown tracking
        self.aircraft_cooldowns = {}  # Dictionary to track cooldown for each aircraft
        self.current_frame = 0        # Frame counter
        
        # Target aircraft pair for learning (will be updated in reset)
        self.pair_index = 0
        self.aircraft_pairs = []
        self.logger = logger
    
    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        
        # Reset the simulation
        state = self.simulation.reset()
        self.logger.debug_print(f"Reset state: {state}")
        self.logger.debug_print(f"Number of aircraft: {len(state['aircraft'])}")
        self.logger.debug_print(f"Aircraft IDs: {[ac['id'] for ac in state['aircraft']]}")
        self.logger.debug_print(f"Aircraft positions: {[ac['location'] for ac in state['aircraft']]}")
    
    def step(self, action):
        """Execute action and advance the environment by one timestep."""
        self.current_frame += 1  # Increment frame counter
        
        # When no aircraft remain - episode is done
        if len(self.simulation.aircraft) == 0:
            observation = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return observation, 0, True, False, {"num_planes_landed": 0, "had_collision": False}
        
        # If no collision risks, all aircraft maintain course
        if not self.aircraft_pairs:
            state, rewards_dict, done, info = self.simulation.step([0] * len(self.simulation.aircraft)) # maintain course
            observation = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            total_reward = sum(rewards_dict.values())
            return observation, total_reward, done, False, {**info, "collision_risk": False}
        
        # Else, if there is collision risk, get the first pair to control
        ac1_id, ac2_id = self.aircraft_pairs[0]  # Always use the first pair
        self.logger.debug_print(f"Controlled aircraft pair: {ac1_id} - {ac2_id}")
        
        # Prepare actions for all aircraft
        all_actions = [0] * len(self.simulation.aircraft)  # Default: all maintain course
        
        # Apply action to the first pair
        for i, ac in enumerate(self.simulation.aircraft):
            if ac.getIdent() == ac1_id:
                all_actions[i] = action
            elif ac.getIdent() == ac2_id:
                all_actions[i] = self._mirror_action(action)
        
        # Execute actions in simulation
        state, rewards_dict, done, info = self.simulation.step(all_actions)

        # NOW set the cooldowns for the aircraft pair we just controlled
        self.aircraft_cooldowns[ac1_id] = self.current_frame
        self.aircraft_cooldowns[ac2_id] = self.current_frame
        self.logger.debug_print(f"Set cooldown for aircraft {ac1_id} and {ac2_id} at frame {self.current_frame}")

        # Calculate reward for the specific aircraft pair we controlled
        pair_reward = 0
        if ac1_id in rewards_dict:
            pair_reward += rewards_dict[ac1_id]
        if ac2_id in rewards_dict:
            pair_reward += rewards_dict[ac2_id]

        # Add additional reward for collision avoidance
        additional_reward = self._calculate_reward(state, ac1_id, ac2_id)
        total_reward = pair_reward + additional_reward

        print(f"Step: {self.simulation.step_count}, Base Reward: {pair_reward:.2f}, Collision Avoidance Reward: {additional_reward:.2f}")
        
        # Get observation for the next step based on the SAME aircraft pair we just controlled
        # This maintains the state-action-reward-next_state continuity
        observation = self._get_observation((ac1_id, ac2_id))
   
        return observation, total_reward, done, False, {
            **info,
            "collision_risk": True
        }
    
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
                    if dist < self.collision_risk_radius: # if the distance is less than the collision risk radius
                        # Skip if either aircraft is on cooldown
                        if (ac1['id'] in self.aircraft_cooldowns and 
                            self.current_frame - self.aircraft_cooldowns[ac1['id']] < self.cooldown_frames):
                            self.logger.debug_print(f"Current frame: {self.current_frame}, Cooldown frame: {self.aircraft_cooldowns[ac1['id']]}")
                            self.logger.debug_print(f"In collision risk, skipping aircraft {ac1['id']} due to cooldown")
                            continue
                        if (ac2['id'] in self.aircraft_cooldowns and 
                            self.current_frame - self.aircraft_cooldowns[ac2['id']] < self.cooldown_frames):
                            self.logger.debug_print(f"Current frame: {self.current_frame}, Cooldown frame: {self.aircraft_cooldowns[ac2['id']]}")
                            self.logger.debug_print(f"In collision risk, skipping aircraft {ac2['id']} due to cooldown")
                            continue
                        
                        # If both aircraft are not on cooldown, add the pair
                        self.aircraft_pairs.append((ac1['id'], ac2['id']))
                        self.logger.debug_print(f"Adding aircraft pair: {ac1['id']} - {ac2['id']}, Distance: {dist:.2f}")
        
        # Get observation for the first pair if any exist
        if self.aircraft_pairs:
            observation = self._get_observation(self.aircraft_pairs[0])
            self.logger.debug_print(f"Feeding observation for pair: {self.aircraft_pairs[0]}, Observation: {observation}")
            return observation
        else:
            observation = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            self.logger.debug_print(f"No collision risks, no pairs to update, returning observation {np.zeros(self.observation_space.shape[0], dtype=np.float32)}")
            return observation
                    
    def _get_observation(self, aircraft_pair):
        """Get observation for a pair of aircraft including speed information and separation rate."""
        ac1_id, ac2_id = aircraft_pair
        ac1 = self._find_aircraft_by_id(ac1_id)
        ac2 = self._find_aircraft_by_id(ac2_id)
        
        if ac1 is None or ac2 is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Calculate existing observation components
        distance = Utility.locDist(ac1.getLocation(), ac2.getLocation())
        angle = self._calculate_angle(ac1.getLocation(), ac2.getLocation())
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
        
        return np.array([
            distance, 
            angle, 
            rel_heading,
            destination_angle,
            angular_diff_to_dest,
            relative_speed,    # NEW: Relative speed of ac2 to ac1
            separation_rate    # NEW: Rate of separation/approach
        ], dtype=np.float32)
    
    def _calculate_reward(self, state, ac1_id, ac2_id):
        """Calculate reward based on aircraft states, including speed adjustment factors."""
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
            print("Only one aircraft remaining, no reward calculation")
            return 0
        
        # Get default speed from config
        default_speed = conf.get()['aircraft']['speed_default']
        
        # Calculate distance
        distance = Utility.locDist(ac1_state['location'], ac2_state['location'])
        
        # 1. Distance reward - continuous function instead of zones
        radius = self.collision_risk_radius
        distance_factor = 1.0 - min(1.0, distance/radius)
        distance_reward = -300 * (distance_factor ** 2)  # Quadratic penalty
        
        # 2. Destination alignment reward
        dest_alignment_reward = 0
        if len(ac1_state['waypoints']) > 0:
            # Get destination
            dest = ac1_state['waypoints'][-1]
            
            # Calculate angle to destination
            dest_angle = self._calculate_angle(ac1_state['location'], dest)
            
            # Calculate angular difference between current heading and destination
            hdg_to_dest_diff = abs((ac1_state['heading'] - dest_angle) % 360)
            if hdg_to_dest_diff > 180:
                hdg_to_dest_diff = 360 - hdg_to_dest_diff
                
            # Higher reward for pointing toward destination
            dest_alignment_reward = 50 * (1 - hdg_to_dest_diff/180)
        
        # 3. Heading away from each other reward
        heading_reward = 0
        if distance < radius * 3:  # Only when relatively close
            # Calculate relative heading between aircraft
            heading_diff = abs((ac1_state['heading'] - ac2_state['heading']) % 360)
            if heading_diff > 180:
                heading_diff = 360 - heading_diff
                
            # Reward diverging headings (closer to 180 degrees)
            heading_reward = 80 * (heading_diff / 180)
        
        # 4. Calculate separation rate (are they moving toward or away from each other?)
        # First, get velocity components of both aircraft
        hdg1_rad = math.radians(ac1_state['heading'])
        hdg2_rad = math.radians(ac2_state['heading'])
        
        vx1 = ac1_state['speed'] * math.sin(hdg1_rad)
        vy1 = -ac1_state['speed'] * math.cos(hdg1_rad)
        vx2 = ac2_state['speed'] * math.sin(hdg2_rad)
        vy2 = -ac2_state['speed'] * math.cos(hdg2_rad)
        
        # Calculate relative position vector (from ac1 to ac2)
        dx = ac2_state['location'][0] - ac1_state['location'][0]
        dy = ac2_state['location'][1] - ac1_state['location'][1]
        
        # Calculate relative velocity vector
        rel_vx = vx2 - vx1
        rel_vy = vy2 - vy1
        
        # Calculate dot product (negative means separating, positive means approaching)
        dot_product = dx * rel_vx + dy * rel_vy
        
        # Determine if aircraft are separating
        separating = dot_product < 0
        
        # Calculate normalized separation rate (-1 to 1)
        distance_speed_product = distance * (ac1_state['speed'] + ac2_state['speed'])
        if distance_speed_product > 0.1:  # Avoid division by very small numbers
            separation_rate = dot_product / distance_speed_product
            # Clamp to range [-1, 1]
            separation_rate = max(-1.0, min(1.0, separation_rate))
        else:
            separation_rate = 0.0
        
        # 5. Speed adjustment reward
        speed_reward = 0
        
        # Check if ac1 has reduced speed
        ac1_reduced = ac1_state['speed'] < default_speed * 0.9  # Allow some margin
        
        # Check if ac2 has normal speed (asymmetric speed adjustment)
        ac2_normal = ac2_state['speed'] > default_speed * 0.9
        
        # Case 1: Effective speed reduction - reduced speed AND increasing separation AND reasonable alignment to destination
        if ac1_reduced and separating:
            # Base reward for effective speed reduction
            speed_reward += 40
            
            # Bonus for good destination alignment despite reduced speed
            if hdg_to_dest_diff < 90:  # Within 90 degrees of destination heading
                speed_reward += 20 * (1 - hdg_to_dest_diff/90)
                
            # Extra bonus for asymmetric response (one slows, one maintains speed)
            if ac2_normal:
                speed_reward += 15
        
        # Case 2: Ineffective speed reduction - reduced speed BUT not separating
        elif ac1_reduced and not separating:
            # Penalty for ineffective speed reduction
            speed_reward -= 30
            
            # Additional penalty if badly aligned with destination
            if hdg_to_dest_diff > 90:
                speed_reward -= 10
        
        # Case 3: Both aircraft reduced speed - usually suboptimal
        if ac1_reduced and not ac2_normal:
            speed_reward -= 20
        
        # 6. Time penalty - discourages excessive maneuvering
        step_count = self.simulation.step_count
        time_penalty = -0.5 * (1 + step_count / 500)
        
        # If speed is reduced, increase time penalty slightly to discourage unnecessary speed reductions
        if ac1_reduced:
            time_penalty *= 1.2
        
        # Log the reward components for debugging
        print(f"Reward components - Distance: {distance_reward:.2f}, Heading Away: {heading_reward:.2f}, "
            f"Dest Alignment: {dest_alignment_reward:.2f}, Speed: {speed_reward:.2f}, Time: {time_penalty:.2f}")
        
        # Balance components with more emphasis on collision avoidance and effective speed usage
        reward = (1.0 * distance_reward) + \
                (1.2 * heading_reward) + \
                (0.7 * dest_alignment_reward) + \
                (1.0 * speed_reward) + \
                (0.2 * time_penalty)
        
        return max(min(reward, 50), -100)  # Cap reward
    
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
    
    def _find_aircraft_by_id(self, aircraft_id):
        """Find an aircraft by its ID."""
        for ac in self.simulation.aircraft:
            if ac.getIdent() == aircraft_id:
                return ac
        return None
     
    def _calculate_angle(self, point1, point2):
        """Calculate the angle from point1 to point2 in degrees (0-360)."""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        # Calculate angle in radians and convert to degrees
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Convert to 0-360 range
        angle = (angle + 360) % 360
        
        return angle
