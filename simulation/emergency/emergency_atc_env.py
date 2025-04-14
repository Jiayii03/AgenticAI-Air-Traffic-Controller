import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from simulation.emergency.emergency_simulation import EmergencySimulation 
from core.utility import Utility
from core import conf

class EmergencyATCEnv(gym.Env):
    """
    Emergency Air Traffic Control Environment for training an RL agent to reroute aircraft away 
    from emergency airports.
    
    In this environment:
        - When an aircraft’s destination is flagged as emergency, the agent must choose one of 
            the safe alternative destinations.
        - The observation vector (9 features) includes:
            1. Distance from aircraft to the emergency destination.
            2. Angle difference (in degrees) between the aircraft’s heading and the direction to the emergency destination.
            3–8. For each safe alternative destination (3 alternatives):
                a. Distance from the aircraft.
                b. Angle difference between aircraft’s heading and the direction to that destination.
            9. The aircraft’s normalized speed.
        - The action space is Discrete(3) – each action corresponds to selecting one of the safe alternatives.
    
    The reward system is designed so that:
        - A bonus is provided when the agent switches the aircraft’s destination from the emergency to a safe one.
        - If the aircraft later lands at a safe destination, a positive reward is given.
        - If the aircraft lands at the emergency destination or if no safe alternative is available, a penalty is applied.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, logger, num_planes=3, num_obstacles=2, screen_size=(800, 800)):
        super(EmergencyATCEnv, self).__init__()
        
        # Use the emergency simulation
        self.simulation = EmergencySimulation(logger, num_planes, num_obstacles, screen_size)
        
        # Action space: choose one of 3 safe alternative destinations.
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 9 features as described.
        # For distances we assume a max of 150 pixels, angle differences from 0 to 180,
        # speed normalized between 0 and 2.
        obs_low = np.array([0, 0] + [0, 0]*3 + [0], dtype=np.float32)
        obs_high = np.array([150, 180] + [150, 180]*3 + [2], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Emergency-specific parameters from configuration (if needed)
        self.collision_risk_radius = conf.get()['rl_agent']['collision_risk_radius']
        self.cooldown_frames = conf.get()['rl_agent']['cooldown_frames']
        
        # Tracking for cooldown and emergency control (not used in this focused env, but kept for consistency)
        self.aircraft_cooldowns = {}
        self.current_frame = 0
        self.logger = logger

    def reset(self, seed=None, options=None):
        """Reset the emergency environment."""
        state = self.simulation.reset()
        self.current_frame = 0
        self.logger.debug_print(f"[EmergencyATCEnv] Reset state: {state}")
        self.logger.debug_print(f"[EmergencyATCEnv] Number of aircraft: {len(state['aircraft'])}")
        # Return an initial observation. We pick the first aircraft that is in emergency.
        obs = self._build_observation()
        return obs

    def step(self, action):
        self.current_frame += 1

        emergency_ac = None
        for ac in self.simulation.aircraft:
            if len(ac.waypoints) > 0 and hasattr(ac.waypoints[-1], 'isEmergency') and ac.waypoints[-1].isEmergency():
                emergency_ac = ac
                break

        if emergency_ac is None:
            state, rewards_dict, done, info = self.simulation.step([0] * len(self.simulation.aircraft))
            obs = self._build_observation()
            total_reward = sum(rewards_dict.values())
            return obs, total_reward, done, False, {**info, "emergency_risk": False, "rerouted": False}

        # Identify safe alternatives
        safe_dests = [dest for dest in self.simulation.destinations if not dest.isEmergency()]
        safe_dests = sorted(safe_dests, key=lambda d: Utility.locDist(emergency_ac.getLocation(), d.getLocation()))
        while len(safe_dests) < 3:
            safe_dests.append(safe_dests[-1])
        safe_dests = safe_dests[:3]

        # Determine if reroute happens
        original_dest = emergency_ac.waypoints[-1]
        chosen_dest = safe_dests[action]
        rerouted = original_dest != chosen_dest

        emergency_ac.waypoints[-1] = chosen_dest
        self.logger.debug_print(f"[EmergencyATCEnv] Aircraft {emergency_ac.getIdent()} rerouted to {chosen_dest.text} via action {action}")

        state, rewards_dict, done, info = self.simulation.step([0] * len(self.simulation.aircraft))
        
        reward = rewards_dict.get(emergency_ac.getIdent(), 0)

        # ✅ IMMEDIATE reward for rerouting
        if rerouted:
            reward += 50

        if done:
            if len(emergency_ac.waypoints) > 0:
                landed_dest = emergency_ac.waypoints[-1]
                if hasattr(landed_dest, 'isEmergency') and not landed_dest.isEmergency():
                    reward += 100  # Landed at safe destination
                else:
                    reward -= 200
            else:
                reward -= 50
        else:
            reward -= 1  # time penalty

        obs = self._build_observation()
        return obs, reward, done, False, {**info, "emergency_risk": True, "rerouted": rerouted}

    def _build_observation(self):
        """
        Build an observation vector for the emergency aircraft.
        Observation vector (9 dimensions):
                [dist_emerg, ang_diff_emerg,
                dist_safe1, ang_diff_safe1,
                dist_safe2, ang_diff_safe2,
                dist_safe3, ang_diff_safe3,
                normalized_speed]
        If no emergency aircraft exists, returns zeros.
        """
        emergency_ac = None
        for ac in self.simulation.aircraft:
            if len(ac.waypoints) > 0 and hasattr(ac.waypoints[-1], 'isEmergency') and ac.waypoints[-1].isEmergency():
                emergency_ac = ac
                break
        if emergency_ac is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        current_loc = emergency_ac.getLocation()
        current_heading = emergency_ac.getHeading()
        # Features for the current (emergency) destination.
        emerg_dest = emergency_ac.waypoints[-1]
        emerg_loc = emerg_dest.getLocation()
        dist_emerg = Utility.locDist(current_loc, emerg_loc)
        ang_to_emerg = self._calculate_angle(current_loc, emerg_loc)
        ang_diff_emerg = abs((current_heading - ang_to_emerg + 360) % 360)
        if ang_diff_emerg > 180:
            ang_diff_emerg = 360 - ang_diff_emerg
        
        # Compute features for safe alternative destinations.
        safe_dests = [dest for dest in self.simulation.destinations if not dest.isEmergency()]
        safe_dests = sorted(safe_dests, key=lambda d: Utility.locDist(current_loc, d.getLocation()))
        if len(safe_dests) > 3:
            safe_dests = safe_dests[:3]
        elif len(safe_dests) < 3:
            while len(safe_dests) < 3:
                safe_dests.append(safe_dests[-1])
        safe_features = []
        for dest in safe_dests:
            d_loc = dest.getLocation()
            dist = Utility.locDist(current_loc, d_loc)
            ang = self._calculate_angle(current_loc, d_loc)
            ang_diff = abs((current_heading - ang + 360) % 360)
            if ang_diff > 180:
                ang_diff = 360 - ang_diff
            safe_features.extend([dist, ang_diff])
        
        # Normalize speed (assume default speed from config is maximum)
        default_speed = conf.get()['aircraft']['speed_default']
        norm_speed = emergency_ac.getSpeed() / default_speed
        
        observation = np.array([dist_emerg, ang_diff_emerg] + safe_features + [norm_speed], dtype=np.float32)

        # Ensure the observation vector has exactly 9 features
        assert len(observation) == 9, f"Observation vector has incorrect size: {len(observation)} (expected 9)"
        return observation

    def _calculate_angle(self, point1, point2):
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return (angle + 360) % 360

    def render(self, mode='human'):
        return self.simulation.render(mode)

    def close(self):
        pygame.quit()
    
    def _find_aircraft_by_id(self, aircraft_id):
        for ac in self.simulation.aircraft:
            if ac.getIdent() == aircraft_id:
                return ac
        return None

    def _mirror_action(self, action):
        """
        Mirror an action for a paired aircraft.
        For this emergency environment, we simply return the same action for non-controlled aircraft.
        """
        return action
