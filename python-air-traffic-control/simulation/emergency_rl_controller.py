import os
import torch
import numpy as np
from utility import Utility
from waypoint import Waypoint
from simulation.emergency_dqn_agent import EmergencyDQNAgent

class EmergencyRLController:
    """Handles emergency rerouting using a trained DQN model."""

    def __init__(self, model_path='models/emergency_dqn_model.pth', state_dim=9, action_dim=3, debug=True):
        """
        Initialize the Emergency RL Controller with the trained model.
        - model_path: Path to the trained emergency DQN model.
        - state_dim: Size of the observation space.
        - action_dim: Number of possible actions (safe destinations).
        - debug: Whether to print debug information.
        """
        self.debug = debug

        # Initialize the Emergency DQN model
        self.emergency_agent = EmergencyDQNAgent(state_dim=state_dim, action_dim=action_dim)
        if os.path.exists(model_path):
            try:
                self.emergency_agent.load(model_path)  # Use the load method of EmergencyDQNAgent
                self.emergency_agent.q_network.eval()  # Set Q-network to evaluation mode
                print(f"Emergency RL agent loaded from: {model_path}")
            except Exception as e:
                print(f"Error loading emergency model: {e}")
                print("Emergency RL agent will use untrained model.")
        else:
            print(f"Warning: Emergency model file {model_path} not found. Emergency RL agent will use untrained model.")

    def get_observation(self, aircraft, emergency_destination, safe_destinations):
        """
        Build the observation vector for the emergency agent.
        - aircraft: The aircraft to reroute.
        - emergency_destination: The current emergency destination.
        - safe_destinations: List of alternative safe destinations.
        """
        obs = []

        # Distance and angle to the emergency destination
        obs.append(Utility.locDist(aircraft.getLocation(), emergency_destination.getLocation()))
        obs.append(Utility.angleBetween(aircraft.getHeading(), aircraft.getLocation(), emergency_destination.getLocation()))

        # Ensure exactly 3 safe destinations (pad or truncate as needed)
        safe_destinations = safe_destinations[:3]  # Take the first 3 destinations
        while len(safe_destinations) < 3:
            safe_destinations.append(safe_destinations[-1])  # Repeat the last destination if fewer than 3

        # Distance and angle to each safe alternative destination
        for dest in safe_destinations:
            obs.append(Utility.locDist(aircraft.getLocation(), dest.getLocation()))
            obs.append(Utility.angleBetween(aircraft.getHeading(), aircraft.getLocation(), dest.getLocation()))

        # Aircraft's normalized speed
        obs.append(aircraft.getSpeed() / 1000.0)  # Normalize by max speed (1000)

        # Ensure the observation vector has exactly 9 features
        assert len(obs) == 9, f"Observation vector has incorrect size: {len(obs)} (expected 9)"
        return np.array(obs, dtype=np.float32)

    def select_action(self, observation):
        """
        Select the best action (safe destination) based on the observation.
        - observation: The observation vector for the emergency scenario.
        """
        # Convert observation to tensor
        state_tensor = torch.FloatTensor(observation).unsqueeze(0)

        # Get action with the highest Q-value
        with torch.no_grad():
            q_values = self.emergency_agent.q_network(state_tensor)  # Use q_network instead of emergency_agent
            action = torch.argmax(q_values).item()

            if self.debug:
                print(f"Emergency RL Q-values: {q_values.numpy()}")
                print(f"Selected action: {action}")

        return action

    def apply_action(self, aircraft, action, safe_destinations):
        """
        Apply the selected action to reroute the aircraft.
        - aircraft: The aircraft to reroute.
        - action: The action (index of the safe destination).
        - safe_destinations: List of alternative safe destinations.
        """
        if action < len(safe_destinations):
            new_destination = safe_destinations[action]
            aircraft.update_destination(new_destination)  # Update the aircraft's destination
            if self.debug:
                print(f"Aircraft {aircraft.getIdent()} rerouted to {new_destination.text}")  # Use 'text' instead of 'name'
        else:
            if self.debug:
                print(f"Invalid action {action} for aircraft {aircraft.getIdent()}. No rerouting applied.")