# test_rl.py - place this in your project root
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn_agent import QNetwork

class ModelTester:
    def __init__(self, model_path):
        # Load model
        self.state_dim = 5
        self.action_dim = 5
        self.hidden_dim = 128
        self.q_network = QNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        
        try:
            checkpoint = torch.load(model_path)
            if "q_network" in checkpoint:
                self.q_network.load_state_dict(checkpoint["q_network"])
            else:
                self.q_network.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            
        self.q_network.eval()
        
    def test_scenarios(self):
        # Test observations
        test_observations = [
            # [distance, angle, rel_heading, destination_angle, angular_diff_to_dest]
            np.array([50.0, 0.0, 180.0, 180.0, 180.0], dtype=np.float32),
            np.array([75.0, 90.0, 270.0, 0.0, 0.0], dtype=np.float32),
            np.array([75.0, 270.0, 90.0, 90.0, 90.0], dtype=np.float32),
            np.array([30.0, 180.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([120.0, 45.0, 225.0, 270.0, 90.0], dtype=np.float32)
        ]
        
        scenarios = [
            "Head-on collision risk, destination behind",
            "Aircraft approaching from right, destination ahead",
            "Aircraft approaching from left, destination right",
            "Aircraft very close from behind, destination ahead",
            "Aircraft far away crossing path, destination left"
        ]
        
        actions = {
            0: "Maintain course",
            1: "Medium left (90°)",
            2: "Slight left (45°)",
            3: "Medium right (90°)",
            4: "Slight right (45°)"
        }
        
        # Test each observation
        for i, obs in enumerate(test_observations):
            print(f"\nScenario {i+1}: {scenarios[i]}")
            print(f"Observation: {obs}")
            
            # Get Q-values and action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values).item()
            
            print(f"Selected action: {action} - {actions[action]}")
            print("Q-values for each action:")
            for a, q in enumerate(q_values[0].numpy()):
                print(f"  Action {a} ({actions[a]}): {q:.6f}")

if __name__ == "__main__":
    tester = ModelTester("../models/dqn_atc_model.pth")
    tester.test_scenarios()