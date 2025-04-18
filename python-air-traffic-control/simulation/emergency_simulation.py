import pygame
import numpy as np
import sys
import os

# Add parent directory to path so we can import game components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.simulation import Simulation
from destination import Destination
from utility import Utility
import conf

class EmergencySimulation(Simulation):
    """
    A specialized simulation environment for emergency rerouting.
    In this mode, one or more destinations are flagged as emergency,
    and the environment monitors aircraft that are assigned to these
    destinations. When an aircraft’s destination is in emergency,
    an alternative safe destination is chosen.
    """
    def __init__(self, logger, num_planes=5, num_obstacles=3, screen_size=(800, 800)):
        # Call the base Simulation initializer
        super(EmergencySimulation, self).__init__(logger, num_planes, num_obstacles, screen_size)
        # Additional initialization for emergency scenarios can be added here

    def reset(self):
        """
        Reset the simulation to the initial state and then mark a destination (or more) as emergency.
        Here we mark the first destination as emergency.
        """
        state = super(EmergencySimulation, self).reset()
        if self.destinations:
            # Mark the first destination as emergency
            self.destinations[0].setEmergency(True)
            print(f"Destination {self.destinations[0].text} has been marked as EMERGENCY.")
        return state

    def step(self, actions):
        """
        Execute a simulation step as usual, then check for emergency scenarios.
        If any aircraft’s final destination is flagged as emergency,
        trigger an emergency rerouting by selecting a safe destination.
        """
        # Run the normal simulation step
        state, step_rewards_dict, done, info = super(EmergencySimulation, self).step(actions)

        # Initialize rewards_dict with step rewards
        rewards_dict = dict(step_rewards_dict)

        # Ensure all current aircraft in self.aircraft have an entry in rewards_dict
        for ac in self.aircraft:
            if ac.getIdent() not in rewards_dict:
                rewards_dict[ac.getIdent()] = 0

        # Check each aircraft for emergency destination assignment.
        for ac in self.aircraft:
            if len(ac.waypoints) > 0:
                dest = ac.waypoints[-1]
                # Check if the destination is flagged as emergency.
                if hasattr(dest, 'isEmergency') and dest.isEmergency():
                    print(f"Aircraft {ac.getIdent()} is heading to an EMERGENCY destination: {dest.text}")
                    # Trigger emergency rerouting: choose a new, safe destination.
                    new_dest = self.get_alternative_destination(ac)
                    if new_dest:
                        if ac.getIdent() in rewards_dict:
                            rewards_dict[ac.getIdent()] += 50  # bonus reward
                        print(f"Aircraft {ac.getIdent()} rerouted to safe destination: {new_dest.text}")
                        ac.waypoints[-1] = new_dest
                    else:
                        if ac.getIdent() in rewards_dict:
                            rewards_dict[ac.getIdent()] -= 100  # penalty
                        print(f"Aircraft {ac.getIdent()} could not find a safe destination!")

        return state, rewards_dict, done, info

    def get_alternative_destination(self, aircraft):
        """
        Returns the nearest safe destination (one not in emergency).
        If none exist, returns None.
        """
        current_loc = aircraft.getLocation()
        safe_dests = [dest for dest in self.destinations if not dest.isEmergency()]
        if not safe_dests:
            return None
        # Return the safe destination with minimum distance
        nearest = min(safe_dests, key=lambda d: Utility.locDist(current_loc, d.getLocation()))
        return nearest

# If running this file directly, you can do a simple test.
if __name__ == "__main__":
    # Basic setup for testing
    import conf
    from logger import Logger

    logger = Logger(log_dir='logs', prefix='emergency_sim', debug=True).start()
    sim = EmergencySimulation(logger, num_planes=3, num_obstacles=2, screen_size=(800, 800))
    
    state = sim.reset()
    # In a test loop, we simply call step with a dummy action list.
    done = False
    while not done:
        # Here we use action 0 (maintain course) for all aircraft.
        actions = [0] * len(sim.aircraft)
        state, rewards, done, info = sim.step(actions)
        print("Step rewards:", rewards)
        pygame.time.delay(100)
    
    sim.render()
    pygame.quit()
