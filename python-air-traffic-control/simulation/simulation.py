# simulation/simulation.py
import pygame
import numpy as np
import sys
import os

# Add parent directory to path so we can import game components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircraft import Aircraft
from destination import Destination
from obstacle import Obstacle
from waypoint import Waypoint
from utility import Utility
import conf

class Simulation:
    """A wrapper around the air traffic control simulation that allows for RL interaction."""
    
    def __init__(self, num_planes=5, num_obstacles=3, screen_size=(800, 800)):
        """Initialize pygame."""
        if not pygame.get_init():
            pygame.init()
            pygame.font.init()  # Specifically initialize the font module
            
        # Set up a display mode (required for image operations)
        # Use pygame.HIDDEN flag for headless operation if available
        try:
            # For newer pygame versions
            self.screen = pygame.display.set_mode(screen_size, flags=pygame.HIDDEN)
        except:
            # Fallback for older pygame versions
            self.screen = pygame.display.set_mode(screen_size)
        
        """Initialize the simulation with configuration parameters."""
        self.num_planes = num_planes
        self.num_obstacles = num_obstacles
        self.screen_size = screen_size
        
        # Initialize components from the original game
        self.destinations = Destination.generateGameDestinations(screen_size[0], screen_size[1])
        self.obstacles = Obstacle.generateGameObstacles(screen_size[0], screen_size[1], self.destinations)
        self.aircraft = []
        
        # Track simulation state
        self.active_aircraft = []
        self.collision_pairs = []
        self.done = False
        self.step_count = 0
        
        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()
            
        # Create a headless surface for simulation without rendering
        self.surface = pygame.Surface(screen_size)
    
    def reset(self):
        """Reset the simulation to initial state."""
        self.aircraft = []
        self.active_aircraft = []
        self.collision_pairs = []
        self.done = False
        self.step_count = 0
        
        # Spawn initial aircraft
        self._spawn_aircraft(self.num_planes)
        
        return self._get_state()
    
    def step(self, actions):
        """Execute actions and advance simulation by one step."""
        self.step_count += 1
        rewards_dict = {ac.getIdent(): 0 for ac in self.aircraft}
        
        print(f"Debug - initial rewards: {rewards_dict}")
                
        # Apply actions to aircraft
        for i, ac in enumerate(self.aircraft):
            if i < len(actions):
                self._apply_action(ac, actions[i])
        
        # Update aircraft positions
        landed_aircraft = []
        for ac in self.aircraft:
            # Update aircraft position
            reached_destination = ac.update()
            if reached_destination:
                landed_aircraft.append(ac)
                rewards_dict[ac.getIdent()] += 200  # Reward for successful landing
                print(f"Aircraft {ac.getIdent()} landed successfully, +200 reward")
        
        print(f"Debug - rewards after landing: {rewards_dict}")
        
        # Remove landed aircraft
        for ac in landed_aircraft:
            self.aircraft.remove(ac)
        
        # Check for collisions
        self.collision_pairs = self._detect_collisions()
        for pair in self.collision_pairs:
            ac1, ac2 = pair
            rewards_dict[ac1.getIdent()] -= 500  # Penalty for collision
            rewards_dict[ac2.getIdent()] -= 500
            print(f"Collision detected between {ac1.getIdent()} and {ac2.getIdent()}, -500 reward")
            self.done = True  # End episode on collision, critical failure in air traffic control
        
        print(f"Debug - rewards after collision check: {rewards_dict}")
        
        # Check obstacle collisions
        for obs in self.obstacles:
            collided = obs.collideAircraft(self.aircraft)
            for ac in self.aircraft:
                if ac in obs.colliding and ac.getIdent() in rewards_dict:
                    rewards_dict[ac.getIdent()] -= 100  # Penalty for obstacle collision
                    print(f"Obstacle collision detected for {ac.getIdent()}, -20 reward")
                    
        print(f"Debug - rewards after obstacle collision check: {rewards_dict}")
        
        # Get current state
        state = self._get_state()
        
        # Check if simulation is done
        if len(self.aircraft) == 0:
            self.done = True
        
        return state, rewards_dict, self.done, {}
    
    def _spawn_aircraft(self, count):
        """Spawn specified number of aircraft."""
        for i in range(count):
            # Generate random spawn location on edge of screen
            side = np.random.randint(1, 5)
            if side == 1:  # Top
                loc = (np.random.randint(0, self.screen_size[0]), 0)
            elif side == 2:  # Right
                loc = (self.screen_size[0], np.random.randint(0, self.screen_size[1]))
            elif side == 3:  # Bottom
                loc = (np.random.randint(0, self.screen_size[0]), self.screen_size[1])
            else:  # Left
                loc = (0, np.random.randint(0, self.screen_size[1]))
            
            # Select random destination
            dest = np.random.choice(self.destinations)
            
            # Create aircraft
            speed = conf.get()['aircraft']['speed_default']
            ident = f"AC{i+1}"
            ac = Aircraft(None, loc, speed, dest, ident)  # Pass None as game reference
            self.aircraft.append(ac)
    
    def _apply_action(self, aircraft, action):
        """Apply an action to an aircraft.
        
        Actions:
        0: Maintain course (N)
        1: Hard left (HL)
        2: Medium left (ML)
        3: Medium right (MR)
        4: Hard right (HR)
        """
        # Current location and heading
        curr_loc = aircraft.getLocation()
        curr_heading = aircraft.getHeading()
        
        # Calculate new heading based on action
        if action == 0:  # N - maintain course
            new_heading = curr_heading
        elif action == 1:  # HL - hard left
            new_heading = (curr_heading - 90) % 360
        elif action == 2:  # ML - medium left
            new_heading = (curr_heading - 45) % 360
        elif action == 3:  # MR - medium right
            new_heading = (curr_heading + 45) % 360
        elif action == 4:  # HR - hard right
            new_heading = (curr_heading + 90) % 360
        
        # Calculate new waypoint location based on heading and distance
        distance = 50  # Distance to place waypoint
        rad_heading = np.radians(new_heading)
        dx = distance * np.sin(rad_heading)
        dy = -distance * np.cos(rad_heading)
        
        new_waypoint = (int(curr_loc[0] + dx), int(curr_loc[1] + dy))
        
        # Create and add waypoint to aircraft
        waypoint = Waypoint(new_waypoint)
        
        # Clear existing waypoints and add new one
        aircraft.waypoints = []
        aircraft.addWaypoint(waypoint)
    
    def _detect_collisions(self):
        """Detect aircraft pairs that are in potential collision."""
        collision_pairs = []
        for i, ac1 in enumerate(self.aircraft):
            for j, ac2 in enumerate(self.aircraft):
                if i < j:  # Check each pair only once
                    dist_sq = Utility.locDistSq(ac1.getLocation(), ac2.getLocation())
                    if dist_sq < (conf.get()['aircraft']['collision_radius'] ** 2):
                        collision_pairs.append((ac1, ac2))
        
        return collision_pairs
    
    def _get_state(self):
        """Get the current state of the simulation.
        
        Returns a dictionary with information about aircraft positions,
        headings, speeds, and proximity to other aircraft and obstacles.
        """
        state = {
            'aircraft': [],
            'collision_pairs': self.collision_pairs
        }
        
        for ac in self.aircraft:
            ac_state = {
                'id': ac.getIdent(),
                'location': ac.getLocation(),
                'heading': ac.getHeading(),
                'speed': ac.getSpeed(),
                'waypoints': [wp.getLocation() for wp in ac.getWaypoints()],
                'nearest_aircraft': None,
                'nearest_aircraft_dist': float('inf'),
                'nearest_obstacle': None,
                'nearest_obstacle_dist': float('inf')
            }
            
            # Find nearest aircraft
            for other_ac in self.aircraft:
                if other_ac != ac:
                    dist = Utility.locDist(ac.getLocation(), other_ac.getLocation())
                    if dist < ac_state['nearest_aircraft_dist']:
                        ac_state['nearest_aircraft_dist'] = dist
                        ac_state['nearest_aircraft'] = other_ac.getIdent()
            
            # Find nearest obstacle
            for obs in self.obstacles:
                dist = Utility.locDist(ac.getLocation(), obs.location)
                if dist < ac_state['nearest_obstacle_dist']:
                    ac_state['nearest_obstacle_dist'] = dist
                    ac_state['nearest_obstacle'] = obs
            
            state['aircraft'].append(ac_state)
        
        return state
    
    def render(self, mode='human'):
        """Render the current state of the simulation."""
        # Clear surface
        self.surface.fill((0, 0, 0))
        
        # Draw obstacles
        for obs in self.obstacles:
            obs.draw(self.surface)
        
        # Draw destinations
        for dest in self.destinations:
            dest.draw(self.surface)
        
        # Draw aircraft
        for ac in self.aircraft:
            ac.draw(self.surface)
        
        if mode == 'human':
            # Display on screen if pygame display is initialized
            try:
                screen = pygame.display.get_surface()
                if screen is None:
                    pygame.display.set_mode(self.screen_size)
                    screen = pygame.display.get_surface()
                screen.blit(self.surface, (0, 0))
                pygame.display.flip()
            except pygame.error:
                pass
        
        return self.surface