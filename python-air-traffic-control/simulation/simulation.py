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
    
    def __init__(self, logger, num_planes=5, num_obstacles=3, screen_size=(800, 800)):
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
        self.logger = logger
        self.num_planes = num_planes
        self.num_obstacles = num_obstacles
        self.screen_size = screen_size
        
        # Initialize components from the original game
        self.destinations = Destination.generateGameDestinations(screen_size[0], screen_size[1])
        self.obstacles = Obstacle.generateGameObstacles(screen_size[0], screen_size[1], self.destinations)
        self.aircraft = []
        self.collision_radius = conf.get()['aircraft']['collision_radius']
        self.tactical_waypoint_distance = conf.get()['rl_agent']['tactical_waypoint_distance']
        
        # Track simulation state
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
        self.recent_collisions = {}
        self.done = False
        self.step_count = 0
        
        # Initialize spawn schedule instead of spawning all aircraft immediately
        self.total_aircraft_to_spawn = self.num_planes
        self.aircraft_spawned = 0
        self.next_spawn_time = 0  # Spawn first aircraft immediately
        self.spawn_interval_range = (20, 100)  # Frames between spawns
        
        # Spawn the first aircraft immediately
        if self.total_aircraft_to_spawn > 0:
            self._spawn_single_aircraft()
            self.aircraft_spawned += 1
        
        return self._get_state()
    
    def step(self, actions):
        """Execute actions and advance simulation by one step."""
        self.step_count += 1
        had_collision = False
        num_planes_landed = 0
        rewards_dict = {ac.getIdent(): 0 for ac in self.aircraft}
        
        # Check if it's time to spawn a new aircraft
        if (self.aircraft_spawned < self.total_aircraft_to_spawn and 
            self.step_count >= self.next_spawn_time):
            self._spawn_single_aircraft()
            self.aircraft_spawned += 1
            
            # Schedule next spawn
            next_interval = np.random.randint(
                self.spawn_interval_range[0], 
                self.spawn_interval_range[1]
            )
            self.next_spawn_time = self.step_count + next_interval
            self.logger.debug_print(f"Next aircraft spawn scheduled at step {self.next_spawn_time}")
            
        # Apply actions to aircraft
        for i, ac in enumerate(self.aircraft):
            if i < len(actions):
                self._apply_action(ac, actions[i])
                
        # track aircraft progress
        for ac in self.aircraft:
            if len(ac.waypoints) > 0:
                # Distance to final destination
                dest = ac.waypoints[-1].getLocation()
                current = ac.getLocation()
                dist_to_dest = Utility.locDist(current, dest)
                
                # Distance to next waypoint (tactical or destination)
                next_wp = ac.waypoints[0].getLocation()
                dist_to_next = Utility.locDist(current, next_wp)
                
                self.logger.debug_print(f"Aircraft {ac.getIdent()} current position: {current} Distance to next waypoint: {dist_to_next:.2f}, to destination: {dist_to_dest:.2f}")
        
        # Update aircraft positions
        landed_aircraft = []
        for ac in self.aircraft:
            # Update aircraft position
            reached_destination = ac.update()
            if reached_destination:
                landed_aircraft.append(ac)
                num_planes_landed += 1
        
        # Remove landed aircraft
        for ac in landed_aircraft:
            self.aircraft.remove(ac)
        
        # Check for collisions
        self.collision_pairs = self._detect_collisions()
        for pair in self.collision_pairs:
            ac1, ac2 = pair
            # Add safety checks before applying penalties
            if ac1.getIdent() in rewards_dict:
                rewards_dict[ac1.getIdent()] -= 200  # Penalty for collision
            else:
                print(f"Warning: Aircraft {ac1.getIdent()} not found in rewards_dict")
                
            if ac2.getIdent() in rewards_dict:
                rewards_dict[ac2.getIdent()] -= 200
            else:
                print(f"Warning: Aircraft {ac2.getIdent()} not found in rewards_dict")
                
            print(f"Collision detected between {ac1.getIdent()} and {ac2.getIdent()}, -200 reward")
            had_collision = True
            self.done = True
        
        # Get current state
        state = self._get_state()
        
        # Check if simulation is done
        if len(self.aircraft) == 0:
            self.done = True
        
        return state, rewards_dict, self.done, {"num_planes_landed": num_planes_landed, "had_collision": had_collision}
    
    def _spawn_single_aircraft(self):
        """Spawn a single aircraft with minimum separation from existing aircraft."""
        min_separation = 2.5 * self.collision_radius
        max_attempts = 10
        valid_position = False
        attempts = 0
        
        while not valid_position and attempts < max_attempts:
            attempts += 1
            
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
            
            # Check distance to all existing aircraft
            valid_position = True
            for existing_ac in self.aircraft:
                if Utility.locDist(loc, existing_ac.getLocation()) < min_separation:
                    valid_position = False
                    break
        
        # Select random destination
        dest = np.random.choice(self.destinations)
        
        # Create aircraft
        speed = conf.get()['aircraft']['speed_default']
        ident = f"AC{self.aircraft_spawned+1}"
        ac = Aircraft(None, loc, speed, dest, ident)
        self.aircraft.append(ac)
        
        self.logger.debug_print(f"Spawned {ident} at {loc} heading to {dest.getLocation()}")
    
    def _apply_action(self, aircraft, action):
        """Apply an action to an aircraft."""
        # Get current location and destination
        curr_loc = aircraft.getLocation()
        
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
            self.logger.debug_print(f"Aircraft {aircraft.getIdent()}: Maintaining existing waypoints, Waypoints={[wp.getLocation() for wp in aircraft.waypoints]}")
            return
        
        # Clear existing waypoints only if we're going to modify them
        aircraft.waypoints = []
        
        # Add destination back if it exists
        if destination:
            aircraft.addWaypoint(destination)
        else:
            # If no destination exists, just maintain current heading
            print(f"Aircraft {aircraft.getIdent()}: No destination found!")
            return
        
        # Apply tactical waypoint if action is not "maintain course"
        if action != 0:
            # Apply the selected action's heading change
            curr_heading = aircraft.getHeading()
            
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
            
            self.logger.debug_print(f"Aircraft {aircraft.getIdent()}: Action={action}, Applied tactical waypoint")
        else:
            self.logger.debug_print(f"Aircraft {aircraft.getIdent()}: Direct routing to destination")
        
        self.logger.debug_print(f"Aircraft {aircraft.getIdent()}: Waypoints={[wp.getLocation() for wp in aircraft.waypoints]}")
    
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