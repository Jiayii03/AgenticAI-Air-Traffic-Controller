#	File: aircraftspawnevent.py

import conf
import random
import math

class AircraftSpawnEvent:

    def __init__(self, spawnpoint, destination):
        self.spawnpoint = spawnpoint
        self.destination = destination

    def getSpawnPoint(self):
        return self.spawnpoint

    def getDestination(self):
        return self.destination

    def __str__(self):
        return "<" + str(self.spawnpoint) + ", " + str(self.destination.getLocation()) + ">"

    @staticmethod
    def valid_destinations(destinations, test1, test2):
        d = [item for item in destinations if test1(item)]
        if (len(d) == 0):
            return destinations
        else:
            return d

    @staticmethod
    def dist_between_points(p1, p2):
        """Calculate distance between two points"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    @staticmethod
    def generateGameSpawnEvents(screen_w, screen_h, destinations):
        # All aircraft will spawn at time 1
        spawn_time = 1
        n_aircraft = conf.get()['game']['n_aircraft']
        
        # Create a list with the same spawn time for all aircraft
        randtime = [spawn_time] * n_aircraft
        
        randspawnevent = []
        assigned_destinations = set()  # Track which destinations have been assigned
        existing_spawn_points = []  # Track existing spawn points for separation check
        
        # Minimum distance between spawn points (adjust as needed)
        min_spawn_separation = 400
        
        for i in range(n_aircraft):
            # Generate a spawn point with minimum separation from existing points
            randspawn, side = AircraftSpawnEvent.__generateSeparatedSpawnPoint(
                screen_w, screen_h, existing_spawn_points, min_spawn_separation)
            
            # Add the new spawn point to the list
            existing_spawn_points.append(randspawn)
            
            # Define side-specific filtering functions
            if (side == 1):  # Top
                def t1(d): 
                    l = d.getLocation()
                    return l[1] > screen_h/2
                def t2(p1,p2):
                    return 1
            elif (side == 2):  # Right
                def t1(d): 
                    l = d.getLocation()
                    return l[0] < screen_w/2
                def t2(p1,p2):
                    return 1
            elif (side == 3):  # Bottom
                def t1(d): 
                    l = d.getLocation()
                    return l[1] < screen_h/2
                def t2(p1,p2):
                    return 1
            elif (side == 4):  # Left
                def t1(d): 
                    l = d.getLocation()
                    return l[0] > screen_w/2
                def t2(p1,p2):
                    return 1
            
            # Get valid destinations based on spawn side
            valid_dests = AircraftSpawnEvent.valid_destinations(destinations, t1, t2)
            
            # Filter out already assigned destinations
            available_dests = [d for d in valid_dests if d not in assigned_destinations]
            
            # If no unassigned destinations remain, fall back to all valid destinations
            if not available_dests:
                print(f"Warning: No unassigned destinations available for aircraft {i+1}. Choosing randomly from all valid destinations.")
                available_dests = valid_dests
            
            # Choose a destination
            randdest = random.choice(available_dests)
            
            # Track this destination as assigned if it wasn't already
            assigned_destinations.add(randdest)
            
            # Create spawn event
            randspawnevent.append(AircraftSpawnEvent(randspawn, randdest))
            
        print(f"Generated {n_aircraft} aircraft to spawn simultaneously at time {spawn_time}")
        return (randtime, randspawnevent)

    @staticmethod
    def __generateSeparatedSpawnPoint(screen_w, screen_h, existing_points, min_separation):
        """Generate a spawn point that is at least min_separation away from existing points"""
        max_attempts = 50  # Maximum number of attempts to find a separated point
        
        for attempt in range(max_attempts):
            # Generate a random spawn point
            spawn_point, side = AircraftSpawnEvent.__generateRandomSpawnPoint(screen_w, screen_h)
            
            # If no existing points, return this one
            if not existing_points:
                return spawn_point, side
            
            # Check if this point is far enough from all existing points
            is_separated = True
            for existing_point in existing_points:
                distance = AircraftSpawnEvent.dist_between_points(spawn_point, existing_point)
                if distance < min_separation:
                    is_separated = False
                    break
            
            # If this point is separated enough, return it
            if is_separated:
                return spawn_point, side
        
        # If we couldn't find a well-separated point after max attempts,
        # just return the last generated point and print a warning
        print(f"Warning: Could not find a well-separated spawn point after {max_attempts} attempts.")
        return spawn_point, side

    @staticmethod
    def __generateRandomSpawnPoint(screen_w, screen_h):
        side = random.randint(1, 4)
        
        if side == 1:  # Top
            loc = (random.randint(20, screen_w - 20), 0)
        elif side == 2:  # Right
            loc = (screen_w, random.randint(20, screen_h - 20))
        elif side == 3:  # Bottom
            loc = (random.randint(20, screen_w - 20), screen_h)
        elif side == 4:  # Left
            loc = (0, random.randint(20, screen_h - 20))
        
        return loc, side