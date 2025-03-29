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
        # Get configuration values
        n_aircraft = conf.get()['game']['n_aircraft']
        spawn_interval = conf.get()['game']['spawn_interval']
        
        # Create staggered spawn times for better separation
        randtime = [1 + i * spawn_interval for i in range(n_aircraft)]
        
        randspawnevent = []
        assigned_destinations = set()  # Track which destinations have been assigned
        existing_spawn_points = []  # Track existing spawn points for separation check
        
        # Minimum distance between spawn points
        min_spawn_separation = 400
        max_overall_attempts = 100  # Maximum attempts to generate a valid configuration
        
        # First, ensure all destinations get assigned before allowing repetition
        destinations_to_assign = list(destinations)
        
        for overall_attempt in range(max_overall_attempts):
            success = True
            randspawnevent = []
            assigned_destinations = set()
            existing_spawn_points = []
            trajectories = []  # Store spawn points and destinations for trajectory checking
            
            # First pass: Assign unique destinations to aircraft (up to the number of available destinations)
            first_pass_count = min(n_aircraft, len(destinations_to_assign))
            
            for i in range(first_pass_count):
                valid_spawn_dest_found = False
                
                for attempt in range(50):  # Try up to 50 times to find a valid spawn/destination combo
                    # Generate a spawn point with minimum separation from existing points
                    randspawn, side = AircraftSpawnEvent.__generateSeparatedSpawnPoint(
                        screen_w, screen_h, existing_spawn_points, min_spawn_separation)
                    
                    # Define side-specific filtering functions
                    if (side == 1):  # Top
                        def t1(d): 
                            l = d.getLocation()
                            return l[1] > screen_h/2
                    elif (side == 2):  # Right
                        def t1(d): 
                            l = d.getLocation()
                            return l[0] < screen_w/2
                    elif (side == 3):  # Bottom
                        def t1(d): 
                            l = d.getLocation()
                            return l[1] < screen_h/2
                    elif (side == 4):  # Left
                        def t1(d): 
                            l = d.getLocation()
                            return l[0] > screen_w/2
                    
                    # Get valid destinations based on spawn side that haven't been assigned yet
                    valid_dests = [d for d in destinations_to_assign if d not in assigned_destinations and t1(d)]
                    
                    # If no valid destinations for this side, try any unassigned destination
                    if not valid_dests:
                        valid_dests = [d for d in destinations_to_assign if d not in assigned_destinations]
                    
                    # If still no valid destinations, this attempt fails
                    if not valid_dests:
                        continue
                        
                    # Try each valid destination to see if trajectories don't intersect
                    for dest in valid_dests:
                        dest_loc = dest.getLocation()
                        
                        # Check for trajectory conflicts with existing aircraft
                        has_conflict = False
                        for spawn_point, dest_point in trajectories:
                            if AircraftSpawnEvent.trajectory_will_intersect(
                                randspawn, dest_loc, 
                                spawn_point, dest_point
                            ):
                                has_conflict = True
                                break
                        
                        # If no conflicts, we found a valid combination
                        if not has_conflict:
                            existing_spawn_points.append(randspawn)
                            assigned_destinations.add(dest)
                            trajectories.append((randspawn, dest_loc))
                            randspawnevent.append(AircraftSpawnEvent(randspawn, dest))
                            valid_spawn_dest_found = True
                            break
                    
                    if valid_spawn_dest_found:
                        break
                
                # If we couldn't find a valid spawn/destination for this aircraft, restart
                if not valid_spawn_dest_found:
                    success = False
                    break
            
            # If first pass successful, continue with second pass if needed
            if success and first_pass_count < n_aircraft:
                # Second pass: Allow repetition of destinations for remaining aircraft
                remaining_count = n_aircraft - first_pass_count
                
                for i in range(remaining_count):
                    valid_spawn_dest_found = False
                    
                    for attempt in range(50):
                        randspawn, side = AircraftSpawnEvent.__generateSeparatedSpawnPoint(
                            screen_w, screen_h, existing_spawn_points, min_spawn_separation)
                        
                        # Define side-specific filtering functions as before
                        if (side == 1):  # Top
                            def t1(d): 
                                l = d.getLocation()
                                return l[1] > screen_h/2
                        elif (side == 2):  # Right
                            def t1(d): 
                                l = d.getLocation()
                                return l[0] < screen_w/2
                        elif (side == 3):  # Bottom
                            def t1(d): 
                                l = d.getLocation()
                                return l[1] < screen_h/2
                        elif (side == 4):  # Left
                            def t1(d): 
                                l = d.getLocation()
                                return l[0] > screen_w/2
                        
                        # Now we can use ALL destinations, but prefer those matching the side filter
                        valid_dests = AircraftSpawnEvent.valid_destinations(destinations, t1, lambda p1, p2: 1)
                        
                        # If no valid destinations, this attempt fails (should be rare)
                        if not valid_dests:
                            continue
                        
                        # Try each valid destination to see if trajectories don't intersect
                        random.shuffle(valid_dests)  # Randomize order to avoid bias
                        for dest in valid_dests:
                            dest_loc = dest.getLocation()
                            
                            # Check for trajectory conflicts
                            has_conflict = False
                            for spawn_point, dest_point in trajectories:
                                if AircraftSpawnEvent.trajectory_will_intersect(
                                    randspawn, dest_loc, 
                                    spawn_point, dest_point
                                ):
                                    has_conflict = True
                                    break
                            
                            # If no conflicts, we found a valid combination
                            if not has_conflict:
                                existing_spawn_points.append(randspawn)
                                trajectories.append((randspawn, dest_loc))
                                randspawnevent.append(AircraftSpawnEvent(randspawn, dest))
                                valid_spawn_dest_found = True
                                break
                        
                        if valid_spawn_dest_found:
                            break
                    
                    # If we couldn't find a valid spawn/destination for this aircraft, restart
                    if not valid_spawn_dest_found:
                        success = False
                        break
            
            # If we successfully assigned all aircraft, we're done
            if success:
                break
        
        if not success:
            print("Warning: Could not generate a complete valid configuration after multiple attempts.")
            print(f"Number of aircraft successfully configured: {len(randspawnevent)}")
        
        # Return whatever we have, even if incomplete
        print(f"Generated {len(randspawnevent)} aircraft with staggered spawn times")
        return (randtime[:len(randspawnevent)], randspawnevent)
    
    
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
    
    def trajectory_will_intersect(spawn1, dest1, spawn2, dest2, time_window=60, safety_radius=200):
        """Check if two aircraft trajectories will come too close within a time window"""
        # Calculate vectors and speeds
        vector1 = (dest1[0] - spawn1[0], dest1[1] - spawn1[1])
        vector2 = (dest2[0] - spawn2[0], dest2[1] - spawn2[1])
        
        length1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        length2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
        
        # Normalize and scale by speed (you may need to adjust this based on your aircraft speed)
        speed = conf.get()['aircraft']['speed_default']
        unit_v1 = (vector1[0]/length1 * speed, vector1[1]/length1 * speed)
        unit_v2 = (vector2[0]/length2 * speed, vector2[1]/length2 * speed)
        
        # Check multiple points along the trajectories
        for t in range(time_window):
            pos1 = (spawn1[0] + unit_v1[0]*t, spawn1[1] + unit_v1[1]*t)
            pos2 = (spawn2[0] + unit_v2[0]*t, spawn2[1] + unit_v2[1]*t)
            
            if AircraftSpawnEvent.dist_between_points(pos1, pos2) < safety_radius:
                return True
        
        return False