#   File: aircraft.py

import math
import pygame
import os
from core import conf
from core.waypoint import *
from core.utility import *

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))

class FlightState:
    def __init__(self):
        # Initialize any necessary attributes
        pass

    def updateAllFields(self):
        # Update logic...
        pass

    def select(self):
        # Logic for selecting the flight state
        pass

    def deselect(self):
        # Logic for deselecting the flight state
        pass

class Aircraft:

    AC_IMAGE_NORMAL = pygame.image.load(os.path.join(ASSETS_DIR, 'aircraft.png'))
    AC_IMAGE_SELECTED = pygame.image.load(os.path.join(ASSETS_DIR, 'aircraft_sel.png'))
    AC_IMAGE_NEAR = pygame.image.load(os.path.join(ASSETS_DIR, 'aircraft_near.png'))
    AC_IMAGE_REROUTE = pygame.image.load(os.path.join(ASSETS_DIR, 'aircraft_reroute.png'))

    AC_STATE_NORMAL = 1
    AC_STATE_SELECTED = 2
    AC_STATE_NEAR = 3
    AC_STATE_REROUTE = 4

    EVENT_CLICK_AC = 0
    EVENT_CLICK_FS = 1
    
    FS_FONTSIZE = 18

    # Constructor
    def __init__(self, game, location, speed, destination, ident):
        self.game = game
        self.location = location
        self.speed = speed
        self.altitude = 24000  # hardwired for now; measured in ft
        self.waypoints = []
        self.collisionRisk = []
        self.destination = destination  # Add destination attribute
        self.original_destination = destination  # Track the original destination
        self.waypoints.append(destination)  # Add the destination as the first waypoint
        self.ident = ident
        self.selected = False
        self.state = Aircraft.AC_STATE_NORMAL
        self.heading = self.__calculateHeading(self.location, self.waypoints[0].getLocation())
        self.rerouted = False
        Aircraft.AC_IMAGE_NORMAL.convert_alpha()
        Aircraft.AC_IMAGE_SELECTED.convert_alpha()

        # Image/font vars
        self.image = Aircraft.AC_IMAGE_NORMAL
        self.font = pygame.font.Font(None, Aircraft.FS_FONTSIZE)
        self.fs_font_color = (255, 255, 255)

        # Initialize the flight state
        self.fs = FlightState()

    def update_destination(self, new_destination):
        """Update the aircraft's destination."""
        self.original_destination = self.destination
        self.destination = new_destination  # This should be a Destination object

        # Clear existing waypoints and add the new destination
        self.waypoints = []
        self.addWaypoint(Waypoint(new_destination.getLocation()))  # Create a Waypoint from the destination location

    # Add a new waypoint in the specified index in the list
    def addWaypoint(self, waypoint, index=0):
        if len(self.waypoints) < conf.get()['aircraft']['max_waypoints'] + 1:
            self.waypoints.insert(index, waypoint)
            self.heading = self.__calculateHeading(self.location, self.waypoints[0].getLocation())

    # Get the specified waypoint from the list
    def getWaypoint(self, index):
        return self.waypoints[index]
    
    def getWaypoints(self):
        return self.waypoints

    # Return current location
    def getLocation(self):
        return self.location

    # Return current heading
    def getHeading(self):
        ret = 0
        if self.heading < 0:
            ret = 360 + self.heading
        else:
            ret = self.heading
        return ret
        
    def getHeadingStr(self):
        hdg = str(self.getHeading())
        hdg_str = hdg.split(".")[0]
        return hdg_str
        
    def getIdent(self):
        return self.ident

    def getSpeed(self):
        return self.speed

    # Set speed in pixels per frame
    def setSpeed(self, newspeed):
        self.speed = newspeed

    # Set whether I am the selected aircraft or not
    def setSelected(self, selected):
        self.selected = selected
        if selected:
            self.image = Aircraft.AC_IMAGE_SELECTED
            self.fs.select()
        else:
            self.image = Aircraft.AC_IMAGE_NORMAL
            self.fs.deselect()
    
    def setRerouted(self, rerouted=True):
        """Set whether this aircraft is being rerouted."""
        self.rerouted = rerouted
        if rerouted:
            self.image = Aircraft.AC_IMAGE_REROUTE
        else:
            # Revert to normal image
            if self.selected:
                self.image = Aircraft.AC_IMAGE_SELECTED
            else:
                self.image = Aircraft.AC_IMAGE_NORMAL
            
    def requestSelected(self):
        self.game.requestSelected(self)

    def draw(self, surface):
        # Choose the appropriate image based on state
        if hasattr(self, 'rerouted') and self.rerouted:
            rot_image = pygame.transform.rotate(Aircraft.AC_IMAGE_REROUTE, -self.heading)
            rect = rot_image.get_rect()
            rect.center = self.location
            surface.blit(rot_image, rect)
        
        elif hasattr(self, 'rl_controlled') and self.rl_controlled:
            # Use the selected image for RL-controlled aircraft
            rot_image = pygame.transform.rotate(Aircraft.AC_IMAGE_SELECTED, -self.heading)
            rect = rot_image.get_rect()
            rect.center = self.location
            surface.blit(rot_image, rect)
            
            # Draw orange circle to indicate RL control (larger than the collision radius)
            pygame.draw.circle(surface, (255, 165, 0), self.location, 25, 2)
            
            # Draw lines and waypoints if aircraft is RL controlled
            point_list = []
            point_list.append(self.location)
            for x in range(0, len(self.waypoints)-1):
                point_list.append(self.waypoints[x].getLocation())
                self.waypoints[x].draw(surface)
            point_list.append(self.waypoints[-1].getLocation())
            # Draw lines in orange to distinguish from normal selection (which uses yellow)
            pygame.draw.lines(surface, (255, 165, 0), False, point_list)
        else:
            # Original draw code for non-RL controlled aircraft
            rot_image = pygame.transform.rotate(self.image, -self.heading)
            rect = rot_image.get_rect()
            rect.center = self.location
            surface.blit(rot_image, rect)

            if conf.get()['aircraft']['draw_radius']:
                pygame.draw.circle(surface, (255, 255, 0), self.location, conf.get()['aircraft']['collision_radius'], 1)

            # Draw lines and waypoints if selected
            if self.selected:
                point_list = []
                point_list.append(self.location)
                for x in range(0, len(self.waypoints)-1):
                    point_list.append(self.waypoints[x].getLocation())
                    self.waypoints[x].draw(surface)
                point_list.append(self.waypoints[-1].getLocation())
                pygame.draw.lines(surface, (255, 255, 0), False, point_list)

        # Draw the ident string next to the aircraft
        x = self.location[0] + 20
        y = self.location[1]
        list = [self.ident, "FL" + str(self.altitude/100), str(self.speed) + "kts"]
        for line in list:
            id = self.font.render(line, False, self.fs_font_color)
            r = surface.blit(id, (x,y))
            y = y + self.font.get_height()

    # Location/heading update function
    def update(self):
        """Update the aircraft's position and check if it has reached its destination."""
        if self.__reachedWaypoint(self.location, self.waypoints[0].getLocation()):
            self.rl_controlled = False
            # Reached next waypoint, pop it
            self.waypoints.pop(0)
            if len(self.waypoints) == 0:
                # Reached destination, return True
                return True
            else:
                # Update the current destination
                self.destination = self.waypoints[0]
        
        # Keep moving towards waypoint
        self.heading = self.__calculateHeading(self.location, self.waypoints[0].getLocation())
        self.location = self.__calculateNewLocation(self.location, self.heading, self.speed)
        self.fs.updateAllFields()
        return False
    
    def isFollowingRLWaypoint(self):
        return self.rl_controlled and len(self.waypoints) > 0

    def getClickDistanceSq(self, clickpos):
        return Utility.locDistSq(clickpos, self.location)
        
    def setFS(self, fs):
        self.fs = fs
        
    def getFS(self):
        return self.fs

    # Calculate heading based on current position and waypoint
    def __calculateHeading(self, location, waypoint):
        x_diff = waypoint[0] - location[0]
        y_diff = waypoint[1] - location[1]
        # Heading measured in degrees relative to North direction
        heading = math.degrees(math.atan2(y_diff, x_diff) + (math.pi / 2))
        return heading

    # Calculate new location based on current location, heading and speed
    def __calculateNewLocation(self, location, heading, speed):
        x_diff = (speed / conf.get()['aircraft']['speed_scalefactor']) * math.sin(math.radians(heading))
        y_diff = -(speed / conf.get()['aircraft']['speed_scalefactor']) * math.cos(math.radians(heading))
        location = (location[0] + x_diff, location[1] + y_diff)
        return location

    # Check whether I have reached the given waypoint
    def __reachedWaypoint(self, location, waypoint):
        if Utility.locDistSq(location, waypoint) < ((self.speed/conf.get()['aircraft']['speed_scalefactor']) ** 2):
            return True
        else:
            return False

    def click(self, clickpos):
        if Utility.locDistSq(clickpos, self.location) <= 100:
            return True
        else:
            return False