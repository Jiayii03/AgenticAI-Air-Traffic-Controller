# File: destination.py
# Author: Tom Woolfrey

import pygame
import random
import math
import conf
from waypoint import Waypoint
from utility import Utility

class Destination(Waypoint):

    COLOR_DEST = (192, 192, 192)

    def __init__(self, location, text):
        self.location = location
        self.text = text
        font = pygame.font.Font(None, 20)
        self.font_img = font.render(text, True, Destination.COLOR_DEST)

    def draw(self, surface):
        pygame.draw.circle(surface, Destination.COLOR_DEST, self.location, 5, 0)
        surface.blit(self.font_img, (self.location[0] + 8, self.location[1] + 8))

    def clickedOn(self, clickpos):
        return False
		
    @staticmethod
    def generateGameDestinations(screen_w, screen_h):
        ret = []
        # Define minimum distance between destinations
        min_distance = conf.get().get('destinations', {}).get('min_distance', 100)
        # Maximum attempts to place a destination
        max_attempts = 100
        
        # Margin from screen edges
        margin = 40
        
        for x in range(0, conf.get()['game']['n_destinations']):
            # Try to place each destination with safe distance
            placed = False
            attempts = 0
            
            while not placed and attempts < max_attempts:
                # Generate random position within screen bounds (with margin)
                randx = random.randint(margin, screen_w - margin)
                randy = random.randint(margin, screen_h - margin)
                candidate_loc = (randx, randy)
                
                # Check distance to all existing destinations
                valid_position = True
                for existing_dest in ret:
                    dist = Utility.locDist(candidate_loc, existing_dest.location)
                    if dist < min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    # Create new destination and add to list
                    dest = Destination(candidate_loc, "D" + str(x))
                    ret.append(dest)
                    placed = True
                    print(f"Placed destination D{x} at {candidate_loc}")
                
                attempts += 1
            
            # If couldn't place after max attempts, place anyway but print warning
            if not placed:
                randx = random.randint(margin, screen_w - margin)
                randy = random.randint(margin, screen_h - margin)
                dest = Destination((randx, randy), "D" + str(x))
                ret.append(dest)
                print(f"Warning: Could not place destination D{x} with safe distance after {max_attempts} attempts")
        
        return ret