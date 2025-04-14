# File: destination.py
import pygame
import random
from core import conf
from core.waypoint import Waypoint
from core.utility import Utility

class Destination(Waypoint):

    COLOR_DEST = (192, 192, 192)
    COLOR_EMERGENCY = (255, 0, 0)  # Red for emergency

    def __init__(self, location, text):
        self.location = location
        self.text = text
        self.emergency = False  # New flag for emergency status
        font = pygame.font.Font(None, 20)
        self.font_img = font.render(text, True, Destination.COLOR_DEST)

    def draw(self, surface):
        # Change the drawing color if this destination is in emergency mode.
        color = Destination.COLOR_EMERGENCY if self.emergency else Destination.COLOR_DEST
        pygame.draw.circle(surface, color, self.location, 5, 0)
        surface.blit(self.font_img, (self.location[0] + 8, self.location[1] + 8))

    def clickedOn(self, clickpos):
        return False

    def setEmergency(self, status=True):
        self.emergency = status
        # Optionally update the font image to reflect emergency (e.g., by re-rendering with a different color)
        font = pygame.font.Font(None, 20)
        self.font_img = font.render(self.text, True, Destination.COLOR_EMERGENCY if status else Destination.COLOR_DEST)

    def isEmergency(self):
        return self.emergency

    @staticmethod
    def generateGameDestinations(screen_w, screen_h):
        ret = []
        min_distance = conf.get().get('destinations', {}).get('min_distance', 100)
        max_attempts = 100
        margin = 40
        
        for x in range(0, conf.get()['game']['n_destinations']):
            placed = False
            attempts = 0
            
            while not placed and attempts < max_attempts:
                randx = random.randint(margin, screen_w - margin)
                randy = random.randint(margin, screen_h - margin)
                candidate_loc = (randx, randy)
                
                valid_position = True
                for existing_dest in ret:
                    dist = Utility.locDist(candidate_loc, existing_dest.location)
                    if dist < min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    dest = Destination(candidate_loc, "D" + str(x))
                    ret.append(dest)
                    placed = True
                    print(f"Placed destination D{x} at {candidate_loc}")
                
                attempts += 1
            
            if not placed:
                randx = random.randint(margin, screen_w - margin)
                randy = random.randint(margin, screen_h - margin)
                dest = Destination((randx, randy), "D" + str(x))
                ret.append(dest)
                print(f"Warning: Could not place destination D{x} with safe distance after {max_attempts} attempts")
        
        return ret
