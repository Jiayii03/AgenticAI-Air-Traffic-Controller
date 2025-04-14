#!/usr/bin/env python
#   File: main.py

import pygame
from game import Game
from core import conf
import random_state

# main entry point
if __name__ == '__main__':
    # Initialize pygame
    pygame.init()
    
    # Get the scenario seed from config
    scenario_seed = conf.get().get('game', {}).get('scenario_seed', 0)
    
    # Set the global seed
    used_seed = random_state.set_global_seed(scenario_seed)
    
    # Set up the display
    if(conf.get()['game']['fullscreen'] == True):
        screen = pygame.display.set_mode((1024, 768), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((1024, 768))
    
    # Set the window caption
    pygame.display.set_caption('Air Traffic Control Simulation')
    
    # Create and start the game
    game = Game(screen, False)  # False means not in demo mode
    game.start()
    
    # Clean up
    pygame.quit()