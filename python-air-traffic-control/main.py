#!/usr/bin/env python
#   File: main.py

from pygame import *
from game import *
import menu_base
import conf
import random_state

STATE_MENU = 1
STATE_GAME = 2
STATE_DEMO = 3
STATE_HIGH = 4
STATE_KILL = 5
STATE_AGES = 6

class Main:

    BG_COLOR = (0, 0, 0)

    def __init__(self):
        #Init the modules we need
        display.init()
        pygame.mixer.init()
        font.init()
        
        if(conf.get()['game']['fullscreen'] == True):
            self.screen = display.set_mode((1024, 768), pygame.FULLSCREEN)
        else:
            self.screen = display.set_mode((1024, 768))
            
        display.set_caption('ATC Version 2')

        self.menu = menu_base.menu_base(self.screen,150,25)
        self.menu.from_file('main_menu')
        self.ages = menu_base.menu_base(self.screen,150,25)
        self.ages.from_file('ages_menu')
        # self.high = HighScore(self.screen)
        # self.infologger = info_logger.info_logger()
        #Current visitor number
        # self.id = int(self.infologger.get_id())

    def run(self):
        state = STATE_MENU
        exit = 0
        score = 0       
        game = Game(self.screen, False)
        (gameEndCode, score) = game.start()

# main entry point
if __name__ == '__main__':
    # Get the scenario seed from config
    scenario_seed = conf.get().get('game', {}).get('scenario_seed', 0)
    
    # Set the global seed - if it's 0, a random seed will be generated
    used_seed = random_state.set_global_seed(scenario_seed)
    
    game_main = Main()
    game_main.run()
