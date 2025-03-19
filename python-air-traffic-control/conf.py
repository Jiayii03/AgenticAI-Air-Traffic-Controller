# conf.py
import json
import os

# Get the directory where conf.py is located
base_dir = os.path.dirname(os.path.abspath(__file__))
configfile = os.path.join(base_dir, "config_game.json")
loaded = ""

def writeout():
    with open(configfile, "w") as f:
        json.dump(loaded, f)
    
def load():
    global loaded
    with open(configfile) as f:
        loaded = json.load(f)

def get():
    global loaded
    return loaded
    
load()