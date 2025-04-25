# AgenticAI-Air-Traffic-Controller

COMP3071 Designing Intelligent Agents

## Project Overview

This project implements an intelligent air traffic control system using reinforcement learning. The agents handle both collision avoidance and emergency destination rerouting.

## Directory Structure

```
AgenticAI-Air-Traffic-Controller
├── assets
│   ├── sounds/
│   └── themes/
├── core
│   ├── aircraft.py
│   ├── aircraftspawnevent.py
│   ├── conf.py
│   ├── destination.py
│   ├── flightstrippane.py
│   ├── obstacle.py
│   ├── utility.py
│   └── waypoint.py
├── game
│   ├── game.py
│   ├── main.py
│   └── random_state.py
├── models
│   ├── collision_avoidance_dqn_model.pth
│   └── emergency_dqn_model.pth
├── pgu
│   └── gui/
├── simulation
│   ├── collision_avoidance/
│   ├── common/
│   ├── emergency/
│   ├── evaluation/
│   └── logs/
│   └── results/
├── config_game.json
├── requirements.txt
├── README.md
└── .gitignore
```

## Quickstart

1. Clone the repository and navigate to the root directory:
```bash
cd AgenticAI-Air-Traffic-Controller
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the frontend simulation:
```bash
cd game
python main.py
```

The Pygame-based air traffic control simulation will launch, allowing you to observe real-time agent decisions for both collision avoidance and emergency response.

## Key Features

- Collision avoidance using a DQN agent
- Emergency rerouting using a separate DQN agent
- Intuitive frontend visualisation to simulate real-world ATC
- Visual feedback for emergency situations

## Collision Avoidance System

### Observation Space

The collision avoidance agent uses the following state representation:

```
[
    distance_between_pair,      # Distance between the two aircraft in the pair
    angle_to_intruder,          # Angle from aircraft 1 to aircraft 2
    relative_heading,           # Relative heading of aircraft 2 compared to aircraft 1
    angle_to_destination,       # Angle from aircraft 1 to its destination
    angular_diff_to_dest,       # Angular difference between current heading and destination angle
    relative_speed,             # Speed of aircraft 2 relative to aircraft 1
    separation_rate             # Rate of separation/approach (-1 to 1)
]
```

### Action Space

The collision avoidance agent can perform 9 different actions:

- 0: Maintain course and speed (N)
- 1: Medium left turn (90°) with normal speed (ML)
- 2: Slight left turn (45°) with normal speed (SL)
- 3: Medium right turn (90°) with normal speed (MR)
- 4: Slight right turn (45°) with normal speed (SR)
- 5: Medium left turn (90°) with reduced speed (ML-S)
- 6: Slight left turn (45°) with reduced speed (SL-S)
- 7: Medium right turn (90°) with reduced speed (MR-S)
- 8: Slight right turn (45°) with reduced speed (SR-S)

### Reward Function

The collision avoidance reward function considers:

- Distance between aircraft (quadratic penalty for close proximity)
- Heading alignments (reward for diverging headings)
- Destination alignment (reward for maintaining heading toward destination)
- Speed adjustment effectiveness (reward for effective use of speed reductions)
- Time penalties (small penalty to encourage efficient resolution)

## Emergency Rerouting System

### Observation Space

The emergency agent uses the following state representation (9 dimensions):

```
[
    distance_to_emergency,      # Distance from aircraft to emergency destination
    angle_diff_to_emergency,    # Angle difference between heading and emergency destination
    
    # For each of 3 safe alternative destinations:
    distance_to_safe1,          # Distance to first safe destination
    angle_diff_to_safe1,        # Angle difference to first safe destination
    distance_to_safe2,          # Distance to second safe destination
    angle_diff_to_safe2,        # Angle difference to second safe destination
    distance_to_safe3,          # Distance to third safe destination
    angle_diff_to_safe3,        # Angle difference to third safe destination
    
    normalized_speed            # Aircraft's speed normalized between 0-1
]
```

### Action Space

The emergency agent selects one of 3 actions:

- 0: Reroute to first safe destination
- 1: Reroute to second safe destination
- 2: Reroute to third safe destination

### Reward Function

The emergency agent reward function considers:

- Immediate reward for successful rerouting (+50)
- Bonus for landing at a safe destination (+100)
- Penalty for landing at emergency destination (-200)
- Small time penalty to encourage quick decisions (-1 per step)

## Configuration

The system is parameterised and configurable through `config_game.json`, where you can change:

- Number of aircraft and destinations
- Collision risk parameters
- RL model paths
- Maximum number of emergencies
- Random seed for reproducibility
