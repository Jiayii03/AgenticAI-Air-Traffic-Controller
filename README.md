# AgenticAI-Air-Traffic-Controller

COMP3071 Designing Intelligent Agents

## Project Overview

This project implements an intelligent air traffic control system using reinforcement learning. The agents handle both collision avoidance and emergency destination rerouting.

## Features

- Collision avoidance using a DQN agent
- Emergency rerouting using a separate DQN agent
- Visual feedback for emergency situations
- Reproducible simulation using seed-based randomization

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

The system is configurable through `config_game.json`, where you can set:

- Number of aircraft and destinations
- Collision risk parameters
- Maximum number of emergencies
- Random seed for reproducibility

## Seed Numbers for Demonstration

Head over to `config_game.json` and change the `scenario_seed`. Suitable config as follows:

### Without Emergency
1. 331280 (5 aircrafts, 5 destinations)
2. 350072 
3. 379381 

### With Emergency
1. 508861 (5 aircrafts, 5 destinations)
2. 776740 (7 aircrafts, 5 destinations)
3. 959018, 478953 

## TODO:

1. ✓ Try train RL using DQN method with neural network
2. ✓ Integrate collision avoidance
3. ✓ Integrate emergency situation
4. Benchmark against baseline, e.g. random or straight line
5. Parameterised experiment and rigorous testing