# AgenticAI-Air-Traffic-Controller

COMP3071 Designing Intelligent Agents

### TODOs:

1. Try train RL using current SARSA (tabular) method and existing observation and action space. If cannot then need change to neural network or other RL methods.

2. Incorporate time efficiency
 - Add a small negative reward per timestep (-0.1 per step).
 - Scale landing rewards inversely with time (faster landing = higher reward).
 - Include a "direct path" component in your reward function that encourages staying close to the direct path to destination.

3. Incorporate emergency situation

### Observation space

Defines the format and boundaries of the state information that your agent will receive from the environment. 

```python
"""
1. distance to nearest aircraft: [0, 50] pixels
2. angle to nearest aircraft: [0, 360) degrees
3. relative heading of nearest aircraft: [0, 360) degrees
4. distance to nearest obstacle: [0, 50] pixels
5.  to nearest obstacle: [0, 360) degrees
"""
self.observation_space = spaces.Box(
    low=np.array([0, 0, 0, 0, 0]),    # Added two dimensions for obstacle info
    high=np.array([50, 360, 360, 50, 360]),
    dtype=np.float32
)
```

For tabular reinforcement learning methods like SARSA, you need to convert continuous state values into discrete indices to use as keys in your Q-table. "Buckets" are the discrete bins you divide each continuous dimension into.
```python
bucket_sizes = [50, 36, 36, 50, 36]
value_ranges = [[0, 50], [0, 360], [0, 360], [0, 50], [0, 360]]
```

For the `distance` dimension:
`bucket_sizes[0] = 50` means you're dividing the range into 50 buckets
`value_ranges[0] = [0, 50]` means the range spans from 0 to 50 pixels
Each bucket has a width of 1 pixel (50 ÷ 50 = 1)

For the `angle` dimension:
`bucket_sizes[1] = 36` means you're dividing the angle into 36 buckets
`value_ranges[1] = [0, 360]` means the range spans 0 to 360 degrees
Each bucket has a width of 10 degrees (360 ÷ 36 = 10)

If you expand your observation space to include obstacle information with the bucket sizes we discussed, your Q-table size would increase dramatically:
50 × 36 × 36 × 50 × 36 = 116,640,000 state combinations

### Action Space

0: N (Maintain course) - Continue straight ahead
1: HL (Hard Left) - Make a sharp left turn (90° left)
2: ML (Medium Left) - Make a moderate left turn (45° left)
3: MR (Medium Right) - Make a moderate right turn (45° right)
4: HR (Hard Right) - Make a sharp right turn (90° right)

```python
self.action_space = spaces.Discrete(5)  # 5 possible actions
```

### Environment state:
```json
{
  "aircraft": [
    {
      "id": "AC1",
      "location": [
        800,
        692
      ],
      "heading": 328.7082428175321,
      "speed": 200,
      "waypoints": [
        [
          411,
          52
        ]
      ],
      "nearest_aircraft": "AC4",
      "nearest_aircraft_dist": 215.94675269612182,
      "nearest_obstacle": "<obstacle.Obstacle object at 0x000001A741BC7680>",
      "nearest_obstacle_dist": 204.38444167793205
    },
    {
      "id": "AC2",
      "location": [
        5,
        0
      ],
      "heading": 142.06672971640108,
      "speed": 200,
      "waypoints": [
        [
          164,
          204
        ]
      ],
      "nearest_aircraft": "AC3",
      "nearest_aircraft_dist": 385.0324661635691,
      "nearest_obstacle": "<obstacle.Obstacle object at 0x000001A7416EED80>",
      "nearest_obstacle_dist": 223.2129028528593
    },
    {
      "id": "AC3",
      "location": [
        0,
        385
      ],
      "heading": 46.18730731572882,
      "speed": 200,
      "waypoints": [
        [
          197,
          196
        ]
      ],
      "nearest_aircraft": "AC2",
      "nearest_aircraft_dist": 385.0324661635691,
      "nearest_obstacle": "<obstacle.Obstacle object at 0x000001A741BC76E0>",
      "nearest_obstacle_dist": 210.60151946270474
    },
    {
      "id": "AC4",
      "location": [
        613,
        800
      ],
      "heading": 345.98728662500656,
      "speed": 200,
      "waypoints": [
        [
          470,
          227
        ]
      ],
      "nearest_aircraft": "AC1",
      "nearest_aircraft_dist": 215.94675269612182,
      "nearest_obstacle": "<obstacle.Obstacle object at 0x000001A741BC7680>",
      "nearest_obstacle_dist": 258.940147524481
    },
    {
      "id": "AC5",
      "location": [
        800,
        333
      ],
      "heading": 216.20945681718607,
      "speed": 200,
      "waypoints": [
        [
          677,
          501
        ]
      ],
      "nearest_aircraft": "AC1",
      "nearest_aircraft_dist": 359.0,
      "nearest_obstacle": "<obstacle.Obstacle object at 0x000001A741BC7680>",
      "nearest_obstacle_dist": 255.16269319788896
    }
  ],
  "collision_pairs": []
}
```