# Balloon Outreach Simulation

This repository contains a balloon simulation environment for testing navigation strategies.

## Running in Google Colab

To run the simulation in Google Colab, follow these steps:

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Copy and paste the following code into separate cells:

### Cell 1-1 (for public repository): Clone repository and install dependencies 
```python
# Clone the repository
!git clone https://github.com/sdean-group/balloon-outreach.git
%cd balloon-outreach
!git checkout v0

# Install required packages
!pip install numpy matplotlib
```
### Cell 1-2 (for private repository): Clone repository and install dependencies 
```python
# Clone the repository
from getpass import getpass
username = "your-github-username"  
token = getpass("Enter your GitHub Personal Access Token: ")

repo_url = f"https://{username}:{token}@github.com/sdean-group/balloon-outreach.git"

!git clone $repo_url
%cd balloon-outreach
!git checkout v0

# Install required packages
!pip install numpy matplotlib scipy
```


### Cell 2: Import modules
```python
import sys
sys.path.append('.')

from env.balloon_env import BalloonEnvironment
from agent.random_agent import RandomAgent
import numpy as np
import time
import matplotlib.pyplot as plt
%matplotlib inline
```

### Cell 3: Run simulation with random agent
```python

# Run simulation
# Create environment and agent
env = BalloonEnvironment()
# agent = RandomAgent()
agent = GoalDirectedAgent(target_lat=env.target_lat, target_lon=env.target_lon, target_alt=env.target_alt)
# Reset environment
state = env.reset()
done = False
total_reward = 0
max_episode_steps = 100
# Run simulation
for _ in range(max_episode_steps):
    # Select action
    action = agent.select_action(state)

    # Take step in environment
    state, reward, done, info = env.step(action)
    total_reward += reward

    # Render the environment
    env.render()
    if done:
      break

    # Add a small delay to make the visualization smoother
    time.sleep(0.1)

print(f"\nSimulation completed with total reward: {total_reward:.2f}")
```


4. Run each cell in sequence

## Action Space

The simulation uses a single continuous action in the range [-1, 1]:
- Negative values: Drop sand (magnitude determines amount)-> UP
- Positive values: Vent gas (magnitude determines rate) -> DOWN

## Visualization

The simulation shows three plots:
1. 3D trajectory of the balloon
2. Resource levels (volume and sand)
3. Wind speed profile

## Environment Details

- The balloon starts at (0, 0) at 10km altitude
- The target is at (5, 5) at 10km altitude
- The simulation runs for up to 24 hours
- The balloon has limited helium and sand resources 