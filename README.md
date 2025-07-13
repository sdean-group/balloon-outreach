# Balloon Outreach Simulation

This repository contains a balloon simulation environment for testing navigation strategies.

## Running in Google Colab

To run the demo notebooks in Google Colab, follow these steps:

1. Open [Google Colab](https://colab.research.google.com)
2. Under **Open notebook** click on **Github >**
3. Locate the relevant demo notebook on the `main` branch

To run the simulation in Google Colab, follow these steps:

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Copy and paste the following code 

```python
# Clone the repository
!git clone https://github.com/sdean-group/balloon-outreach.git
%cd balloon-outreach
!git checkout main

# Install required packages
!pip install numpy matplotlib
```

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
