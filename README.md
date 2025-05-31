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
!pip install numpy matplotlib
```


### Cell 2: Import modules
```python
import sys
sys.path.append('.')

from env.balloon_env import BalloonEnvironment
from agent.random_agent import RandomAgent
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

### Cell 3: Run simulation with random agent
```python
# Create environment and agent
env = BalloonEnvironment()
agent = RandomAgent()

# Reset environment
state = env.reset()
done = False
total_reward = 0

# Run simulation
while not done:
    # Select action
    action = agent.select_action(state)
    
    # Take step in environment
    state, reward, done, info = env.step(action)
    total_reward += reward
    
    # Render the environment
    env.render()
    
    # Print current state
    print(f"\nStep completed:")
    print(f"Action: {action[0]:.2f}")
    print(f"Reward: {reward:.2f}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    # Add a small delay to make the visualization smoother
    import time
    time.sleep(0.1)

print(f"\nSimulation completed with total reward: {total_reward:.2f}")
```

### Cell 4: Try different actions
```python
# Reset environment
state = env.reset()
done = False
total_reward = 0

# Run simulation with fixed action
action = np.array([-0.5])  # Try different values between -1 and 1

while not done:
    # Take step in environment
    state, reward, done, info = env.step(action)
    total_reward += reward
    
    # Render the environment
    env.render()
    
    # Print current state
    print(f"\nStep completed:")
    print(f"Action: {action[0]:.2f}")
    print(f"Reward: {reward:.2f}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    # Add a small delay
    time.sleep(0.1)

print(f"\nSimulation completed with total reward: {total_reward:.2f}")
```

4. Run each cell in sequence

## Action Space

The simulation uses a single continuous action in the range [-1, 1]:
- Negative values: Drop sand (magnitude determines amount)
- Positive values: Vent gas (magnitude determines rate)

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