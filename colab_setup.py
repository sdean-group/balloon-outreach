# Clone the repository
!git clone https://github.com/sdean-group/balloon-outreach.git
%cd balloon-outreach
!git checkout v0

# Install required packages
!pip install numpy matplotlib scipy
# Import necessary modules
import sys
sys.path.append('.')

from env.balloon_env import BalloonEnvironment
from agent.random_agent import RandomAgent
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

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

# Try different actions
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