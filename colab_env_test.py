## Should produce an identical sequence to `test_up_sequence_from_default` (in `test_action_sequence')


from env.balloon_env import BalloonEnvironment
import numpy as np
np.random.seed(0)  # Set random seed to 0 for reproducibility.
import time
import matplotlib.pyplot as plt

# Create environment and agent
env = BalloonEnvironment()

# Try different actions
# Reset environment
state = env.reset()
done = False
total_reward = 0

# Run simulation with fixed action
action = np.array([1.0])  # Try different values between -1 and 1

t = 0
max_timestep = 5

while not done and t < max_timestep:
    # Take step in environment
    state, reward, done, info = env.step(action)
    total_reward += reward
    
    # Render the environment
    env.render()
    
    # Print current state
    print(f"\nStep completed:")
    print(f"Current state: {state}")
    print(f"Action: {action[0]:.2f}")
    print(f"Reward: {reward:.2f}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    # Add a small delay
    time.sleep(0.1)

    # Increment timestep
    t += 1

print(f"\nSimulation completed with total reward: {total_reward:.2f}") 