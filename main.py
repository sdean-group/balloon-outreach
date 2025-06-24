import numpy as np
from env.balloon_env import BalloonEnvironment
from agent.random_agent import RandomAgent
from agent.goal_agent import GoalDirectedAgent
import matplotlib.pyplot as plt

def run_episode(env: BalloonEnvironment, agent: RandomAgent, max_steps: int = 100) -> float:
    """Run one episode with the given agent"""
    state = env.reset()
    total_reward = 0
    
    # Store trajectory for plotting
    trajectory = [(state[0], state[1])]  # (lat, lon) pairs
    altitudes = [state[2]]  # Store altitudes
    
    for step in range(max_steps):
        # Get action from agent
        action = agent.select_action(state)
        # print(action)
        # Take step
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Store position and altitude
        trajectory.append((state[0], state[1]))
        altitudes.append(state[2])
        print(f"Step {step}: lat: {state[0]:.2f}, lon: {state[1]:.2f}, alt: {state[2]:.2f}")
        
        # Render every 2 hours
        # if step % 2 == 0:
        env.render()
        
        if done:
            print(f"\nEpisode terminated: {info}")
            break
    
    # Plot final trajectory
    plt.figure(figsize=(12, 5))
    
    # Position plot
    plt.subplot(1, 2, 1)
    lats, lons = zip(*trajectory)
    plt.plot(lons, lats, 'b-', alpha=0.5)
    plt.plot(lons[0], lats[0], 'go', label='Start')
    plt.plot(lons[-1], lats[-1], 'ro', label='End')
    plt.grid(True)
    plt.title('Balloon Trajectory')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    
    # Altitude plot
    plt.subplot(1, 2, 2)
    plt.plot(altitudes, 'b-')
    plt.grid(True)
    plt.title('Altitude Profile')
    plt.xlabel('Time Step')
    plt.ylabel('Altitude (km)')
    
    plt.tight_layout()
    plt.show()
    
    return total_reward

def main():
    # Create environment and agent
    env = BalloonEnvironment()
    agent = RandomAgent()
    # agent = GoalDirectedAgent(target_lat=env.target_lat, target_lon=env.target_lon, target_alt=env.target_alt)
    # Run one episode
    reward = run_episode(env, agent)
    print(f"Episode finished with total reward: {reward:.2f}")

if __name__ == "__main__":
    main()