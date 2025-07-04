import numpy as np
from env.balloon_env import BalloonERAEnvironment
from agent.mppi_agent import MPPIAgentWithCostFunction, MPPIAgent
import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt   
import time

def run_episode(env: BalloonERAEnvironment, agent:MPPIAgent, max_steps: int = 100) -> float:
    """Run one episode with the given agent"""
    state = env.reset()
    total_reward = 0
    
    # Store trajectory for plotting
    trajectory = [(state[0], state[1])]  # (lat, lon) pairs
    altitudes = [state[2]]  # Store altitudes

    actions = []
    velocities = []
    helium_mass = []
    sands = []
    avg_opt_time = 0

    for step in range(max_steps):
        start = time.time()
        # Get action from agent
        action = agent.select_action(state, env)
        end = time.time()
        # Take step
        print(f"action: {action}, vertical_velocity: {env.balloon.vertical_velocity}")
        state, reward, done, info = env.step(action)
        total_reward += reward
        avg_opt_time += end-start
        
        actions.append(float(action[0]) if isinstance(action, np.ndarray) else float(action))
        velocities.append(env.balloon.vertical_velocity)
        helium_mass.append(env.balloon.helium_mass)
        sands.append(env.balloon.sand)

        # Store position and altitude
        trajectory.append((state[0], state[1]))
        altitudes.append(state[2])
        print(f"Step {step}: lat: {state[0]:.2f}, lon: {state[1]:.2f}, alt: {state[2]:.2f}")
        
        # env.render()
        
        if done:
            print(f"\nEpisode terminated: {info}")
            avg_opt_time /= step+1
            break  
    avg_opt_time /= max_steps
    print(f"Average time to get one action: {avg_opt_time}") 
    # Plot final trajectory
    plt.figure(figsize=(12, 5))
    
    # Position plot
    plt.subplot(1, 2, 1)
    lats, lons = zip(*trajectory)
    plt.plot(lons, lats, 'b-', alpha=0.5)
    plt.plot(lons[0], lats[0], 'go', label='Start')
    plt.plot(lons[-1], lats[-1], 'ro', label='End')
    if agent.objective == 'target':
        plt.plot(env.target_lon, env.target_lat, 'rx', label='Target End')
    plt.grid(True)
    plt.title(f'Balloon Trajectory in {max_steps} max steps')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    
    # Altitude plot
    plt.subplot(1, 2, 2)
    plt.plot(altitudes, 'b-')
    if agent.objective == 'target':
        plt.axhline(y=env.target_alt,linewidth=1, color='r', label='Target End Altitude')
    plt.grid(True)
    plt.title(f'Altitude Profile using {env.dt} delta_time')
    plt.xlabel('Time Step')
    plt.ylabel('Altitude (km)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('balloon_trajectory_and_altitude_test.png')
    plt.close()

    # 추가: Target velocity (action) vs. Current velocity, Resource 변화
    plt.figure(figsize=(12, 4))

    # (1) Target velocity (action) vs. Current velocity
    plt.subplot(1, 2, 1)
    plt.plot(actions, label='Target velocity (action)')
    plt.plot(velocities, label='Current vertical velocity')
    plt.xlabel('Step')
    plt.ylabel('Velocity (m/s)')
    plt.title('Target vs. Current Vertical Velocity')
    plt.legend()
    plt.grid(True)

    # (2) Resource 변화 (volume, sand)
    plt.subplot(1, 2, 2)
    plt.plot(helium_mass, label='Helium Mass')
    plt.plot(sands, label='Sand')
    plt.xlabel('Step')
    plt.ylabel('Resource')
    plt.title('Resource Change')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('balloon_velocity_and_resource_test.png')
    plt.close()

    return total_reward

def main():
    # Create environment and agent
    ds = xr.open_dataset("era5_data.nc", engine="netcdf4")
    # 2. pick a reference start_time (should match your dataset's first valid_time)
    start_time = dt.datetime(2024, 7, 1, 0, 0)

    #This is Ithaca
    initial_lat = 42.6
    initial_lon = -76.5
    initial_alt = 10.0
    target_lat = 70
    target_lon = -90
    target_alt = 12.0
    max_steps = 1440 #1 day for 60 minutes
    time_step = 120 #120 minutes
    noise_std = 1
    env = BalloonERAEnvironment(ds=ds, start_time=start_time, initial_lat=initial_lat, initial_lon=initial_lon, initial_alt=initial_alt, target_lat=target_lat, target_lon=target_lon,target_alt=target_alt, dt=time_step)
    agent = MPPIAgentWithCostFunction(target_lat=target_lat, target_lon=target_lon, target_alt=target_alt, num_samples=10, noise_std=noise_std, num_iterations=1, horizon=1, objective='target')
    
    # Run one episode
    start = time.time()
    reward = run_episode(env, agent, max_steps=max_steps)
    print(f"Episode finished with total reward: {reward:.2f}")
    end=time.time()
    print(f"time took:{end-start}")


if __name__ == "__main__":
    main()