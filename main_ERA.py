## Dependencies: jax, opensimplex

import xarray as xr
import numpy as np
import datetime as dt
from env.balloon_env import BalloonERAEnvironment
from agent.random_agent import RandomAgent
from agent.goal_agent import GoalDirectedAgent
import matplotlib.pyplot as plt
from env.visualize import plot_wind_field, plot_trajectory_earth
from pathlib import Path
import sys
import importlib.resources as pkg_resources

def run_episode(env: BalloonERAEnvironment, agent: RandomAgent, max_steps: int = 100) -> float:
    """Run one episode with the given agent"""
    state = env.reset()
    total_reward = 0
    
    # Store trajectory for plotting
    trajectory = [(state[0], state[1])]  # (lat, lon) pairs
    altitudes = [state[2]]  # Store altitudes

    # 추가: 기록용 리스트
    actions = []
    velocities = []
    helium_mass = []
    sands = []

    for step in range(max_steps):
        # Get action from agent
        action = agent.select_action(state, max_steps, step)
        # Take step
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        actions.append(float(action[0]) if isinstance(action, np.ndarray) else float(action))
        velocities.append(env.balloon.vertical_velocity)
        helium_mass.append(env.balloon.helium_mass)
        sands.append(env.balloon.sand)

        # Store position and altitude
        trajectory.append((state[0], state[1]))
        altitudes.append(state[2])
        print(f"Step {step}: lat: {state[0]:.2f}, lon: {state[1]:.2f}, alt: {state[2]:.2f}")
        
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
    plt.savefig('balloon_trajectory_and_altitude.png')
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
    plt.savefig('balloon_velocity_and_resource.png')
    plt.close()

    # # Wind field visualization
    # pressure_levels = [1000, 500, 200]  # hPa
    # times = [0, 12]  # hours
    
    # for p in pressure_levels:
    #     for t in times:
    #         plot_wind_field(env.wind_field, p, t)
    # plt.savefig(f'wind_field.png')
    # plt.close()

    # 3D visualization – robust texture path resolution
    try:
        texture_path = pkg_resources.files("env").joinpath("figs/2k_earth_daymap.jpg")
    except Exception:
        texture_path = Path(__file__).resolve().parent / "env" / "figs" / "2k_earth_daymap.jpg"
    plot_trajectory_earth(lats, lons, altitudes, texture_path=str(texture_path), lon_offset_deg=210, flip_lat=True)
    return total_reward

def main():
    # 1. load your ERA5 file
    ds = xr.open_dataset("era5_data.nc", engine="netcdf4")
    # 2. pick a reference start_time (should match your dataset's first valid_time)
    start_time = dt.datetime(2024, 7, 1, 0, 0)
    # Create environment and agent
    env = BalloonERAEnvironment(ds=ds, start_time=start_time)
    agent = RandomAgent()
    # agent = GoalDirectedAgent(target_lat=env.target_lat, target_lon=env.target_lon, target_alt=env.target_alt)
    # Run one episode
    reward = run_episode(env, agent)
    print(f"Episode finished with total reward: {reward:.2f}")

if __name__ == "__main__":
    main()