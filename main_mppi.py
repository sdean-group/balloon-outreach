import numpy as np
from env.balloon_env import BalloonERAEnvironment
from agent.mppi_agent import MPPIAgentWithCostFunction, MPPIAgent
import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt   
import time
from env.util import haversine_distance

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
    avg_step_time = 0
    total_time = 0
    for step in range(max_steps):
        start = time.time()
        # Get action from agent
        action = agent.select_action(state, env, step)
        end = time.time()
        # Take step
        avg_opt_time += end-start
        total_time += end-start
        start = time.time()
        state, reward, done, info = env.step(action)
        end = time.time()
        avg_step_time += end-start
        total_reward += reward
        
        actions.append(float(action[0]) if isinstance(action, np.ndarray) else float(action))
        velocities.append(env.balloon.vertical_velocity)
        helium_mass.append(env.balloon.helium_mass)
        sands.append(env.balloon.sand)

        # Store position and altitude
        trajectory.append((state[0], state[1]))
        altitudes.append(state[2])
        print(f"Step {step}: lat: {state[0]:.2f}, lon: {state[1]:.2f}, alt: {state[2]:.2f}")
        # print(f"Average time to get one action: {avg_opt_time/(step+1)}")
        # print(f"Average time to take one step: {avg_step_time/(step+1)}")
        # env.render()
        
        if done:
            print(f"\nEpisode terminated: {info}")
            avg_opt_time /= step+1
            break  
    avg_opt_time /= max_steps
    print(f"Average time to get one action: {avg_opt_time}") 
    # --- Combined 2x2 Summary Plot ---
    initial_pos = [round(trajectory[0][0],1), round(trajectory[0][1],1)]
    target_pos = [round(env.target_lat,1), round(env.target_lon,1)]
    end_pos = [round(trajectory[-1][0],1), round(trajectory[-1][1],1)]
    distance = haversine_distance(target_pos[0], target_pos[1], end_pos[0], end_pos[1])
    print(f"Initial position: {initial_pos}, Target position: {target_pos}, End position: {end_pos}, Distance: {distance} km")
    print(f"num_samples: {agent.num_samples}, acc_bounds: {agent.acc_bounds}, noise_std: {agent.noise_std}, num_iterations: {agent.num_iterations}, horizon: {agent.horizon}")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1,1) Balloon Trajectory (Position)
    ax = axes[0, 0]
    lats, lons = zip(*trajectory)
    ax.plot(lons, lats, 'b-', alpha=0.5)
    ax.plot(lons[0], lats[0], 'go', label='Start')
    ax.plot(lons[-1], lats[-1], 'ro', label='End')
    if hasattr(agent, 'objective') and agent.objective == 'target':
        ax.plot(env.target_lon, env.target_lat, 'rx', label='Target End')
        ax.text(env.target_lon, env.target_lat+0.1, f"({env.target_lat:.1f}, {env.target_lon:.1f})", color='red', fontsize=10, ha='center', va='bottom')
        ax.text(env.target_lon, env.target_lat-0.4, f"error: {distance:.2f} km", color='black', fontsize=10, ha='center', va='bottom')
        
    # Annotate start
    ax.text(lons[0], lats[0]+0.1, f"({lats[0]:.1f}, {lons[0]:.1f})", color='green', fontsize=10, ha='center', va='bottom')
    # Annotate end
    ax.text(lons[-1], lats[-1]+0.1, f"({lats[-1]:.1f}, {lons[-1]:.1f})", color='red', fontsize=10, ha='center', va='bottom')

    ax.grid(True)
    ax.set_title(f'Balloon Trajectory in {max_steps} max steps')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(-83, -75)
    ax.set_ylim(42,80)
    ax.legend()

    # (1,2) Altitude Profile
    ax = axes[0, 1]
    ax.plot(altitudes, 'b-')
    if hasattr(agent, 'objective') and agent.objective == 'target':
        ax.axhline(y=env.target_alt, linewidth=1, color='r', label='Target End Altitude')
        ax.legend()
    ax.grid(True)
    ax.set_title(f'Altitude Profile using {env.dt} delta_time')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Altitude (km)')

    # (2,1) Target velocity (action) vs. Current velocity
    ax = axes[1, 0]
    ax.plot(actions, label='Target velocity (action)')
    ax.plot(velocities, label='Current vertical velocity')
    ax.set_xlabel('Step')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Target vs. Current Vertical Velocity')
    ax.legend()
    ax.grid(True)

    # (2,2) Resource Change (helium, sand)
    ax = axes[1, 1]
    ax.plot(helium_mass, label='Helium Mass')
    ax.plot(sands, label='Sand')
    ax.set_xlabel('Step')
    ax.set_ylabel('Resource')
    ax.set_title('Resource Change')
    ax.legend()
    ax.grid(True)

    # Add info text
    info_text = (
        f"Initial: [{initial_pos[0]}, {initial_pos[1]}]\n"
        f"Target: [{target_pos[0]}, {target_pos[1]}]\n"
        f"End: [{end_pos[0]}, {end_pos[1]}]\n"
        f"Distance: {distance:.2f} km\n"
        f"samples: {agent.num_samples}, acc_bounds: {agent.acc_bounds}\n"
        f"noise_std: {agent.noise_std}, iter: {agent.num_iterations}, horizon: {agent.horizon}\n"
        f"temperature: {agent.temperature}\n"
        f"total duration: {total_time:.2f}"
    )
    # Place the text in the upper right of the plot area
    ax.text(
        0.01, 0.5, info_text,
        transform=ax.transAxes,
        fontsize=10,
        va='center', ha='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    plt.tight_layout()
    plt.savefig(f'fig/test/modify_cost/balloon_summary_({target_pos[0]},{target_pos[1]})_({end_pos[0]},{end_pos[1]})_({agent.num_samples}s,{agent.num_iterations}it,{agent.horizon}h,{agent.temperature}t)_err_{distance:.2f}km.png')
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

##############
    # target_lat = 64.6
    # target_lon = -77.7

    # target_lat = 64.6
    # target_lon = -79.1

    # target_lat = 76
    # target_lon = -81.5

    target_lat = 77.3
    target_lon = -80.5

    target_alt = 12.0
    time_step = 120 #120 seconds
    # max_steps = 30 
    max_steps = int(1440/(time_step/60)) #1 day
    
    noise_std = 0.05
    acc_bounds= (-0.1, 0.1)
    objective = 'target'
    # For target
    horizon=10
    # For fly
    # horizon = 5
    num_samples=10
    num_iterations=1
    env = BalloonERAEnvironment(ds=ds, start_time=start_time, initial_lat=initial_lat, initial_lon=initial_lon, initial_alt=initial_alt, target_lat=target_lat, target_lon=target_lon,target_alt=target_alt, objective=objective, dt=time_step, viz=False)
    agent = MPPIAgentWithCostFunction(target_lat=target_lat, target_lon=target_lon, target_alt=target_alt, num_samples=num_samples, acc_bounds= acc_bounds,  noise_std=noise_std, num_iterations=num_iterations, horizon=horizon,visualize=False, objective=objective)
    # Run one episode



    start = time.time()
    reward = run_episode(env, agent, max_steps=max_steps)
    print(f"Episode finished with total reward: {reward:.2f}")
    end=time.time()
    print(f"time took:{end-start}")


if __name__ == "__main__":
    main()









    # 