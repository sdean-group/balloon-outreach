import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append("C:\\Users\\sc3377\\Documents\\balloon-outreach")

from env.util import haversine_distance
from env.balloon_env import BalloonERAEnvironment
from agent.mppi_agent import MPPIAgentWithCostFunction, MPPIAgent

def plot_expert_summary(
        agent: MPPIAgentWithCostFunction,
        trajectory: list,
        altitudes: list,
        actions: list,
        velocities: list,
        helium_mass: list,
        sands: list,
        policy_name: str,
        max_steps: int,
        dt: float
        ):
    # --- Combined 2x2 Summary Plot ---
    initial_pos = [round(trajectory[0][0],1), round(trajectory[0][1],1)]
    target_pos = [round(agent.target_lat,1), round(agent.target_lon,1)]
    end_pos = [round(trajectory[-1][0],1), round(trajectory[-1][1],1)]
    # distance = haversine_distance(initial_pos[1], initial_pos[0], end_pos[1], end_pos[0])
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
        ax.plot(agent.target_lon, agent.target_lat, 'rx', label='Target End')
        ax.text(agent.target_lon, agent.target_lat+0.1, f"({agent.target_lat:.1f}, {agent.target_lon:.1f})", color='red', fontsize=10, ha='center', va='bottom')
        ax.text(agent.target_lon, agent.target_lat-0.4, f"error: {distance:.2f} km", color='black', fontsize=10, ha='center', va='bottom')
        
    # Annotate start
    ax.text(lons[0], lats[0]+0.1, f"({lats[0]:.1f}, {lons[0]:.1f})", color='green', fontsize=10, ha='center', va='bottom')
    # Annotate end
    ax.text(lons[-1], lats[-1]+0.1, f"({lats[-1]:.1f}, {lons[-1]:.1f})", color='red', fontsize=10, ha='center', va='bottom')

    ax.grid(True)
    ax.set_title(f'Balloon Trajectory in {max_steps} max steps')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # ax.set_xlim(-83, -75)
    # ax.set_ylim(42,80)
    ax.legend()

    # (1,2) Altitude Profile
    ax = axes[0, 1]
    ax.plot(altitudes, 'b-')
    if hasattr(agent, 'objective') and agent.objective == 'target':
        ax.axhline(y=agent.target_alt, linewidth=1, color='r', label='Target End Altitude')
        ax.legend()
    ax.grid(True)
    ax.set_title(f'Altitude Profile using {dt} delta_time')
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
        # f"total duration: {total_time:.2f}"
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

    fig_dir = 'figs/'

    if hasattr(agent, 'objective') and agent.objective == 'target':
        if not os.path.isdir(fig_dir + 'modify_cost'):
            os.makedirs(fig_dir+'modify_cost')
        plt.savefig(os.path.join(fig_dir+'modify_cost', f'balloon_summary_{policy_name}_({target_pos[0]},{target_pos[1]})_({end_pos[0]},{end_pos[1]})_({agent.num_samples}s,{agent.num_iterations}it,{agent.horizon}h,{agent.temperature}t)_err_{distance:.2f}km.png'))

    else:
        if not os.path.isdir(fig_dir + 'fly'):
            os.makedirs(fig_dir+'fly')
        plt.savefig(os.path.join(fig_dir+'fly', f'balloon_summary_{policy_name}_(({end_pos[0]},{end_pos[1]})_({agent.num_samples}s,{agent.num_iterations}it,{agent.horizon}h,{agent.temperature}t)_err_{distance:.2f}km.png'))

    plt.close()

def plot_agent_trajectory(
        trajectory: list,
        altitudes: list,
        actions: list,
        velocities: list,
        helium_mass: list,
        sands: list,
        policy_name: str,
        max_steps: int,
        dt: float, 
        objective: str = 'target',
        target_lon: float = None,
        target_lat: float = None,
        target_alt: float = None
        ):
    
    # Plot final trajectory
    plt.figure(figsize=(12, 5))
    
    # Position plot
    plt.subplot(1, 2, 1)
    lats, lons = zip(*trajectory)
    plt.plot(lons, lats, 'b-', alpha=0.5)
    plt.plot(lons[0], lats[0], 'go', label='Start')
    plt.plot(lons[-1], lats[-1], 'ro', label='End')
    if objective == 'target':
        plt.plot(target_lon, target_lat, 'rx', label='Target End')
    plt.grid(True)
    plt.title(f'Balloon Trajectory in {max_steps} max steps')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    
    # Altitude plot
    plt.subplot(1, 2, 2)
    plt.plot(altitudes, 'b-')
    if objective == 'target':
        plt.axhline(y=target_alt, linewidth=1, color='r', label='Target End Altitude')
    plt.grid(True)
    plt.title(f'Altitude Profile using {dt} delta_time')
    plt.xlabel('Time Step')
    plt.ylabel('Altitude (km)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'balloon_trajectory_and_altitude_{policy_name}.png')
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
    plt.savefig(f'balloon_velocity_and_resource_test_{policy_name}.png')
    plt.close()

# Run an episode from an expert and collect its behavior

def run_expert_episode(
        env: BalloonERAEnvironment, 
        agent:MPPIAgent, 
        max_steps: int = 100,
        policy_name: str='expert'):
    """
    Run one episode with the given agent,
    collect state-action pair of the agent,
    and plot its trajectory in the given environment.
    """
    # Save state-action pairs from expert policy
    initial_states = []
    initial_actions = []

    state = env.reset()
    total_reward = 0
    
    # Store trajectory for plotting
    trajectory = [(state[0], state[1])]  # (lat, lon) pairs
    altitudes = [state[2]]  # Store altitudes

    actions = []
    velocities = []
    helium_mass = []
    sands = []
    for step in range(max_steps):
        # Get action from agent
        action = agent.select_action(state, env, step)
        
        # record state and expert action
        initial_states.append(state)
        initial_actions.append(action)
        
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
        # env.render()
        
        if done:
            print(f"\nEpisode terminated: {info}")
            break   

    # Convert to arrays
    states_np = np.array(initial_states, dtype=np.float32)
    actions_np = np.array(initial_actions, dtype=np.float32)
    print(f"Collected {len(initial_states)} state-action pairs from expert.")

    # Plot final trajectory
    plot_agent_trajectory(trajectory=trajectory,
                          altitudes=altitudes,
                          actions=actions,
                          velocities=velocities,
                          helium_mass=helium_mass,
                          sands=sands,
                          policy_name=policy_name,
                          max_steps=max_steps,
                          dt=env.dt,
                          objective=agent.objective,
                          target_lon=env.target_lon,
                          target_lat=env.target_lat,
                          target_alt=env.target_alt
                          )

    return total_reward, states_np, actions_np

# Evaluate a policy on the environment
def evaluate_policy(env: BalloonERAEnvironment, 
                    policy: nn.Module,
                    max_steps: int,
                    policy_name: str,
                    expert_avg_total_reward: float):
    policy.eval()
    if policy.training:
        print("→ policy is in training mode")
    else:
        print("→ policy is in evaluation mode")

    state = env.reset()
    total_reward = 0.0

    # Store trajectory for plotting
    trajectory = [(state[0], state[1])]  # (lat, lon) pairs
    altitudes = [state[2]]  # Store altitudes

    actions = []
    velocities = []
    helium_mass = []
    sands = []

    for step in range(max_steps):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)      # shape (1, 21)
        action_pred = policy(state_tensor).item()                        # continuous action
        state, reward, done, info = env.step(action_pred)
        total_reward += reward

        actions.append(float(action_pred) if isinstance(action_pred, np.ndarray) else float(action_pred))
        velocities.append(env.balloon.vertical_velocity)
        helium_mass.append(env.balloon.helium_mass)
        sands.append(env.balloon.sand)

        # Store position and altitude
        trajectory.append((state[0], state[1]))
        altitudes.append(state[2])

        if done:
            print(f"\nEpisode terminated: {info}")
            break
    print(f"Total reward obtained from current policy: {total_reward:.2f}")
    print(f"Expert policy reward - Current policy reward: {expert_avg_total_reward-total_reward:.2f}")

    # Plot final trajectory
    plot_agent_trajectory(trajectory=trajectory,
                          altitudes=altitudes,
                          actions=actions,
                          velocities=velocities,
                          helium_mass=helium_mass,
                          sands=sands,
                          policy_name=policy_name,
                          max_steps=max_steps,
                          dt=env.dt,
                          objective='target',
                          target_lon=env.target_lon,
                          target_lat=env.target_lat,
                          target_alt=env.target_alt
                          )

    return total_reward
