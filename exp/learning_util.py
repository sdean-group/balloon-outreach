import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append("C:\\Users\\sc3377\\Documents\\balloon-outreach")

from env.balloon_env import BalloonERAEnvironment
from agent.mppi_agent import MPPIAgentWithCostFunction, MPPIAgent

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