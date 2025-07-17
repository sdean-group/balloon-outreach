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
    if hasattr(agent, 'objective') and agent.objective == 'target':
        target_pos = [round(agent.target_lat,1), round(agent.target_lon,1)]
    end_pos = [round(trajectory[-1][0],1), round(trajectory[-1][1],1)]

    if hasattr(agent, 'objective') and agent.objective == 'target':
        distance = haversine_distance(target_pos[0], target_pos[1], end_pos[0], end_pos[1])
        print(f"Initial position: {initial_pos}, Target position: {target_pos}, End position: {end_pos}, Distance: {distance} km")
    else:
        distance = haversine_distance(initial_pos[1], initial_pos[0], end_pos[1], end_pos[0])
        print(f"Initial position: {initial_pos}, End position: {end_pos}, Distance: {distance} km")
    
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
    if hasattr(agent, 'objective') and agent.objective == 'target':
        info_text = (
            f"Initial: [{initial_pos[0]}, {initial_pos[1]}]\n"
            f"Target: [{target_pos[0]}, {target_pos[1]}]\n"
            f"End: [{end_pos[0]}, {end_pos[1]}]\n"
            f"Distance: {distance:.2f} km\n"
        )
    else:
        info_text = (
            f"Initial: [{initial_pos[0]}, {initial_pos[1]}]\n"
            f"End: [{end_pos[0]}, {end_pos[1]}]\n"
            f"Distance: {distance:.2f} km\n"
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
        if not os.path.isdir(fig_dir + 'target'):
            os.makedirs(fig_dir+'target')
        plot_fnm = os.path.join(fig_dir+'target', f'balloon_summary_{policy_name}_({target_pos[0]},{target_pos[1]})_({end_pos[0]},{end_pos[1]})_({agent.num_samples}s,{agent.num_iterations}it,{agent.horizon}h,{agent.temperature}t)_err_{distance:.2f}km.png')
        plt.savefig(plot_fnm)

    else:
        if not os.path.isdir(fig_dir + 'fly'):
            os.makedirs(fig_dir+'fly')
        plot_fnm = os.path.join(fig_dir+'fly', f'balloon_summary_{policy_name}_(({end_pos[0]},{end_pos[1]})_({agent.num_samples}s,{agent.num_iterations}it,{agent.horizon}h,{agent.temperature}t)_err_{distance:.2f}km.png')
        plt.savefig(plot_fnm)

    plt.close()

    return plot_fnm

def plot_agent_summary(
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
    
    # --- Combined 2x2 Summary Plot ---
    initial_pos = [round(trajectory[0][0],1), round(trajectory[0][1],1)]
    if objective == 'target':
        target_pos = [round(target_lat,1), round(target_lon,1)]
    end_pos = [round(trajectory[-1][0],1), round(trajectory[-1][1],1)]

    if objective == 'target':
        distance = haversine_distance(target_pos[0], target_pos[1], end_pos[0], end_pos[1])
        print(f"Initial position: {initial_pos}, Target position: {target_pos}, End position: {end_pos}, Distance: {distance} km")
    else:
        distance = haversine_distance(initial_pos[1], initial_pos[0], end_pos[1], end_pos[0])
        print(f"Initial position: {initial_pos}, End position: {end_pos}, Distance: {distance} km")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Policy: {policy_name}", fontsize=16)

    # (1,1) Balloon Trajectory (Position)
    ax = axes[0, 0]
    lats, lons = zip(*trajectory)
    ax.plot(lons, lats, 'b-', alpha=0.5)
    ax.plot(lons[0], lats[0], 'go', label='Start')
    ax.plot(lons[-1], lats[-1], 'ro', label='End')

    if objective == 'target':
        ax.plot(target_lon, target_lat, 'rx', label='Target End')
        ax.text(target_lon, target_lat+0.1, f"({target_lat:.1f}, {target_lon:.1f})", color='red', fontsize=10, ha='center', va='bottom')
        ax.text(target_lon, target_lat-0.4, f"error: {distance:.2f} km", color='black', fontsize=10, ha='center', va='bottom')
        
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
    if objective == 'target':
        ax.axhline(y=target_alt, linewidth=1, color='r', label='Target End Altitude')
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
    if objective == 'target':
        info_text = (
            f"Initial: [{initial_pos[0]}, {initial_pos[1]}]\n"
            f"Target: [{target_pos[0]}, {target_pos[1]}]\n"
            f"End: [{end_pos[0]}, {end_pos[1]}]\n"
            f"Distance: {distance:.2f} km\n"
        )
    else:
        info_text = (
            f"Initial: [{initial_pos[0]}, {initial_pos[1]}]\n"
            f"End: [{end_pos[0]}, {end_pos[1]}]\n"
            f"Distance: {distance:.2f} km\n"
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

    # Save the figure
    if objective == 'target':
        save_dir = 'figs/target'
    else:
        save_dir = 'figs/fly'
    os.makedirs(save_dir, exist_ok=True)
    filename = f"balloon_summary_{policy_name}_({end_pos[0]},{end_pos[1]})_err_{distance:.2f}km.png"
    plot_fnm = os.path.join(save_dir, filename)
    plt.savefig(plot_fnm)
    plt.close()

    return plot_fnm

def plot_agent_summary(
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
    
    # --- Combined 2x2 Summary Plot ---
    initial_pos = [round(trajectory[0][0],1), round(trajectory[0][1],1)]
    if objective == 'target':
        target_pos = [round(target_lat,1), round(target_lon,1)]
    end_pos = [round(trajectory[-1][0],1), round(trajectory[-1][1],1)]

    if objective == 'target':
        distance = haversine_distance(target_pos[0], target_pos[1], end_pos[0], end_pos[1])
        print(f"Initial position: {initial_pos}, Target position: {target_pos}, End position: {end_pos}, Distance: {distance} km")
    else:
        distance = haversine_distance(initial_pos[1], initial_pos[0], end_pos[1], end_pos[0])
        print(f"Initial position: {initial_pos}, End position: {end_pos}, Distance: {distance} km")

    # Create figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Add a super title using policy_name
    fig.suptitle(f"Policy: {policy_name}", fontsize=16)

    # (1,1) Balloon Trajectory (Position)
    ax = axes[0, 0]
    lats, lons = zip(*trajectory)
    ax.plot(lons, lats, 'b-', alpha=0.5)
    ax.plot(lons[0], lats[0], 'go', label='Start')
    ax.plot(lons[-1], lats[-1], 'ro', label='End')

    if objective == 'target':
        ax.plot(target_lon, target_lat, 'rx', label='Target End')
        ax.text(target_lon, target_lat+0.1, f"({target_lat:.1f}, {target_lon:.1f})", color='red', fontsize=10, ha='center', va='bottom')
        ax.text(target_lon, target_lat-0.4, f"error: {distance:.2f} km", color='black', fontsize=10, ha='center', va='bottom')

    # Annotate start and end points
    ax.text(lons[0], lats[0]+0.1, f"({lats[0]:.1f}, {lons[0]:.1f})", color='green', fontsize=10, ha='center', va='bottom')
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
    if objective == 'target':
        ax.axhline(y=target_alt, linewidth=1, color='r', label='Target End Altitude')
        ax.legend()
    ax.grid(True)
    ax.set_title(f'Altitude Profile using {dt} delta_time')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Altitude (km)')

    # (2,1) Target vs Current Velocity
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

    # Add info text box
    if objective == 'target':
        info_text = (
            f"Initial: [{initial_pos[0]}, {initial_pos[1]}]\n"
            f"Target: [{target_pos[0]}, {target_pos[1]}]\n"
            f"End: [{end_pos[0]}, {end_pos[1]}]\n"
            f"Distance: {distance:.2f} km\n"
        )
    else:
        info_text = (
            f"Initial: [{initial_pos[0]}, {initial_pos[1]}]\n"
            f"End: [{end_pos[0]}, {end_pos[1]}]\n"
            f"Distance: {distance:.2f} km\n"
        )
    ax.text(
        0.01, 0.5, info_text,
        transform=ax.transAxes,
        fontsize=10,
        va='center', ha='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # Adjust layout to accommodate the super title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    if objective == 'target':
        save_dir = 'figs/target'
    else:
        save_dir = 'figs/fly'
    os.makedirs(save_dir, exist_ok=True)
    filename = f"balloon_summary_{policy_name}_({end_pos[0]},{end_pos[1]})_err_{distance:.2f}km.png"
    plot_fnm = os.path.join(save_dir, filename)
    plt.savefig(plot_fnm)
    plt.close()

    return plot_fnm


# Run an episode from an expert and collect its behavior
def run_expert_episode(
        env: BalloonERAEnvironment,
        agent:MPPIAgent,
        max_steps: int = 100,
        dt: int = 60,
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
        # print(f"Step {step}: lat: {state[0]:.2f}, lon: {state[1]:.2f}, alt: {state[2]:.2f}")

        if done:
            print(f"\nEpisode terminated: {info}")
            break

    # Convert to arrays
    states_np = np.array(initial_states, dtype=np.float32)
    actions_np = np.array(initial_actions, dtype=np.float32)
    print(f"Collected {len(initial_states)} state-action pairs from expert.")

    # --- Combined 2x2 Summary Plot ---
    plot_fnm = plot_expert_summary(agent, trajectory, altitudes, actions, velocities, helium_mass, sands, policy_name, max_steps, dt)

    return total_reward, states_np, actions_np, plot_fnm

# Evaluate a policy on the environment
def evaluate_policy(env: BalloonERAEnvironment,
                    policy: nn.Module,
                    objective: str,
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

    # Plot summary (figs/)
    plot_fnm = plot_agent_summary(
        trajectory=trajectory,
        altitudes=altitudes,
        actions=actions,
        velocities=velocities,
        helium_mass=helium_mass,
        sands=sands,
        policy_name=policy_name,
        max_steps=max_steps,
        dt=env.dt,
        objective=objective,
        target_lon=env.target_lon,
        target_lat=env.target_lat,
        target_alt=env.target_alt
        )

    return total_reward, plot_fnm

# === Training Function Using Loader ===
def train_one_epoch(loader, policy, optimizer, loss_fn):
    policy.train()
    if policy.training:
        print("→ policy is in training mode")
    else:
        print("→ policy is in evaluation mode")

    total_loss = 0.0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = policy(x_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss