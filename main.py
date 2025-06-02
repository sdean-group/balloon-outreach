import numpy as np
from env.balloon_env import BalloonEnvironment
from agent.random_agent import RandomAgent
import matplotlib.pyplot as plt
from agent.goal_agent import GoalDirectedAgent


def run_episode(env: BalloonEnvironment, agent: GoalDirectedAgent, max_steps: int = 50, target_lat: float = None, target_lon: float = None) -> float:
    state = env.reset()

    # If not passed in, define target from initial state
    if target_lat is None:
        target_lat = state[0] + 3.0
    if target_lon is None:
        target_lon = state[1] + 5.0
    target_alt = state[2]

    total_reward = 0
    trajectory = [(state[0], state[1])]
    altitudes = [state[2]]

    for step in range(max_steps):
        print(f"[State shape] {state.shape}")
        action = agent.select_action(state)

        state, reward, done, info = env.step(action)
        # reward = np.clip(reward, -500, 100)
        total_reward += reward

        trajectory.append((state[0], state[1]))
        altitudes.append(state[2])
        print(f"Step {step}: lat={state[0]:.2f}, lon={state[1]:.2f}, alt={state[2]:.2f}")

        if step % 2 == 0:
            env.render()

        if done:
            print(f"\nEpisode terminated: {info}")
            break

    # Turn off interactive mode before final plotting
    plt.ioff()

    # ==== Plotting ====
    plt.figure(figsize=(12, 5))

# Trajectory plot
    plt.subplot(1, 2, 1)
    lats, lons = zip(*trajectory)

    plt.plot(lons, lats, 'b-', alpha=0.6, label='Trajectory')
    plt.plot(lons[0], lats[0], 'go', label='Start')
    plt.plot(lons[-1], lats[-1], 'ro', label='End')

    # Target line
    if target_lat is not None and target_lon is not None:
        plt.plot(
            [lons[0], target_lon],
            [lats[0], target_lat],
            'k--',
            label='Start → Target',
            linewidth=2,
            zorder=5
        )
        plt.plot(target_lon, target_lat, 'kx', markersize=10, label='Target', zorder=6)

    # Safe axis adjustment
    all_lats = list(lats) + [target_lat]
    all_lons = list(lons) + [target_lon]

    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_min, lat_max = min(all_lats), max(all_lats)

    if abs(lon_max - lon_min) < 0.01:
        lon_min -= 0.005
        lon_max += 0.005
    else:
        lon_margin = 0.1 * (lon_max - lon_min)
        lon_min -= lon_margin
        lon_max += lon_margin

    if abs(lat_max - lat_min) < 0.01:
        lat_min -= 0.005
        lat_max += 0.005
    else:
        lat_margin = 0.1 * (lat_max - lat_min)
        lat_min -= lat_margin
        lat_max += lat_margin

    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.title("Balloon Trajectory")
    plt.legend()

    # Altitude plot
    plt.subplot(1, 2, 2)
    plt.plot(altitudes, 'b-')
    plt.grid(True)
    plt.title('Altitude Profile')
    plt.xlabel('Time Step')
    plt.ylabel('Altitude (m)')

    plt.tight_layout()
    plt.show()

    return total_reward


def main():
    # Create environment and agent
    env = BalloonEnvironment()
    # agent = RandomAgent()
    agent = GoalDirectedAgent(
        env.target_lat ,     # example
        env.target_lon,
        env.target_altitude     # in meters
    )
   
    reward = run_episode(env, agent,target_lat=env.target_lat,
                         target_lon=env.target_lon)
    print(f"Episode finished with total reward: {reward:.2f}")

    
    # # Run one episode
    # reward = run_episode(env, agent)
    # print(f"Episode finished with total reward: {reward:.2f}")

if __name__ == "__main__":
    main()