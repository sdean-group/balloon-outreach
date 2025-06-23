## Test script to see how lat/long/alt changes with different action sequences.
## Uses transition model embedded in TreeSearchAgent.

from agent.tree_search_agent import TreeSearchAgent
from env.balloon_env import BalloonEnvironment
import matplotlib.pyplot as plt
import numpy as np
# Set random seed to 0 for reproducibility.
np.random.seed(0)

def plot_state_sequence(state_sequence : np.ndarray, plot_prefix : str):
    """
    Plot the sequence of states over time (lat, lon, alt). Plot lat/long as
    x/y, and annotate points with alittudes.
    
    Args:
        state_sequence (list of np.ndarray): List of states to plot.
    """
    latitudes = [state[0] for state in state_sequence]
    longitudes = [state[1] for state in state_sequence]
    altitudes = [state[2] for state in state_sequence]
    fig, ax = plt.subplots()
    ax.scatter(latitudes, longitudes, c='blue', label='Explored States')
    # show initial state
    ax.scatter(latitudes[0], longitudes[0], c='green', label='Initial State', marker='o')
    # Write altitudes on the lat/long points.
    for i, (lat, lon, alt) in enumerate(zip(latitudes, longitudes, altitudes)):
        ax.text(lat, lon, f'{alt:.2f}', fontsize=8, ha='right', va='bottom', color='black')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_title('State Sequence.')
    ax.legend()
    plt.show()
    plt.savefig(f'{plot_prefix}.png')


def run_action_sequence(initial_state : np.ndarray, action_sequence : list, plot_prefix: str):
    """
    Run a sequence of actions starting from an initial state.
    
    Args:
        initial_state (np.ndarray): Initial state of the balloon [lat, lon, alt, time].
        action_sequence (list): List of actions to perform.
        agent (TreeSearchAgent): The agent to apply actions with.
    
    Returns:
        np.ndarray: Final state after applying the action sequence.
    """
    env = BalloonEnvironment()
    agent = TreeSearchAgent(balloon_env=env, distance='euclidean', heuristic='zero')
    # Set the balloon's initial state.
    env.balloon.lat, env.balloon.lon, env.balloon.alt = initial_state[:3]

    # Execute the action sequence
    current_state = initial_state
    current_balloon = env.balloon
    state_sequence = [current_state]
    print(f"Initial State [lat/long/alt/t]: {current_state}")
    for action in action_sequence:
        current_state, current_balloon = agent.apply_action(current_state, action, current_balloon)
        state_sequence.append(current_state)
        print(f"Action: {action}, Current State [lat/long/alt/t]: {current_state}")

    # Plot the state sequence
    plot_state_sequence(np.array(state_sequence), plot_prefix)


def test_action_sequence():
    # ## Test sequence of 5 'stay' actions, starting from some initial state.
    # initial_state = [500,-100,12,0]
    # action_sequence = ['stay'] * 5  # Repeat 'stay' action 5 times
    # print("Testing action sequence with initial state:", initial_state)
    # run_action_sequence(np.array(initial_state), action_sequence, plot_prefix="test_stay_sequence")

    # # Experiment. Add noise_val to the initial state. but otherwise same.
    # print("-------------------------")
    # initial_state = [500.02,-99.98,12.02,0]
    # action_sequence = ['stay'] * 5  # Repeat 'stay' action 5 times
    # print("Testing action sequence with initial state:", initial_state)
    # run_action_sequence(np.array(initial_state), action_sequence, plot_prefix="test_stay_sequence_with_noise")

    # # Experiment. fix lat/long, step altitude from 1 to 10.
    # print("-------------------------")
    # for altitude in range(1, 11):
    #     initial_state = [500, -100, altitude, 0]
    #     action_sequence = ['stay'] * 5  # Repeat 'stay' action 5 times
    #     print(f"Testing action sequence with initial state: {initial_state}")
    #     run_action_sequence(np.array(initial_state), action_sequence, plot_prefix=f"test_stay_sequence_altitude_{altitude}")

    # # Experiment. Test sequence of 5 'ascend' actions, starting from some initial state.
    # print("-------------------------")
    # initial_state = [500, -100, 12, 0]
    # action_sequence = ['ascend'] * 5  # Repeat 'up' action 5 times
    # print("Testing action sequence with initial state:", initial_state)
    # run_action_sequence(np.array(initial_state), action_sequence, plot_prefix="test_up_sequence")

    # # Experiment. Test sequence of 5 'down' actions, starting from some initial state.
    # print("-------------------------")
    # initial_state = [500, -100, 12, 0]
    # action_sequence = ['descend'] * 5  # Repeat 'descend' action 5 times
    # print("Testing action sequence with initial state:", initial_state)
    # run_action_sequence(np.array(initial_state), action_sequence, plot_prefix="test_down_sequence")

    # Test sequence of 5 'ascend' actions, starting from [0,0,10,0] initial state (same as env default).
    print("-------------------------")
    initial_state = [0, 0, 10, 0]
    action_sequence = ['ascend'] * 5  # Repeat 'up' action 5 times
    print("Testing action sequence with initial state:", initial_state)
    run_action_sequence(np.array(initial_state), action_sequence, plot_prefix="test_up_sequence_from_default")

if __name__ == "__main__":
    test_action_sequence()
    print("Test completed.")