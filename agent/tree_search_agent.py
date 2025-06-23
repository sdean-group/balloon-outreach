import numpy as np
from env.balloon_env import Balloon,BalloonEnvironment
from env.balloon_ERA_env import BalloonERAEnvironment
import copy
import matplotlib.pyplot as plt

## Run using: python -m agent.tree_search_agent (from within balloon-outreach directory.)

# Set random seed to 0 for reproducibility.
np.random.seed(0)

# Copied over from env/balloon_env.py in branch v0_edit.
def haversine_distance(state_start, state_end):
    """Calculate great-circle distance between two (lat, lon) points in meters."""
    import math
    R = 6371e3  # Earth radius in meters

    # extract balloon state.
    # state is a numpy array [lat, lon, alt, t]
    (lat_start, lon_start) = state_start[:2]
    (lat_end, lon_end) = state_end[:2]

    phi1 = math.radians(lat_start)
    phi2 = math.radians(lat_end)
    delta_phi = math.radians(lat_end - lat_start)
    delta_lambda = math.radians(lon_end - lon_start)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def haversine_heuristic(state, target_lat, target_lon, target_alt):
    """
    A heuristic function for A* that estimates the cost to reach the target state.
    Uses Haversine distance in 2D space (ignoring altitude).
    
    Args:
        state: Current state as a numpy array [lat, lon, alt, t]
        target_lat: Target latitude
        target_lon: Target longitude
    
    Returns:
        Estimated cost to reach the target state.
    """
    return haversine_distance(state, np.array([target_lat, target_lon, target_alt, -1])) # -1 for time, since we ignore it in the heuristic.


def euclidean_heuristic(state, target_lat, target_lon, target_alt):
    """
    A heuristic function for A* that estimates the cost to reach the target state.
    Uses Euclidean distance in 3D space.
    
    Args:
        state: Current state as a numpy array [lat, lon, alt, t]
        target_lat: Target latitude
        target_lon: Target longitude
        target_alt: Target altitude
    
    Returns:
        Estimated cost to reach the target state.
    """
    return np.sqrt((state[0] - target_lat) ** 2 + 
                   (state[1] - target_lon) ** 2 + 
                   (state[2] - target_alt) ** 2)

def euclidean_distance(state1, state2):
    """
    Calculate the Euclidean distance between two states.

    TODO replace with another distance metric?
    
    Args:
        state1: First state as a numpy array [lat, lon, alt, t]
        state2: Second state as a numpy array [lat, lon, alt, t]
    
    Returns:
        Euclidean distance between [lat, long, alt]
    """
    # Ignore time component for distance calculation.
    return np.linalg.norm(state1[:3] - state2[:3])

class TreeSearchAgent:
    """
    A simple tree search agent that constructs a search tree (nodes = states, edges = actions)
    to find the optimal path from current state to goal state.

    Tasks:
    - Go to target location (specified by the balloon environment.)
    - Fly as far as possible [not yet implemented]

    State: [lat, long, alt, t]
    Action: {stay, ascend, descend}

    Algorithm: A*
    """
    def __init__(self, balloon_env=None, distance='euclidean', heuristic='euclidean'):
        if balloon_env is not None:                     # NOTE: this represents the balloon environment for the root node only.
            self.balloon_env = balloon_env
            self.target_lat = balloon_env.target_lat
            self.target_lon = balloon_env.target_lon
            self.target_alt = balloon_env.target_alt
        else:
            print("No environment provided. Initialization failed.")
            return
    
        if distance == 'euclidean':
            self.distance = euclidean_distance
        elif distance == 'haversine':
            self.distance = haversine_distance
        else:
            raise ValueError(f"Unknown distance metric: {distance}. Supported metrics: 'euclidean', 'haversine'.")

        if heuristic == 'euclidean':
            self.heuristic = euclidean_heuristic
        elif heuristic == 'zero':
            # Zero heuristic (Equivalent to Dijkstra's algorithm.)
            self.heuristic = lambda state, target_lat, target_lon, target_alt: 0.0
        elif heuristic == 'haversine':
            self.heuristic = haversine_heuristic
        else:
            raise ValueError(f"Unknown heuristic: {heuristic}. Supported heuristics: 'euclidean', 'zero'.")

    def is_goal_state(self, state: np.ndarray, atols: np.ndarray) -> bool:
        """
        Check if the current state is the goal state.

        Args:
            state: Current state of the environment as a numpy array [lat, lon, alt, t]
        
        Returns:
            True if the current state matches the target state within a tolerance, False otherwise.
        """
        return (np.isclose(state[0], self.target_lat, atol=atols[0]) and
                np.isclose(state[1], self.target_lon, atol=atols[1]) and
                np.isclose(state[2], self.target_alt, atol=atols[2]))

    def get_possible_actions(self, state: np.ndarray) -> list:
        """
        Get possible actions from the current state.
        
        Args:
            state: Current state of the environment as a numpy array [lat, lon, alt, t]
        
        Returns:
            A list of possible actions. In this case, actions are represented as strings.
        """
        return ['stay', 'ascend', 'descend']

    def apply_action(self, action: str, current_balloonenv : BalloonEnvironment) -> np.ndarray:
        """
        Apply an action to the current state and return the new state (adapted from BalloonEnvironment.step())
        
        Args:
            action: Action to apply ('stay', 'ascend', 'descend')
            current_balloonenv: Current BalloonEnvironment instance to apply the action on (contains the balloon state).
        
        Returns:
            New state after applying the action [lat, lon, alt, t], and the updated BalloonEnvironment instance.
        """
        # Convert action string to numerical value.
        if action == 'stay':
            action_value = 0.0
        elif action == 'ascend':
            action_value = 1.0
        elif action == 'descend':
            action_value = -1.0

        # Create deep-copy of the current balloon environment.
        balloon_env = copy.deepcopy(current_balloonenv)
        
        # Step the balloon environment with the action.
        state, _, _, _ = balloon_env.step(action_value)
        
        # Extract lat/long/alt/t from the new state, and return the updated balloon environment.
        new_state = np.array([state[0], state[1], state[2], state[6]])

        return new_state, balloon_env


    def reconstruct_path(self, came_from: dict, current_state: tuple) -> np.ndarray:
        """
        Reconstruct the path from the current state to the initial state using the 'came_from' mapping.

        Args:
            current_state: The current state as a tuple (lat, lon, alt)
        
        Returns:
            action_sequence: A sequence of (state, action) pairs leading to the target state.
        """
        assert came_from is not None, "came_from mapping cannot be None"
        path = []
        working_action = None
        working_state = current_state
        while working_state is not None:
            path.append((working_state, working_action))
            working_state, working_action = came_from[working_state] if working_state in came_from else (None, None)
        return path[::-1]

    def plot_astar_tree(self, init_state: np.ndarray, g_score: dict, lat_long_atol: float = 1e-2, plot_suffix: str = ""):
        """
        Plot the A* search tree.

        Args:
            init_state: Initial state of the environment as a numpy array [lat, lon, alt, t]
            g_score: Dictionary mapping states to their g-scores (cost from start to state)
            lat_long_atol: Tolerance for latitude and longitude to define the goal region
        """
        latitudes = [state[0] for state in g_score.keys()]
        longitudes = [state[1] for state in g_score.keys()]
        altitudes = [state[2] for state in g_score.keys()]
        fig, ax = plt.subplots()
        ax.scatter(latitudes, longitudes, c='blue', label='Explored States')
        ax.scatter(self.target_lat, self.target_lon, c='red', label='Target State', marker='x')
        # show initial state
        ax.scatter(init_state[0], init_state[1], c='green', label='Initial State', marker='o')
        # Plot a green circle with radius lat_long_atol around the target state.
        circle = plt.Circle((self.target_lat, self.target_lon), lat_long_atol, color='green', fill=False, linestyle='--', label='Goal Region')
        # Write altitudes on the lat/long points.
        for i, (lat, lon, alt) in enumerate(zip(latitudes, longitudes, altitudes)):
            ax.text(lat, lon, f'{alt:.2f}', fontsize=8, ha='right', va='bottom', color='black')
        ax.add_artist(circle)
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Longitude')
        ax.set_title('Explored States in A* Search')
        ax.legend()
        plt.show()
        plt.savefig(f'explored_states_{plot_suffix}.png')

    def select_action_sequence(self, init_state: np.ndarray, plot_suffix : str = '') -> np.ndarray:
        """
        Perform A* starting from an initial state to find a path to the target.
        Also saves a plot of explored states to disk.

        Args
            init_state: Initial state of the environment as a numpy array [lat, lon, alt, t]
        
        Returns
            action_sequence: A sequence of (state, action) pairs leading to the target state.
        """
        max_iterations = 1000  # Limit the number of iterations to prevent A* from running indefinitely.
        # Tolerances (assume lat/long tolerance is the same; altitude tolerance is different.)
        lat_long_atol = 1e-2  # 0.01 degrees in latitude/longitude.
        alt_atol = 0.02  # 20 cm in altitude.

        # Initialize the root node with the initial state
        open_set = [tuple(init_state)]  # Open set of nodes to explore
        action_sequence = []
        came_from = {tuple(init_state): (None,None)}  # To reconstruct the path later
        g_score = {tuple(init_state): 0}
        f_score = {tuple(init_state): self.heuristic(init_state, self.target_lat, self.target_lon, self.target_alt)}
        # We also need a lookup table from each state to a Balloon instance.
        state_to_balloon_env = {tuple(init_state): self.balloon_env}
        it = 0
        while open_set:
            # Get the node with the lowest value (cost-to-go + A* heuristic)
            current_state = min(open_set, key=lambda state: f_score.get(state, np.inf))
            open_set.remove(current_state)
            # Print the came_from action (if it exists)
            # print(f"State: {current_state}, Came-from action: {came_from[tuple(current_state)][1]}")

            # Check if we reached the goal state
            if self.is_goal_state(current_state, atols=np.array([lat_long_atol, lat_long_atol, alt_atol])):
                action_sequence = self.reconstruct_path(came_from, current_state)
                self.plot_astar_tree(init_state, g_score, lat_long_atol=lat_long_atol, plot_suffix=plot_suffix)
                return action_sequence

            # Generate children nodes for possible actions
            for action in self.get_possible_actions(current_state):
                child_state, child_state_balloon_env = self.apply_action(action, state_to_balloon_env[current_state])
                tentative_g_score = g_score[tuple(current_state)] + self.distance(current_state, child_state)
                if tentative_g_score < g_score.get(tuple(child_state), np.inf):
                    # record the better path.
                    came_from[tuple(child_state)] = (current_state, action)
                    g_score[tuple(child_state)] = tentative_g_score
                    f_score[tuple(child_state)] = tentative_g_score + self.heuristic(child_state, self.target_lat, self.target_lon, self.target_alt)
                    # Update the balloon for this child state.
                    state_to_balloon_env[tuple(child_state)] = child_state_balloon_env
                    if tuple(child_state) not in open_set:
                        open_set.append(tuple(child_state))

            # Increment iteration count and check for max iterations.
            it += 1
            print(f"Iteration {it}/{max_iterations}")
            if it >= max_iterations:
                print("Max iterations reached. Stopping search.")
                break

        # In case of search failure, plot the all the lat/long tuples in the g_score mapping.
        print("A* failed. Plotting explored states...")
        self.plot_astar_tree(init_state, g_score, lat_long_atol=lat_long_atol, plot_suffix=plot_suffix)


def run_astar(env, initial_lat: float, initial_long: float, initial_alt: float, target_lat: float, target_lon: float, target_alt: float,
              distance='euclidean', heuristic='euclidean', plot_suffix: str = ""):
    """
    Run A* search from an initial state to a target state.

    Returns a sequence of actions to reach the target state.
    """
    agent = TreeSearchAgent(balloon_env=env, distance=distance, heuristic=heuristic)
    # Set the balloon's initial state.
    initial_state = np.array([initial_lat, initial_long, initial_alt, env.current_time])  # Starting at (lat=0, lon=0, alt=0, t=current_time)
    env.balloon = Balloon(initial_lat=initial_state[0],
                          initial_lon=initial_state[1],
                          initial_alt=initial_state[2],)
    agent.target_lat = target_lat
    agent.target_lon = target_lon
    agent.target_alt = target_alt
    action_sequence = agent.select_action_sequence(initial_state, plot_suffix=plot_suffix)
    print(f"Action sequence to target: {action_sequence}")
    return action_sequence


def test1():
    # Case 1 (initial state = target state.)
    print("------ Case 1: Initial state = target state ---")
    env = BalloonEnvironment()
    run_astar(env, initial_lat=0, initial_long=0, initial_alt=10.0,
              target_lat=0, target_lon=0, target_alt=10.0,
              distance='euclidean', heuristic='zero',
              plot_suffix="test1")

def test2():
    # Case 2 (initial state = target state with some drift).
    print("------ Case 2: Initial state = target state with noise ---")
    env = BalloonEnvironment()
    run_astar(env, initial_lat=0, initial_long=0, initial_alt=10,
              target_lat=0.16, target_lon=0.16, target_alt=10,
              distance='euclidean', heuristic='euclidean',
              plot_suffix="test2")

def test3():
    # Case 3 [test Haversine distance metric and heuristic, otherwise same as Case 2.]
    print("------ Case 3: Initial state = target state with noise, using Haversine distance ---")
    env = BalloonEnvironment()
    run_astar(env, initial_lat=0, initial_long=0, initial_alt=10,
              target_lat=0.16, target_lon=0.16, target_alt=10,
              distance='haversine', heuristic='haversine',
              plot_suffix="test3")
    
def test_era():
    # Case 4: Test A* with BalloonERAEnvironment.
    print("------ Case 4: Test A* with BalloonERAEnvironment ---")

    # BalloonERAEnvironment initialization (see main_ERA.py)
    import xarray as xr
    import datetime as dt
    # 1. load your ERA5 file
    ds = xr.open_dataset("era5_data.nc", engine="netcdf4")
    # 2. pick a reference start_time (should match your dataset’s first valid_time)
    start_time = dt.datetime(2024, 7, 1, 0, 0)
    # Create environment and agent
    env = BalloonERAEnvironment(ds=ds, start_time=start_time)

    # Run same test case as case 3.
    run_astar(env, initial_lat=0, initial_long=0, initial_alt=10,
              target_lat=0.18, target_lon=2.4, target_alt=10,
              distance='haversine', heuristic='haversine',
              plot_suffix="test_era")


if __name__=="__main__":
    ## NEW TEST CASES (6/16/2025).

    # Case 1 (initial state = target state.)
    # test1()

    # # Case 2 (initial state close to target state; expecting to get sequence of 'stay' actions.)
    # test2()

    # # Case 3 [test Haversine distance metric, otherwise same as Case 2.]
    # test3()

    # Case 4: Test A* with BalloonERAEnvironment.
    test_era()

    # Expected outputs from Cases 1-3:
    # ------ Case 1: Initial state = target state ---
    # Action sequence to target: [((np.float64(0.0), np.float64(0.0), np.float64(10.0), np.float64(0.0)), None)]
    # ------ Case 2: Initial state = target state with noise ---
    # Action sequence to target: [((np.float64(0.0), np.float64(0.0), np.float64(10.0), np.float64(0.0)), 'stay'), ((np.float64(0.03194087297407955), np.float64(0.028071626935738263), np.float64(10.000054055658351), np.float64(0.016666666666666666)), 'stay'), ((np.float64(0.0637754093993097), np.float64(0.05741264959936474), np.float64(9.99982956253404), np.float64(0.03333333333333333)), 'stay'), ((np.float64(0.09550437678071091), np.float64(0.08802261846718662), np.float64(9.99979730199271), np.float64(0.05)), 'stay'), ((np.float64(0.12712857331563762), np.float64(0.11990106212115517), np.float64(9.999804818339697), np.float64(0.06666666666666667)), 'stay'), ((np.float64(0.1586488301314381), np.float64(0.15304748657786632), np.float64(9.999808634033268), np.float64(0.08333333333333333)), None)]
    # ------ Case 3: Initial state = target state with noise, using Haversine distance ---
    # Action sequence to target: [((np.float64(0.0), np.float64(0.0), np.float64(10.0), np.float64(0.0)), 'ascend'), ((np.float64(0.03074732806206913), np.float64(0.027918130248106156), np.float64(10.028240761498703), np.float64(0.016666666666666666)), 'stay'), ((np.float64(0.0615009936937044), np.float64(0.05717205027069738), np.float64(10.037375501346354), np.float64(0.03333333333333333)), 'ascend'), ((np.float64(0.09226261009070276), np.float64(0.08776143920750092), np.float64(10.069463552260146), np.float64(0.05)), 'descend'), ((np.float64(0.12303399984336755), np.float64(0.11968626113176586), np.float64(10.036209859091501), np.float64(0.06666666666666667)), 'stay'), ((np.float64(0.15381691689843763), np.float64(0.15294577331388504), np.float64(10.015281067642064), np.float64(0.08333333333333333)), None)]
    # ------ Case 4: Test A* with BalloonERAEnvironment ---
    # Action sequence to target: [((np.float64(0.0), np.float64(0.0), np.float64(10.0), np.float64(0.0)), 'ascend'), ((np.float64(0.00010084852098865782), np.float64(0.577479588035151), np.float64(10.028240761498703), np.float64(0.016666666666666666)), 'stay'), ((np.float64(0.04461073719514106), np.float64(1.2136785730361956), np.float64(10.037375501346354), np.float64(0.03333333333333333)), 'stay'), ((np.float64(0.1058041539878507), np.float64(1.8356345987464802), np.float64(10.067342758256219), np.float64(0.05)), 'stay'), ((np.float64(0.189743420549645), np.float64(2.4069581477187407), np.float64(10.009109374288483), np.float64(0.06666666666666667)), None)]