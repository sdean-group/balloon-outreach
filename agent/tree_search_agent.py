import numpy as np
from env.balloon import Balloon
from env.balloon_env import BalloonEnvironment,BalloonERAEnvironment,BalloonState
import copy
import matplotlib.pyplot as plt

## Run using: python -m agent.tree_search_agent (from within balloon-outreach directory.)

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
    def __init__(self, balloon_env=None, distance='euclidean', heuristic='euclidean', simplified_step=False,\
                 lat_long_atol=1e-2, alt_atol=0.02, max_iter=1000):
        if balloon_env is not None:
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
        
        self.simplified_step = simplified_step  # If True, use simplified step function.

        # Tolerances for goal state checking.
        self.lat_long_atol = lat_long_atol  # Tolerance for latitude and longitude
        self.alt_atol = alt_atol            # Tolerance for altitude

        # Max number of iterations.
        self.max_iter = max_iter

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

    def apply_action(self, action: str, current_balloonstate: BalloonState) -> tuple[np.ndarray, BalloonState]:
        """
        Apply an action to the current state and return the new state (adapted from BalloonEnvironment.step())
        
        Args:
            action: Action to apply ('stay', 'ascend', 'descend')
            current_balloonstate: Current BalloonState instance to apply the action on.

        Returns:
            New state after applying the action [lat, lon, alt, t], and the updated BalloonState instance.
        """
        # Convert action string to numerical value.
        if action == 'stay':
            action_value = 0.0
        elif action == 'ascend':
            action_value = 1.0
        elif action == 'descend':
            action_value = -1.0

        # Push the current state into the balloon environment object associated with this instance.
        self.balloon_env.set_balloon_state(current_balloonstate)
        
        # Step the balloon environment with the action.
        if self.simplified_step:
            # Use simplified step function.
            state, _, _, _ = self.balloon_env.simplified_step(action_value)
        else:
            state, _, _, _ = self.balloon_env.step(action_value)

        # Extract lat/long/alt/t from the new state, and return the updated balloon environment.
        new_state = np.array([state[0], state[1], state[2], state[6]])
        new_balloon_state = self.balloon_env.get_balloon_state()

        return new_state, new_balloon_state

    def discretize_state(self, state: np.ndarray, decimals: int = 1) -> tuple:
        """
        Discretize a continuous state by rounding to a specified number of decimal places.
        
        This function is useful for reducing the state space in A* search by grouping
        similar states together. For example, states (1.234, 2.345, 3.456) and 
        (1.235, 2.344, 3.457) would be considered the same when discretized to 1 decimal place.
        
        Args:
            state: Continuous state as a numpy array [lat, lon, alt, t]
            decimals: Number of decimal places to round to (default: 1)
                     - 0: Round to nearest integer
                     - 1: Round to 0.1 precision
                     - 2: Round to 0.01 precision
                     - etc.
        
        Returns:
            Discretized state as a tuple of rounded values
            
        Example:
            >>> agent = TreeSearchAgent(env)
            >>> state = np.array([1.234, 2.345, 3.456, 0.0])
            >>> discretized = agent.discretize_state(state, decimals=1)
            >>> print(discretized)
            (1.2, 2.3, 3.5, 0.0)
        """
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # Round each component to the specified number of decimal places
        discretized = np.round(state, decimals=decimals)
        
        # Convert to tuple for use as dictionary key
        return tuple(discretized)
    
    def get_discretization_info(self, decimals: int = 1) -> dict:
        """
        Get information about the discretization process for a given precision.
        
        This helps understand how the discretization affects the state space.
        
        Args:
            decimals: Number of decimal places to round to
            
        Returns:
            Dictionary containing discretization information
            
        Example:
            >>> agent = TreeSearchAgent(env)
            >>> info = agent.get_discretization_info(decimals=1)
            >>> print(f"Precision: {info['precision']}")
            >>> print(f"Example discretization: {info['example']}")
        """
        precision = 10 ** (-decimals)
        
        # Create example states to show discretization
        example_states = [
            np.array([1.234, 2.345, 3.456, 0.0]),
            np.array([1.235, 2.344, 3.457, 0.0]),
            np.array([1.236, 2.343, 3.458, 0.0])
        ]
        
        discretized_examples = [self.discretize_state(state, decimals) for state in example_states]
        
        return {
            'decimals': decimals,
            'precision': precision,
            'description': f"States are rounded to {precision} precision",
            'example_states': example_states,
            'example_discretized': discretized_examples,
            'example': f"States {example_states[0]} and {example_states[1]} both become {discretized_examples[0]} when discretized"
        }
        
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

    def plot_astar_tree(self, init_state: np.ndarray, expanded_set: list, plot_suffix: str = ""):
        """
        Plot the A* search tree. Also plot the altitude distribution in a separate plot.

        Args:
            init_state: Initial state of the environment as a numpy array [lat, lon, alt, t]
            g_score: Dictionary mapping states to their g-scores (cost from start to state)
        """
        latitudes = [state[0] for state in expanded_set]
        longitudes = [state[1] for state in expanded_set]
        altitudes = [state[2] for state in expanded_set]
        fig, ax = plt.subplots()
        # longitudes should be plotted on the x-axis, latitudes on the y-axis.
        ax.scatter(longitudes, latitudes, c='blue', label='Explored States')
        ax.scatter(self.target_lon, self.target_lat, c='red', label='Target State', marker='x')
        # show initial state
        ax.scatter(init_state[1], init_state[0], c='green', label='Initial State', marker='o')
        # Plot a green circle with radius lat_long_atol around the target state.
        circle = plt.Circle((self.target_lon, self.target_lat), self.lat_long_atol, color='green', fill=False, linestyle='--', label='Goal Region')
        # Write altitudes on the lat/long points.
        for i, (lat, lon, alt) in enumerate(zip(latitudes, longitudes, altitudes)):
            ax.text(lon, lat, f'{alt:.2f}', fontsize=8, ha='right', va='bottom', color='black')
        ax.add_artist(circle)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Explored States in A* Search')
        ax.legend()
        plt.show()
        plt.savefig(f'explored_states_{plot_suffix}.png')

        # Plot altitude distribution
        fig, ax2 = plt.subplots()
        ax2.hist(altitudes, bins=20, color='blue', alpha=0.7)
        ax2.set_xlabel('Altitude (km)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Altitude Distribution of Explored States')
        plt.savefig(f'altitude_distribution_{plot_suffix}.png')

    def select_action_sequence(self, init_state: np.ndarray, plot_suffix : str = '') -> np.ndarray:
        """
        Perform A* starting from an initial state to find a path to the target.
        Also saves a plot of explored states to disk.

        Args
            init_state: Initial state of the environment as a numpy array [lat, lon, alt, t]
        
        Returns
            action_sequence: A sequence of (state, action) pairs leading to the target state.
        """
        # Initialize the root node with the initial state
        open_set = [tuple(init_state)]  # Open set of nodes to explore
        expanded_set = []               # List of expanded nodes (for tracking A* progress)
        action_sequence = []
        came_from = {tuple(init_state): (None,None)}  # To reconstruct the path later
        g_score = {tuple(init_state): 0}
        h_score = {tuple(init_state): self.heuristic(init_state, self.target_lat, self.target_lon, self.target_alt)}
        f_score = {tuple(init_state): g_score[tuple(init_state)] + h_score[tuple(init_state)]}
        # Initialize lookup table from each state to a BalloonState instance.
        state_to_balloon_state = {tuple(init_state): self.balloon_env.get_balloon_state()}

        # open_set = [self.discretize_state(init_state,decimals=1)]  # Open set of nodes to explore
        # came_from = {self.discretize_state(init_state,decimals=1): (None, None)}
        # g_score = {self.discretize_state(init_state,decimals=1): 0}
        # f_score = {self.discretize_state(init_state,decimals=1): self.heuristic(init_state, self.target_lat, self.target_lon, self.target_alt)}
        # state_to_balloon = {self.discretize_state(init_state,decimals=1): self.balloon_env.balloon}
        it = 0
        while open_set:
            # Get the node with the lowest value (cost-to-go + A* heuristic)
            current_state = min(open_set, key=lambda state: f_score.get(state, np.inf))
            open_set.remove(current_state)
            expanded_set.append(current_state)
            # Print the came_from action (if it exists)
            # print(f"State: {current_state}, Came-from action: {came_from[tuple(current_state)][1]}")

            # Check if we reached the goal state
            if self.is_goal_state(current_state, atols=np.array([self.lat_long_atol, self.lat_long_atol, self.alt_atol])):
                action_sequence = self.reconstruct_path(came_from, current_state)
                self.plot_astar_tree(init_state, g_score, plot_suffix=plot_suffix)
                return action_sequence

            # Generate children nodes for possible actions
            for action in self.get_possible_actions(current_state):
                child_state, child_balloon_state = self.apply_action(action, state_to_balloon_state[current_state])
                tentative_g_score = g_score[tuple(current_state)] + self.distance(current_state, child_state)
                if tentative_g_score < g_score.get(tuple(child_state), np.inf):
                    # record the better path.
                    came_from[tuple(child_state)] = (current_state, action)
                    g_score[tuple(child_state)] = tentative_g_score
                    h_score[tuple(child_state)] = self.heuristic(child_state, self.target_lat, self.target_lon, self.target_alt)
                    f_score[tuple(child_state)] = g_score[tuple(child_state)] + h_score[tuple(child_state)]
                    # Update the balloon for this child state.
                    state_to_balloon_state[tuple(child_state)] = child_balloon_state
                    if tuple(child_state) not in open_set:
                        open_set.append(tuple(child_state))
            # for action in self.get_possible_actions(current_state):
            #     child_state, child_state_balloon = self.apply_action(current_state, action, state_to_balloon[current_state])
            #     child_state_disc = self.discretize_state(child_state)
            #     tentative_g_score = g_score[current_state] + self.distance(current_state, child_state)
            #     if tentative_g_score < g_score.get(child_state_disc, np.inf):
            #         came_from[child_state_disc] = (current_state, action)
            #         g_score[child_state_disc] = tentative_g_score
            #         f_score[child_state_disc] = tentative_g_score + self.heuristic(child_state, self.target_lat, self.target_lon, self.target_alt)
            #         state_to_balloon[child_state_disc] = child_state_balloon
            #         if child_state_disc not in open_set:
            #             open_set.append(child_state_disc)
            # Increment iteration count and check for max iterations.
            it += 1
            print(f"Iteration {it}/{self.max_iter}")
            if it >= self.max_iter:
                print("Max iterations reached. Stopping search.")
                break

        # In case of search failure, plot the all the lat/long tuples in the g_score mapping.
        print("A* failed. Plotting explored states...")
        self.plot_astar_tree(init_state, expanded_set, plot_suffix=plot_suffix)

        ## Even if A* fails, still return a path whose last state is
        ## as close to the target as possible.
        print("A* failed. Returning path to the closest state to the target.")
        best_state = min(h_score, key=h_score.get)
        action_sequence = self.reconstruct_path(came_from, best_state)
        return action_sequence

def run_astar(env, initial_lat: float, initial_long: float, initial_alt: float, target_lat: float, target_lon: float, target_alt: float,
              distance='euclidean', heuristic='euclidean', plot_suffix: str = "", simplified_step: bool = False,
              lat_long_atol: float = 1e-2, alt_atol: float = 0.02, max_iter: int = 1000):
    """
    Run A* search from an initial state to a target state.

    Returns a sequence of actions to reach the target state.
    """
    agent = TreeSearchAgent(balloon_env=env, distance=distance, heuristic=heuristic, simplified_step=simplified_step,
                            lat_long_atol=lat_long_atol, alt_atol=alt_atol, max_iter=max_iter)
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

## NOTE: because of seeding right now, test cases 1-3 all have to be run in order in order to work correctly.
## But test case 4 (test_era) can be run independently.
def test1():
    # Case 1 (initial state = target state.)
    print("------ Case 1: Initial state = target state ---")
    # Set random seed to 0 for reproducibility.
    np.random.seed(0)
    env = BalloonEnvironment()
    run_astar(env, initial_lat=0, initial_long=0, initial_alt=10.0,
              target_lat=0, target_lon=0, target_alt=10.0,
              distance='euclidean', heuristic='zero',
              plot_suffix="test1")

def test2():
    # Case 2 (initial state = target state with some drift).
    print("------ Case 2: Initial state = target state with noise ---")
    # Set random seed to 0 for reproducibility.
    np.random.seed(0)
    env = BalloonEnvironment()
    run_astar(env, initial_lat=0, initial_long=0, initial_alt=10,
              target_lat=0.03, target_lon=0.16, target_alt=10,
              distance='euclidean', heuristic='euclidean',
              plot_suffix="test2")

def test3():
    # Case 3 [test Haversine distance metric and heuristic, otherwise same as Case 2.]
    print("------ Case 3: Initial state = target state with noise, using Haversine distance ---")
    # Set random seed to 0 for reproducibility.
    np.random.seed(0)
    env = BalloonEnvironment()
    run_astar(env, initial_lat=0, initial_long=0, initial_alt=10,
              target_lat=0.011, target_lon=0, target_alt=10,
              distance='haversine', heuristic='haversine',
              plot_suffix="test3")
    
def test_era():
    # Case 4: Test A* with BalloonERAEnvironment.
    print("------ Case 4: Test A* with BalloonERAEnvironment ---")

    # Set random seed to 0 for reproducibility (later: use BalloonERAEnvironment's random seed.)
    np.random.seed(0)

    # BalloonERAEnvironment initialization (see main_ERA.py)
    import xarray as xr
    import datetime as dt
    # 1. load your ERA5 file
    ds = xr.open_dataset("era5_data.nc", engine="netcdf4")
    # 2. pick a reference start_time (should match your dataset’s first valid_time)
    start_time = dt.datetime(2024, 7, 1, 0, 0)
    # Create environment and agent
    env = BalloonERAEnvironment(ds=ds, start_time=start_time, viz=False)
    # Run same test case as case 3.
    run_astar(env, initial_lat=0, initial_long=0, initial_alt=10,
              target_lat=-0.0001, target_lon=0.04, target_alt=10,
              distance='haversine', heuristic='haversine',
              plot_suffix="test_era")


if __name__=="__main__":
    ## NEW TEST CASES (6/16/2025).

    # # Case 1 (initial state = target state.)
    test1()

    # # # Case 2 (initial state close to target state; expecting to get sequence of 'stay' actions.)
    test2()

    # # # Case 3 [test Haversine distance metric, otherwise same as Case 2.]
    test3()

    # Case 4: Test A* with BalloonERAEnvironment.
    test_era()

def test_discretization():
    """
    Test the improved discretization function to demonstrate its clarity and functionality.
    """
    print("------ Testing Discretization Function ---")
    
    # Create a simple environment for testing
    np.random.seed(0)
    env = BalloonEnvironment()
    agent = TreeSearchAgent(balloon_env=env)
    
    # Test different precision levels
    test_state = np.array([1.234567, 2.345678, 3.456789, 0.0])
    
    print(f"Original state: {test_state}")
    
    for decimals in [0, 1, 2, 3]:
        discretized = agent.discretize_state(test_state, decimals)
        print(f"Discretized to {decimals} decimal places: {discretized}")
    
    # Test the discretization info function
    print("\n--- Discretization Information ---")
    info = agent.get_discretization_info(decimals=1)
    print(f"Precision: {info['precision']}")
    print(f"Description: {info['description']}")
    print(f"Example: {info['example']}")
    
    # Show how similar states get grouped together
    print("\n--- State Grouping Example ---")
    similar_states = [
        np.array([1.234, 2.345, 3.456, 0.0]),
        np.array([1.235, 2.344, 3.457, 0.0]),
        np.array([1.236, 2.343, 3.458, 0.0])
    ]
    
    for i, state in enumerate(similar_states):
        discretized = agent.discretize_state(state, decimals=1)
        print(f"State {i+1}: {state} -> {discretized}")
    
    print("All three states become the same when discretized to 1 decimal place!")


if __name__=="__main__":
    ## NEW TEST CASES (6/16/2025).

    # # Case 1 (initial state = target state.)
    test1()

    # # # Case 2 (initial state close to target state; expecting to get sequence of 'stay' actions.)
    test2()

    # # # Case 3 [test Haversine distance metric, otherwise same as Case 2.]
    test3()

    # Case 4: Test A* with BalloonERAEnvironment.
    test_era()

    # Test the improved discretization function
    test_discretization()
    # Expected outputs from Cases 1-3:
    # ------ Case 1: Initial state = target state ---
    # Action sequence to target: [((np.float64(0.0), np.float64(0.0), np.float64(10.0), np.float64(0.0)), None)]
    # ------ Case 2: Initial state = target state with noise ---
    # Action sequence to target: [((np.float64(0.0), np.float64(0.0), np.float64(10.0), np.float64(0.0)), 'stay'), ((np.float64(0.028296857220620968), np.float64(0.030217999015719896), np.float64(10.000006162530978), np.float64(0.016666666666666666)), 'stay'), ((np.float64(0.056536064465265115), np.float64(0.06175999003859439), np.float64(9.999997987460512), np.float64(0.03333333333333333)), 'stay'), ((np.float64(0.084719169413003), np.float64(0.09462547588485093), np.float64(9.999998116572215), np.float64(0.05)), 'stay'), ((np.float64(0.11284772158361407), np.float64(0.12881396562205658), np.float64(9.99999789138437), np.float64(0.06666666666666667)), 'stay'), ((np.float64(0.1409232725619879), np.float64(0.16432497443867367), np.float64(9.999997884086225), np.float64(0.08333333333333333)), None)]
    # ------ Case 3: Initial state = target state with noise, using Haversine distance ---
    # Action sequence to target: [((np.float64(0.0), np.float64(0.0), np.float64(10.0), np.float64(0.0)), 'descend'), ((np.float64(0.028296857220620968), np.float64(0.030217999015719896), np.float64(9.976661860657844), np.float64(0.016666666666666666)), 'stay'), ((np.float64(0.05653602698671051), np.float64(0.06175990621982107), np.float64(9.940651320675139), np.float64(0.03333333333333333)), 'stay'), ((np.float64(0.08471903271482123), np.float64(0.0946251978266461), np.float64(9.93978505832379), np.float64(0.05)), 'stay'), ((np.float64(0.11284748019902958), np.float64(0.12881350929071972), np.float64(9.963307754266621), np.float64(0.06666666666666667)), 'ascend'), ((np.float64(0.14092296497152726), np.float64(0.16432442080656626), np.float64(9.997590219824646), np.float64(0.08333333333333333)), None)]
    # ------ Case 4: Test A* with BalloonERAEnvironment ---
    # Action sequence to target: [((np.float64(0.0), np.float64(0.0), np.float64(10.0), np.float64(0.0)), 'descend'), ((np.float64(0.00010084852098865782), np.float64(0.577479588035151), np.float64(9.976661860657844), np.float64(0.016666666666666666)), 'stay'), ((np.float64(0.01691783033279263), np.float64(1.1620614711529997), np.float64(9.940651320675139), np.float64(0.03333333333333333)), 'stay'), ((np.float64(0.034743348664190134), np.float64(1.704066487561494), np.float64(9.93978505832379), np.float64(0.05)), 'ascend'), ((np.float64(0.05644705627335142), np.float64(2.2493163255951414), np.float64(9.981959535668265), np.float64(0.06666666666666667)), None)]
