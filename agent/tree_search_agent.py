import numpy as np
from env.balloon_env import Balloon,BalloonEnvironment
import copy
import matplotlib.pyplot as plt

## Run using: python -m agent.tree_search_agent (from within balloon-outreach directory.)

## TODOs:
## - Change heuristic to use Haversine distance.
## - Change distance metric to Haversine distance.

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

    def apply_action(self, state: np.ndarray, action: str, current_balloon : Balloon) -> np.ndarray:
        """
        Apply an action to the current state and return the new state (adapted from BalloonEnvironment.step())
        
        Args:
            state: Current state of the environment as a numpy array [lat, lon, alt, t]
            action: Action to apply ('stay', 'ascend', 'descend')
        
        Returns:
            New state after applying the action [lat, lon, alt, t], and the updated Balloon instance.
        """
        # NOTE: we should not be changing the state of the BalloonEnvironment directly,
        # except through the Balloon instance.

        # Unpack the current state.
        lat, lon, alt, current_time = state

        # Convert action string to numerical value.
        if action == 'stay':
            action_value = 0.0
        elif action == 'ascend':
            action_value = 1.0
        elif action == 'descend':
            action_value = -1.0

        # Create a new balloon and push it into the current balloon environment.
        balloon = copy.copy(current_balloon)  # Create a copy of the current balloon.
        self.balloon_env.balloon = balloon  # Update the balloon in the environment.

        # # Get current pressure based on altitude
        pressure = self.balloon_env.balloon.altitude_to_pressure(self.balloon_env.balloon.alt)

        # # Get wind at current position and time
        wind = self.balloon_env.wind_field.get_wind(
            self.balloon_env.balloon.lat,
            self.balloon_env.balloon.lon,
            pressure,
            current_time
        )
        # Update balloon state.
        # print("Starting balloon lat, lon, alt:", self.balloon_env.balloon.lat, self.balloon_env.balloon.lon, self.balloon_env.balloon.alt)
        # print("Applying action:", action, "with value:", action_value)
        print(f"Wind: {wind}, dt: {self.balloon_env.dt}, Action value: {action_value}")
        self.balloon_env.balloon.step(wind, self.balloon_env.dt, action_value)

        # Extract state from balloon, and also increment time.
        new_state = np.array([self.balloon_env.balloon.lat,
                              self.balloon_env.balloon.lon,
                              self.balloon_env.balloon.alt,
                              current_time + self.balloon_env.dt / 3600])
        return new_state, self.balloon_env.balloon  # Return the new state and the updated balloon instance.

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

    def plot_astar_tree(self, init_state: np.ndarray, g_score: dict, lat_long_atol: float = 1e-2):
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
        plt.savefig('explored_states.png')

    def select_action_sequence(self, init_state: np.ndarray) -> np.ndarray:
        """
        Perform A* starting from an initial state to find a path to the target.

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
        state_to_balloon = {tuple(init_state): self.balloon_env.balloon}
        it = 0
        while open_set:
            # Get the node with the lowest value (cost-to-go + A* heuristic)
            current_state = min(open_set, key=lambda state: f_score.get(state, np.inf))
            open_set.remove(current_state)
            # Print the came_from action (if it exists)
            print(f"State: {current_state}, Came-from action: {came_from[tuple(current_state)][1]}")

            # Check if we reached the goal state
            if self.is_goal_state(current_state, atols=np.array([lat_long_atol, lat_long_atol, alt_atol])):
                action_sequence = self.reconstruct_path(came_from, current_state)
                self.plot_astar_tree(init_state, g_score, lat_long_atol=lat_long_atol)
                return action_sequence

            # Generate children nodes for possible actions
            for action in self.get_possible_actions(current_state):
                child_state, child_state_balloon = self.apply_action(current_state, action, state_to_balloon[current_state])
                tentative_g_score = g_score[tuple(current_state)] + self.distance(current_state, child_state)
                if tentative_g_score < g_score.get(tuple(child_state), np.inf):
                    # record the better path.
                    came_from[tuple(child_state)] = (current_state, action)
                    g_score[tuple(child_state)] = tentative_g_score
                    f_score[tuple(child_state)] = tentative_g_score + self.heuristic(child_state, self.target_lat, self.target_lon, self.target_alt)
                    # Update the balloon for this child state.
                    state_to_balloon[tuple(child_state)] = child_state_balloon
                    if tuple(child_state) not in open_set:
                        open_set.append(tuple(child_state))

            # Increment iteration count and check for max iterations.
            it += 1
            if it >= max_iterations:
                print("Max iterations reached. Stopping search.")
                break

        # In case of search failure, plot the all the lat/long tuples in the g_score mapping.
        print("A* failed. Plotting explored states...")
        self.plot_astar_tree(init_state, g_score, lat_long_atol=lat_long_atol)


def test1():
    # Case 1 (initial state = target state.)
    print("------ Case 1: Initial state = target state ---")
    env = BalloonEnvironment()
    agent = TreeSearchAgent(balloon_env=env, distance='euclidean', heuristic='zero')
    initial_state = np.array([env.target_lat, env.target_lon, env.target_alt, env.current_time])  # Starting at (lat=0, lon=0, alt=0, t=current_time)
    # Set the balloon's initial state.
    env.balloon.lat, env.balloon.lon, env.balloon.alt = initial_state[:3]
    action_sequence = agent.select_action_sequence(initial_state)
    print(f"Action sequence to target: {action_sequence}")

if __name__=="__main__":
    ## NEW TEST CASES (6/16/2025).

    # Case 1 (initial state = target state.)
    test1()

    # Case 2 (initial state = target state with some noise + hacking).
    # Also, changed altitude tolerance to be very large (10 km) for testing.
    print("------ Case 2: Initial state = target state with noise ---")
    env = BalloonEnvironment()      # it's important that you re-initialize this before every re-plan.
    agent = TreeSearchAgent(balloon_env=env, distance='euclidean', heuristic='zero')    # and by consequence, re-initialize this.
    # otherwise the tree search agent will have the wrong initial balloon (since it uses self.balloon_env.balloon).
    noise_val = 0.02
    # initial_state = np.array([env.target_lat + noise_val, env.target_lon + noise_val, env.target_alt + noise_val, env.current_time])  # Starting at (lat=0, lon=0, alt=0, t=current_time)
    initial_state = np.array([env.target_lat + noise_val, env.target_lon + noise_val, env.target_alt + noise_val, env.current_time])  # Starting at (lat=0, lon=0, alt=0, t=current_time)
    # Set the balloon's initial state.
    env.balloon.lat, env.balloon.lon, env.balloon.alt = initial_state[:3]
    
    # HACK: change the target state to one that is currently feasible for A*.
    agent.target_lat = 500.625
    agent.target_lon = -100.09
    agent.target_alt = 5.0
    print(f"Initial state: {initial_state[:-1]}, Target state: {[agent.target_lat, agent.target_lon, agent.target_alt]}")
    action_sequence = agent.select_action_sequence(initial_state)
    print(f"Action sequence to target: {action_sequence}")

    # Case 3 [test Haversine distance metric, otherwise same as Case 2.]
    # Currently not working...
    # env = BalloonEnvironment()
    # agent = TreeSearchAgent(balloon_env=env, distance='haversine', heuristic='zero')
    # initial_state = np.array([env.target_lat + noise_val, env.target_lon + noise_val, env.target_alt + noise_val, env.current_time])  # Starting at (lat=0, lon=0, alt=0, t=current_time)
    # # Set the balloon's initial state.
    # env.balloon.lat, env.balloon.lon, env.balloon.alt = initial_state[:3]
    # # HACK: change the target state to one that is currently feasible for A*.
    # agent.target_lat = 499.6
    # agent.target_lon = -99.86
    # agent.target_alt = 10.0
    # action_sequence = agent.select_action_sequence(initial_state)
    # print(f"Action sequence to target: {action_sequence}")

    ## OLD TEST CASES.
    # # Example usage. Test case 1 (ascend.)
    # agent = TreeSearchAgent(target_lat=50, target_lon=50, target_alt=10)
    # initial_state = np.array([50, 50, 0])  # Starting at (lat=0, lon=0, alt=0)
    # # lat and lon do not change in our transition model.
    # action_sequence = agent.select_action_sequence(initial_state)
    # print("Action sequence to target:", action_sequence)

    # # Test case 2 (descend.)
    # initial_state = np.array([50, 50, 20])  # Starting at (lat=0, lon=0, alt=20)
    # action_sequence = agent.select_action_sequence(initial_state)
    # print("Action sequence to target:", action_sequence)

    # # Test case 3 (stay.)
    # # (it won't explicitly use the 'stay' action because we do goal-checking prior to action application.)
    # initial_state = np.array([50, 50, 10])  # Starting at (lat=0, lon=0, alt=10)
    # action_sequence = agent.select_action_sequence(initial_state)
    # print("Action sequence to target:", action_sequence)

    # # Action sequence to target: [((50, 50, 0), 'ascend'), ((50, 50, 1), 'ascend'), ((50, 50, 2), 'ascend'), ((50, 50, 3), 'ascend'), ((50, 50, 4), 'ascend'), ((50, 50, 5), 'ascend'), ((50, 50, 6), 'ascend'), ((50, 50, 7), 'ascend'), ((50, 50, 8), 'ascend'), ((50, 50, 9), 'ascend'), ((50, 50, 10), None)]
    # # Action sequence to target: [((50, 50, 20), 'descend'), ((50, 50, 19), 'descend'), ((50, 50, 18), 'descend'), ((50, 50, 17), 'descend'), ((50, 50, 16), 'descend'), ((50, 50, 15), 'descend'), ((50, 50, 14), 'descend'), ((50, 50, 13), 'descend'), ((50, 50, 12), 'descend'), ((50, 50, 11), 'descend'), ((50, 50, 10), None)]
    # # Action sequence to target: [((50, 50, 10), None)]