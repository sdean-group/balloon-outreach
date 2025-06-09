import numpy as np

## TODOs:
## - Change transition model to include wind field (so lat, long, alt change properly.)
## - Change heuristic to use Haversine distance.
## - Change distance metric to Haversine distance? 

def heuristic(state, target_lat, target_lon, target_alt):
    """
    A heuristic function for A* that estimates the cost to reach the target state.
    Uses Euclidean distance in 3D space.

    TODO replace with Haversine distance?
    
    Args:
        state: Current state as a numpy array [lat, lon, alt]
        target_lat: Target latitude
        target_lon: Target longitude
        target_alt: Target altitude
    
    Returns:
        Estimated cost to reach the target state.
    """
    return np.sqrt((state[0] - target_lat) ** 2 + 
                   (state[1] - target_lon) ** 2 + 
                   (state[2] - target_alt) ** 2)

def distance(state1, state2):
    """
    Calculate the Euclidean distance between two states.

    TODO replace with another distance metric?
    
    Args:
        state1: First state as a numpy array [lat, lon, alt]
        state2: Second state as a numpy array [lat, lon, alt]
    
    Returns:
        Euclidean distance between the two states.
    """
    return np.linalg.norm(state1 - state2)

class TreeSearchAgent:
    """
    A simple tree search agent that constructs a search tree (nodes = states, edges = actions)
    to find the optimal path from current state to goal state.

    Tasks:
    - Go to target location
    - Fly as far as possible [not yet incorporated]

    State: [lat, long, alt]
    Action: {stay, ascend, descend}

    Algorithm: A*
    """
    def __init__(self, target_lat=100, target_lon=100, target_alt=12):
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.target_alt = target_alt

    def is_goal_state(self, state: np.ndarray) -> bool:
        """
        Check if the current state is the goal state.

        Args:
            state: Current state of the environment as a numpy array [lat, lon, alt]
        
        Returns:
            True if the current state matches the target state within a tolerance, False otherwise.
        """
        return (np.isclose(state[0], self.target_lat, atol=1e-2) and
                np.isclose(state[1], self.target_lon, atol=1e-2) and
                np.isclose(state[2], self.target_alt, atol=1e-2))

    def get_possible_actions(self, state: np.ndarray) -> list:
        """
        Get possible actions from the current state.
        
        Args:
            state: Current state of the environment as a numpy array [lat, lon, alt]
        
        Returns:
            A list of possible actions. In this case, actions are represented as strings.
        """
        return ['stay', 'ascend', 'descend']

    def apply_action(self, state: np.ndarray, action: str) -> np.ndarray:
        """
        Apply an action to the current state and return the new state.

        # TODO replace with actual transition model.
        
        Args:
            state: Current state of the environment as a numpy array [lat, lon, alt]
            action: Action to apply ('stay', 'ascend', 'descend')
        
        Returns:
            New state after applying the action.
        """
        new_state = np.copy(state)
        if action == 'ascend':
            new_state[2] += 1
        elif action == 'descend':
            new_state[2] -= 1
        # 'stay' action does not change the state
        return new_state

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

    def select_action_sequence(self, init_state: np.ndarray) -> np.ndarray:
        """
        Perform A* starting from an initial state to find a path to the target.

        Args
            init_state: Initial state of the environment as a numpy array [lat, lon, alt]
        
        Returns
            action_sequence: A sequence of (state, action) pairs leading to the target state.
        """
        # Initialize the root node with the initial state
        open_set = [tuple(init_state)]  # Open set of nodes to explore
        action_sequence = []
        came_from = {tuple(init_state): (None,None)}  # To reconstruct the path later
        g_score = {tuple(init_state): 0}
        f_score = {tuple(init_state): heuristic(init_state, self.target_lat, self.target_lon, self.target_alt)}
        while open_set:
            # Get the node with the lowest value (cost-to-go + A* heuristic)
            current_state = min(open_set, key=lambda state: f_score.get(state, np.inf))
            open_set.remove(current_state)

            # Check if we reached the goal state
            if self.is_goal_state(current_state):
                action_sequence = self.reconstruct_path(came_from, current_state)
                return action_sequence

            # Generate children nodes for possible actions
            for action in self.get_possible_actions(current_state):
                child_state = self.apply_action(current_state, action)
                tentative_g_score = g_score[tuple(current_state)] + distance(current_state, child_state)
                if tentative_g_score < g_score.get(tuple(child_state), np.inf):
                    # record the better path.
                    came_from[tuple(child_state)] = (current_state, action)
                    g_score[tuple(child_state)] = tentative_g_score
                    f_score[tuple(child_state)] = tentative_g_score + heuristic(child_state, self.target_lat, self.target_lon, self.target_alt)
                    if tuple(child_state) not in open_set:
                        open_set.append(tuple(child_state))
        
if __name__=="__main__":
    # Example usage. Test case 1 (ascend.)
    agent = TreeSearchAgent(target_lat=50, target_lon=50, target_alt=10)
    initial_state = np.array([50, 50, 0])  # Starting at (lat=0, lon=0, alt=0)
    # lat and lon do not change in our transition model.
    action_sequence = agent.select_action_sequence(initial_state)
    print("Action sequence to target:", action_sequence)

    # Test case 2 (descend.)
    initial_state = np.array([50, 50, 20])  # Starting at (lat=0, lon=0, alt=20)
    action_sequence = agent.select_action_sequence(initial_state)
    print("Action sequence to target:", action_sequence)

    # Test case 3 (stay.)
    # (it won't explicitly use the 'stay' action because we do goal-checking prior to action application.)
    initial_state = np.array([50, 50, 10])  # Starting at (lat=0, lon=0, alt=10)
    action_sequence = agent.select_action_sequence(initial_state)
    print("Action sequence to target:", action_sequence)