import numpy as np

class RandomAgent:
    """A simple agent that selects random actions"""
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select a random action
        
        Args:
            state: Current state of the environment
            
        Returns:
            action: Single continuous action in range [-1, 1]
                   Negative values: drop sand (magnitude determines amount)
                   Positive values: vent gas (magnitude determines rate)
        """
        # Generate a single random action in range [-1, 1]
        action = np.random.uniform(-1.0, 1.0)
        action = 0
        return np.array([action])  # Return as numpy array for compatibility 