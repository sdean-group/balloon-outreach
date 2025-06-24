import numpy as np

class RandomAgent:
    """A simple agent that selects random actions"""
    
    def select_action(self, state: np.ndarray, max_steps: int=100, time: int=0) -> np.ndarray:
        """
        Select a random action
        
        Args:
            state: Current state of the environment
            
        Returns:
            action: Single continuous action in range [-1, 1] [m/s]
                   
        """
        # Generate a single random action in range [-1, 1]
        action = np.random.uniform(-1.0, 1.0)
        action = np.sin(time*3/max_steps*np.pi)
        return np.array([action])  # Return as numpy array for compatibility 