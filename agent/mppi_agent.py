import numpy as np
from typing import Tuple, List
import copy
import scipy

class MPPIAgent:
    """
    Model Predictive Path Integral (MPPI) agent for balloon navigation.
    
    The agent samples multiple control sequences, evaluates them through the environment,
    and returns the optimal action based on weighted averaging of the best trajectories.
    """
    
    def __init__(self, 
                 horizon: int = 10,
                 num_samples: int = 100,
                 temperature: float = 1.0,
                 noise_std: float = 0.5,
                 action_bounds: Tuple[float, float] = (-1.0, 1.0)):
        """
        Initialize MPPI agent.
        
        Args:
            horizon: Planning horizon (number of steps to look ahead)
            num_samples: Number of control sequences to sample
            temperature: Temperature parameter for trajectory weighting (β in exp(-β*S))
            noise_std: Standard deviation of noise for control sampling
            action_bounds: (min_action, max_action) for vertical velocity control
        """
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.noise_std = noise_std
        self.action_bounds = action_bounds
        
        # Initialize control sequence (all zeros initially)
        self.control_sequence = np.zeros(horizon)
        
    def select_action(self, state: np.ndarray, env) -> np.ndarray:
        """
        Select optimal action using MPPI algorithm.
        
        Args:
            state: Current state of the environment
            env: Environment instance for rollouts
            
        Returns:
            Optimal action as numpy array
        """
        # Sample control sequences
        self.vertical_velocity = copy.deepcopy(env.balloon.vertical_velocity)
        acc_samples, vel_samples = self._sample_control_sequences()
        
        # Evaluate each control sequence
        costs = []
        for i in range(self.num_samples):
            cost = self._evaluate_control_sequence(acc_samples[i], vel_samples[i], state, env)
            costs.append(cost)
        
        # Convert costs to weights
        costs = np.array(costs)
        
        # Check for NaN costs and replace with high cost
        if np.any(~np.isfinite(costs)):
            print("Warning: NaN costs detected, replacing with high cost")
            costs = np.where(np.isfinite(costs), costs, 1e6)
        
        weights = self._compute_weights(costs)
        
        # Check for NaN weights
        if np.any(~np.isfinite(weights)):
            print("Warning: NaN weights detected, using uniform weights")
            weights = np.ones_like(weights) / len(weights)
        
        # Compute optimal control by weighted averaging
        optimal_acc = np.average(acc_samples, axis=0, weights=weights)
        #print(f"{int(len(optimal_acc)/2)}, {int(len(optimal_acc)/4)}")
        # if len(optimal_acc) >= 4:
        #     optimal_acc = scipy.signal.savgol_filter(optimal_acc, int(len(optimal_acc)/2), int(len(optimal_acc)/4))
        
        # Check for NaN in optimal control
        if not np.isfinite(optimal_acc[0]):
            print("Warning: NaN optimal control detected, using zero action")
            optimal_acc[0] = 0.0
        
        # Update control sequence (shift and append)
        # min_cost_idx = np.argmin(costs)
        # self.control_sequence = np.roll(self.control_sequence, -1)
        self.control_sequence = np.roll(optimal_acc, -1)
        self.control_sequence[-1] = optimal_acc[0]
        
        optimal_vel = np.array([self.vertical_velocity + optimal_acc[0]])
        # Return first action from optimal sequence
        # return np.array([optimal_acc[0]])
        return optimal_vel
    
    def _sample_control_sequences(self) -> np.ndarray:
        """
        Sample control sequences by adding noise to the current control sequence.
        
        Returns:
            Array of shape (num_samples, horizon) containing control sequences
        """
        # Start with current control sequence
        base_sequence = self.control_sequence.copy()
        
        # Add noise to create samples
        noise = np.random.normal(0, self.noise_std, (self.num_samples, self.horizon))
    
        acc_samples = base_sequence + noise
        acc_samples = np.clip(acc_samples, self.acc_bounds[0], self.acc_bounds[1])
        vel_samples = self.vertical_velocity + acc_samples
        # Clip to action bounds
        vel_samples = np.clip(vel_samples, self.vel_bounds[0], self.vel_bounds[1])
        
        return acc_samples, vel_samples
    
    
    def _evaluate_control_sequence(self, control_seq: np.ndarray, 
                                 initial_state: np.ndarray, env) -> float:
        """
        Evaluate a control sequence by rolling it out through the environment.
        
        Args:
            control_seq: Control sequence to evaluate
            initial_state: Initial state of the environment
            env: Environment instance
            
        Returns:
            Total cost of the trajectory
        """
        # Create a lightweight copy of the environment (reuses wind_field)
        
        env_copy = env.shallow_copy()
        total_cost = 0.0
        
        # Rollout the control sequence
        for t in range(min(self.horizon, len(control_seq))):
            action = np.array([control_seq[t]])
            
            # Take step in environment
            next_state, reward, done, _ = env_copy.step(action)
            
            # Accumulate cost (negative reward)
            total_cost -= reward
            
            # Early termination if episode ends
            if done:
                break
        
        return total_cost
    
    def _compute_weights(self, costs: np.ndarray) -> np.ndarray:
        """
        Compute weights for control sequences based on their costs.
        
        Args:
            costs: Array of costs for each control sequence
            
        Returns:
            Array of weights for each control sequence
        """
        # Shift costs to prevent numerical issues
        costs_shifted = costs - np.min(costs)
        
        # Compute weights using softmax
        weights = np.exp(-self.temperature * costs_shifted)
        weights = weights / np.sum(weights)
        
        
        return weights
    
    def reset(self):
        """Reset the agent's internal state."""
        self.control_sequence = np.zeros(self.horizon)


class MPPIAgentWithCostFunction(MPPIAgent):
    """
    Enhanced MPPI agent with custom cost function for balloon navigation.
    """
    
    def __init__(self, 
                 horizon: int = 1,
                 num_samples: int = 100,
                 temperature: float = 1.0,
                 noise_std: float = 0.1,
                 acc_bounds: Tuple[float, float] = (-0.1, 0.1),
                 vel_bounds: Tuple[float, float] = (-1.0, 1.0),
                 target_lat: float = 500.0,
                 target_lon: float = -100.0,
                 target_alt: float = 12.0):
        """
        Initialize MPPI agent with custom cost function.
        
        Args:
            horizon: Planning horizon
            num_samples: Number of control sequences to sample
            temperature: Temperature parameter for trajectory weighting
            noise_std: Standard deviation of noise for control sampling
            action_bounds: (min_action, max_action) for vertical velocity control
            target_lat: Target latitude
            target_lon: Target longitude
            target_alt: Target altitude
        """
        super().__init__(horizon, num_samples, temperature, noise_std, vel_bounds)
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.target_alt = target_alt
        self.acc_bounds = acc_bounds
        self.vel_bounds = vel_bounds
        self.vertical_velocity = 0.0
    
    def _evaluate_control_sequence(self, acc_seq: np.ndarray, vel_seq: np.ndarray, 
                                 initial_state: np.ndarray, env) -> float:
        """
        Evaluate control sequence with custom cost function for balloon navigation.
        
        Args:
            control_seq: Control sequence to evaluate
            initial_state: Initial state of the environment
            env: Environment instance
            
        Returns:
            Total cost of the trajectory
        """
        # Create a lightweight copy of the environment (reuses wind_field)
        
        env_copy = env.shallow_copy()
        total_cost = 0.0
        
        # Rollout the control sequence
        lat_diff = env.balloon.lat - self.target_lat
        lon_diff = env.balloon.lon - self.target_lon
        distance = np.sqrt(lat_diff**2 + lon_diff**2)
        self.initial_goal_cost = distance
        for t in range(min(self.horizon, len(vel_seq))):
            action = np.array([vel_seq[t]])
            acc = np.array([acc_seq[t]])
            # Take step in environment
            next_state, _, done, _ = env_copy.step(action)
            # Custom cost function
            cost = self._compute_step_cost(next_state, acc[0], t)
            total_cost += cost
            
            # Early termination if episode ends
            if done:
                # Add penalty for early termination
                total_cost += 1000.0
                break
        
        return total_cost
    
    def _compute_step_cost(self, state: np.ndarray, acc: float, step: int) -> float:
        """
        Compute cost for a single step.
        
        Args:
            state: Current state [lat, lon, alt, volume, sand, vel, time, ...]
            acc:  (vertical acceleration)
            step: Current step in the sequence
            
        Returns:
            Cost for this step
        """
        # Extract state components
        w1,w2,w3,w4,w5 = 30,25,1,1,1
        lat, lon, alt = state[0], state[1], state[2]
        volume_ratio, sand_ratio = state[3], state[4]
        
        # Distance to target
        lat_diff = lat - self.target_lat
        lon_diff = lon - self.target_lon
        distance = np.sqrt(lat_diff**2 + lon_diff**2)
        distance_cost = distance - self.initial_goal_cost
        # Altitude deviation from target
        alt_diff = abs(alt - self.target_alt)
        
        # Resource penalty (encourage conservation)
        resource_penalty = 0.1 * (1.0 - volume_ratio) + 0.1 * (1.0 - sand_ratio)
        
        # Action penalty (encourage smooth control)
        acc_penalty = acc**2
        
        # Time penalty (encourage efficiency)
        time_penalty = 0.01 * step
        
        # Total cost
        cost = (w1*distance_cost + 
                w2*(alt_diff)**2 + 
                w3*resource_penalty + 
                w4*acc_penalty + 
                w5*time_penalty)
        
        return cost