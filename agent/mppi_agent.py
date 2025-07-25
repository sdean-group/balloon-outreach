import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from env.balloon_env import BaseBalloonEnvironment
from mpl_toolkits import mplot3d
from env.util import haversine_distance
import time
class MPPIAgent:
    """
    Model Predictive Path Integral (MPPI) agent for balloon navigation.
    
    The agent samples multiple control sequences, evaluates them through the environment,
    and returns the optimal action based on weighted averaging of the best trajectories.
    """
    
    def __init__(self, 
                 horizon: int = 10,
                 num_samples: int = 100,
                 num_iterations: int = 1,
                 temperature: float = 1.0,
                 noise_std: float = 0.1,
                 acc_bounds: Tuple[float, float] = (-0.1, 0.1),
                 vel_bounds: Tuple[float, float] = (-1.0, 1.0),
                 visualize: bool = False,
                 objective:str = 'target'):
        """
        Initialize MPPI agent.
        
        Args:
            horizon: Planning horizon (number of steps to look ahead)
            num_samples: Number of control sequences to sample
            num_iterations: Number of times to update optimal control sequence
            temperature: Temperature parameter for trajectory weighting (β in exp(-β*S))
            noise_std: Standard deviation of noise for control sampling
            acc_bounds: (min_amt, max_amt) for acceleration control
            vel_bounds: (min_action, max_action) for vertical velocity control
            visualize: Whether or not to visualize planned trajectories
            objective: Should either be 'target' or 'fly'. Indicates which task the agent is completing
        """
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.noise_std = noise_std
        self.acc_bounds = acc_bounds
        self.vel_bounds = vel_bounds
        self.visualize = visualize
        self.objective = objective
        self.actual_trajectory = []
        self.view_angle = (30, -37.5)
        
        # Initialize control sequence (all zeros initially)
        self.control_sequence = np.zeros(horizon)
        
    def select_action(self, state: np.ndarray, env:BaseBalloonEnvironment, step_num: int = 0) -> np.ndarray:
        """
        Select optimal action using MPPI algorithm.
        
        Args:
            state: Current state of the environment
            env: Environment instance for rollouts
            
        Returns:
            Optimal action as numpy array
        """
        self.actual_trajectory.append([env.balloon.lat, env.balloon.lon, env.balloon.alt])
        self.vertical_velocity = env.balloon.vertical_velocity
        optimal_acc_idx = step_num % self.num_iterations
        # Run new MPPI loop 
        if optimal_acc_idx == 0:
            self.control_sequence = np.roll(self.control_sequence, -self.num_iterations)
            self.control_sequence[-self.num_iterations:] = self.control_sequence[np.max([-self.horizon, -self.num_iterations-1])]
            acc_samples, vel_samples = self._sample_control_sequences()
            
            # Evaluate each control sequence
            costs = []
            trajectories = []
            start_time = time.time()
            for i in range(self.num_samples):
                cost, trajectory = self._evaluate_control_sequence(acc_samples[i], vel_samples[i], state, env)
                # clean_traj = [
                #     (round(float(lat), 2), round(float(lon), 2), round(float(alt), 2))
                #     for lat, lon, alt in trajectory
                # ]
                clean_traj = [
                    round(float(vel), 2)
                    for vel in vel_samples[i]
                ]
                # print(f"sample {i} \n cost: {cost:.2f}\n vel: {clean_traj}\n ")
                # print(f"sample {i} \n cost: {cost:.2f}\n")
                costs.append(cost)
                trajectories.append(trajectory)
            end_time = time.time()
            #print(f"MPPI evaluation time took: {end_time - start_time}")
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
        
            # Check for NaN in optimal control
            if not np.isfinite(optimal_acc[0]):
                print("Warning: NaN optimal control detected, using zero action")
                optimal_acc[0] = 0.0
            
            # Update control sequence (shift and append)
            # min_cost_idx = np.argmin(costs)
            # self.control_sequence = np.roll(self.control_sequence, -1)
            # self.control_sequence = np.roll(optimal_acc, -1)
            #roll through optimal acc. but then we add 1 from acc to the running contorl seq
            # self.control_sequence[-1] = optimal_acc[0]
            self.control_sequence = optimal_acc
        if self.visualize:
            final = []
            curr_velocity = self.vertical_velocity
            final.append(curr_velocity)
            for acc in self.control_sequence:
                curr_velocity += acc
                final.append(curr_velocity)
            _, control_trajectory = self._evaluate_control_sequence(optimal_acc, final, state, env)
            target_state = [env.target_lat, env.target_lon, env.target_alt]
            self._visualize_trajectories(target_state, env.init_state, trajectories, control_trajectory, self.actual_trajectory)

        optimal_vel = np.array([self.vertical_velocity + self.control_sequence[optimal_acc_idx]])
        # Return first action from optimal sequence
        # return np.array([optimal_acc[0]])
        return optimal_vel
    
    def _sample_control_sequences(self) -> np.ndarray:
        """
        Sample control sequences by adding noise to the current control sequence. Velocity samples are created by accumulating acceleration samples that were generated from a normal distribution.
        
        Returns:
            acc_samples: Array of shape (num_samples, horizon) containing acceleration sequences
            vel_samples: Array of shape (num_samples, horizon) containing velocity sequences
        """
        # Start with current control sequence
        base_sequence = self.control_sequence.copy()
        
        # Add noise to create samples
        noise = np.random.normal(0, self.noise_std, (self.num_samples, self.horizon))
    
        acc_samples = base_sequence + noise
        acc_samples = np.clip(acc_samples, self.acc_bounds[0], self.acc_bounds[1])
        # vel_samples = self.vertical_velocity + acc_samples
        accumulated_acc_samples = np.cumsum(acc_samples, axis=1)
        vel_samples = self.vertical_velocity + accumulated_acc_samples
        # Clip to action bounds
        vel_samples = np.clip(vel_samples, self.vel_bounds[0], self.vel_bounds[1])
        
        return acc_samples, vel_samples
    
    def _evaluate_control_sequence(self, acc_seq: np.ndarray, control_seq: np.ndarray, 
                                 initial_state: np.ndarray, env:BaseBalloonEnvironment) -> Tuple[float, List[float]]:
        """
        Evaluate a control sequence by rolling it out through the environment.
        
        Args:
            acc_seq: Sequence of changes in velocity (unused)
            control_seq: Control sequence to evaluate
            initial_state: Initial state of the environment (unused)
            env: Environment instance
            
        Returns:
            Total cost of the trajectory, trajectory
        """
        return env.rollout_sequence_mppi(control_seq, min(self.horizon, len(control_seq)))

    
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

    def _visualize_trajectories(self, target_state:np.ndarray, init_state:np.ndarray, trajectories: np.ndarray, final_trajectory: np.ndarray, actual_trajectory: np.ndarray=None) -> None:
        """
        Visualize the sampled trajectories and the final average trajectory for one step.
        
        Args:
            target_state: The target point to reach (lat, lon, alt) if we are using the target objective
            init_state: The balloon's original starting point
            trajectories: List of sampled control sequences
            final_trajectory: Weight average control sequence as a list of tuples [(lat1,lon1,alt1),(lat2,lon2,alt2)...]
            view_angle: Tuple specifying the 3D view angle (elev, azim)
            
        """
        # Plot final trajectory
        plt.figure(figsize=(12, 5))
        ax = plt.axes(projection='3d')

        # Set custom view angle
        ax.view_init(elev=self.view_angle[0], azim=self.view_angle[1])
        
        # Position plot
        for trajectory in trajectories:
            lats, lons, alts = zip(*trajectory)
            ax.plot3D(lons, lats,alts, 'b-', alpha=0.3)
        lats,lons,alts = zip(*final_trajectory)
        ax.plot3D(lons, lats,alts, 'r-', alpha=1)
        plt.plot(lons[0], lats[0],alts[0], 'bo', label='Current Point')
        plt.plot(init_state[1], init_state[0], init_state[2], 'go', label='Start')
        plt.plot(init_state[1], init_state[0], np.linspace(0, init_state[2], 10), 'g-')
        plt.plot(lons[0], lats[0], np.linspace(0, alts[0], 10), 'b-')
        if self.objective == 'target':
            plt.plot(target_state[1], target_state[0], target_state[2], 'rx', label='Target End')
        if actual_trajectory is not None:
            self.actual_trajectory.append([lats[0], lons[0], alts[0]])
            lats, lons, alts = zip(*actual_trajectory)
            plt.plot(lons, lats, alts, 'y-', alpha=1, label='Trajectory')
        plt.grid(True)
        ax.set_title(f'Balloon Trajectory with MPPI')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('trajectory.png')
        plt.show()
        plt.close()
    
    def reset(self):
        """Reset the agent's internal state."""
        self.actual_trajectory = []
        self.control_sequence = np.zeros(self.horizon)


class MPPIAgentWithCostFunction(MPPIAgent):
    """
    Enhanced MPPI agent with custom cost function for balloon navigation.
    """
    
    def __init__(self, 
                 horizon: int = 10,
                 num_samples: int = 10,
                 num_iterations:int = 1,
                 temperature: float = 10,
                 noise_std: float = 0.1,
                 acc_bounds: Tuple[float, float] = (-0.1, 0.1),
                 vel_bounds: Tuple[float, float] = (-1.0, 1.0),
                 target_lat: float = 500.0,
                 target_lon: float = -100.0,
                 target_alt: float = 12.0,
                 visualize: bool = False,
                 objective:str = 'target'):
        """
        Initialize MPPI agent with custom cost function.
        
        Args:
            horizon: Planning horizon
            num_samples: Number of control sequences to sample
            num_iterations: Number of times to update optimal control sequence
            temperature: Temperature parameter for trajectory weighting
            noise_std: Standard deviation of noise for control sampling
            action_bounds: (min_action, max_action) for vertical velocity control
            target_lat: Target latitude
            target_lon: Target longitude
            target_alt: Target altitude
            visualize: Whether or not to visualize planned trajectories
            objective: Should either be 'target' or 'fly'. Indicates which task the agent is completing
        """
        super().__init__(horizon, num_samples, num_iterations, temperature, noise_std, acc_bounds, vel_bounds, visualize,objective)
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.target_alt = target_alt
        self.acc_bounds = acc_bounds
        self.vel_bounds = vel_bounds
        self.vertical_velocity = 0.0
    
    def _evaluate_control_sequence(self, acc_seq: np.ndarray, vel_seq: np.ndarray, 
                                 initial_state: np.ndarray, env:BaseBalloonEnvironment, visualize:bool=False) -> Tuple[float, List[float]]:
        """
        Evaluate control sequence with custom cost function for balloon navigation.
        
        Args:
            acc_seq: Sequence of changes in velocity
            vel_seq: Velocity sequence to evaluate
            initial_state: Initial state of the environment
            env: Environment instance
            
        Returns:
            Total cost of the trajectory, trajectory
        """
        if self.objective == 'target':
            target_state = np.array([env.target_lat, env.target_lon, env.target_alt])
            # lat_diff = env.balloon.lat - env.target_lat
            # lon_diff = env.balloon.lon - env.target_lon
            # initial_goal_cost = np.sqrt(lat_diff**2 + lon_diff**2)
            initial_goal_cost = haversine_distance(env.balloon.lon, env.balloon.lat, env.target_lon, env.target_lat)
            
            return env.rollout_sequence_mppi_with_cost(vel_seq, acc_seq, min(self.horizon, len(vel_seq)), self._compute_step_cost_target2, target_state, initial_goal_cost)
        else:
            cost, trajectory = env.rollout_sequence_mppi_with_cost(vel_seq, acc_seq, min(self.horizon, len(vel_seq)), self._compute_step_cost_fly2, env.init_state)
            # Discourage looping
            backward_penalty = 0
            for lat,lon, alt in trajectory:
                prev_dx = lon - env.balloon.lon
                prev_dy = lat - env.balloon.lat
                prev_r = np.sqrt(prev_dx**2 + prev_dy**2)
                dr = cost - prev_r
            # Penalize backward motion
                backward_penalty += -min(0, dr)  # Only nonzero if dr < 0
            
            return cost + 10*backward_penalty, trajectory
    
    def _compute_step_cost_target(self, target_state:np.ndarray, initial_goal_cost:float, state: np.ndarray, acc: float, step: int) -> float:
        """
        Compute cost for a single step with the target objective.
        
        Args:
            target_state: Target balloon state [target_lat, target_lon, target_alt]
            initial_goal_cost: Euclidean distance to target point from current state
            state: Current state [lat, lon, alt, volume, sand, vel, time, ...]
            acc:  (vertical acceleration)
            step: Current step in the sequence
            
        Returns:
            Cost for this step
        """
        # Extract state components
        w1,w2,w3,w4,w5 = 30,0.1,1,1,1
        lat, lon, alt = state[0], state[1], state[2]
        volume_ratio, sand_ratio = state[3], state[4]
        
        # Euclidean Distance to target
        lat_diff = lat - target_state[0]
        lon_diff = lon - target_state[1]
        distance = np.sqrt(lat_diff**2 + lon_diff**2)
        # distance_cost = distance
        distance_cost = distance - initial_goal_cost
        # Altitude deviation from target. Low weight to prioritize selection of best altitude
        alt_diff = (abs(alt - target_state[2]))**2
        
        # Resource penalty (encourage conservation)
        resource_penalty = 0.1 * (1.0 - volume_ratio) + 0.1 * (1.0 - sand_ratio)
        
        # Action penalty (encourage smooth control)
        acc_penalty = acc**2
        
        # Time penalty (encourage efficiency)
        time_penalty = 0.01 * step
        
        # Total cost
        cost = (w1*distance_cost + 
                w2*alt_diff + 
                w3*resource_penalty + 
                w4*acc_penalty + 
                w5*time_penalty)
        
        return cost
    def _compute_step_cost_target2(self, target_state:np.ndarray, initial_goal_cost:float, state: np.ndarray, acc: float, step: int) -> float:
        """
        Compute cost for a single step with the target objective.
        
        Args:
            target_state: Target balloon state [target_lat, target_lon, target_alt]
            initial_goal_cost: Euclidean distance to target point from current state
            state: Current state [lat, lon, alt, volume, sand, vel, time, ...]
            acc:  (vertical acceleration)
            step: Current step in the sequence
            
        Returns:
            Cost for this step
        """
        # Extract state components
        w1,w2,w3,w4,w5 = 30,1,1,1,1
        lat, lon, alt = state[0], state[1], state[2]
        volume_ratio, sand_ratio = state[3], state[4]
        
        # Euclidean Distance to target
        lat_diff = lat - target_state[0]
        lon_diff = lon - target_state[1]
        distance = haversine_distance(target_state[1], target_state[0], lon,lat)
   
        distance_cost = distance - initial_goal_cost
        # Penalize altitudes close to limits
        if alt >= 18 or alt <= 8:
            alt_cost = 10
        else:
            alt_cost = 0
        # Resource penalty (encourage conservation)
        resource_penalty = 0.1 * (1.0 - volume_ratio) + 0.1 * (1.0 - sand_ratio)
        
        # Action penalty (encourage smooth control)
        acc_penalty = acc**2
        
        # Time penalty (encourage efficiency)
        time_penalty = 0.01 * step
        
        # Total cost
        cost = (w1*distance_cost + 
                w2*alt_cost + 
                w3*resource_penalty + 
                w4*acc_penalty + 
                w5*time_penalty)
        
        return cost
    
    def _compute_step_cost_fly(self, init_state:np.ndarray, state: np.ndarray, acc: float, step: int) -> float:
        """
        Compute cost for a single step with the fly as far objective.
        
        Args:
            init_state: Initial balloon state [init_lat, init_lon, init_alt]
            state: Current state [lat, lon, alt, volume, sand, vel, time, ...]
            acc:  (vertical acceleration)
            step: Current step in the sequence
            
        Returns:
            Cost for this step
        """
        # Extract state components
        w1,w2,w3,w4,w5 = -50,-0.1,0,1,5
        lat, lon, alt = state[0], state[1], state[2]
        volume_ratio, sand_ratio = state[3], state[4]
        
        
        # Encourage distance away from init to target
        distance = haversine_distance(init_state[1], init_state[0], lon, lat)
        # Encourage altitude deviation from target. 
        alt_diff = (alt - init_state[2])**2
        
        # Resource penalty (encourage conservation)
        resource_penalty = 0.1 * (1.0 - volume_ratio) + 0.1 * (1.0 - sand_ratio)
        
        # Action penalty (encourage smooth control)
        acc_penalty = acc**2
        
        # Time penalty (encourage efficiency)
        time_penalty = 0.01 * step
        
        # Total cost
        cost = (w1*distance + 
                w2*alt_diff + 
                w3*resource_penalty + 
                w4*acc_penalty + 
                w5*time_penalty)
        
        return cost
    
    def _compute_step_cost_fly2(self, init_state:np.ndarray, state: np.ndarray, acc: float, step: int) -> float:
        """
        Compute cost for a single step with the fly as far objective.
        
        Args:
            init_state: Initial balloon state [init_lat, init_lon, init_alt]
            state: Current state [lat, lon, alt, volume, sand, vel, time, ...]
            acc:  (vertical acceleration)
            step: Current step in the sequence
            
        Returns:
            Cost for this step
        """
        # Extract state components
        if not hasattr(self, 'volume_ratio_prev'):
            self.volume_ratio_prev = state[3]
            self.sand_ratio_prev = state[4]

        
        w1,w2,w3,w4,w5 = 10,1,10,1,0
        lat, lon, alt = state[0], state[1], state[2]
        volume_ratio, sand_ratio = state[3], state[4]
        
        
        # Encourage distance away from init to target
        distance = 1/max(haversine_distance(init_state[1], init_state[0], lon, lat),1)
        # Encourage altitude deviation from target. 
        if alt >= 18 or alt <= 8:
            alt_cost = 10
        else:
            alt_cost = 0
        # Resource penalty (encourage conservation)
        # resource_penalty = (1.0 - volume_ratio) + (1.0 - sand_ratio)
        resource_penalty =(self.volume_ratio_prev - volume_ratio + self.sand_ratio_prev - sand_ratio)
        # Action penalty (encourage smooth control)
        acc_penalty = acc**2
        
        # Time penalty (encourage efficiency)
        time_penalty = 0.01 * step
        
        # Total cost
        cost = (w1*distance + 
                w2*alt_cost + 
                w3*resource_penalty + 
                w4*acc_penalty + 
                w5*time_penalty)
        self.volume_ratio_prev = volume_ratio
        self.sand_ratio_prev = sand_ratio
        return cost
    