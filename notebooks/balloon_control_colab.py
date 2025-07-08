# Balloon Navigation Control Demo for Google Colab
# Copy each section into separate cells in Colab

# ============================================================================
# CELL 1: Setup and Installation
# ============================================================================

# Install required packages
!pip install numpy matplotlib xarray netcdf4

# ============================================================================
# CELL 2: Imports
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import Dict, List, Tuple, Optional
import math

# For visualization
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# CELL 3: Tree Search Configuration
# ============================================================================

# Tree Search Configuration Parameters
class TreeSearchConfig:
    def __init__(self):
        # Search parameters
        self.max_iterations = 1000
        self.max_search_time = 60.0  # seconds
        
        # State discretization
        self.lat_decimals = 2      # Latitude precision (degrees)
        self.lon_decimals = 2      # Longitude precision (degrees)
        self.alt_decimals = 1      # Altitude precision (km)
        
        # Goal tolerance
        self.lat_tolerance = 0.01  # degrees
        self.lon_tolerance = 0.01  # degrees
        self.alt_tolerance = 0.02  # km
        
        # Distance metrics
        self.distance_metric = 'haversine'  # 'euclidean' or 'haversine'
        self.heuristic_type = 'haversine'   # 'euclidean', 'haversine', or 'zero'
        
        # Action space
        self.actions = ['stay', 'ascend', 'descend']
        self.action_values = {'stay': 0.0, 'ascend': 1.0, 'descend': -1.0}
        
        # Cost weights
        self.distance_weight = 1.0
        self.altitude_change_weight = 0.5
        self.time_weight = 0.1

# Initialize configuration
tree_config = TreeSearchConfig()
print("Tree Search Configuration:")
for key, value in tree_config.__dict__.items():
    print(f"  {key}: {value}")

# ============================================================================
# CELL 4: Tree Search Utility Functions
# ============================================================================

# Tree Search Utility Functions
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in meters."""
    R = 6371e3  # Earth radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2) ** 2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def euclidean_distance(lat1: float, lon1: float, alt1: float, 
                      lat2: float, lon2: float, alt2: float) -> float:
    """Calculate Euclidean distance in 3D space."""
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2 + (alt1 - alt2) ** 2)

def discretize_state(state: np.ndarray, config: TreeSearchConfig) -> tuple:
    """Discretize state for tree search."""
    lat, lon, alt, t = state
    discretized = (
        round(lat, config.lat_decimals),
        round(lon, config.lon_decimals),
        round(alt, config.alt_decimals),
        t  # Keep time as is
    )
    return discretized

def is_goal_reached(state: np.ndarray, target: np.ndarray, config: TreeSearchConfig) -> bool:
    """Check if current state is close enough to target."""
    lat, lon, alt = state[:3]
    target_lat, target_lon, target_alt = target[:3]
    
    return (abs(lat - target_lat) <= config.lat_tolerance and
            abs(lon - target_lon) <= config.lon_tolerance and
            abs(alt - target_alt) <= config.alt_tolerance)

# ============================================================================
# CELL 5: Tree Search Agent
# ============================================================================

# Simplified Tree Search Agent
class TreeSearchAgent:
    def __init__(self, config: TreeSearchConfig):
        self.config = config
        self.explored_states = set()
        
    def heuristic(self, state: np.ndarray, target: np.ndarray) -> float:
        """Calculate heuristic value for A* search."""
        lat, lon, alt = state[:3]
        target_lat, target_lon, target_alt = target[:3]
        
        if self.config.heuristic_type == 'haversine':
            return haversine_distance(lat, lon, target_lat, target_lon)
        elif self.config.heuristic_type == 'euclidean':
            return euclidean_distance(lat, lon, alt, target_lat, target_lon, target_alt)
        else:  # zero heuristic
            return 0.0
    
    def get_action_cost(self, action: str, current_alt: float, new_alt: float) -> float:
        """Calculate cost of taking an action."""
        base_cost = 1.0
        altitude_change = abs(new_alt - current_alt)
        return base_cost + self.config.altitude_change_weight * altitude_change
    
    def find_path(self, start_state: np.ndarray, target_state: np.ndarray) -> List[Tuple]:
        """Find optimal path using A* search."""
        # Simplified A* implementation for demonstration
        open_set = [(0, tuple(start_state))]  # (f_score, state)
        came_from = {tuple(start_state): None}
        g_score = {tuple(start_state): 0}
        f_score = {tuple(start_state): self.heuristic(start_state, target_state)}
        
        iterations = 0
        while open_set and iterations < self.config.max_iterations:
            current_f, current_state = min(open_set)
            open_set.remove((current_f, current_state))
            
            current_array = np.array(current_state)
            
            if is_goal_reached(current_array, target_state, self.config):
                return self._reconstruct_path(came_from, current_state)
            
            # Generate neighbors (simplified)
            for action in self.config.actions:
                # Simulate action (simplified)
                new_state = self._simulate_action(current_array, action)
                new_state_tuple = tuple(new_state)
                
                tentative_g = g_score[current_state] + self.get_action_cost(
                    action, current_array[2], new_state[2])
                
                if new_state_tuple not in g_score or tentative_g < g_score[new_state_tuple]:
                    came_from[new_state_tuple] = (current_state, action)
                    g_score[new_state_tuple] = tentative_g
                    f_score[new_state_tuple] = tentative_g + self.heuristic(new_state, target_state)
                    
                    if new_state_tuple not in [s[1] for s in open_set]:
                        open_set.append((f_score[new_state_tuple], new_state_tuple))
            
            iterations += 1
        
        return []  # No path found
    
    def _simulate_action(self, state: np.ndarray, action: str) -> np.ndarray:
        """Simulate the effect of an action on the state."""
        lat, lon, alt, t = state
        action_value = self.config.action_values[action]
        
        # Simplified physics simulation
        new_alt = alt + action_value * 0.1  # 0.1 km per step
        new_lat = lat + np.random.normal(0, 0.01)  # Wind drift
        new_lon = lon + np.random.normal(0, 0.01)  # Wind drift
        new_t = t + 0.016  # Time step (1 minute)
        
        return np.array([new_lat, new_lon, new_alt, new_t])
    
    def _reconstruct_path(self, came_from: Dict, goal_state: tuple) -> List[Tuple]:
        """Reconstruct the path from start to goal."""
        path = []
        current = goal_state
        
        while current is not None:
            path.append(current)
            if current in came_from:
                current = came_from[current][0] if came_from[current] else None
            else:
                break
        
        return path[::-1]

# ============================================================================
# CELL 6: PID Control Configuration
# ============================================================================

# PID Control Configuration
class PIDConfig:
    def __init__(self):
        # PID gains for latitude control
        self.lat_kp = 1.0
        self.lat_ki = 0.1
        self.lat_kd = 0.5
        
        # PID gains for longitude control
        self.lon_kp = 1.0
        self.lon_ki = 0.1
        self.lon_kd = 0.5
        
        # PID gains for altitude control
        self.alt_kp = 2.0
        self.alt_ki = 0.2
        self.alt_kd = 1.0
        
        # Control limits
        self.max_altitude_change = 0.5  # km per step
        self.min_altitude = 5.0         # km
        self.max_altitude = 30.0        # km
        
        # Time parameters
        self.dt = 0.016  # Time step (1 minute)
        self.max_time = 3600  # Maximum simulation time (1 hour)
        
        # Wind parameters (simplified)
        self.wind_lat_mean = 0.0
        self.wind_lat_std = 0.01
        self.wind_lon_mean = 0.0
        self.wind_lon_std = 0.01
        
        # Convergence criteria
        self.position_tolerance = 0.01  # degrees
        self.altitude_tolerance = 0.02  # km
        self.velocity_tolerance = 0.001  # degrees/min

# Initialize PID configuration
pid_config = PIDConfig()
print("PID Control Configuration:")
for key, value in pid_config.__dict__.items():
    print(f"  {key}: {value}")

# ============================================================================
# CELL 7: PID Controller Implementation
# ============================================================================

# PID Controller Implementation
class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        
        # State variables
        self.previous_error = 0.0
        self.integral = 0.0
        
    def compute(self, setpoint: float, measured_value: float) -> float:
        """Compute PID control output."""
        error = setpoint - measured_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * self.dt
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / self.dt
        d_term = self.kd * derivative
        
        # Update previous error
        self.previous_error = error
        
        # Total output
        output = p_term + i_term + d_term
        
        return output
    
    def reset(self):
        """Reset controller state."""
        self.previous_error = 0.0
        self.integral = 0.0

# Multi-axis PID Controller for Balloon
class BalloonPIDController:
    def __init__(self, config: PIDConfig):
        self.config = config
        
        # Initialize PID controllers for each axis
        self.lat_controller = PIDController(
            config.lat_kp, config.lat_ki, config.lat_kd, config.dt)
        self.lon_controller = PIDController(
            config.lon_kp, config.lon_ki, config.lon_kd, config.dt)
        self.alt_controller = PIDController(
            config.alt_kp, config.alt_ki, config.alt_kd, config.dt)
        
        # State history
        self.state_history = []
        self.control_history = []
        
    def compute_control(self, current_state: np.ndarray, target_state: np.ndarray) -> float:
        """Compute control output for altitude adjustment."""
        current_lat, current_lon, current_alt = current_state[:3]
        target_lat, target_lon, target_alt = target_state[:3]
        
        # Compute control signals
        lat_control = self.lat_controller.compute(target_lat, current_lat)
        lon_control = self.lon_controller.compute(target_lon, current_lon)
        alt_control = self.alt_controller.compute(target_alt, current_alt)
        
        # Combine controls (simplified - only altitude control for now)
        # In practice, you might use different control strategies
        combined_control = alt_control
        
        # Apply limits
        combined_control = np.clip(combined_control, -self.config.max_altitude_change, 
                                   self.config.max_altitude_change)
        
        return combined_control
    
    def simulate_step(self, current_state: np.ndarray, target_state: np.ndarray) -> np.ndarray:
        """Simulate one control step."""
        # Compute control
        control = self.compute_control(current_state, target_state)
        
        # Update state
        lat, lon, alt, t = current_state
        
        # Apply control to altitude
        new_alt = alt + control * self.config.dt
        new_alt = np.clip(new_alt, self.config.min_altitude, self.config.max_altitude)
        
        # Add wind effects
        wind_lat = np.random.normal(self.config.wind_lat_mean, self.config.wind_lat_std)
        wind_lon = np.random.normal(self.config.wind_lon_mean, self.config.wind_lon_std)
        
        new_lat = lat + wind_lat * self.config.dt
        new_lon = lon + wind_lon * self.config.dt
        new_t = t + self.config.dt
        
        new_state = np.array([new_lat, new_lon, new_alt, new_t])
        
        # Store history
        self.state_history.append(new_state.copy())
        self.control_history.append(control)
        
        return new_state
    
    def is_target_reached(self, current_state: np.ndarray, target_state: np.ndarray) -> bool:
        """Check if target has been reached."""
        current_lat, current_lon, current_alt = current_state[:3]
        target_lat, target_lon, target_alt = target_state[:3]
        
        lat_error = abs(current_lat - target_lat)
        lon_error = abs(current_lon - target_lon)
        alt_error = abs(current_alt - target_alt)
        
        return (lat_error <= self.config.position_tolerance and
                lon_error <= self.config.position_tolerance and
                alt_error <= self.config.altitude_tolerance)
    
    def reset(self):
        """Reset all controllers and history."""
        self.lat_controller.reset()
        self.lon_controller.reset()
        self.alt_controller.reset()
        self.state_history = []
        self.control_history = []

# ============================================================================
# CELL 8: Demo Setup
# ============================================================================

# Demo Setup
np.random.seed(42)  # For reproducible results

# Define start and target states
start_state = np.array([0.0, 0.0, 10.0, 0.0])  # lat, lon, alt, time
target_state = np.array([0.1, 0.1, 12.0, 0.0])  # lat, lon, alt, time

print(f"Start State:  Lat={start_state[0]:.3f}, Lon={start_state[1]:.3f}, Alt={start_state[2]:.1f} km")
print(f"Target State: Lat={target_state[0]:.3f}, Lon={target_state[1]:.3f}, Alt={target_state[2]:.1f} km")
print(f"Distance: {haversine_distance(start_state[0], start_state[1], target_state[0], target_state[1]):.0f} m")

# ============================================================================
# CELL 9: Tree Search Demo
# ============================================================================

# Tree Search Demo
print("=== Tree Search (A*) Demo ===")

# Initialize tree search agent
tree_agent = TreeSearchAgent(tree_config)

# Find optimal path
print("Searching for optimal path...")
optimal_path = tree_agent.find_path(start_state, target_state)

if optimal_path:
    print(f"Found path with {len(optimal_path)} steps")
    print("Path:")
    for i, state in enumerate(optimal_path):
        print(f"  Step {i}: Lat={state[0]:.3f}, Lon={state[1]:.3f}, Alt={state[2]:.1f} km")
else:
    print("No path found within iteration limit")

# ============================================================================
# CELL 10: PID Control Demo
# ============================================================================

# PID Control Demo
print("\n=== PID Control Demo ===")

# Initialize PID controller
pid_controller = BalloonPIDController(pid_config)

# Run simulation
current_state = start_state.copy()
step = 0
max_steps = int(pid_config.max_time / pid_config.dt)

print(f"Starting PID control simulation (max {max_steps} steps)...")

while step < max_steps and not pid_controller.is_target_reached(current_state, target_state):
    current_state = pid_controller.simulate_step(current_state, target_state)
    step += 1
    
    if step % 100 == 0:  # Print progress every 100 steps
        lat, lon, alt = current_state[:3]
        target_lat, target_lon, target_alt = target_state[:3]
        error = haversine_distance(lat, lon, target_lat, target_lon)
        print(f"  Step {step}: Lat={lat:.3f}, Lon={lon:.3f}, Alt={alt:.1f} km, Error={error:.0f} m")

print(f"\nPID Control completed in {step} steps")
print(f"Final state: Lat={current_state[0]:.3f}, Lon={current_state[1]:.3f}, Alt={current_state[2]:.1f} km")
print(f"Target reached: {pid_controller.is_target_reached(current_state, target_state)}")

# ============================================================================
# CELL 11: Visualization
# ============================================================================

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Trajectory comparison
if optimal_path:
    tree_lats = [state[0] for state in optimal_path]
    tree_lons = [state[1] for state in optimal_path]
    ax1.plot(tree_lons, tree_lats, 'b-o', label='Tree Search', markersize=4)

if pid_controller.state_history:
    pid_lats = [state[0] for state in pid_controller.state_history]
    pid_lons = [state[1] for state in pid_controller.state_history]
    ax1.plot(pid_lons, pid_lats, 'r-', label='PID Control', linewidth=2)

ax1.plot(start_state[1], start_state[0], 'go', markersize=10, label='Start')
ax1.plot(target_state[1], target_state[0], 'ro', markersize=10, label='Target')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title('Trajectory Comparison')
ax1.legend()
ax1.grid(True)

# Plot 2: Altitude over time
if optimal_path:
    tree_times = [state[3] for state in optimal_path]
    tree_alts = [state[2] for state in optimal_path]
    ax2.plot(tree_times, tree_alts, 'b-o', label='Tree Search', markersize=4)

if pid_controller.state_history:
    pid_times = [state[3] for state in pid_controller.state_history]
    pid_alts = [state[2] for state in pid_controller.state_history]
    ax2.plot(pid_times, pid_alts, 'r-', label='PID Control', linewidth=2)

ax2.axhline(y=target_state[2], color='r', linestyle='--', alpha=0.7, label='Target Altitude')
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Altitude (km)')
ax2.set_title('Altitude Profile')
ax2.legend()
ax2.grid(True)

# Plot 3: Control signals (PID only)
if pid_controller.control_history:
    times = [i * pid_config.dt for i in range(len(pid_controller.control_history))]
    ax3.plot(times, pid_controller.control_history, 'g-', linewidth=2)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Control Signal')
    ax3.set_title('PID Control Signals')
    ax3.grid(True)

# Plot 4: Error over time
if pid_controller.state_history:
    errors = []
    for state in pid_controller.state_history:
        error = haversine_distance(state[0], state[1], target_state[0], target_state[1])
        errors.append(error)
    
    times = [state[3] for state in pid_controller.state_history]
    ax4.plot(times, errors, 'm-', linewidth=2)
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Distance Error (m)')
    ax4.set_title('Distance Error Over Time')
    ax4.grid(True)

plt.tight_layout()
plt.show()

# ============================================================================
# CELL 12: Parameter Tuning Demo
# ============================================================================

# Parameter Tuning Demo
print("=== Parameter Tuning Demo ===")

# Test different PID parameters
pid_params = [
    {'name': 'Conservative', 'kp': 0.5, 'ki': 0.05, 'kd': 0.2},
    {'name': 'Aggressive', 'kp': 2.0, 'ki': 0.3, 'kd': 1.0},
    {'name': 'Balanced', 'kp': 1.0, 'ki': 0.1, 'kd': 0.5}
]

results = []

for params in pid_params:
    print(f"\nTesting {params['name']} parameters...")
    
    # Create new config with these parameters
    test_config = PIDConfig()
    test_config.lat_kp = params['kp']
    test_config.lat_ki = params['ki']
    test_config.lat_kd = params['kd']
    test_config.lon_kp = params['kp']
    test_config.lon_ki = params['ki']
    test_config.lon_kd = params['kd']
    test_config.alt_kp = params['kp'] * 2
    test_config.alt_ki = params['ki'] * 2
    test_config.alt_kd = params['kd'] * 2
    
    # Test the parameters
    controller = BalloonPIDController(test_config)
    current_state = start_state.copy()
    step = 0
    max_steps = 1000
    
    while step < max_steps and not controller.is_target_reached(current_state, target_state):
        current_state = controller.simulate_step(current_state, target_state)
        step += 1
    
    final_error = haversine_distance(current_state[0], current_state[1], 
                                     target_state[0], target_state[1])
    
    results.append({
        'name': params['name'],
        'steps': step,
        'final_error': final_error,
        'reached_target': controller.is_target_reached(current_state, target_state)
    })
    
    print(f"  Steps: {step}, Final Error: {final_error:.0f} m, Target Reached: {controller.is_target_reached(current_state, target_state)}")

# Display results
print("\n=== Parameter Tuning Results ===")
for result in results:
    print(f"{result['name']:12} | Steps: {result['steps']:4d} | Error: {result['final_error']:6.0f} m | Reached: {result['reached_target']}")

# ============================================================================
# CELL 13: Conclusion
# ============================================================================

print("""
# 5. Conclusion and Next Steps

## Key Takeaways:
1. **Tree Search** is great for finding optimal paths but can be computationally expensive
2. **PID Control** is efficient for real-time control but requires careful tuning
3. **Hybrid approaches** often work best in practice

## Next Steps:
1. **Integrate with your balloon environment** from the main project
2. **Add more sophisticated wind models** using ERA5 data
3. **Implement hybrid control strategies**
4. **Add obstacle avoidance** for real-world scenarios
5. **Optimize parameters** using machine learning techniques

## Experiment Ideas:
- Test with different wind conditions
- Compare performance with varying target distances
- Implement adaptive PID gains
- Add multiple balloon coordination

Happy experimenting! 🎈
""") 