import numpy as np 
# from env.ERA_wind_field import WindField
from env.balloon_env import BalloonEnvironment
# from tree_search_agent import TreeSearchAgent
import matplotlib.pyplot as plt

# class PIDAgent:
#     def __init__(self, target_altitude_km):
#         # Convert km to meters if needed
#         self.controller = AltitudePIDController(target_altitude_km * 1000)
#         self.prev_time = None  # We'll assume constant dt = 3600 seconds (1 hour per step)

#     def select_action(self, state):
#         current_alt_km = state[2]  # altitude in km
#         current_alt_m = current_alt_km * 1000
#         dt = 3600  # 1 hour per time step

#         alt_control = self.controller.compute_action(current_alt=current_alt_m, dt=dt)

#         # Assume only vertical control: gas or ballast
#         # Return 2D action [gas, ballast], map PID output to this
#         gas = max(alt_control, 0.0)
#         ballast = max(-alt_control, 0.0)

#         return np.array([gas, ballast])


# TODO: Gain tuning 

class AltitudePIDController:
    def __init__(self, Kp=0.005, Ki=0.0, Kd=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.last_error = None

    def reset(self):
        self.integral = 0.0
        self.last_error = None

    def compute_action(self, current_alt, target_alt, dt):
        error = target_alt - current_alt
        P_term = self.Kp * error
        self.integral += error * dt
        I_term = self.Ki * self.integral
        D_term = 0.0 if self.last_error is None else self.Kd * (error - self.last_error) / dt
        self.last_error = error
        action = P_term + I_term + D_term
        return np.clip(action, -1.0, 1.0)

class PIDTrackingAgent:
    def __init__(self, balloon_env, plan,wind_field, Kp=0.005, Ki=0.0, Kd=0.0, dt=60):
        """
        balloon_env: BalloonEnvironment instance (for wind field)
        plan: list of (state, action) tuples from tree search [(state, action), ...]
        dt: control timestep in seconds (smaller than tree search dt)
        """
        self.env = balloon_env
        self.wind_field = wind_field
        self.plan = plan
        self.dt = dt
        self.pid = AltitudePIDController(Kp, Ki, Kd)
        self.current_plan_idx = 0

    def reset(self):
        self.pid.reset()
        self.current_plan_idx = 0

    def step(self, state, t):
        """
        state: [lat, lon, alt, t]
        t: current time (hours or seconds, as used in your env)
        Returns: action to apply (ascend/descend command)
        """
        # Find the next target waypoint in the plan
        while (self.current_plan_idx + 1 < len(self.plan) and
               t >= self.plan[self.current_plan_idx + 1][0][3]):
            self.current_plan_idx += 1

        target_state, _ = self.plan[self.current_plan_idx]
        target_alt = target_state[2]  # in km

        # Query wind at current position and time
        lat, lon, alt = state[:3]
        pressure = self.env.altitude_to_pressure(alt)
        wind = self.wind_field.get_wind(lat, lon, pressure, t)

        # PID control (altitudes in meters)
        action = self.pid.compute_action(current_alt=alt, target_alt=target_alt, dt=self.dt)
        # Apply wind and action to update balloon state (example, you may need to adapt this)
        self.env.step(action)

        # Return action for logging or further processing
        return action
    
def generate_synthetic_trajectory(num_points=100, dt=60):
    lats = np.zeros(num_points)  # keep lat constant
    lons = np.zeros(num_points)  # keep lon constant
    # Sine wave for altitude: mean 10 km, amplitude 2 km, period = full trajectory
    alts = 10000 + 2000 * np.sin(np.linspace(0, 2 * np.pi, num_points))  # 8000 to 12000 meters
    times = np.arange(0, num_points * dt, dt) / 3600  # hours
    plan = [(np.array([lat, lon, alt, t]), None) for lat, lon, alt, t in zip(lats, lons, alts, times)]
    return plan

# Dummy wind field class for testing
class DummyWindField:
    def get_wind(self, lat, lon, pressure, t):
        # Return zero wind for simplicity, or you can add a synthetic wind pattern here
        return np.array([0.0, 0.0])

# Example usage:
if __name__ == "__main__":
    # plan = [(state1, action1), (state2, action2), ...]  # from tree search
    # plan = TreeSearchAgent.select_action_sequence()  # Example placeholder, replace with actual plan
    dt = 60  # seconds
    plan = generate_synthetic_trajectory(num_points=100, dt=dt)
    print([state[2] for state, _ in plan])
    initial_state = plan[0][0].copy()

    env = BalloonEnvironment()
    era_windfield = DummyWindField()  # Placeholder for ERA wind field, replace with actual wind field data
    env.balloon.lat = initial_state[0]  # Set initial lat, lon, alt
    env.balloon.lon = initial_state[1]
    env.balloon.alt = initial_state[2]
    pid_agent = PIDTrackingAgent(env, plan, era_windfield,Kp=0.005,Ki=0.0,Kd=0.00001, dt=dt)
    state = initial_state.copy()
    actual_alts = []
    target_alts = []
    times = []

    for i, (target_state, _) in enumerate(plan):
        t = target_state[3]
        action = pid_agent.step(state, t)
        # Update state from env.balloon after step
        state[0] = env.balloon.lat
        state[1] = env.balloon.lon
        state[2] = env.balloon.alt
        actual_alts.append(state[2]/1000)  # Convert to km
        target_alts.append(target_state[2]/1000)
        times.append(t)
        print(f"t={t:.2f}h, target={target_state[2]/1000:.2f}km, actual={state[2]/1000:.2f}km, action={action:.4f}")
    # print(target_alts[:10]) 
    plt.figure(figsize=(10, 5))
    plt.plot(times, target_alts, label='Target Altitude (km)', linestyle='--')
    plt.plot(times, actual_alts, label='PID Altitude (km)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Altitude (km)')
    plt.title('PID Tracking of Synthetic Altitude Trajectory (BalloonEnvironment)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    input("Press Enter to exit and close the plot...")