import numpy as np 
from env.ERA_wind_field import WindField
from env.balloon_env import BalloonEnvironment
from tree_search_agent import TreeSearchAgent

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
        action = self.pid.compute_action(current_alt=alt*1000, target_alt=target_alt*1000, dt=self.dt)
        # Apply wind and action to update balloon state (example, you may need to adapt this)
        self.env.step(wind, self.dt, action)

        # Return action for logging or further processing
        return action

# Example usage:
if __name__ == "__main__":
    # plan = [(state1, action1), (state2, action2), ...]  # from tree search
    plan = TreeSearchAgent.select_action_sequence()  # Example placeholder, replace with actual plan
    env = BalloonEnvironment()
    era_windfield = WindField
    pid_agent = PIDTrackingAgent(env, plan, era_windfield,Kp=0.01, dt=60)
    state = np.array([lat, lon, alt, t])
    for t in np.arange(start_time, end_time, pid_agent.dt/3600):
        action = pid_agent.step(state, t)
    # update state from env.balloon after step, or keep track externally