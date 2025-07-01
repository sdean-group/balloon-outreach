import numpy as np

class PIDAgent:
    def __init__(self, target_alt, Kp, Ki, Kd, deadband):
        self.target_alt = target_alt
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.deadband = deadband
        self.integral = 0.0
        self.last_error = None

    def select_action(self, state, dt):
        """
        state: [lat, lon, alt, t] or [alt]
        dt: timestep in seconds
        """
        # Support both [lat, lon, alt, t] and [alt]
        if isinstance(state, (list, np.ndarray)) and len(state) >= 3:
            alt = state[2]
        else:
            alt = state

        error = self.target_alt - alt
        self.integral += error * dt
        derivative = 0.0 if self.last_error is None else (error - self.last_error) / dt
        self.last_error = error

        action = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Deadband for small errors
        if abs(error) < self.deadband:
            action = 0.0

        # Clamp action to [-1, 1]
        return float(np.clip(action, -1.0, 1.0))

# # Example usage:
# if __name__ == "__main__":
#     agent = PIDAgent(target_alt=12000, Kp=0.05, Ki=0.0, Kd=1.2, deadband=2.0)
#     dt = 60  # seconds
#     alt = 10000
#     for t in range(10):
#         state = [0, 0, alt, t*dt/3600]
#         action = agent.select_action(state, dt)
#         print(f"Step {t}: Altitude={alt:.2f}, Action={action:.3f}")
#         # Simulate altitude change (for demonstration)
#         alt += action * 0.9 * dt  # assuming max_rate=0.9 m/s
