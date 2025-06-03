import numpy as np 

class AltitudePIDController:
    def __init__(self, target_altitude, Kp=0.005, Ki=0.0, Kd=0.0):
        self.target_alt = target_altitude  # meters
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0.0
        self.last_error = None

    def compute_action(self, current_alt, dt):
        error = self.target_alt - current_alt

        # Proportional
        P_term = self.Kp * error

        # Integral
        self.integral += error * dt
        I_term = self.Ki * self.integral

        # Derivative
        if self.last_error is None:
            D_term = 0.0
        else:
            derivative = (error - self.last_error) / dt
            D_term = self.Kd * derivative

        self.last_error = error

        # Combine
        action = P_term + I_term + D_term

        # Clip action to safe bounds [-1, 1]
        action = np.clip(action, -1.0, 1.0)

        return action
