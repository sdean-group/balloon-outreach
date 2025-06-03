import numpy as np
# from env.balloon_env import haversine_distance  # if needed

class GoalDirectedAgent:
    def __init__(self, target_lat, target_lon, target_alt):
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.target_alt = target_alt

    def select_action(self, state):
        lat= state[0]
        lon = state[1]
        alt = state[2]

        # --- Altitude Control ---
        error = self.target_alt - alt
        if abs(error) < 10:
            return 0  # hold
        action = np.clip(error / 100.0, -1, 1)
        return action
