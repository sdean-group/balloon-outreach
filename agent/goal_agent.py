import numpy as np
# from env.balloon_env import haversine_distance  # if needed

class GoalDirectedAgent:
    def __init__(self, target_lat=100, target_lon=100, target_alt=12):
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.target_alt = target_alt

    def select_action(self, state):
        lat= state[0]
        lon = state[1]
        alt = state[2]

        # --- Altitude Control ---
        if alt < self.target_alt - 1:
            return -1   # ascend
        elif alt > self.target_alt + 1:
            return 1  # descend
        else:
            return 0   # hold
