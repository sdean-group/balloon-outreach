# import numpy as np
# import matplotlib.pyplot as plt

# class Balloon:
#     def __init__(self, alt=10000, max_volume=1000, max_sand=20):
#         self.alt = alt  # meters
#         self.volume = 600  # Start with less helium for better control
#         self.sand = max_sand
#         self.max_volume = max_volume
#         self.max_sand = max_sand
#         self.min_volume = 50.0  # Allow more venting
#         self.balloon_mass = 80.0  # Heavier balloon for easier descent
#         self.helium_density = 0.1786
#         self.air_density0 = 1.225
#         self.gravity = 9.81
#         self.vertical_velocity = 0.0
#         self.max_vent_rate = 0.05   # m³/s (more effective venting)
#         self.max_sand_rate = 0.01  # kg/s

#     def get_air_density(self, altitude):
#         return self.air_density0 * np.exp(-altitude / 7000.0)

#     def step(self, dt, action):
#         rho_air = self.get_air_density(self.alt)
#         helium_mass = self.helium_density * self.volume
#         total_mass = self.balloon_mass + self.sand + helium_mass
#         buoyancy_force = rho_air * self.volume * self.gravity
#         weight_force = total_mass * self.gravity
#         net_force = buoyancy_force - weight_force

#         drag_coefficient = 1.5
#         cross_section = np.pi * (self.volume / np.pi) ** (2/3) + 1e-6

#         if abs(net_force) < 0.1:
#             self.vertical_velocity = 0.0
#         else:
#             v_terminal = np.sqrt(abs(2 * net_force / (drag_coefficient * cross_section * rho_air)))
#             self.vertical_velocity = np.sign(net_force) * v_terminal

#         if abs(action) < 1e-3:
#             self.vertical_velocity *= 0.05  # Damp oscillations
#         else:
#             if action > 0:
#                 dV = self.max_vent_rate * action * dt
#                 self.volume = max(self.min_volume, self.volume - dV)
#             if action < 0:
#                 d_sand = self.max_sand_rate * (-action) * dt
#                 self.sand = max(0.0, self.sand - d_sand)

#         self.alt += self.vertical_velocity * dt
#         self.alt = np.clip(self.alt, 5000.0, 25000.0)

#         if self.alt >= 25000.0 and self.vertical_velocity > 0:
#             self.volume = max(self.min_volume, self.volume - self.max_vent_rate * dt)

#         self.volume = np.clip(self.volume, self.min_volume, self.max_volume)
#         self.sand = np.clip(self.sand, 0.0, self.max_sand)

# class HybridController:
#     def __init__(self, threshold, Kp):
#         self.threshold = threshold
#         self.Kp = Kp

#     def compute(self, current, target, dt):
#         error = target - current
#         if abs(error) > self.threshold:
#             return np.clip(self.Kp * error, -1.0, 1.0)
#         else:
#             return 0.0

# # Simulation parameters
# num_points = 100
# dt = 60
# times = np.arange(num_points) * dt / 3600
# target_alts = np.full(num_points, 10000)  # Constant altitude at 10km

# balloon = Balloon(alt=10000)
# controller = HybridController(threshold=20, Kp=0.003)

# actual_alts = []

# for i in range(num_points):
#     current_alt = balloon.alt
#     target_alt = target_alts[i]
#     action = controller.compute(current_alt, target_alt, dt)
#     balloon.step(dt, action)
#     actual_alts.append(balloon.alt)
#     print(f"t={times[i]:.2f}h, target={target_alt/1000:.2f}km, actual={balloon.alt/1000:.2f}km, action={action:.4f}, sand={balloon.sand:.2f}, volume={balloon.volume:.2f}")

# plt.plot(times, target_alts/1000, label='Target Altitude (km)', linestyle='--')
# plt.plot(times, np.array(actual_alts)/1000, label='Actual Altitude (km)')
# plt.xlabel('Time (hours)')
# plt.ylabel('Altitude (km)')
# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

class SimpleBalloon:
    def __init__(self, alt=10000):
        self.alt = alt
        self.vertical_velocity = 0.0
        self.max_rate = 0.9  # m/s, max climb/descent rate

    def step(self, dt, action):
        # Action is in [-1, 1], directly sets vertical velocity
        self.vertical_velocity = np.clip(action, -1, 1) * self.max_rate
        self.alt += self.vertical_velocity * dt
        self.alt = np.clip(self.alt, 5000.0, 25000.0)

class SimplePID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.last_error = None

    def compute(self, current, target, dt):
        error = target - current
        self.integral += error * dt
        derivative = 0.0 if self.last_error is None else (error - self.last_error) / dt
        self.last_error = error
        action = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        if abs(error) < 2:  # deadband for small errors
            action = 0.0
        return np.clip(action, -1.0, 1.0)

# Simulation parameters
num_points = 100
dt = 60
times = np.arange(num_points) * dt / 3600
# target_alts = np.full(num_points, 10000)  # Constant altitude at 10km
# Sine wave reference: 9km to 11km
target_alts = 10000 + 1000 * np.sin(np.linspace(0, 2 * np.pi, num_points))
balloon = SimpleBalloon(alt=10000)
controller = SimplePID(Kp=0.05, Ki=0.0, Kd=1.2)

actual_alts = []

for i in range(num_points):
    current_alt = balloon.alt
    target_alt = target_alts[i]
    action = controller.compute(current_alt, target_alt, dt)
    balloon.step(dt, action)
    actual_alts.append(balloon.alt)
    print(f"t={times[i]:.2f}h, target={target_alt/1000:.2f}km, actual={balloon.alt/1000:.2f}km, action={action:.4f}")

plt.plot(times, target_alts/1000, label='Target Altitude (km)', linestyle='--')
plt.plot(times, np.array(actual_alts)/1000, label='Actual Altitude (km)')
plt.xlabel('Time (hours)')
plt.ylabel('Altitude (km)')
plt.legend()
plt.show()
