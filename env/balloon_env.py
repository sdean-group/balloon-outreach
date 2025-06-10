import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
from env.wind_field import WindField, WindVector
import math
# from env.balloon import balloon
# Enable interactive mode
plt.ion()
import numpy as np

import numpy as np

class WindVector:
    """Wind vector (simple structure)"""
    def __init__(self, u: float, v: float):
        self.u = u  # m/s (latitude direction)
        self.v = v  # m/s (longitude direction)

# class Balloon:
#     def __init__(self,
#                  initial_lat: float,  # degrees
#                  initial_lon: float,  # degrees
#                  initial_alt: float,  # km
#                  max_volume: float = 3000.0,  # m³
#                  max_sand: float = 30):     # kg
#         self.lat = initial_lat
#         self.lon = initial_lon
#         self.alt = initial_alt * 1000.0  # convert km → m
#         self.volume = max_volume
#         self.sand = max_sand
#         self.max_volume = max_volume
#         self.max_sand = max_sand

#         # Constants
#         self.EARTH_RADIUS = 6371  # km
#         self.DEG_TO_RAD = np.pi / 180.0
#         self.balloon_mass = 50.0  # kg
#         self.helium_density = 0.1786  # kg/m³ at STP
#         self.air_density0 = 1.225  # kg/m³ at sea level
#         self.gravity = 9.81  # m/s²
#         self.vertical_velocity = 0.0
#         # Control rates
#         self.max_vent_rate = 0.1  # m³/s (maximum venting rate)
#         self.max_sand_rate = 0.1  # kg/s (maximum sand dropping rate)

#     def get_air_density(self, altitude: float) -> float:
#         """Compute air density at the given altitude (exponential decay model)"""
#         return self.air_density0 * np.exp(-altitude / 7000.0)

#     def get_helium_density(self, altitude: float) -> float:
#         """Compute helium density at altitude (optional, for more precision)"""
#         return self.helium_density * np.exp(altitude / 7000.0)

#     def step(self, wind: WindVector, dt: float, action: float = 0.0) -> None:
#         """
#         Update the balloon's state using terminal velocity equilibrium:
#         - action=0: altitude remains nearly steady (±10m)
#         - action>0: volume reduction → always descending
#         - action<0: sand reduction → always ascending
#         """
#         # 1️⃣ Horizontal motion (latitude & longitude update)
#         self.lat += wind.u * dt / (self.EARTH_RADIUS * 1000 * self.DEG_TO_RAD)
#         self.lon += wind.v * dt / (self.EARTH_RADIUS * 1000 * self.DEG_TO_RAD)

#         # 2️⃣ Compute buoyancy and weight forces
#         rho_air = self.get_air_density(self.alt)
#         helium_mass = self.helium_density * self.volume
#         total_mass = self.balloon_mass + self.sand + helium_mass

#         buoyancy_force = rho_air * self.volume * self.gravity
#         weight_force = total_mass * self.gravity
#         net_force = buoyancy_force - weight_force

#         # 3️⃣ Compute drag-limited terminal vertical velocity
#         drag_coefficient = 1.5  # stronger drag to stabilize vertical motion
#         cross_section = np.pi * (self.volume / np.pi) ** (2/3)
#         if net_force == 0:
#             self.vertical_velocity = 0.0
#         else:
#             v_terminal = np.sqrt(
#                 abs(2 * net_force / (drag_coefficient * cross_section * rho_air))
#             )
#             self.vertical_velocity = np.sign(net_force) * v_terminal

#         # 4️⃣ When no action, dampen oscillation (±10m)
#         if abs(action) < 1e-3:
#             self.vertical_velocity *= 0.1
#         else:
#             # action>0: venting → decrease volume → decrease buoyancy → descend
#             if action > 0:
#                 dV = self.max_vent_rate * action * dt
#                 self.volume = max(0.0, self.volume - dV)
#             # action<0: dropping sand → decrease weight → ascend
#             if action < 0:
#                 d_sand = self.max_sand_rate * (-action) * dt
#                 self.sand = max(0.0, self.sand - d_sand)
#         if self.alt <= 5000.0:
#             self.vertical_velocity = max(self.vertical_velocity, 0.0)  # No falling below 5km

#         # 5️⃣ Update altitude
#         self.alt += self.vertical_velocity * dt
#         # Prevent falling below 5000 meters
#         if self.alt < 5000.0:
#             self.alt = 5000.0
#             self.vertical_velocity = 0.0

#         # Clamp altitude to [5 km, 25 km]
#         self.alt = np.clip(self.alt, 5000.0, 25000.0)

#         # Debug output
#         print(f"Lat: {self.lat:.6f}°, Lon: {self.lon:.6f}°, Alt: {self.alt/1000:.2f} km")
#         print(f"Volume: {self.volume:.2f} m³, Sand: {self.sand:.2f} kg, Vertical Vel.: {self.vertical_velocity:.2f} m/s")
class Balloon:
    def __init__(self,
                 initial_lat: float,  # degrees
                 initial_lon: float,  # degrees
                 initial_alt: float,  # meters (no more confusion!)
                 max_volume: float = 3000.0,  # m³
                 max_sand: float = 30.0):     # kg
        self.lat = initial_lat
        self.lon = initial_lon
        self.alt = initial_alt  # meters, no multiplication!
        self.volume = max_volume
        self.sand = max_sand
        self.max_volume = max_volume
        self.max_sand = max_sand

        # Constants
        self.EARTH_RADIUS = 6371e3  # meters
        self.DEG_TO_RAD = np.pi / 180.0
        self.balloon_mass = 50.0  # kg
        self.helium_density = 0.1786  # kg/m³ at STP
        self.air_density0 = 1.225  # kg/m³ at sea level
        self.gravity = 9.81  # m/s²
        self.vertical_velocity = 0.0
        # Control rates
        self.max_vent_rate = 0.1  # m³/s
        self.max_sand_rate = 0.001  # kg/s - 1g/s right now

    def get_air_density(self, altitude: float) -> float:
        """Exponential decay model for air density with altitude."""
        return self.air_density0 * np.exp(-altitude / 7000.0)

    def get_helium_density(self, altitude: float) -> float:
        """Helium density can also decrease (optional, for more realism)."""
        return self.helium_density * np.exp(altitude / 7000.0)

    def step(self, wind: WindVector, dt: float, action: float = 0.0) -> None:
        """Update balloon state."""
        # 1️⃣ Horizontal motion (latitude & longitude update)
        self.lat += wind.u * dt / (self.EARTH_RADIUS * self.DEG_TO_RAD)
        self.lon += wind.v * dt / (self.EARTH_RADIUS * self.DEG_TO_RAD)

        # 2️⃣ Forces
        rho_air = self.get_air_density(self.alt)
        helium_mass = self.helium_density * self.volume
        total_mass = self.balloon_mass + self.sand + helium_mass

        buoyancy_force = rho_air * self.volume * self.gravity
        weight_force = total_mass * self.gravity
        net_force = buoyancy_force - weight_force

        # 3️⃣ Drag-limited terminal velocity
        drag_coefficient = 1.5
        cross_section = np.pi * (self.volume / np.pi) ** (2/3)
        # damping_force = -0.5 * self.vertical_velocity
        # net_force += damping_force

        if net_force == 0:
            self.vertical_velocity = 0.0
        else:
            v_terminal = np.sqrt(abs(2 * net_force / (drag_coefficient * cross_section * rho_air)))
            self.vertical_velocity = np.sign(net_force) * v_terminal

        # 4️⃣ Action Handling
        if abs(action) < 1e-3:
            self.vertical_velocity *= 0.05  # Damp oscillations
        else:
            if action > 0:
                dV = self.max_vent_rate * action * dt
                self.volume = max(0.0, self.volume - dV)
            if action < 0:
                d_sand = self.max_sand_rate * (-action) * dt
                self.sand = max(0.0, self.sand - d_sand)

        # 5️⃣ Altitude Update
        self.alt += self.vertical_velocity * dt
        if self.alt < 5000.0:
            self.alt = 5000.0
            self.vertical_velocity = 0.0

        # Clamp
        self.alt = np.clip(self.alt, 5000.0, 25000.0)

        # Debug
        print(f"Lat: {self.lat:.6f}°, Lon: {self.lon:.6f}°, Alt: {self.alt/1000:.2f} km")
        print(f"Volume: {self.volume:.2f} m³, Sand: {self.sand:.2f} kg, Vertical Vel.: {self.vertical_velocity:.2f} m/s")

class BalloonEnvironment:
    """Environment for balloon navigation"""
    def __init__(self):
        self.wind_field = WindField()
        self.balloon = Balloon(
            initial_lat=np.random.uniform(30.0, 45.0),
            initial_lon=np.random.uniform(-120.0, -70.0),
            initial_alt=18000.0,  # meters
            max_volume=3000.0,
            max_sand=30.0
        )
        self.dt = 300  #  time step
        self.target_lat = self.balloon.lat + 3.0
        self.target_lon = self.balloon.lon + 5.0
        self.target_altitude = 18000.0  # meters
        self.current_time = 0.0  # hours

        # Plotting
        self.fig = plt.figure(figsize=(15, 5))
        self.ax1 = self.fig.add_subplot(131, projection='3d')
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        plt.tight_layout()
        self.trajectory = {'lat': [], 'lon': [], 'alt': []}

   
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.balloon = Balloon(
            initial_lat=np.random.uniform(30.0, 45.0),
            initial_lon=np.random.uniform(-120.0, -70.0),
            initial_alt=18000.0,
            max_volume=3000.0,
            max_sand=30.0
        )
        self.current_time = 0.0
        self.trajectory = {'lat': [], 'lon': [], 'alt': []}
        return self._get_state()
    
    def haversine_distance(self):
        """Calculate great-circle distance between two (lat, lon) points in meters."""
        R = 6371e3  # Earth radius in meters
        phi1 = math.radians(self.balloon.lat)
        phi2 = math.radians(self.target_lat)
        delta_phi = math.radians(self.target_lat - self.balloon.lat)
        delta_lambda = math.radians(self.target_lon - self.balloon.lon)

        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * \
            math.sin(delta_lambda / 2) ** 2

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c


    def _get_wind_column(self) -> np.ndarray:
        """Get wind vectors at all pressure levels for current position and time"""
        wind_column = []
        for pressure in self.wind_field.pressure_levels:
            wind = self.wind_field.get_wind(
                self.balloon.lon,
                self.balloon.lat,
                pressure,
                self.current_time
            )
            wind_column.extend([wind.u, wind.v])
        return np.array(wind_column)

    def _is_done(self) -> Tuple[bool, str]:
        """Check if episode is done and return the reason"""
        # Episode ends if balloon is out of bounds or at target
        lat_diff = self.balloon.lat - self.target_lat
        lon_diff = self.balloon.lon - self.target_lon
        distance = np.sqrt(lat_diff**2 + lon_diff**2)

        if self.balloon.volume <= 0:
            return True, "No helium left"
        elif self.balloon.sand <= 0:
            return True, "No sand left"
        elif distance < 0.1:
            return True, "Reached target"
        elif self.current_time >= 24:
            return True, "Time limit reached"
        else:
            return False, ""

    def _get_pressure(self, altitude_km: float) -> float:
        """
        Convert altitude (km) to pressure (hPa) using the barometric formula

        Args:
            altitude: Altitude in kilometers

        Returns:
            pressure: Pressure in hectopascals (hPa)
        """
        # Constants for barometric formula
        P0 = 1013.25  # Sea level pressure in hPa
        L = 0.0065    # Temperature lapse rate in K/m
        T0 = 288.15   # Sea level temperature in K
        g = 9.80665   # Gravitational acceleration in m/s^2
        R = 287.05    # Gas constant for dry air in J/(kg·K)

        # Convert altitude from km to m
        h = altitude_km * 1000

        # Calculate pressure using barometric formula
        pressure = P0 * (1 - (L * h) / T0) ** (g / (R * L))

        return pressure
    def altitude_to_pressure(self,altitude_m: float) -> float:
        """
        고도(m)를 입력받아, 대기압(hPa)을 반환하는 함수입니다.
        국제표준대기모델(ISA)의 간단한 지수 감소식을 기반으로 합니다.
        """
        # 상수
        P0 = 1013.25  # 해수면 기준 대기압 (hPa)
        M = 0.029  # 공기 평균 분자량 (kg/mol)
        g = 9.81  # 중력 가속도 (m/s^2)
        R = 8.314  # 기체 상수 (J/(mol·K))
        T = 220  # 평균 기온 (K, 10 km 부근)
        # 지수 감소 인자
        exponent = - (M * g * altitude_m) / (R * T)

        # 압력 계산
        pressure_hPa = P0 * np.exp(exponent)

        return pressure_hPa

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment

        Args:
            action: Single continuous action in range [-1, 1]
                   Can be passed as numpy array with single value
                   Negative values: drop sand (magnitude determines amount)
                   Positive values: vent gas (magnitude determines rate)

        Returns:
            state: Current state
            reward: Reward for the step
            done: Whether the episode is done
            info: Additional information
        """
        # Convert numpy array to float if necessary
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)

        # print(f"\nEnvironment step:")
        # print(f"Action: {action_value:.2f}")
        # print(f"Current time: {self.current_time:.2f} hours")

        # Get current pressure based on altitude
        # pressure = self._get_pressure(self.balloon.alt)
        pressure = self.altitude_to_pressure(self.balloon.alt)

        # Get wind at current position and time
        wind = self.wind_field.get_wind(
            self.balloon.lat,
            self.balloon.lon,
            pressure,
            self.current_time
        )

        
        

        # Update balloon state
        self.balloon.step(wind, self.dt, action_value)

        # Update time
        self.current_time += self.dt / 3600  # Convert seconds to hours

        # Get new state
        state = self._get_state()
        # state = self._get_state()
        self.state = state.copy() 

        # Calculate reward
        reward = self._get_reward()

        # --- Goal tracking ---
        pos_error = self.haversine_distance()  # meters
        alt_error = abs(self.balloon.alt - self.target_altitude)  # meters

        # --- Reward shaping ---
        # You want higher rewards when closer to the goal (negative error penalty)
        # Normalization ranges (approximate max values you expect)
        pos_norm = pos_error / 100000  # scale 0–1 over 100 km
        alt_norm = alt_error / 10000   # scale 0–1 over 10 km

        reward = -5.0 * pos_norm - 2.0 * alt_norm  # tunable weights

        #exponential reward shaping
        # reward = np.exp(-pos_error / 20000) * np.exp(-alt_error / 5000)
        # reward *= 10  # scale up

        # # Optional terminal bonus
        # if pos_error < 1000 and alt_error < 100:
        #     done = True
        #     reward += 100
        #     info = "Target position and altitude reached"
        # else:
        #     done = False
        #     info = {}


        # Optional bonus for reaching the goal
        if pos_error < 1000 and alt_error < 100:
            done = True
            reward += 100  # Big terminal bonus
            info = "Target position and altitude reached"
        else:
            done = False
            info = {}


        
        # Check if done
        # done, reason = self._is_done()
        if done:
            print(f"\nEpisode terminated: {info}")

        return state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """Get current state as numpy array"""
        # Balloon state
        balloon_state = np.array([
            self.balloon.lat,
            self.balloon.lon,
            self.balloon.alt,
            self.balloon.volume / self.balloon.max_volume,  # Normalized volume
            self.balloon.sand / self.balloon.max_sand,      # Normalized sand
            self.balloon.vertical_velocity,
            self.current_time
        ])

        # Wind column at all pressure levels
        wind_column = self._get_wind_column()

        # Combine balloon state and wind column
        return np.concatenate([balloon_state, wind_column])

    def _get_reward(self) -> float:
        """Calculate reward based on distance to target"""
        lat_diff = self.balloon.lat - self.target_lat
        lon_diff = self.balloon.lon - self.target_lon
        distance = np.sqrt(lat_diff**2 + lon_diff**2)
        return -distance  # Negative distance as reward

    def render(self) -> None:
        """Render current state"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Update trajectory
        self.trajectory['lat'].append(self.balloon.lat)
        self.trajectory['lon'].append(self.balloon.lon)
        self.trajectory['alt'].append(self.balloon.alt)

        # Convert lists to numpy arrays for plotting
        lats = np.array(self.trajectory['lat'])
        lons = np.array(self.trajectory['lon'])
        alts = np.array(self.trajectory['alt'])

        # Plot 3D trajectory
        if len(lats) > 1:  # Only plot line if we have more than one point
            self.ax1.plot(lons, lats, alts, 'b-', alpha=0.5, label='Trajectory')

        # Plot current position
        self.ax1.scatter(lons[-1], lats[-1], alts[-1],
                        c='b', marker='o', s=100, label='Current Position')

        # Plot start and target
        self.ax1.scatter(0, 0, 10, c='g', marker='o', s=100, label='Start')
        self.ax1.scatter(self.target_lon, self.target_lat, 10, c='r', marker='o', s=100, label='Target')

        # Set labels and title
        self.ax1.set_xlabel('Longitude')
        self.ax1.set_ylabel('Latitude')
        self.ax1.set_zlabel('Altitude (km)')
        self.ax1.set_title('Balloon Trajectory')

        # Set view limits
        self.ax1.set_xlim(min(lons) - 1, max(lons) + 1)
        self.ax1.set_ylim(min(lats) - 1, max(lats) + 1)
        self.ax1.set_zlim(0, 20)  # Altitude range

        # Set equal aspect ratio for lat/lon
        self.ax1.set_box_aspect([1, 1, 1])

        # Add legend
        self.ax1.legend()

        # Plot resources
        # self.ax2.bar(['Volume', 'Sand'],
        #              [self.balloon.volume/self.balloon.max_volume,
        #               self.balloon.sand/self.balloon.max_sand])
        # self.ax2.set_ylim(0, 1)
        # self.ax2.set_title('Resources')

        volume_ratio = self.balloon.volume / self.balloon.max_volume if self.balloon.max_volume > 0 else 0
        sand_ratio = self.balloon.sand / self.balloon.max_sand if self.balloon.max_sand > 0 else 0

        self.ax2.bar(['Volume', 'Sand'], [volume_ratio, sand_ratio])
        self.ax2.set_ylim(0, 1)
        self.ax2.set_title('Resources')


        # Plot wind column
        wind_column = self._get_wind_column()
        u_winds = wind_column[::2]  # Every other element is u component
        v_winds = wind_column[1::2]  # Every other element is v component
        wind_speeds = np.sqrt(u_winds**2 + v_winds**2)
        self.ax3.plot(wind_speeds, self.wind_field.pressure_levels, 'b-')
        self.ax3.grid(True)
        self.ax3.set_title('Wind Speed Profile')
        self.ax3.set_xlabel('Wind Speed (m/s)')
        self.ax3.set_ylabel('Pressure (hPa)')
        self.ax3.invert_yaxis()  # Invert y-axis to show pressure decreasing upward

        # Update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.5)  # Increased pause time for better visualization

