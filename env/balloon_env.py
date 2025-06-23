import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
from env.wind_field import WindField, WindVector
# Enable interactive mode
plt.ion()
import numpy as np

import numpy as np

class WindVector:
    """Wind vector (simple structure)"""
    def __init__(self, u: float, v: float):
        self.u = u  # m/s (latitude direction)
        self.v = v  # m/s (longitude direction)

class Balloon:
        

    def __init__(self,
                 initial_lat: float,  # km
                 initial_lon: float,  # km
                 initial_alt: float,  # km
                 max_volume: float = 1000.0,  # m³ reflect BLE parameters on initial volume and mass (max_volume 1800, load 92.5, balloon 68.5)
                 max_sand: float = 100):     # kg
        self.lat = initial_lat
        self.lon = initial_lon
        self.alt = initial_alt # km   
        self.volume = max_volume
        self.volume_pre = max_volume
        self.sand = max_sand
        self.max_volume = max_volume
        self.max_sand = max_sand

        # Constants
        self.EARTH_RADIUS = 6371  # km
        self.DEG_TO_RAD = np.pi / 180.0
        self.balloon_mass = 68.5
        self.helium_density = 0.1786  # kg/m³ at STP
        self.air_density0 = 1.225  # kg/m³ at sea level
        self.gravity = 9.81  # m/s²
        self.vertical_velocity = 0.0
        self.temperature = self.get_temperature(initial_alt*1000)  # K : changing depending on altitude
        self.R = 287.05  # Gas constant for dry air in J/(kg·K)
        # Control rates (reduced for more gentle control)   
        self.max_vent_rate = self.max_volume * 0.01  # m³/dt
        self.max_sand_rate = self.max_sand * 0.01  # kg/dt
        self.pressure_pre = self.altitude_to_pressure(self.alt * 1000)
        self.max_velocity = 1.0  # m/s

    def get_air_density(self, altitude: float) -> float:
        """Compute air density at the given altitude (exponential decay model)"""
        return self.air_density0 * np.exp(-altitude / 7000.0)

    def altitude_to_pressure(self, altitude_m: float) -> float:
        """
        Calculate pressure at given altitude using the barometric formula.
        
        Args:
            altitude_m: Altitude in meters
            
        Returns:
            pressure: Pressure in hectopascals (hPa)
            
        Formula: P = P₀ * exp(-Mgh/RT)
        where:
        - P₀ = sea level pressure (1013.25 hPa)
        - M = molar mass of air (0.02896 kg/mol)
        - g = gravitational acceleration (9.81 m/s²)
        - h = height above sea level (m)
        - R = universal gas constant (8.314 J/(mol·K))
        - T = temperature (K)
        """
        # Constants
        P0 = 1013.25  # Sea level pressure (hPa)
        M = 0.02896   # Molar mass of air (kg/mol)
        R = 8.314     # Universal gas constant (J/(mol·K))
        
        pressure = P0 * np.exp(-M * self.gravity * altitude_m / (R * self.temperature))
        
        return pressure
    def get_temperature(self, altitude: float) -> float:
        """Compute temperature at the given altitude (exponential decay model)"""
        T0 = 288.15   # Sea level temperature (K)
        L = 0.0065    # Temperature lapse rate (K/m)
        T = T0 - L * altitude
        return T
    
    def get_helium_density(self, altitude: float) -> float:
        """Compute helium density at altitude (optional, for more precision)"""
        return self.helium_density * np.exp(altitude / 7000.0)

    def internal_controller(self, v_init:float, v_des: float, dt: float, pressure: float):
        """
        Stable internal controller with reduced oscillations.
        Uses gentler control gains and better damping.
        """
        delta_t = 1  # control step in seconds
        v_current = v_init
        pressure_current = pressure
        pressure_pre = pressure
        alt_current = self.alt  # Track current altitude in controller
        
        # Store initial values for restoration
        initial_volume = self.volume
        initial_sand = self.sand
        
        # Control gains - much gentler to reduce oscillations
        Kp_vel = 0.1    # Reduced from aggressive control
        Kd_vel = 0.02   # Small derivative gain for damping
        
        # Alternative gain sets you can try (uncomment one):
        # Option 1: Faster convergence, more aggressive
        # Kp_vel = 0.2    # Higher proportional gain
        # Kd_vel = 0.05   # Higher derivative gain
        
        # Option 2: Very fast convergence, might overshoot slightly
        # Kp_vel = 0.3    # Even higher proportional gain
        # Kd_vel = 0.08   # Higher derivative gain
        
        # Option 3: Conservative but better than current
        # Kp_vel = 0.15   # Moderate proportional gain
        # Kd_vel = 0.03   # Moderate derivative gain
        
        # Initialize error tracking for derivative control
        vel_error_prev = 0.0
        
        # Add integral control to eliminate steady state error
        vel_error_integral = 0.0
        Ki_vel = 0.01   # Integral gain for steady state error elimination
        # print(f"delta_t:{delta_t}, dt:{dt}")
        for i in range(0, dt, delta_t):
            # Smooth velocity target (linear interpolation)
            t_progress = i / dt
            v_target = v_init + t_progress * (v_des - v_init)
            self.volume = self.volume * (pressure_pre / pressure_current)
            # Calculate target altitude from velocity target
            alt_target = alt_current + v_target * delta_t
            
            helium_mass = self.helium_density * self.volume
            total_mass = self.balloon_mass + self.sand + helium_mass
            # rho_air = (pressure_current * 100) / (self.R * self.temperature)
            rho_air = (pressure_current * 100) / (self.R * 220)

            # Calculate current forces
            buoyancy = rho_air * self.volume * self.gravity
            weight = total_mass * self.gravity
            drag = self.drag_force(pressure_current, net_force=buoyancy-weight)
            F_current = buoyancy - weight + drag

            # Velocity control with damping
            velocity_error = v_target - v_current
            velocity_error_rate = (velocity_error - vel_error_prev) / delta_t
            
            # Accumulate integral error (with anti-windup)
            vel_error_integral += velocity_error * delta_t
            vel_error_integral = np.clip(vel_error_integral, -10.0, 10.0)  # Anti-windup
            
            # PID control for better convergence
            acc_des = Kp_vel * velocity_error + Ki_vel * vel_error_integral + Kd_vel * velocity_error_rate
            
            # Calculate required force
            F_des = total_mass * acc_des
            
            # Calculate altitude error for resource control
            alt_error = alt_target - alt_current
            
            # Calculate required changes in volume and mass
            dSand = 0.0
            dVolume = 0.0
            
            # Enhanced resource control with much gentler gains
            if F_des < F_current:  # Need to reduce buoyancy (vent gas)
                # Base volume change from force requirement - much gentler
                dVolume_force = (F_current - F_des) / (rho_air * self.gravity)
                
                # Additional volume change based on altitude error - reduced factor
                alt_factor = 1.0 + 0.05 * np.sign(alt_error)  # Reduced from 0.2 to 0.05
                dVolume = dVolume_force * alt_factor
                
                # Ensure we don't vent too much if we're already too low
                if alt_error < -0.2:  # Reduced threshold from 0.5 to 0.2
                    dVolume *= 0.3  # More aggressive reduction
                
            else:  # Need to reduce weight (drop sand)
                # Base sand change from force requirement - much gentler
                dSand_force = (F_des - F_current) / self.gravity
                
                # Additional sand change based on altitude error - reduced factor
                alt_factor = 1.0 - 0.05 * np.sign(alt_error)  # Reduced from 0.2 to 0.05
                dSand = dSand_force * alt_factor
                
                # Ensure we don't drop too much sand if we're already too high
                if alt_error > 0.2:  # Reduced threshold from 0.5 to 0.2
                    dSand *= 0.3  # More aggressive reduction
            
            # Much more conservative rate limits
            dVolume = np.clip(dVolume, 0.0, self.max_vent_rate * 0.05)  # Reduced from 0.1 to 0.02
            dSand = np.clip(dSand, 0.0, self.max_sand_rate * 0.05)     # Reduced from 0.1 to 0.02
            # print(f"velocity_error:{velocity_error:.2f}, v_current:{v_current:.2f}, v_target:{v_target:.2f}, dVolume:{dVolume:.4f}, dSand:{dSand:.4f}")
            
            # Enhanced damping when close to targets - adjusted for better convergence
            if abs(velocity_error) < 0.02:  # Reduced threshold for more precise control
                dVolume *= 0.3  # Less aggressive damping
                dSand *= 0.3
            
            # Additional damping based on velocity magnitude - reduced for faster convergence
            if abs(v_current) < 0.05:  # Reduced threshold from 0.1 to 0.05
                dVolume *= 0.7  # Less aggressive damping
                dSand *= 0.7
            
            # Apply control actions
            self.volume -= dVolume
            self.sand -= dSand

            if self.volume <= 0 or self.sand <= 0:
                break
            
            # Update system state
            helium_mass = self.helium_density * self.volume
            total_mass = self.balloon_mass + self.sand + helium_mass
            
            buoyancy_force = rho_air * self.volume * self.gravity
            weight_force = total_mass * self.gravity
            net_force = buoyancy_force - weight_force
            drag_force = self.drag_force(pressure_current, net_force=net_force)
            
            # Update velocity and altitude
            acceleration = (net_force + drag_force) / total_mass
            v_current += acceleration * delta_t
            alt_current += v_current * delta_t / 1000  # Convert to km
            
            # Clamp altitude
            alt_current = np.clip(alt_current, 5.0, 25.0)
            
            # Update pressure for next iteration
            pressure_pre = pressure_current
            self.temperature = self.get_temperature(alt_current * 1000)
            pressure_current = self.altitude_to_pressure(alt_current * 1000)
            
            # Store error for next iteration
            vel_error_prev = velocity_error
        
        # Calculate total changes over the entire time step
        total_dVolume = initial_volume - self.volume
        total_dSand = initial_sand - self.sand
        self.alt = alt_current
        self.vertical_velocity = v_current
        
        return total_dVolume, total_dSand

    def drag_force(self, pressure: float, net_force: float, vel: float=None):
        if vel is None:
            vel = self.vertical_velocity

        cross_section = np.pi * (3 * self.volume / (4 * np.pi)) ** (2/3)
        rho_air = (pressure * 100) / (self.R * self.temperature)
        # drag = 0.5 * self.get_air_density(self.alt*1000) * self.vertical_velocity**2 * 0.47 * cross_section
        # Drag coefficient (varies with Reynolds number)
        # Reynolds number depends on air density and viscosity
        kinematic_viscosity = 1.5e-5 * (1013.25/pressure)  # Viscosity increases with altitude
        
        # Handle zero velocity case
        if abs(vel) < 1e-6:
            reynolds_number = 1e-6  # Small non-zero value to avoid division by zero
        else:
            reynolds_number = abs(vel) * np.sqrt(cross_section) / kinematic_viscosity
        
        # Drag coefficient varies with Reynolds number
        if reynolds_number < 1:
            drag_coefficient = 24.0  # Stokes flow limit
        elif reynolds_number < 1000:
            drag_coefficient = 24/reynolds_number * (1 + 0.15 * reynolds_number**0.687)  # Transitional
        else:
            drag_coefficient = 0.47  # Turbulent flow
            
        # Add altitude effect on drag coefficient (reduced effect)
        drag_coefficient *= (1 + 0.05 * (self.alt/10))
        
        # Add base drag coefficient to make drag stronger overall
        # drag_coefficient += 0.5  # Base drag coefficient
        
        # Handle zero velocity case for drag force
        if abs(vel) < 1e-6:
            drag_force = 0.0
        else:
            # Drag force (F_d = 0.5 * ρ * v² * C_d * A)
            drag_force = 0.5 * rho_air * vel**2 * drag_coefficient * cross_section
            
            # Add a linear drag component for better low-velocity behavior
            linear_drag = 0.1 * rho_air * abs(vel) * cross_section
            drag_force += linear_drag
        # return 0
        drag_force *= np.sign(net_force)*-1
        return drag_force

    def step(self, wind: WindVector, dt: float, action: float = 0.0) -> None:
        """
        Update the balloon's state using terminal velocity equilibrium:
        - action=0: altitude remains nearly steady (±10m)
        - action>0: volume reduction → always descending
        - action<0: sand reduction → always ascending
        """
        # 1️⃣ Horizontal motion (latitude & longitude updates in km)
        # print(f"target velocity: {action:.2f} m/s")
        self.lat += wind.u * dt / 1000
        self.lon += wind.v * dt / 1000
        pressure = self.altitude_to_pressure(self.alt*1000)  # hPa
        dVolume, dSand = self.internal_controller(self.vertical_velocity, action, dt, pressure)
        # print(f"currnet velocity: {self.vertical_velocity:.2f} m/s | desired velocity: {action:.2f} m/s\n--------------------------------")


class BalloonEnvironment:
    """Environment for balloon navigation"""
    def __init__(self):
        self.wind_field = WindField()
        self.balloon = Balloon(initial_lat=0.0, initial_lon=0.0, initial_alt=8.0)
        self.dt = 60  # 1 minute time step (reduced from 1 hour)
        self.target_lat = 500  # km
        self.target_lon = -100  # km
        self.target_alt = 12  # km
        self.current_time = 0.0  # hours

        # Initialize figure for real-time plotting
        self.fig = plt.figure(figsize=(15, 5))
        self.ax1 = self.fig.add_subplot(131, projection='3d')  # 3D Position plot
        self.ax2 = self.fig.add_subplot(132)  # Resources plot
        self.ax3 = self.fig.add_subplot(133)  # Wind profile plot
        plt.tight_layout()

        # Store trajectory for 3D plot
        self.trajectory = {'lat': [], 'lon': [], 'alt': []}
        # Add initial position to trajectory
        self.trajectory['lat'].append(self.balloon.lat)
        self.trajectory['lon'].append(self.balloon.lon)
        self.trajectory['alt'].append(self.balloon.alt)

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.balloon = Balloon(initial_lat=0.0, initial_lon=0.0, initial_alt=10.0)
        self.current_time = 0.0
        # Clear trajectory
        self.trajectory = {'lat': [], 'lon': [], 'alt': []}
        # Add initial position to trajectory
        self.trajectory['lat'].append(self.balloon.lat)
        self.trajectory['lon'].append(self.balloon.lon)
        self.trajectory['alt'].append(self.balloon.alt)
        return self._get_state()

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

        pressure = self.balloon.altitude_to_pressure(self.balloon.alt)

        # Get wind at current position and time
        wind = self.wind_field.get_wind(
            self.balloon.lat,
            self.balloon.lon,
            pressure,
            self.current_time
        )
        # Update balloon state
        # print(f"Wind: {wind}, dt: {self.dt}, Action value: {action_value}")
        self.balloon.step(wind, self.dt, action_value)

        # Update time
        self.current_time += self.dt / 3600  # Convert seconds to hours

        # Get new state
        state = self._get_state()

        # Calculate reward
        reward = self._get_reward()

        # Check if done
        done, reason = self._is_done()
        if done:
            print(f"\nEpisode terminated: {reason}")

        return state, reward, done, reason

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

        # Update trajectory with current position
        self.trajectory['lat'].append(self.balloon.lat)
        self.trajectory['lon'].append(self.balloon.lon)
        self.trajectory['alt'].append(self.balloon.alt)

        # Convert lists to numpy arrays for plotting
        lats = np.array(self.trajectory['lat'])
        lons = np.array(self.trajectory['lon'])
        alts = np.array(self.trajectory['alt'])

        # Plot trajectory
        if len(lats) > 1:
            self.ax1.plot(lons, lats, alts, 'b-', linewidth=2, label='Trajectory')
        # Plot current position
        self.ax1.scatter(lons[-1], lats[-1], alts[-1],
                        c='red', marker='o', s=150, label='Current Position')

        # Plot target
        self.ax1.scatter(self.target_lon, self.target_lat, 10,
                        c='green', marker='*', s=200, label='Target')

        # Set labels and title
        self.ax1.set_xlabel('Longitude (km)')
        self.ax1.set_ylabel('Latitude (km)')
        self.ax1.set_zlabel('Altitude (km)')
        self.ax1.set_title('Balloon Navigation')

        # Set view limits with some padding
        # self.ax1.set_xlim(min(lons) - 10, max(lons) + 10)
        # self.ax1.set_ylim(min(lats) - 10, max(lats) + 10)

        self.ax1.set_xlim(-10, 10)
        self.ax1.set_ylim(-10, 10)
        self.ax1.set_zlim(0, 25)  # Altitude range in km

        # Add legend
        self.ax1.legend()

        # Plot resources
        self.ax2.bar(['Volume', 'Sand'],
                     [self.balloon.volume/self.balloon.max_volume,
                      self.balloon.sand/self.balloon.max_sand])
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
        plt.pause(0.1)  # Reduced pause time for smoother animation


