import numpy as np
from env.util import wind_displacement_to_position
class WindVector:
    """Wind vector (simple structure)"""
    def __init__(self, u: float, v: float):
        self.u = u  # m/s (latitude direction)
        self.v = v  # m/s (longitude direction)

class Balloon:
    def __init__(self,
                 initial_lat: float,
                 initial_lon: float,
                 initial_alt: float,
                 initial_volume: float = 1000.0,
                 initial_sand: float = 100,
                 max_volume: float = 1500.0):
        self.lat = initial_lat # deg
        self.lon = initial_lon # deg
        self.alt = initial_alt  # km
        self.volume = initial_volume
        self.initial_volume = initial_volume
        self.sand = initial_sand
        self.initial_sand = initial_sand
        self.max_volume = max_volume
        self.EARTH_RADIUS = 6371  # km
        self.DEG_TO_RAD = np.pi / 180.0
        self.balloon_mass = 68.5
        self.helium_density = 0.1786  # kg/m^3
        self.air_density0 = 1.225  # kg/m^3
        self.gravity = 9.81  # m/s^2
        self.vertical_velocity = 0.0
        self.T0 = 288.15 #Standard sea-level temperature K
        self.L = 0.0065 #Lapse rate K/m
        self.temperature = self.get_temperature(initial_alt * 1000)  # K
        self.temperature_pre = self.temperature
        self.R = 287.05
        self.R_u = 8.314 #Universal Gas Constant J/(mol·K)
        self.helium_molar_mass = 0.0040026 #kg/mol
        self.air_molar_mass = 0.02896 #kg/mol
        self.P0 = 1013.25 #Standard sea-level pressure hPa
        self.max_vent_rate = initial_volume * 0.01  # m^3/dt
        self.max_sand_rate = initial_sand * 0.01  # kg/dt
        self.pressure_pre = self.altitude_to_pressure(self.alt * 1000)
        self.max_velocity = 1.0  # m/s
        self.delta_t = 1

        # New state
        self.initial_helium_mass = self.helium_density * initial_volume
        self.helium_mass = self.helium_density * initial_volume

        buoyancy, weight, net_force, drag_force, rho_air, total_mass = self.calc_forces(self.pressure_pre)
        acceleration = (net_force + drag_force) / total_mass
        self.vertical_velocity += acceleration * self.delta_t
        print(f"vertical_velocity: {self.vertical_velocity}")
            

    def get_air_density(self, altitude: float) -> float:
        return self.air_density0 * np.exp(-altitude / 7000.0)

    def altitude_to_pressure(self, altitude_m: float) -> float:
        temperature = self.get_temperature(altitude_m)
        return self.P0 * np.exp(-self.air_molar_mass * self.gravity * altitude_m / (self.R_u * temperature))

    def get_temperature(self, altitude: float) -> float:
        return self.T0 - self.L * altitude

    def internal_controller(self, v_init, v_des, dt, pressure):
        # delta_t = 1
        v_current = v_init
        pressure_current = pressure
        pressure_pre = pressure
        alt_current = self.alt

        initial_mass = self.helium_mass
        initial_sand = self.sand

        Kp_vel = 0.1 #0.1
        Kd_vel = 0.05 #0.02
        Ki_vel = 0.001 #0.01

        vel_error_prev = 0.0
        vel_error_integral = 0.0

        for i in range(0, dt, self.delta_t):
            t_progress = i / dt
            v_target = v_init + t_progress * (v_des - v_init)
            self.volume = self.volume * (pressure_pre / pressure_current) / (self.temperature_pre / self.temperature)
            self.helium_density = self.helium_mass / self.volume
            alt_target = alt_current + v_target * self.delta_t

            buoyancy, weight, net_force, drag_force, rho_air, total_mass = self.calc_forces(pressure_current)
            F_current = buoyancy - weight + drag_force


            velocity_error = v_target - v_current
            if np.isfinite(vel_error_prev):
                velocity_error_rate = (velocity_error - vel_error_prev) / self.delta_t
            else:
                velocity_error_rate = 0.0
            vel_error_integral += velocity_error * self.delta_t
            vel_error_integral = np.clip(vel_error_integral, -10.0, 10.0)

            acc_des = Kp_vel * velocity_error + Ki_vel * vel_error_integral + Kd_vel * velocity_error_rate
            F_des = total_mass * acc_des
            alt_error = alt_target - alt_current

            dSand = 0.0
            dMass = 0.0

            if F_des < F_current:
                dVolume = (F_current - F_des) / (rho_air * self.gravity)
                alt_factor = 1.0 + 0.05 * np.sign(alt_error)
                dVolume *= alt_factor
                if alt_error < -0.2:
                    dVolume *= 0.3
                dVolume = np.clip(dVolume, 0.0, self.max_vent_rate * 0.05)
                delta_n = (pressure_current * 100) * dVolume / (self.R_u * self.temperature)
                dMass = delta_n * self.helium_molar_mass
                self.helium_mass -= dMass
                self.volume = self.helium_mass / self.helium_density
            else:
                dSand = (F_des - F_current) / self.gravity
                alt_factor = 1.0 - 0.05 * np.sign(alt_error)
                dSand *= alt_factor
                if alt_error > 0.2:
                    dSand *= 0.3
                dSand = np.clip(dSand, 0.0, self.max_sand_rate * 0.05)
                self.sand -= dSand
            
            if abs(velocity_error) < 0.02:
                dMass *= 0.3
                dSand *= 0.3
            if abs(v_current) < 0.05:
                dMass *= 0.7
                dSand *= 0.7

            self.volume = max(self.volume, 1e-6)
            buoyancy, weight, net_force, drag_force, rho_air, total_mass = self.calc_forces(pressure_current)
            acceleration = (net_force + drag_force) / total_mass
            v_current += acceleration * self.delta_t
            v_current = np.clip(v_current, -self.max_velocity, self.max_velocity)
            alt_current += v_current * self.delta_t / 1000
            # alt_current = np.clip(alt_current, 5.0, 25.0)

            pressure_pre = pressure_current
            self.temperature_pre = self.temperature
            self.temperature = self.get_temperature(alt_current * 1000)
            pressure_current = self.altitude_to_pressure(alt_current * 1000)
            vel_error_prev = velocity_error

        total_dMass = initial_mass - self.helium_mass
        total_dSand = initial_sand - self.sand
        self.alt = alt_current
        self.vertical_velocity = v_current
        return total_dMass, total_dSand

    def simplified_internal_controller(self, v_init, v_des, dt, pressure):
        """
        Simplified internal controller that eliminates the inner loop for faster computation.
        Uses direct calculations instead of iterative PID control.
        """
        # Simple proportional control based on desired velocity
        velocity_error = v_des - v_init
        
        # Direct calculation of control actions without inner loop
        if v_des < 0:  # Want to go up (ascend)
            # Drop sand to reduce weight
            sand_factor = abs(velocity_error) * 0.1  # Simplified relationship
            dSand = min(sand_factor * dt, self.max_sand_rate * dt)
            dSand = min(dSand, self.sand)  # Can't drop more sand than we have
            dMass = 0.0
        else:  # Want to go down or stay steady
            # Vent helium to reduce volume
            volume_factor = abs(velocity_error) * 0.05  # Simplified relationship
            dVolume = min(volume_factor * dt, self.max_vent_rate * dt)
            dVolume = min(dVolume, self.volume * 0.1)  # Can't vent more than 10% of volume
            
            # Convert volume change to mass change
            delta_n = (pressure * 100) * dVolume / (self.R_u * self.temperature)
            dMass = delta_n * self.helium_molar_mass
            dMass = min(dMass, self.helium_mass)  # Can't vent more helium than we have
            dSand = 0.0
        
        # Apply changes directly
        self.helium_mass -= dMass
        self.sand -= dSand
        
        alt_change = v_des * dt / 1000  # Convert to km
        self.alt += alt_change
        self.alt = np.clip(self.alt, 5.0, 25.0)
        
        # Update temperature and pressure
        self.temperature = self.get_temperature(self.alt * 1000)
        
        # Update vertical velocity
        self.vertical_velocity = v_des
        
        return dMass, dSand

    def drag_force(self, pressure: float, net_force: float, vel: float = None):
        if vel is None:
            vel = self.vertical_velocity
        if not np.isfinite(vel) or not np.isfinite(pressure):
            return 0.0
        cross_section = np.pi * (3 * self.volume / (4 * np.pi)) ** (2/3)
        rho_air = (pressure * 100) / (self.R * self.temperature)
        kinematic_viscosity = 1.5e-5 * (self.P0 / pressure)

        if abs(vel) < 1e-6:
            reynolds_number = 1e-6
        else:
            reynolds_number = abs(vel) * np.sqrt(cross_section) / kinematic_viscosity

        if reynolds_number < 1:
            drag_coefficient = 24.0
        elif reynolds_number < 1000:
            drag_coefficient = 24/reynolds_number * (1 + 0.15 * reynolds_number**0.687)
        else:
            drag_coefficient = 0.47

        drag_coefficient *= (1 + 0.05 * (self.alt / 10))
        if abs(vel) < 1e-6:
            drag_force = 0.0
        else:
            drag_force = 0.5 * rho_air * vel**2 * drag_coefficient * cross_section
            linear_drag = 0.1 * rho_air * abs(vel) * cross_section
            drag_force += linear_drag

        drag_force *= np.sign(net_force) * -1
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
        
        self.du_km = wind.u * dt / 1000
        self.dv_km = wind.v * dt / 1000
        self.lat, self.lon = wind_displacement_to_position(self.lat, self.lon, self.du_km, self.dv_km)
        
        pressure = self.altitude_to_pressure(self.alt*1000)  # hPa
        dVolume, dSand = self.internal_controller(self.vertical_velocity, action, dt, pressure)

    def simplified_step(self, wind: WindVector, dt: float, action: float = 0.0) -> None:
        """
        Simplified step function that uses the simplified internal controller
        for faster computation with less accuracy.
        """
        # Horizontal motion (same as original)
        self.du_km = wind.u * dt / 1000
        self.dv_km = wind.v * dt / 1000
        self.lat, self.lon = wind_displacement_to_position(self.lat, self.lon, self.du_km, self.dv_km)
        
        # Get current pressure
        pressure = self.altitude_to_pressure(self.alt*1000)  # hPa
        
        # Use simplified controller
        dMass, dSand = self.simplified_internal_controller(self.vertical_velocity, action, dt, pressure)
 
   
    def calc_forces(self, pressure: float):
        rho_air = (pressure * 100) / (self.R * self.temperature)
        total_mass = self.balloon_mass + self.sand + self.helium_mass
        buoyancy = rho_air * self.volume * self.gravity
        weight = total_mass * self.gravity
        net_force = buoyancy - weight
        drag_force = self.drag_force(pressure, net_force)
        return buoyancy, weight, net_force, drag_force, rho_air, total_mass
        
    def step_with_resource(self, dt: float, dHelium: float = 0., dSand: float = 0., wind: WindVector = None):
        # Handle wind with default value
        if wind is None:
            wind = WindVector(0., 0.)
        # Handle horizontal motion
        self.du_km = wind.u * dt / 1000
        self.dv_km = wind.v * dt / 1000
        self.lat, self.lon = wind_displacement_to_position(self.lat, self.lon, self.du_km, self.dv_km)
        alt_current = self.alt
        v_current = self.vertical_velocity  # Initialize v_current

        
        delta_Helium = dHelium / dt * self.delta_t
        delta_Sand = dSand / dt * self.delta_t
        pressure_current = self.altitude_to_pressure(self.alt*1000)

        for _ in range(0, dt, self.delta_t):
            self.sand -= delta_Sand
            self.helium_mass -= delta_Helium 

            self.volume = self.helium_mass / self.helium_density
            self.volume = max(self.volume, 1e-6)
            buoyancy, weight, net_force, drag_force, rho_air, total_mass = self.calc_forces(pressure_current)
            
            acceleration = (net_force + drag_force) / total_mass
            v_current += acceleration * self.delta_t
            v_current = np.clip(v_current, -self.max_velocity, self.max_velocity)
            alt_current += v_current * self.delta_t / 1000

            self.pressure_pre = pressure_current
            self.temperature_pre = self.temperature
            self.temperature = self.get_temperature(alt_current * 1000)
            pressure_current = self.altitude_to_pressure(alt_current * 1000)
            self.volume = self.volume * (self.pressure_pre / pressure_current) / (self.temperature_pre / self.temperature)
            self.helium_density = self.helium_mass / self.volume
            # Check if resources are depleted
            if self.helium_mass <= 0:
                dHelium, delta_Helium = 0.0, 0.0
            if self.sand <= 0:
                dSand, delta_Sand = 0.0, 0.0
                
            if alt_current <= 0:
                return f"Balloon Crashed", self.helium_mass, self.sand,[buoyancy, -weight, drag_force, net_force+drag_force]
        self.alt = alt_current
        self.vertical_velocity = v_current
        if self.volume > self.max_volume:
            return f"Balloon burst : {self.volume}", self.helium_mass, self.sand,[buoyancy, -weight, drag_force, net_force+drag_force]
        if self.helium_mass <= 0:
            return f"Running out of resource: helium", self.helium_mass, self.sand, [buoyancy, -weight, drag_force, net_force+drag_force]
        if self.sand <= 0:
            return f"Running out of resource: sand", self.helium_mass, self.sand, [buoyancy, -weight, drag_force, net_force+drag_force]
        return None, self.helium_mass, self.sand, [buoyancy, -weight, drag_force, net_force+drag_force]
