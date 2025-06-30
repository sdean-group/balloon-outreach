import numpy as np

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
                 max_volume: float = 1000.0,
                 max_sand: float = 100):
        self.lat = initial_lat
        self.lon = initial_lon
        self.alt = initial_alt  # km
        self.volume = max_volume
        self.sand = max_sand
        self.max_volume = max_volume
        self.max_sand = max_sand

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
        
        self.R = 287.05
        self.R_u = 8.314 #Universal Gas Constant J/(mol·K)
        self.helium_molar_mass = 0.0040026 #kg/mol
        self.air_molar_mass = 0.02896 #kg/mol
        self.P0 = 1013.25 #Standard sea-level pressure hPa
        self.max_vent_rate = self.max_volume * 0.01  # m^3/dt
        self.max_sand_rate = self.max_sand * 0.01  # kg/dt
        self.pressure_pre = self.altitude_to_pressure(self.alt * 1000)
        self.max_velocity = 1.0  # m/s

        # New state
        self.initial_helium_mass = self.helium_density * max_volume
        self.helium_mass = self.helium_density * max_volume

    def get_air_density(self, altitude: float) -> float:
        return self.air_density0 * np.exp(-altitude / 7000.0)

    def altitude_to_pressure(self, altitude_m: float) -> float:
        return self.P0 * np.exp(-self.air_molar_mass * self.gravity * altitude_m / (self.R_u * self.temperature))

    def get_temperature(self, altitude: float) -> float:
        return self.T0 - self.L * altitude

    def internal_controller(self, v_init, v_des, dt, pressure):
        delta_t = 1
        v_current = v_init
        pressure_current = pressure
        pressure_pre = pressure
        alt_current = self.alt

        initial_mass = self.helium_mass
        initial_sand = self.sand

        Kp_vel = 0.09 #0.1
        Kd_vel = 0.06 #0.02
        Ki_vel = 0.005 #0.01

        vel_error_prev = 0.0
        vel_error_integral = 0.0

        for i in range(0, dt, delta_t):
            t_progress = i / dt
            v_target = v_init + t_progress * (v_des - v_init)
            self.volume = self.volume * (pressure_pre / pressure_current)
            self.helium_density = self.helium_mass / self.volume
            alt_target = alt_current + v_target * delta_t

            helium_mass = self.helium_mass
            total_mass = self.balloon_mass + self.sand + helium_mass
            rho_air = (pressure_current * 100) / (self.R * 220)

            buoyancy = rho_air * self.volume * self.gravity
            weight = total_mass * self.gravity
            drag = self.drag_force(pressure_current, buoyancy - weight)
            F_current = buoyancy - weight + drag

            velocity_error = v_target - v_current
            #print(f"vel_error {velocity_error}, vert vel {self.vertical_velocity}")
            velocity_error_rate = (velocity_error - vel_error_prev) / delta_t
            vel_error_integral += velocity_error * delta_t
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

            total_mass = self.balloon_mass + self.sand + self.helium_mass
            net_force = rho_air * self.volume * self.gravity - total_mass * self.gravity
            drag_force = self.drag_force(pressure_current, net_force)

            acceleration = (net_force + drag_force) / total_mass
            v_current += acceleration * delta_t
            alt_current += v_current * delta_t / 1000
            alt_current = np.clip(alt_current, 5.0, 25.0)

            pressure_pre = pressure_current
            self.temperature = self.get_temperature(alt_current * 1000)
            pressure_current = self.altitude_to_pressure(alt_current * 1000)
            vel_error_prev = velocity_error

        total_dMass = initial_mass - self.helium_mass
        total_dSand = initial_sand - self.sand
        self.alt = alt_current
        v_current = np.clip(v_current, -self.max_velocity, self.max_velocity)
        if v_current < -1e100:
            v_current = 0.0
        #print(f"vcurrent{v_current}")
        self.vertical_velocity = v_current
        return total_dMass, total_dSand

    def drag_force(self, pressure: float, net_force: float, vel: float = None):
        if vel is None:
            vel = self.vertical_velocity
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

#add nan check here?
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
        self.lat += wind.u * dt / 1000
        self.lon += wind.v * dt / 1000
        pressure = self.altitude_to_pressure(self.alt*1000)  # hPa
        dVolume, dSand = self.internal_controller(self.vertical_velocity, action, dt, pressure)
 