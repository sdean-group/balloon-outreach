import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
from env.wind_field import WindField as DefaultWindField, WindVector as DefaultWindVector
from env.ERA_wind_field import WindField as ERAWindField, WindVector as ERAWindVector
from env.balloon import Balloon
import xarray as xr
import datetime as dt
import copy

      
class BaseBalloonEnvironment:
    def __init__(self, balloon: Balloon, dt: float = 60, target_lat: float = 500, target_lon: float = -100, target_alt: float = 12):
        self.balloon = balloon
        self.init_state = np.array([balloon.lat, balloon.lon, balloon.alt])
        self.dt = dt
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.target_alt = target_alt
        self.current_time = 0.0
        self.trajectory = {'lat': [], 'lon': [], 'alt': []}
        self.trajectory['lat'].append(self.balloon.lat)
        self.trajectory['lon'].append(self.balloon.lon)
        self.trajectory['alt'].append(self.balloon.alt)
    def reset(self) -> np.ndarray:
        self.current_time = 0.0
        self.trajectory = {'lat': [], 'lon': [], 'alt': []}
        self.trajectory['lat'].append(self.balloon.lat)
        self.trajectory['lon'].append(self.balloon.lon)
        self.trajectory['alt'].append(self.balloon.alt)
        return self._get_state()
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, str]:
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        pressure = self.balloon.altitude_to_pressure(self.balloon.alt)
        wind = self.wind_field.get_wind(
            self.balloon.lon,
            self.balloon.lat,
            pressure,
            self.current_time
        )
        self.balloon.step(wind, self.dt, action_value)
        self.current_time += self.dt / 3600
        state = self._get_state()
        reward = self._get_reward()
        done, reason = self._is_done()
        return state, reward, done, reason
    def simplified_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, str]:
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        pressure = self.balloon.altitude_to_pressure(self.balloon.alt)
        wind = self.wind_field.get_wind(
            self.balloon.lon,
            self.balloon.lat,
            pressure,
            self.current_time
        )
        self.balloon.simplified_step(wind, self.dt, action_value)
        self.current_time += self.dt / 3600
        state = self._get_state()
        reward = self._get_reward()
        done, reason = self._is_done()
        return state, reward, done, reason
    
    def _is_done(self) -> Tuple[bool, str]:
        lat_diff = self.balloon.lat - self.target_lat
        lon_diff = self.balloon.lon - self.target_lon
        distance = np.sqrt(lat_diff**2 + lon_diff**2)
        if self.balloon.helium_mass <= 0:
            return True, "No helium left"
        elif self.balloon.sand <= 0:
            return True, "No sand left"
        elif distance < 0.1:
            return True, "Reached target"
        elif self.current_time >= 24:
            return True, "Time limit reached"
        else:
            return False, ""
    def _get_balloon_state(self) -> np.ndarray:
        balloon_state = np.array([
            self.balloon.lat,
            self.balloon.lon,
            self.balloon.alt,
            self.balloon.volume / self.balloon.max_volume,
            self.balloon.sand / self.balloon.max_sand,
            self.balloon.vertical_velocity,
            self.current_time
        ])
        return balloon_state
    def _get_state(self) -> np.ndarray:
        balloon_state = self._get_balloon_state()
        wind_column = self._get_wind_column()
        return np.concatenate([balloon_state, wind_column])
    def _get_wind_column(self) -> np.ndarray:
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
    
    def _get_reward(self) -> float:
        lat_diff = self.balloon.lat - self.target_lat
        lon_diff = self.balloon.lon - self.target_lon
        distance = np.sqrt(lat_diff**2 + lon_diff**2)
        return -distance
    
    def rollout_sequence_mppi_target(self,control_seq:np.ndarray, max_steps:int) -> Tuple[float, List[float]]:
        """
        Executes a sequence of control steps in the environment with the target objective. Resets environment to initial state before returning.
        Args:
            control_seq: Control sequence to evaluate
            max_steps: Maximum amount of iterations to step through

        Returns:
            Total cost of trajectory, trajectory
        """
        # Saving initial states before stepping
        alt = self.balloon.alt
        lon = self.balloon.lon
        lat = self.balloon.lat
        temperature = self.balloon.temperature
        vertical_velocity = self.balloon.vertical_velocity
        volume = self.balloon.volume
        helium_density = self.balloon.helium_density
        helium_mass = self.balloon.helium_mass
        sand = self.balloon.sand
        current_time = self.current_time

        # Rollout the control sequence
        total_cost = 0.0
        trajectory = []
        # Rollout the control sequence
        for t in range(max_steps):
            action = np.array([control_seq[t]])
            
            # Take step in environment
            state, reward, done, _ = self.simplified_step(action)
            trajectory.append((state[0], state[1]))
            
            # Accumulate cost (negative reward)
            total_cost -= reward
            
            # Early termination if episode ends
            if done:
                break

        # Reset states
        self.balloon.alt = alt
        self.balloon.lon = lon
        self.balloon.lat = lat
        self.balloon.temperature = temperature
        self.balloon.vertical_velocity = vertical_velocity
        self.balloon.volume = volume
        self.balloon.helium_density = helium_density
        self.balloon.helium_mass = helium_mass
        self.balloon.sand = sand
        self.current_time = current_time
        
        return total_cost, trajectory
    
    def rollout_sequence_mppi_fly(self,control_seq:np.ndarray, max_steps:int) -> Tuple[float, List[float]]:
        """
        Executes a sequence of control steps in the environment with the fly as far objective. Resets environment to initial state before returning.
        Args:
            control_seq: Control sequence to evaluate
            max_steps: Maximum amount of iterations to step through

        Returns:
            Total cost of trajectory, trajectory
        """
        # Saving initial states before stepping
        alt = self.balloon.alt
        lon = self.balloon.lon
        lat = self.balloon.lat
        temperature = self.balloon.temperature
        vertical_velocity = self.balloon.vertical_velocity
        volume = self.balloon.volume
        helium_density = self.balloon.helium_density
        helium_mass = self.balloon.helium_mass
        sand = self.balloon.sand
        current_time = self.current_time

        # Rollout the control sequence
        total_cost = 0.0
        trajectory = []
        # Rollout the control sequence
        for t in range(max_steps):
            action = np.array([control_seq[t]])
            
            # Take step in environment
            state, reward, done, _ = self.simplified_step(action)
            trajectory.append((state[0], state[1]))
            
            # Accumulate cost (euclidean distance from current point and initial point)
            total_cost -= np.linalg.norm(state[:3] - self.init_state)
            
            # Early termination if episode ends
            if done:
                break

        # Reset states
        self.balloon.alt = alt
        self.balloon.lon = lon
        self.balloon.lat = lat
        self.balloon.temperature = temperature
        self.balloon.vertical_velocity = vertical_velocity
        self.balloon.volume = volume
        self.balloon.helium_density = helium_density
        self.balloon.helium_mass = helium_mass
        self.balloon.sand = sand
        self.current_time = current_time
        
        return total_cost, trajectory

    def rollout_sequence_mppi_with_cost(self,vel_seq:np.ndarray, acc_seq:np.ndarray, max_steps:int, cost_fcn, *params) -> Tuple[float, List[float]]:
        """
        Executes a sequence of control steps in the environment with a more complex cost function. The cost function should handle the target or fly as far objective. Resets environment to initial state before returning.
        Args:
            vel_seq: Array of planned velocities.
            acc_seq: Array of planned accelerations.
            max_steps: Maximum amount of iterations to step through
            cost_fcn: `cost_fcn` can have any arguments, but must end with next_state (np.ndarray), acceleration(float), timestep(int)
            *params: Any additional arguments that cost_fcn takes

        Returns:
            Total cost of trajectory, trajectory
        """
        # Saving initial states before stepping
        alt = self.balloon.alt
        lon = self.balloon.lon
        lat = self.balloon.lat
        temperature = self.balloon.temperature
        vertical_velocity = self.balloon.vertical_velocity
        volume = self.balloon.volume
        helium_density = self.balloon.helium_density
        helium_mass = self.balloon.helium_mass
        sand = self.balloon.sand
        current_time = self.current_time

        total_cost = 0.0
        trajectory = []
        for t in range(max_steps):
            # Casting action and acc from scalars to arrays
            action = np.array([vel_seq[t]])
            acc = np.array([acc_seq[t]])
            # Take step in environment
            # next_state, _, done, _ = self.step(action)
            next_state, _, done, _ = self.simplified_step(action)
            
            trajectory.append((next_state[0], next_state[1]))
            # Custom cost function
            cost = cost_fcn(*params, next_state, acc[0], t)
            total_cost += cost
            
            # Early termination if episode ends
            if done:
                # Add penalty for early termination
                total_cost += 1000.0
                break
        # Reset states
        #print(f"Alt before: {alt}, Alt after: {self.balloon.alt}")
        self.balloon.alt = alt
        self.balloon.lon = lon
        self.balloon.lat = lat
        self.balloon.temperature = temperature
        self.balloon.vertical_velocity = vertical_velocity
        self.balloon.volume = volume
        self.balloon.helium_density = helium_density
        self.balloon.helium_mass = helium_mass
        self.balloon.sand = sand
        self.current_time = current_time
        
        return total_cost, trajectory
    
    def render(self):
        # Optionally implement or override in child
        pass

class BalloonEnvironment(BaseBalloonEnvironment):
    def __init__(self, initial_lat=0.0, initial_lon=0.0, initial_alt=10.0, dt=60, target_lat=500, target_lon=-100, target_alt=12):
        balloon = Balloon(initial_lat=initial_lat, initial_lon=initial_lon, initial_alt=initial_alt)
        super().__init__(balloon=balloon, dt=dt, target_lat=target_lat, target_lon=target_lon, target_alt=target_alt)
        self.wind_field = DefaultWindField()


class BalloonERAEnvironment(BaseBalloonEnvironment):
    """Environment for balloon navigation using ERA5 wind field"""
    def __init__(self, ds: xr.Dataset, start_time: dt.datetime, noise_seed: int = None, initial_lat: float = 0.0, initial_lon: float = 0.0, initial_alt: float = 10.0, target_lat: float = 17, target_lon: float = 17, target_alt: float = 12, dt=60, viz = True):
        # balloon = Balloon(initial_lat=42.6, initial_lon=-76.5, initial_alt=10.0)
        balloon = Balloon(initial_lat=initial_lat, initial_lon=initial_lon, initial_alt=initial_alt)
        super().__init__(balloon=balloon, dt=dt, target_lat=target_lat, target_lon=target_lon, target_alt=target_alt)
        self.wind_field = ERAWindField(ds=ds, start_time=start_time, noise_seed=noise_seed)
        self.ds = ds
        self.start_time = start_time
        self.noise_seed = noise_seed
        self.viz = viz
        if self.viz:
            self.fig = plt.figure(figsize=(15, 5))
            self.ax1 = self.fig.add_subplot(131, projection='3d')  # 3D Position plot
            self.ax2 = self.fig.add_subplot(132)  # Resources plot
            self.ax3 = self.fig.add_subplot(133)  # Wind profile plot
            plt.tight_layout()

    def render(self) -> None:
        if self.viz:
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.trajectory['lat'].append(self.balloon.lat)
            self.trajectory['lon'].append(self.balloon.lon)
            self.trajectory['alt'].append(self.balloon.alt)
            lats = np.array(self.trajectory['lat'])
            lons = np.array(self.trajectory['lon'])
            alts = np.array(self.trajectory['alt'])
            if len(lats) > 1:
                self.ax1.plot(lons, lats, alts, 'b-', linewidth=2, label='Trajectory')
            self.ax1.scatter(lons[-1], lats[-1], alts[-1], c='red', marker='o', s=150, label='Current Position')
            self.ax1.scatter(self.target_lon, self.target_lat, 10, c='green', marker='*', s=200, label='Target')
            self.ax1.set_xlabel('Longitude (km)')
            self.ax1.set_ylabel('Latitude (km)')
            self.ax1.set_zlabel('Altitude (km)')
            self.ax1.set_title('Balloon Navigation')
            self.ax1.set_xlim(-10, 10)
            self.ax1.set_ylim(-10, 10)
            self.ax1.set_zlim(0, 25)
            self.ax1.legend()
            self.ax2.bar(['Helium', 'Sand'], [self.balloon.helium_mass/self.balloon.initial_helium_mass, self.balloon.sand/self.balloon.max_sand])
            self.ax2.set_ylim(0, 1)
            self.ax2.set_title('Resources')
            wind_column = self._get_wind_column()
            u_winds = wind_column[::2]
            v_winds = wind_column[1::2]
            wind_speeds = np.sqrt(u_winds**2 + v_winds**2)
            self.ax3.plot(wind_speeds, self.wind_field.pressure_levels, 'b-')
            self.ax3.grid(True)
            self.ax3.set_title('Wind Speed Profile')
            self.ax3.set_xlabel('Wind Speed (m/s)')
            self.ax3.set_ylabel('Pressure (hPa)')
            self.ax3.invert_yaxis()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.1)

