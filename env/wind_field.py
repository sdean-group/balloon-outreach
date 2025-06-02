# import numpy as np
# from dataclasses import dataclass
# from typing import Tuple, Dict, List
#
# @dataclass
# class WindVector:
#     """Wind vector with u (east-west) and v (north-south) components"""
#     u: float  # m/s, positive is eastward
#     v: float  # m/s, positive is northward
#
# @dataclass
# class WindPoint:
#     """A point in the wind field with position, pressure, time and wind vector"""
#     x: float  # longitude
#     y: float  # latitude
#     pressure: float  # hPa
#     time: float  # hours
#     wind: WindVector
#
# class WindField:
#     """Wind field that stores and interpolates wind vectors at specific points"""
#     def __init__(self):
#         # Store wind points in a dictionary for quick lookup
#         self.wind_points: Dict[Tuple[float, float, float, float], WindVector] = {}
#
#         # Grid parameters
#         self.x_range = (-180, 180)  # longitude range
#         self.y_range = (-90, 90)    # latitude range
#         self.pressure_levels = [1000, 850, 700, 500, 300, 200, 100]  # hPa
#         self.time_steps = np.arange(0, 24, 1)  # hourly steps
#
#         # Initialize with some example data
#         self._initialize_example_data()
#
#     def _initialize_example_data(self):
#         """Initialize the wind field with some example data"""
#         # Create a simple example wind field
#         for x in np.linspace(self.x_range[0], self.x_range[1], 10):  # 10 points in longitude
#             for y in np.linspace(self.y_range[0], self.y_range[1], 10):  # 10 points in latitude
#                 for p in self.pressure_levels:
#                     for t in self.time_steps:
#                         # Create some example wind patterns
#                         u = 10 * np.sin(x/30) * np.cos(y/30) * (1 + 0.2 * np.sin(2*np.pi*t/24))
#                         v = 10 * np.cos(x/30) * np.sin(y/30) * (1 + 0.2 * np.sin(2*np.pi*t/24))
#
#                         # Store the wind point
#                         self.wind_points[(x, y, p, t)] = WindVector(u, v)
#
#     def _find_nearest_points(self, x: float, y: float, pressure: float, time: float) -> List[WindPoint]:
#         """Find the 16 nearest points in the 4D space for interpolation"""
#         points = []
#
#         # Find nearest grid points
#         x_points = sorted([p for p in np.linspace(self.x_range[0], self.x_range[1], 10)
#                           if abs(p - x) <= 20])[:2]
#         y_points = sorted([p for p in np.linspace(self.y_range[0], self.y_range[1], 10)
#                           if abs(p - y) <= 20])[:2]
#         p_points = sorted([p for p in self.pressure_levels
#                           if abs(p - pressure) <= 200])[:2]
#         t_points = sorted([t for t in self.time_steps
#                           if abs(t - time) <= 3])[:2]
#
#         # Create all combinations of nearest points
#         for xp in x_points:
#             for yp in y_points:
#                 for pp in p_points:
#                     for tp in t_points:
#                         if (xp, yp, pp, tp) in self.wind_points:
#                             points.append(WindPoint(xp, yp, pp, tp,
#                                                   self.wind_points[(xp, yp, pp, tp)]))
#
#         return points
#
#     def _trilinear_interpolation(self, points: List[WindPoint],
#                                 x: float, y: float,
#                                 pressure: float, time: float) -> WindVector:
#         """Perform trilinear interpolation between points"""
#         if not points:
#             return WindVector(0, 0)  # Return zero wind if no points found
#
#         # Calculate weights for each point
#         total_weight = 0
#         u_sum = 0
#         v_sum = 0
#
#         for point in points:
#             # Calculate distance in 4D space
#             dx = abs(point.x - x) / 20  # Normalize by typical grid spacing
#             dy = abs(point.y - y) / 20
#             dp = abs(point.pressure - pressure) / 200
#             dt = abs(point.time - time) / 3
#
#             # Weight decreases with distance
#             weight = 1 / (1 + dx + dy + dp + dt)
#
#             u_sum += point.wind.u * weight
#             v_sum += point.wind.v * weight
#             total_weight += weight
#
#         if total_weight == 0:
#             return WindVector(0, 0)
#
#         return WindVector(u_sum/total_weight, v_sum/total_weight)
#
#     def get_wind(self, x: float, y: float, pressure: float, time: float) -> WindVector:
#         """
#         Get wind vector at given position, pressure, and time using interpolation
#
#         Args:
#             x: Longitude in degrees
#             y: Latitude in degrees
#             pressure: Pressure in hPa
#             time: Time in hours
#
#         Returns:
#             WindVector: Interpolated wind vector
#         """
#         # Find nearest points
#         nearest_points = self._find_nearest_points(x, y, pressure, time)
#
#         # Interpolate between points
#         return self._trilinear_interpolation(nearest_points, x, y, pressure, time)
#
#     def add_wind_point(self, x: float, y: float, pressure: float,
#                       time: float, u: float, v: float):
#         """Add a new wind point to the field"""
#         self.wind_points[(x, y, pressure, time)] = WindVector(u, v)

import numpy as np
from dataclasses import dataclass
from scipy.interpolate import interpn


@dataclass
class WindVector:
    """Wind vector with u (east-west) and v (north-south) components."""
    u: float  # m/s, positive eastward
    v: float  # m/s, positive northward


class WindField:
    """
    Wind field using 4D (time, lat, lon, pressure) grid-based data.
    Smooth variations in space/time and no zero-wind points.
    """
    def __init__(self,
                 grid_t: np.ndarray = None,
                 grid_y: np.ndarray = None,
                 grid_x: np.ndarray = None,
                 grid_p: np.ndarray = None):
        """
        Initialize the wind field with optional grids.
        If grids are not provided, default values are used.
        """
        # Default grids
        if grid_t is None:
            grid_t = np.arange(0, 24, 1)  # hourly (24 hours)
        if grid_y is None:
            grid_y = np.linspace(-90, 90, 10)  # 10 latitudes
        if grid_x is None:
            grid_x = np.linspace(-180, 180, 10)  # 10 longitudes
        if grid_p is None:
            grid_p = np.array([1000, 850, 700, 500, 300, 200, 100])  # 7 levels

        self.grid = (grid_t, grid_y, grid_x, grid_p)
        self.u_data = None
        self.v_data = None
        self.pressure_levels = grid_p
        self.initialize_wind_field()

    def initialize_wind_field(self):
        """
        Create a physically plausible wind field:
        - Smooth temporal and spatial variations
        - Opposite directions at different pressure levels
        - Small random noise for realism
        - Offset to ensure no (u=0, v=0) points
        """
        T, Y, X, P = np.meshgrid(self.grid[0], self.grid[1],
                                  self.grid[2], self.grid[3],
                                  indexing='ij')

        # Smooth variation in time, latitude, longitude
        time_wave = np.sin(2 * np.pi * T / self.grid[0][-1])
        lat_wave = np.sin(np.deg2rad(Y))
        lon_wave = np.cos(np.deg2rad(X))

        # Pressure-dependent sign for opposite direction
        pressure_sign = np.where(P >= 500, 1, -1)

        # Base wind speed amplitude
        base_speed = 5.0  # m/s

        # Core wind fields
        u_base = base_speed * time_wave * lat_wave * pressure_sign
        v_base = base_speed * time_wave * lon_wave * pressure_sign

        # Add small noise for natural variability
        noise_scale = 0.1
        u_noise = noise_scale * np.random.randn(*u_base.shape)
        v_noise = noise_scale * np.random.randn(*v_base.shape)

        # Add a small offset to ensure no (0,0) points
        min_offset = 0.5  # m/s
        u_offset = min_offset * np.ones_like(u_base)
        v_offset = min_offset * np.ones_like(v_base)

        self.u_data = u_base + u_noise + u_offset
        self.v_data = v_base + v_noise + v_offset

    # def get_wind(self, x: float, y: float, pressure: float, time: float) -> WindVector:
    #     """
    #     Interpolate wind vector at a given (lon, lat, pressure, time).
    #     """
    #     point = np.array([time, y, x, pressure])

    #     # Interpolate u and v components
    #     u_interp = interpn(self.grid, self.u_data, point,
    #                        method='linear', bounds_error=False, fill_value=None)
    #     v_interp = interpn(self.grid, self.v_data, point,
    #                        method='linear', bounds_error=False, fill_value=None)

    #     return WindVector(float(u_interp), float(v_interp))
    def get_wind(self, x: float, y: float, pressure: float, time: float) -> WindVector:
    # """
    # Return wind for testing.
    # """
        return WindVector(np.random.uniform(20, 40), np.random.uniform(-2, 5))