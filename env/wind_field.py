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

    def get_wind(self, x: float, y: float, pressure: float, time: float) -> WindVector:
        """
        Interpolate wind vector at a given (lon, lat, pressure, time).
        """
        point = np.array([time, y, x, pressure])

        # Interpolate u and v components
        u_interp = interpn(self.grid, self.u_data, point,
                           method='linear', bounds_error=False, fill_value=None)
        v_interp = interpn(self.grid, self.v_data, point,
                           method='linear', bounds_error=False, fill_value=None)

        return WindVector(float(u_interp), float(v_interp))