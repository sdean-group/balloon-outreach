import xarray as xr
import numpy as np
from dataclasses import dataclass
import datetime as dt
import jax
from jax import numpy as jnp

import env.simplex_wind_noise as simplex_wind_noise
import env.units as units

@dataclass
class WindVector:
    """Wind vector with u (east-west) and v (north-south) components."""
    u: float  # m/s, positive eastward
    v: float  # m/s, positive northward

class WindField:
    """
    Wind field backed by an ERA5 xarray.Dataset, with added Simplex noise.
    Users must provide a fixed start_time; queries use elapsed_time (timedelta).

    Interface:
      - self.pressure_levels: iterable of pressure levels (hPa)
      - get_wind(lon, lat, pressure, elapsed_time) -> WindVector (with noise)
    """
    def __init__(
        self,
        ds: xr.Dataset,
        start_time: dt.datetime,
        noise_seed: int = None
    ):
        """
        Parameters
        ----------
        ds : xarray.Dataset
            Must contain variables 'u' and 'v' with dims
            (valid_time, pressure_level, latitude, longitude).
        start_time : datetime.datetime
            Reference time corresponding to zero elapsed_time.
        noise_seed : int, optional
            Seed for the noise model. If None, a random seed is chosen.
        """
        self.ds = ds
        self.start_time = start_time
        self.time_coord   = "valid_time"
        self.plevel_coord = "pressure_level"
        self.lat_coord    = "latitude"
        self.lon_coord    = "longitude"
        self.pressure_levels = self.ds[self.plevel_coord].values

        # Initialize and seed the noise model with a JAX PRNGKey + start_time
        self.noise_model = SimplexWindNoise()
        seed = noise_seed if noise_seed is not None else np.random.randint(0, 2**31 - 1)
        key = jax.random.PRNGKey(seed)
        self.noise_model.reset(key, self.start_time)

    def reset_noise(self, noise_seed: int = None):
        """
        Re-seed the underlying noise model for a fresh realization,
        preserving the original start_time reference.
        """
        seed = noise_seed if noise_seed is not None else np.random.randint(0, 2**31 - 1)
        key = jax.random.PRNGKey(seed)
        self.noise_model.reset(key, self.start_time)

    def get_wind(
        self,
        lon: float,
        lat: float,
        pressure: float,
        elapsed_time: dt.timedelta
    ) -> WindVector:
        """
        Interpolate and return the noisy wind vector at a given location and elapsed time.

        Parameters
        ----------
        lon : float
            Longitude in degrees_east.
        lat : float
            Latitude in degrees_north.
        pressure : float
            Pressure level in hPa (must be in self.pressure_levels).
        elapsed_time : datetime.timedelta
            Time difference from `start_time` given at init.

        Returns
        -------
        WindVector
            Eastward (u) and northward (v) wind components in m/s,
            with added Simplex noise.
        """
        # Compute actual timestamp for interpolation
        current_time = self.start_time + elapsed_time

        # Linear interpolation from ERA5
        u_interp = self.ds["u"].interp(
            {self.lon_coord: lon,
             self.lat_coord: lat,
             self.plevel_coord: pressure,
             self.time_coord: current_time},
            method="linear",
            kwargs={"fill_value": "extrapolate"}
        )
        v_interp = self.ds["v"].interp(
            {self.lon_coord: lon,
             self.lat_coord: lat,
             self.plevel_coord: pressure,
             self.time_coord: current_time},
            method="linear",
            kwargs={"fill_value": "extrapolate"}
        )

        # Base forecast values
        u0 = float(u_interp.values.item())
        v0 = float(v_interp.values.item())

        # Convert lon/lat (in degrees) into Distance for noise
        # Approximate: 1° ≈ 111 km
        x_dist = units.Distance(kilometers=lon * 111.0)
        y_dist = units.Distance(kilometers=lat * 111.0)

        # Generate noise using Distance and elapsed_time
        noise = self.noise_model.get_wind_noise(
            x_dist,
            y_dist,
            pressure,
            elapsed_time
        )
        # Extract m/s from Velocity objects
        u_noise = noise.u.meters_per_second
        v_noise = noise.v.meters_per_second

        return WindVector(
            u=u0 + u_noise,
            v=v0 + v_noise
        )

class SimplexWindNoise:
    """Wrapper for BLE’s U/V NoisyWindComponent models."""

    def __init__(self):
        self.noise_u = simplex_wind_noise.NoisyWindComponent(which='u')
        self.noise_v = simplex_wind_noise.NoisyWindComponent(which='v')

    def reset(self, key: jnp.ndarray, date_time: dt.datetime) -> None:
        """
        Reset the U and V noise components using the provided JAX PRNGKey.

        Args:
          key: A JAX PRNGKey for randomness.
          date_time: Starting datetime (unused by Simplex noise).
        """
        del date_time
        noise_u_key, noise_v_key = jax.random.split(key, num=2)
        self.noise_u.reset(noise_u_key)
        self.noise_v.reset(noise_v_key)

    def get_wind_noise(
        self,
        x: units.Distance,
        y: units.Distance,
        pressure: float,
        elapsed_time: dt.timedelta
    ) -> WindVector:
        """Generate and return the noise vector (u,v) at elapsed_time."""
        u_n = self.noise_u.get_noise(x, y, pressure, elapsed_time)
        v_n = self.noise_v.get_noise(x, y, pressure, elapsed_time)
        return WindVector(
            u=units.Velocity(meters_per_second=u_n),
            v=units.Velocity(meters_per_second=v_n)
        )
