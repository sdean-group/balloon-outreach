import xarray as xr
import numpy as np
from dataclasses import dataclass
import datetime as dt
import jax
from jax import numpy as jnp
from scipy.interpolate import RegularGridInterpolator

import env.simplex_wind_noise as simplex_wind_noise
import env.units as units

@dataclass
class WindVector:
    """Wind vector with u (east-west) and v (north-south) components."""
    u: float  # m/s, positive eastward
    v: float  # m/s, positive northward

class WindField:
    def __init__(self, ds: xr.Dataset, start_time: dt.datetime, noise_seed: int = None, add_noise: bool = True):
        
        self.ds = ds
        self.start_time = start_time
        self.time_coord   = "valid_time"
        self.plevel_coord = "pressure_level"
        self.lat_coord    = "latitude"
        self.lon_coord    = "longitude"
        self.pressure_levels = self.ds[self.plevel_coord].values
        
        # grab the raw numpy arrays & coordinates
        # convert the datetime64 times into floats in hours since start_time
        time_vals = (
            (ds[self.time_coord].values.astype("datetime64[s]") 
             - np.datetime64(self.start_time, "s"))
            / np.timedelta64(1, "h")
        )
        p_vals   = ds[self.plevel_coord].values
        lat_vals = ds[self.lat_coord].values
        lon_vals = ds[self.lon_coord].values

        u_vals = ds["u"].values  # shape (ntime, nplev, nlat, nlon)
        v_vals = ds["v"].values

        # build two fast interpolators
        self._u_interp = RegularGridInterpolator(
            (time_vals, p_vals, lat_vals, lon_vals),
            u_vals,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        self._v_interp = RegularGridInterpolator(
            (time_vals, p_vals, lat_vals, lon_vals),
            v_vals,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        self.add_noise = add_noise

        # Initialize and seed the noise model with a JAX PRNGKey
        self.noise_model = SimplexWindNoise()
        seed = noise_seed if noise_seed is not None else np.random.randint(0, 2**31 - 1)
        key = jax.random.PRNGKey(seed)
        self.noise_model.reset(key)

    def enable_noise(self, noise_seed: int = None):
        """Turn noise back on (optionally reseeding)."""
        self.add_noise = True
        self.reset_noise(noise_seed)

    def disable_noise(self):
        """Turn noise off completely."""
        self.add_noise = False

    def reset_noise(self, noise_seed: int = None):
        """
        Re-seed the underlying noise model for a fresh realization.
        """
        seed = noise_seed if noise_seed is not None else np.random.randint(0, 2**31 - 1)
        key = jax.random.PRNGKey(seed)
        self.noise_model.reset(key)

    def get_wind(self, lon, lat, pressure, elapsed_time):

        # elapsed_time (float hours) already in hours
        pt = np.array([elapsed_time, pressure, lat, lon], dtype=float)
        u0 = float(self._u_interp(pt))
        v0 = float(self._v_interp(pt))

        if not self.add_noise: 
            # noise turned off
            return WindVector(u=u0, v=v0)
        
        else:
            # Compute actual timestamp for interpolation
            elapsed_time_dt = dt.timedelta(seconds=elapsed_time*3600)

            # Convert lon/lat (in degrees) into Distance for noise
            # Approximate: 1° ≈ 111 km
            x_dist = units.Distance(kilometers=lon * 111.0)
            y_dist = units.Distance(kilometers=lat * 111.0)

            # Generate noise using Distance and elapsed_time
            noise = self.noise_model.get_wind_noise(
                x_dist,
                y_dist,
                pressure,
                elapsed_time_dt
            )
            # Extract m/s from Velocity objects
            u_noise = noise.u.meters_per_second
            v_noise = noise.v.meters_per_second

            return WindVector(u=u0 + u_noise, v=v0 + v_noise)

class SimplexWindNoise:
    """Wrapper for BLE’s U/V NoisyWindComponent models."""

    def __init__(self):
        self.noise_u = simplex_wind_noise.NoisyWindComponent(which='u')
        self.noise_v = simplex_wind_noise.NoisyWindComponent(which='v')

    def reset(self, key: jnp.ndarray) -> None:
        """
        Reset the U and V noise components using the provided JAX PRNGKey.

        Args:
          key: A JAX PRNGKey for randomness.
        """
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
