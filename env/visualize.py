import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Sequence, Optional
from PIL import Image

# Constants
EARTH_RADIUS_KM = 6371  # Radius of the Earth in kilometres

# Pillow is optional – only needed when using texture_path
try:
    from PIL import Image
except ImportError:
    Image = None  # will raise later if texture_path is used without Pillow


def _km_to_degree_lat(lat_km: np.ndarray) -> np.ndarray:
    """Convert north-south displacement in kilometres to degrees of latitude."""
    return lat_km / 111.32  # Roughly km per degree latitude


def _km_to_degree_lon(lon_km: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Convert east-west displacement in kilometres to degrees of longitude at a given latitude."""
    # km per degree longitude shrinks with cos(latitude)
    km_per_deg = 111.32 * np.cos(np.radians(lat_deg))
    # Avoid division by zero near the poles
    km_per_deg = np.where(np.abs(km_per_deg) < 1e-6, 1e-6, km_per_deg)
    return lon_km / km_per_deg


def _ensure_degree_units(lats: Sequence[float], lons: Sequence[float]):
    """Attempt to infer whether the provided lat/lon are in degrees or kilometres.

    Heuristic: if any absolute latitude value exceeds 90, we assume the
    sequence is in kilometres and convert to degrees. The same check is applied to
    longitudes using a 180-degree threshold.
    """
    lats_arr = np.asarray(lats, dtype=float)
    lons_arr = np.asarray(lons, dtype=float)

    # Convert km to degrees if heuristic triggered
    if np.any(np.abs(lats_arr) > 90):
        lats_arr = _km_to_degree_lat(lats_arr)
    if np.any(np.abs(lons_arr) > 180):
        # Need lat degrees for conversion
        if np.any(np.abs(lats_arr) > 90):
            # We have already converted lats to degrees in previous step.
            pass
        lons_arr = _km_to_degree_lon(lons_arr, lats_arr)

    return lats_arr, lons_arr


def plot_trajectory_earth(
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    altitudes_km: Sequence[float],
    add_city_markers: bool = True,
    earth_colors: bool = False,
    texture_path: Optional[str] = None,
    lon_offset_deg: float = 0.0,
    flip_lat: bool = False,
    opacity: float = 1.0,
    use_texture_resolution: bool = True,
):
    """Visualise the balloon trajectory around a 3D abstract Earth sphere.

    Parameters
    ----------
    latitudes, longitudes : Sequence[float]
        The balloon path coordinates. They may be provided either in **degrees**
        or in **kilometres** displacement from the origin. If any value exceeds
        the natural bounds (|lat| > 90° or |lon| > 180°) we assume the input is
        in kilometres and convert to degrees internally.
    altitudes_km : Sequence[float]
        Altitude of the balloon in kilometres corresponding to each lat/lon
        coordinate.
    add_city_markers : bool, default True
        Whether to annotate a few major US cities for reference.
    earth_colors : bool, default False
        Whether to use a realistic Earth colourscale.
    texture_path : Optional[str], default None
        Path to an equirectangular image for texture mapping.
    lon_offset_deg : float, default 0.0
        Longitude offset in degrees for texture mapping.
    flip_lat : bool, default False
        Whether to flip the latitude direction for texture mapping.
    opacity : float, default 1.0
        Opacity of the Earth surface (1.0 = fully opaque).
    use_texture_resolution : bool, default True
        When a texture is supplied, build the sphere mesh with the same
        resolution (rows = image height, cols = width) to avoid any resampling.
    """

    if len(latitudes) == 0:
        raise ValueError("Latitude sequence is empty.")

    # Ensure numpy arrays
    lats_raw = np.asarray(latitudes, dtype=float)
    lons_raw = np.asarray(longitudes, dtype=float)
    alts = np.asarray(altitudes_km, dtype=float)

    # Infer units and convert to degrees if required
    lats_deg, lons_deg = _ensure_degree_units(lats_raw, lons_raw)

    # Convert to radians for Cartesian conversion
    lats_rad = np.radians(lats_deg)
    lons_rad = np.radians(lons_deg)

    # Cartesian coordinates of trajectory
    x_traj = (EARTH_RADIUS_KM + alts) * np.cos(lats_rad) * np.cos(lons_rad)
    y_traj = (EARTH_RADIUS_KM + alts) * np.cos(lats_rad) * np.sin(lons_rad)
    z_traj = (EARTH_RADIUS_KM + alts) * np.sin(lats_rad)

    # Decide colouring / get surface indices ----------------------------------
    if texture_path is not None:
        if Image is None:
            raise ImportError("Pillow is required for texture mapping. Please install pillow>=9.0.0")

        # ----------------------------------------------------------------
        # Load and quantise image to max 256 colours
        # ----------------------------------------------------------------
        tex_img = Image.open(texture_path)

        w_raw, h_raw = tex_img.size
        # Force exact 2:1 ratio for safety
        if abs((w_raw / h_raw) - 2.0) > 1e-2:
            # fallback to resize to closest 2:1 keeping height
            new_w = int(2 * h_raw)
            tex_img = tex_img.resize((new_w, h_raw), resample=Image.BILINEAR)
            w_raw = new_w

        if not use_texture_resolution:
            tex_img = tex_img.resize((360, 180), resample=Image.BILINEAR)
            w_raw, h_raw = tex_img.size

        # Reduce colour count for Plotly colourscale (≤256)
        tex_img = tex_img.convert("P", palette=Image.ADAPTIVE, colors=256)

        palette = tex_img.getpalette()  # length 768 list
        palette = np.array(palette, dtype=np.uint8).reshape(-1, 3)[:256]

        colorscale = [
            [i / 255.0, f"rgb({int(r)},{int(g)},{int(b)})"]
            for i, (r, g, b) in enumerate(palette)
        ]

        # surfacecolor indices
        surf_idx = np.asarray(tex_img, dtype=np.uint8)

        # Apply optional longitude offset (rotate columns)
        w = surf_idx.shape[1]
        shift_pix = int(np.round((lon_offset_deg / 360.0) * w))
        surf_idx = np.roll(surf_idx, -shift_pix, axis=1)

        # Optional flip latitude (north↔south)
        if flip_lat:
            surf_idx = np.flipud(surf_idx)

        surf_idx = np.roll(surf_idx, 180, axis=1)   # shift 180° east-west
        surf_idx = np.flipud(surf_idx)              # N-S flip

    else:
        # Default abstract colour via z value (will be replaced by z_sphere later)
        surf_idx = None

    # ------------------------------------------------------------------
    # Build sphere mesh (after surf_idx computed)
    # ------------------------------------------------------------------
    if texture_path is not None and use_texture_resolution and surf_idx is not None:
        H, W = surf_idx.shape
        phi, theta = np.mgrid[0 : np.pi : complex(H), 0 : 2 * np.pi : complex(W)]
    else:
        phi, theta = np.mgrid[0 : np.pi : 180j, 0 : 2 * np.pi : 360j]

    x_sphere = EARTH_RADIUS_KM * np.sin(phi) * np.cos(theta)
    y_sphere = EARTH_RADIUS_KM * np.sin(phi) * np.sin(theta)
    z_sphere = EARTH_RADIUS_KM * np.cos(phi)

    if surf_idx is None:
        surf_idx = z_sphere  # abstract colouring

    fig = go.Figure()

    # Earth surface
    surface_kwargs = dict(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        surfacecolor=surf_idx,
        showscale=False,
        opacity=opacity,
        lighting=dict(ambient=0.6, diffuse=0.6),
        name="Earth",
    )

    if texture_path is not None:
        surface_kwargs["cmin"] = 0
        surface_kwargs["cmax"] = 255
        surface_kwargs["colorscale"] = colorscale
    else:
        surface_kwargs["colorscale"] = "Earth" if earth_colors else "Viridis"

    fig.add_trace(go.Surface(**surface_kwargs))

    # Trajectory line + markers
    fig.add_trace(
        go.Scatter3d(
            x=x_traj,
            y=y_traj,
            z=z_traj,
            mode="lines+markers",
            marker=dict(size=3, color="blue"),
            line=dict(width=2, color="blue"),
            name="Trajectory",
        )
    )

    # Current position (last point)
    fig.add_trace(
        go.Scatter3d(
            x=[x_traj[-1]],
            y=[y_traj[-1]],
            z=[z_traj[-1]],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="Current Position",
        )
    )

    if add_city_markers:
        cities = {
            "New York": (40.7128, -74.0060),
            "Los Angeles": (34.0522, -118.2437),
            "Chicago": (41.8781, -87.6298),
            "Houston": (29.7604, -95.3698),
        }
        c_lat_deg, c_lon_deg = zip(*cities.values())
        c_lat_rad = np.radians(c_lat_deg)
        c_lon_rad = np.radians(c_lon_deg)
        cx = EARTH_RADIUS_KM * np.cos(c_lat_rad) * np.cos(c_lon_rad)
        cy = EARTH_RADIUS_KM * np.cos(c_lat_rad) * np.sin(c_lon_rad)
        cz = EARTH_RADIUS_KM * np.sin(c_lat_rad)

        fig.add_trace(
            go.Scatter3d(
                x=cx,
                y=cy,
                z=cz,
                mode="markers+text",
                marker=dict(size=4, color="green"),
                text=list(cities.keys()),
                textposition="top center",
                name="Major US Cities",
            )
        )

    fig.update_layout(
        title="Balloon Trajectory Around Abstract Earth",
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
    )

    fig.show()

def plot_wind_field(wind_field, pressure_level=1000, time=0):
    """Plot wind vectors at a specific pressure level and time"""
    # Create a grid of points
    x = np.linspace(-180, 180, 20)  # longitude
    y = np.linspace(-90, 90, 20)    # latitude
    X, Y = np.meshgrid(x, y)
    
    # Get wind vectors at each point
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(len(x)):
        for j in range(len(y)):
            wind = wind_field.get_wind(X[j,i], Y[j,i], pressure_level, time)
            U[j,i] = wind.u
            V[j,i] = wind.v
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.quiver(X, Y, U, V, scale=50)
    plt.title(f'Wind Field at {pressure_level}hPa, t={time}h')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()