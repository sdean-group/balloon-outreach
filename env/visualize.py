import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Sequence, Optional
from PIL import Image
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, FancyArrow
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from math import cos, radians
from datetime import datetime

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

# ---------- Used in MPPI Iterative Optimization notebook ---------------

def plot_accs_vels_samples(acc_samples, vel_samples, horizon):
    """ Plot the samples of accelerations and velocities in MPPI algorithm. """
    fig = plt.figure()

    plt.subplot(2,1,1)
    for acc_sample in acc_samples:
        plt.plot([i for i in range(horizon)], acc_sample, 'b-', alpha=0.3)
    plt.title(f'Acceleration Samples')
    plt.xlabel('Timestep in Horizon')
    plt.ylabel('Acceleration (m/s^2)')

    plt.subplot(2,1,2)
    for vel_sample in vel_samples:
        plt.plot([i for i in range(horizon)], vel_sample, 'b-', alpha=0.3)
    plt.title(f'Velocity Samples')
    plt.xlabel('Timestep in Horizon')
    plt.ylabel('Vertical Velocity (m/s)')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('samples.png')
    plt.show()
    plt.close()

def plot_alts_movement_samples(vert_movement, lateral_movement, horizon, target_state=None):
    """ Plot the vertical and horizontal position of balloon using MPPI samples"""
    fig = plt.figure()
    plt.subplot(1,2,1)
    for alts in vert_movement:
        plt.plot([i for i in range(horizon + 1)], alts, 'b-', alpha=0.3)
    if target_state is not None:
      plt.axhline(y=target_state[2], linewidth=1, color='r', label='Target End Altitude')
      plt.legend()
    plt.title(f'Altitudes of Samples')
    plt.xlabel('Timestep in Horizon')
    plt.ylabel('Altitude (km)')

    # Graph the horizontal position of balloon
    plt.subplot(1,2,2)
    for paths in lateral_movement:
        lats, lons = zip(*paths)
        plt.plot(lons, lats, 'b-', alpha=0.3)
    if target_state is not None:
      plt.plot(target_state[1],target_state[0], 'rx', label='Target End')
      plt.legend()
    plt.title(f'Lateral Movement of Samples')
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('movement.png')
    plt.show()
    plt.close()

def plot_costs_samples(costs):
    """ Plots costs of sampled control sequences. If the trajectories are similar, the costs and weights are also similar. """
    plt.bar([i+1 for i in range(len(costs))], costs)
    plt.title(f'Sample Costs')
    plt.xlabel('Plan Number')
    plt.xticks([i+1 for i in range(len(costs))])
    plt.ylabel('Cost')
    plt.tight_layout()
    plt.savefig('costs.png')
    plt.show()
    plt.close()

def plot_vels_averaged(vel_samples, control_sequence, horizon):
    """ Plots final averaged control sequence along with sampled control sequences."""
    # Plot final trajectory
    plt.figure(figsize=(12, 5))

    # Position plot
    #plt.subplot(1, 2, 1)
    for vel_sample in vel_samples:
        plt.plot([i for i in range(horizon)], vel_sample, 'b-', alpha=0.3)
    plt.plot([i for i in range(horizon)], control_sequence, 'r-', alpha=1, label='Weighted velocity sequence')
    plt.plot(0, control_sequence[0], 'm*', label='Optimal Action')
    plt.grid(True)
    plt.title(f'Balloon Velocity with MPPI')
    plt.xlabel('Timestep in Horizon')
    plt.ylabel('Vertical Velocity (m/s)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('velocity.png')
    plt.show()
    plt.close()



class BalloonTrajectoryAnimator:
    def __init__(self, longitudes, latitudes, altitudes, volumes, forces):
        self.longitudes = longitudes
        self.latitudes = latitudes
        self.altitudes = altitudes
        self.volumes = volumes
        self.forces = forces  # list of (buoyancy, weight, drag, net_force)
        self.max_vol = max(volumes)
        self.min_vol = min(volumes)
        self.vol_range = self.max_vol - self.min_vol
        self.fig_anim = None
        self.ax_alt_lon = None
        self.ax_alt_lat = None
        self.balloon_lon = None
        self.balloon_lat = None
        self.volume_text = None
        self.force_arrows = []
        self.force_labels = []
        self.ani = None
        max_x_range = max((max(latitudes)-min(latitudes)), (max(longitudes)-min(longitudes))) * 0.5 + 0.5
        mean_lat = (max(latitudes) + min(latitudes)) / 2 
        mean_lon = (max(longitudes) + min(longitudes)) / 2 
        self.lat_range = [mean_lat - max_x_range, mean_lat + max_x_range]
        self.lon_range = [mean_lon - max_x_range, mean_lon + max_x_range]
        self.alt_range  = [0, 26]
        self.lat_scale = abs((self.lat_range[1] - self.lat_range[0])) / (self.alt_range[1] - self.alt_range[0])
        self.lon_scale = abs((self.lon_range[1] - self.lon_range[0])) / (self.alt_range[1] - self.alt_range[0])

    def volume_to_size(self, vol, scale):
        height = 4
        width = height * scale
        if self.vol_range == 0:
            return width * 0.1, height * 0.1  # default small size
        width *= vol / 1000 
        height *= vol / 1000
        return width, height

    def draw_force_arrow(self, base_x, base_y, force_val, label, color, scale):
        if color == 'red':
            dy = force_val / 1000.0 
            # dy =  np.sign(force_val) * (1 + 4 * (np.log(abs(force_val)) / np.log(5000)))
        else:
            dy = force_val / 1000.0
        arrow = FancyArrow(base_x, base_y, 0, dy,
                           width=0.3*scale, length_includes_head=True,
                           color=color, zorder=5)
        self.ax_alt_lon.add_patch(arrow)
        text = self.ax_alt_lon.text(base_x - 0.05*scale, base_y + dy + np.sign(dy) * 0.5,
                                    f"{label}\n{int(force_val)}N",
                                    ha='left', va='center', fontsize=10)
        return arrow, text

    def animate(self, interval=500, repeat=True):
        # Create figure and axes
        self.fig_anim, (self.ax_alt_lon, self.ax_alt_lat) = plt.subplots(1, 2, figsize=(16, 8))
        # Set up plots
        self.ax_alt_lon.set_xlabel('Longitude (°)')
        self.ax_alt_lon.set_ylabel('Altitude (km)')
        self.ax_alt_lon.set_title('Balloon Trajectory: Altitude vs Longitude')
        self.ax_alt_lon.grid(True, alpha=0.3)
        self.ax_alt_lon.set_ylim(self.alt_range[0], self.alt_range[1])
        self.ax_alt_lon.set_xlim(self.lon_range[0], self.lon_range[1])

        self.ax_alt_lat.set_xlabel('Latitude (°)')
        self.ax_alt_lat.set_ylabel('Altitude (km)')
        self.ax_alt_lat.set_title('Balloon Trajectory: Altitude vs Latitude')
        self.ax_alt_lat.grid(True, alpha=0.3)
        self.ax_alt_lat.set_ylim(self.alt_range[0], self.alt_range[1])
        self.ax_alt_lat.set_xlim(self.lat_range[0], self.lat_range[1])

        # Plot full trajectory
        self.ax_alt_lon.plot(self.longitudes, self.altitudes, 'b-', alpha=0.3, linewidth=1)
        self.ax_alt_lat.plot(self.latitudes, self.altitudes, 'b-', alpha=0.3, linewidth=1)

        # Create balloon ellipses
        w0_lon, h0_lon = self.volume_to_size(self.volumes[0], self.lon_scale)
        w0_lat, h0_lat = self.volume_to_size(self.volumes[0], self.lat_scale)
        if self.volumes[0] >= 1500:
            balloon_color = 'red'
        else:
            balloon_color = 'blue'
        self.balloon_lon = Ellipse(
            (self.longitudes[0], self.altitudes[0]),
            width=w0_lon,
            height=h0_lon,
            facecolor=balloon_color,
            alpha=0.3,
            edgecolor='black',
            linewidth=1.5,
            zorder=10
        )
        self.balloon_lat = Ellipse(
            (self.latitudes[0], self.altitudes[0]),
            width=w0_lat,
            height=h0_lat,
            facecolor=balloon_color,
            alpha=0.3,
            edgecolor='black',
            linewidth=1.5,
            zorder=10
        )
        self.ax_alt_lon.add_patch(self.balloon_lon)
        self.ax_alt_lat.add_patch(self.balloon_lat)

        # Add a text object for the current volume
        self.volume_text = self.ax_alt_lon.text(
            self.longitudes[0] - 0.2 * self.lon_scale,
            self.altitudes[0],
            f"Volume: {self.volumes[0]:.1f} m³",
            fontsize=12,
            color='black',
            ha='right',
            va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
        )

        # Draw initial force arrows
        b_arrow, b_label = self.draw_force_arrow(self.longitudes[0] + w0_lon*0.5 + 0.1*self.lon_scale, self.altitudes[0], self.forces[0][0], "Buoyancy", 'deepskyblue', self.lon_scale)
        w_arrow, w_label = self.draw_force_arrow(self.longitudes[0] + w0_lon*0.5 + 1.1*self.lon_scale, self.altitudes[0], self.forces[0][1], "Weight", 'deepskyblue', self.lon_scale)
        d_arrow, d_label = self.draw_force_arrow(self.longitudes[0] + w0_lon*0.5 + 2.1*self.lon_scale, self.altitudes[0], self.forces[0][2], "Drag", 'deepskyblue', self.lon_scale)
        net_arrow, net_label = self.draw_force_arrow(self.longitudes[0], self.altitudes[0] + np.sign(self.forces[0][3]) * h0_lon * 0.5, self.forces[0][3], "", 'red', self.lon_scale)
        self.force_arrows = [b_arrow, w_arrow, d_arrow, net_arrow]
        self.force_labels = [b_label, w_label, d_label, net_label]

        # Add start/end markers
        self.ax_alt_lon.plot(self.longitudes[0], self.altitudes[0], 'go', markersize=8, label='Start')
        self.ax_alt_lon.plot(self.longitudes[-1], self.altitudes[-1], 'ro', markersize=8, label='End')
        self.ax_alt_lat.plot(self.latitudes[0], self.altitudes[0], 'go', markersize=8, label='Start')
        self.ax_alt_lat.plot(self.latitudes[-1], self.altitudes[-1], 'ro', markersize=8, label='End')
        self.ax_alt_lon.legend()
        self.ax_alt_lat.legend()

        # # Set x-axis limits with margin
        # lon_margin = (max(self.longitudes) - min(self.longitudes)) * 0.1
        # lat_margin = (max(self.latitudes) - min(self.latitudes)) * 0.1
        # if lon_margin > 0:
        #     self.ax_alt_lon.set_xlim(min(self.longitudes) - lon_margin, max(self.longitudes) + lon_margin)
        # else:
        #     self.ax_alt_lon.set_xlim(self.longitudes[0] - 0.01, self.longitudes[0] + 0.01)
        # if lat_margin > 0:
        #     self.ax_alt_lat.set_xlim(min(self.latitudes) - lat_margin, max(self.latitudes) + lat_margin)
        # else:
        #     self.ax_alt_lat.set_xlim(self.latitudes[0] - 0.01, self.latitudes[0] + 0.01)

        def update(frame):
            lon, lat = self.longitudes[frame], self.latitudes[frame]
            alt = self.altitudes[frame]
            width_lon, height_lon = self.volume_to_size(self.volumes[frame], self.lon_scale)
            width_lat, height_lat = self.volume_to_size(self.volumes[frame], self.lat_scale)
            buoyancy, weight, drag, total_force = self.forces[frame]

            self.balloon_lon.set_center((lon, alt))
            self.balloon_lat.set_center((lat, alt))
            self.balloon_lon.width = width_lon
            self.balloon_lon.height = height_lon
            self.balloon_lat.width = width_lat
            self.balloon_lat.height = height_lat

            # Update the volume text position and content
            self.volume_text.set_position((lon - 0.002, alt))
            self.volume_text.set_text(f"Volume: {self.volumes[frame]:.1f} m³")

            self.ax_alt_lon.set_title(f'Balloon Trajectory: Altitude vs Longitude')
            self.ax_alt_lat.set_title(f'Balloon Trajectory: Altitude vs Latitude')

            # Remove old arrows and labels
            for item in self.force_arrows + self.force_labels:
                item.remove()

            # Redraw force arrows
            b_arrow, b_label = self.draw_force_arrow(lon + width_lon*0.5 + 0.1*self.lon_scale, alt, buoyancy, "Buoyancy", 'deepskyblue', self.lon_scale)
            w_arrow, w_label = self.draw_force_arrow(lon + width_lon*0.5 + 1.1*self.lon_scale, alt, weight, "Weight", 'deepskyblue', self.lon_scale)
            d_arrow, d_label = self.draw_force_arrow(lon + width_lon*0.5 + 2.1*self.lon_scale, alt, drag, "Drag", 'deepskyblue', self.lon_scale)
            net_arrow, net_label = self.draw_force_arrow(lon, alt + np.sign(total_force) * height_lon * 0.5, total_force, "", 'red', self.lon_scale)
            self.force_arrows[:] = [b_arrow, w_arrow, d_arrow, net_arrow]
            self.force_labels[:] = [b_label, w_label, d_label, net_label]

            return self.balloon_lon, self.balloon_lat, self.volume_text, self.force_arrows, self.force_labels

        frames = len(self.altitudes)
        self.ani = animation.FuncAnimation(self.fig_anim, update, frames=frames, interval=interval, blit=False, repeat=repeat)
        plt.tight_layout()
        plt.show()

    def save(self, filename='balloon_traj.mp4', fps=10):
        if self.ani is not None:
            if filename.endswith('.gif'):
                # GIF (slower)
                self.ani.save(filename, writer='pillow', fps=fps)
            else:
                # MP4 (faster)
                from matplotlib.animation import FFMpegWriter
                writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=1800)
                self.ani.save(filename, writer=writer)
        else:
            print('No animation to save. Please run animate() first.')
class BalloonSummaryPlotter:
    def __init__(self, altitudes, volumes, helium_masses, sand_masses, longitudes, latitudes, dt, balloon):
        self.altitudes = altitudes
        self.volumes = volumes
        self.helium_masses = helium_masses
        self.sand_masses = sand_masses
        self.longitudes = longitudes
        self.latitudes = latitudes
        self.dt = dt
        self.balloon = balloon

    def plot(self):
        plot_times = np.arange(0, len(self.altitudes) * self.dt, self.dt)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Altitude trajectory
        ax1.plot(plot_times / 3600, self.altitudes, 'b-', linewidth=2, label='Altitude')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Altitude (km)')
        ax1.set_title('Balloon Altitude Trajectory')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 26)
        ax1.legend()

        # Plot 2: Volume changes
        ax2.plot(plot_times / 3600, self.volumes, 'g-', linewidth=2, label='Volume')
        ax2.axhline(y=self.balloon.max_volume, color='r', linestyle='--', label=f'Max Volume ({self.balloon.max_volume} m³)')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Volume (m³)')
        ax2.set_title('Balloon Volume Over Time')
        ax2.grid(True, alpha=0.3)
        # ax2.set_ylim(0, 1600)
        ax2.legend()

        # Plot 3: Resource consumption
        ax3.plot(plot_times / 3600, self.helium_masses, 'c-', linewidth=2, label='Helium Mass')
        ax3.plot(plot_times / 3600, self.sand_masses, 'orange', linewidth=2, label='Sand Mass')
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Mass (kg)')
        ax3.set_title('Resource Consumption Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, max(self.balloon.initial_helium_mass, self.balloon.initial_sand) + 10)
        ax3.legend()

        # Plot 4: Position changes (if any wind)
        if any(lat != self.latitudes[0] for lat in self.latitudes) or any(lon != self.longitudes[0] for lon in self.longitudes):
            ax4.plot(self.longitudes, self.latitudes, 'purple', linewidth=2, marker='o', markersize=3)
            ax4.plot(self.longitudes[0], self.latitudes[0], 'go', markersize=8, label='Start')
            ax4.plot(self.longitudes[-1], self.latitudes[-1], 'ro', markersize=8, label='End')
            ax4.set_xlabel('Longitude (°)')
            ax4.set_ylabel('Latitude (°)')
            ax4.set_title('Balloon Position Trajectory')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No position change\n(no wind)', 
                     ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Balloon Position Trajectory')

        plt.tight_layout()
        plt.show()

def plot_trajectory_earth(lons, lats):
    fig, ax = plt.subplots(
        figsize=(12,6),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    ax = plt.axes(projection=ccrs.PlateCarree())

    # around the North America
    lon_min, lon_max = -180, 0
    lat_min, lat_max = 0, 90
    ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                crs=ccrs.PlateCarree())

    # add background
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue')
    ax.coastlines('50m', linewidth=0.5)

    # gridlines with labels
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    # plot trajectory
    ax.plot(lons, lats,
            transform=ccrs.PlateCarree(),
            color='blue', linewidth=2,
            label='Trajectory')

    # start and end markers
    ax.scatter(lons[0], lats[0],
            transform=ccrs.PlateCarree(),
            color='green', marker='o', s=60,
            label='Start')
    ax.scatter(lons[-1], lats[-1],
            transform=ccrs.PlateCarree(),
            color='red', marker='X', s=60,
            label='End')

    ax.set_title(f"Balloon Trajectory with Wind Field")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc='lower left')

    plt.show()