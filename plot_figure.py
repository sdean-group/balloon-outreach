import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, FancyArrow
import numpy as np

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
        self.lat_range = [min(latitudes)-0.01, max(latitudes)+0.01]
        self.lon_range = [min(longitudes)-0.01, max(longitudes)+0.01]
        self.alt_range  = [4, 26]
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
            dy = force_val / 1000.0 * 100
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

    def save(self, filename, fps=2):
        if self.ani is not None:
            self.ani.save(filename, writer='pillow', fps=fps)
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
        ax1.set_ylim(4, 26)
        ax1.legend()

        # Plot 2: Volume changes
        ax2.plot(plot_times / 3600, self.volumes, 'g-', linewidth=2, label='Volume')
        ax2.axhline(y=self.balloon.max_volume, color='r', linestyle='--', label=f'Max Volume ({self.balloon.max_volume} m³)')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Volume (m³)')
        ax2.set_title('Balloon Volume Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1600)
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