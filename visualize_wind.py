import numpy as np
import matplotlib.pyplot as plt
from env.wind_field import WindField

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

def main():
    # Create wind field
    wind_field = WindField()
    
    # Plot at different pressure levels
    pressure_levels = [1000, 500, 200]  # hPa
    times = [0, 12]  # hours
    
    for p in pressure_levels:
        for t in times:
            plot_wind_field(wind_field, p, t)

if __name__ == "__main__":
    main() 