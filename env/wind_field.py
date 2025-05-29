import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass
class WindVector:
    """Wind vector with u (east-west) and v (north-south) components"""
    u: float  # m/s, positive is eastward
    v: float  # m/s, positive is northward

@dataclass
class WindPoint:
    """A point in the wind field with position, pressure, time and wind vector"""
    x: float  # longitude
    y: float  # latitude
    pressure: float  # hPa
    time: float  # hours
    wind: WindVector

class WindField:
    """Wind field that stores and interpolates wind vectors at specific points"""
    def __init__(self):
        # Store wind points in a dictionary for quick lookup
        self.wind_points: Dict[Tuple[float, float, float, float], WindVector] = {}
        
        # Grid parameters
        self.x_range = (-180, 180)  # longitude range
        self.y_range = (-90, 90)    # latitude range
        self.pressure_levels = [1000, 850, 700, 500, 300, 200, 100]  # hPa
        self.time_steps = np.arange(0, 24, 1)  # hourly steps
        
        # Initialize with some example data
        self._initialize_example_data()
    
    def _initialize_example_data(self):
        """Initialize the wind field with some example data"""
        # Create a simple example wind field
        for x in np.linspace(self.x_range[0], self.x_range[1], 10):  # 10 points in longitude
            for y in np.linspace(self.y_range[0], self.y_range[1], 10):  # 10 points in latitude
                for p in self.pressure_levels:
                    for t in self.time_steps:
                        # Create some example wind patterns
                        u = 10 * np.sin(x/30) * np.cos(y/30) * (1 + 0.2 * np.sin(2*np.pi*t/24))
                        v = 10 * np.cos(x/30) * np.sin(y/30) * (1 + 0.2 * np.sin(2*np.pi*t/24))
                        
                        # Store the wind point
                        self.wind_points[(x, y, p, t)] = WindVector(u, v)
    
    def _find_nearest_points(self, x: float, y: float, pressure: float, time: float) -> List[WindPoint]:
        """Find the 16 nearest points in the 4D space for interpolation"""
        points = []
        
        # Find nearest grid points
        x_points = sorted([p for p in np.linspace(self.x_range[0], self.x_range[1], 10) 
                          if abs(p - x) <= 20])[:2]
        y_points = sorted([p for p in np.linspace(self.y_range[0], self.y_range[1], 10) 
                          if abs(p - y) <= 20])[:2]
        p_points = sorted([p for p in self.pressure_levels 
                          if abs(p - pressure) <= 200])[:2]
        t_points = sorted([t for t in self.time_steps 
                          if abs(t - time) <= 3])[:2]
        
        # Create all combinations of nearest points
        for xp in x_points:
            for yp in y_points:
                for pp in p_points:
                    for tp in t_points:
                        if (xp, yp, pp, tp) in self.wind_points:
                            points.append(WindPoint(xp, yp, pp, tp, 
                                                  self.wind_points[(xp, yp, pp, tp)]))
        
        return points
    
    def _trilinear_interpolation(self, points: List[WindPoint], 
                                x: float, y: float, 
                                pressure: float, time: float) -> WindVector:
        """Perform trilinear interpolation between points"""
        if not points:
            return WindVector(0, 0)  # Return zero wind if no points found
        
        # Calculate weights for each point
        total_weight = 0
        u_sum = 0
        v_sum = 0
        
        for point in points:
            # Calculate distance in 4D space
            dx = abs(point.x - x) / 20  # Normalize by typical grid spacing
            dy = abs(point.y - y) / 20
            dp = abs(point.pressure - pressure) / 200
            dt = abs(point.time - time) / 3
            
            # Weight decreases with distance
            weight = 1 / (1 + dx + dy + dp + dt)
            
            u_sum += point.wind.u * weight
            v_sum += point.wind.v * weight
            total_weight += weight
        
        if total_weight == 0:
            return WindVector(0, 0)
            
        return WindVector(u_sum/total_weight, v_sum/total_weight)
    
    def get_wind(self, x: float, y: float, pressure: float, time: float) -> WindVector:
        """
        Get wind vector at given position, pressure, and time using interpolation
        
        Args:
            x: Longitude in degrees
            y: Latitude in degrees
            pressure: Pressure in hPa
            time: Time in hours
            
        Returns:
            WindVector: Interpolated wind vector
        """
        # Find nearest points
        nearest_points = self._find_nearest_points(x, y, pressure, time)
        
        # Interpolate between points
        return self._trilinear_interpolation(nearest_points, x, y, pressure, time)
    
    def add_wind_point(self, x: float, y: float, pressure: float, 
                      time: float, u: float, v: float):
        """Add a new wind point to the field"""
        self.wind_points[(x, y, pressure, time)] = WindVector(u, v) 