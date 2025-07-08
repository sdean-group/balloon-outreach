import numpy as np
import math

# Earth's radius in kilometers
EARTH_RADIUS_KM = 6371.0

def degrees_to_km(lon_deg, lat_deg):
    """
    Convert longitude and latitude differences from degrees to kilometers.
    
    Args:
        lon_deg (float): Longitude difference in degrees
        lat_deg (float): Latitude difference in degrees
    
    Returns:
        tuple: (lon_km, lat_km) - Longitude and latitude differences in kilometers
    """
    # Convert degrees to radians
    lon_rad = np.radians(lon_deg)
    lat_rad = np.radians(lat_deg)
    
    # Calculate distances in kilometers
    # For longitude: distance depends on latitude (longer at equator, shorter at poles)
    # For latitude: distance is approximately constant
    lon_km = lon_rad * EARTH_RADIUS_KM * np.cos(np.radians(0))  # At equator
    lat_km = lat_rad * EARTH_RADIUS_KM
    
    return lon_km, lat_km

def km_to_degrees(lon_km, lat_km, ref_lat=0.0):
    """
    Convert longitude and latitude differences from kilometers to degrees.
    
    Args:
        lon_km (float): Longitude difference in kilometers
        lat_km (float): Latitude difference in kilometers
        ref_lat (float): Reference latitude for longitude conversion (default: 0, equator)
    
    Returns:
        tuple: (lon_deg, lat_deg) - Longitude and latitude differences in degrees
    """
    # Convert kilometers to radians
    lon_rad = lon_km / (EARTH_RADIUS_KM * np.cos(np.radians(ref_lat)))
    lat_rad = lat_km / EARTH_RADIUS_KM
    
    # Convert radians to degrees
    lon_deg = np.degrees(lon_rad)
    lat_deg = np.degrees(lat_rad)
    
    return lon_deg, lat_deg

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on Earth
    using the Haversine formula.
    
    Args:
        lon1, lat1 (float): Longitude and latitude of first point in degrees
        lon2, lat2 (float): Longitude and latitude of second point in degrees
    
    Returns:
        float: Distance in kilometers
    """
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return EARTH_RADIUS_KM * c

def point_at_distance(lon, lat, distance_km, bearing_deg):
    """
    Calculate a new point at a given distance and bearing from a starting point.
    
    Args:
        lon, lat (float): Starting longitude and latitude in degrees
        distance_km (float): Distance to travel in kilometers
        bearing_deg (float): Bearing in degrees (0 = North, 90 = East, etc.)
    
    Returns:
        tuple: (new_lon, new_lat) - New longitude and latitude in degrees
    """
    # Convert to radians
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    bearing_rad = np.radians(bearing_deg)
    
    # Angular distance
    angular_distance = distance_km / EARTH_RADIUS_KM
    
    # Calculate new latitude
    new_lat_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(angular_distance) +
        np.cos(lat_rad) * np.sin(angular_distance) * np.cos(bearing_rad)
    )
    
    # Calculate new longitude
    new_lon_rad = lon_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(angular_distance) * np.cos(lat_rad),
        np.cos(angular_distance) - np.sin(lat_rad) * np.sin(new_lat_rad)
    )
    
    # Convert back to degrees
    new_lon = np.degrees(new_lon_rad)
    new_lat = np.degrees(new_lat_rad)
    
    return new_lon, new_lat

def wind_displacement_to_position(lon, lat, du_km, dv_km):
    """
    Calculate the new latitude/longitude position after wind displacement.
    
    Args:
        lon (float): Current longitude in degrees
        lat (float): Current latitude in degrees
        du_km (float): East-West wind component in km (positive = eastward, negative = westward)
        dv_km (float): North-South wind component in km (positive = northward, negative = southward)
    
    Returns:
        tuple: (new_lon, new_lat) - New longitude and latitude in degrees
    """
    # Convert current position to radians
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    
    # Calculate angular displacements
    # For latitude: direct conversion (1 degree ≈ 111 km)
    dlat_rad = dv_km / EARTH_RADIUS_KM
    
    # For longitude: depends on current latitude (longer at equator, shorter at poles)
    dlon_rad = du_km / (EARTH_RADIUS_KM * np.cos(lat_rad))
    
    # Calculate new position
    new_lat_rad = lat_rad + dlat_rad
    new_lon_rad = lon_rad + dlon_rad
    
    # Convert back to degrees
    new_lon = np.degrees(new_lon_rad)
    new_lat = np.degrees(new_lat_rad)
    
    return new_lon, new_lat

def wind_displacement_to_position_haversine(lon, lat, du_km, dv_km):
    """
    Calculate the new latitude/longitude position after wind displacement using
    more accurate spherical geometry calculations.
    
    Args:
        lon (float): Current longitude in degrees
        lat (float): Current latitude in degrees
        du_km (float): East-West wind component in km (positive = eastward, negative = westward)
        dv_km (float): North-South wind component in km (positive = northward, negative = southward)
    
    Returns:
        tuple: (new_lon, new_lat) - New longitude and latitude in degrees
    """
    # Calculate the total displacement distance and bearing
    total_distance = np.sqrt(du_km**2 + dv_km**2)
    
    if total_distance == 0:
        return lon, lat
    
    # Calculate bearing (0 = North, 90 = East, 180 = South, 270 = West)
    # atan2(du, dv) gives us the angle from North
    bearing = np.degrees(np.arctan2(du_km, dv_km))
    
    # Use the point_at_distance function for accurate spherical calculation
    new_lon, new_lat = point_at_distance(lon, lat, total_distance, bearing)
    
    return new_lon, new_lat