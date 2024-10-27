import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from solar_calculations import get_weatherbit_data, calculate_solar_energy


# Function to calculate rooftop area from the image
def calculate_rooftop_area(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create binary image
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

    # Detect contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the rooftop
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the area of the rooftop in pixels
    area_in_pixels = cv2.contourArea(largest_contour)

    # Convert pixel area to real-world area (meters squared)
    # Assuming 1 pixel corresponds to X square meters (you will need to provide scale)
    scale_factor = 0.25  # Placeholder: 1 pixel = 0.25 square meters (adjust as needed)
    area_in_meters = area_in_pixels * scale_factor

    return area_in_meters


# Example usage
if __name__ == "__main__":
    lat = 18.5171  # Latitude of building
    lon = 73.8496  # Longitude of building

    # Path to the rooftop image
    image_path = 'uploads/rooftop.png'

    # Get rooftop area from image
    rooftop_area = calculate_rooftop_area(image_path)
    print(rooftop_area)

    # Get solar data from Weatherbit
    weather_data = get_weatherbit_data(lat, lon)

    # Calculate total solar energy potential for the given period
    total_energy = calculate_solar_energy(weather_data, rooftop_area)

    # Convert to kilowatt-hours (kWh)
    total_energy_kwh = total_energy / 1000
    print(f"Total solar energy potential: {total_energy_kwh:.2f} kWh for the selected period.")
