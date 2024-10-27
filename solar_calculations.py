import requests

from BrightBox import readingCoords


# Function to get solar data from Weatherbit API
def get_weatherbit_data(lat, lon):
    api_key = '92e74a96cc1e4a44912c6508efe0a39d'  # Weatherbit API key
    url = f"https://api.weatherbit.io/v2.0/current?lat={lat}&lon={lon}&key={api_key}&include=solarradiation"
    response = requests.get(url)
    return response.json()

def calculate_solar_energy(weather_data, rooftop_area, panel_efficiency=0.18):
    total_energy = 0  # in watt-hours (Wh)

    # Extract relevant data from the Weatherbit response
    if 'data' in weather_data and len(weather_data['data']) > 0:
        solar_rad = weather_data['data'][0]['solar_rad']  # Solar radiation in W/m²
        cloud_cover = weather_data['data'][0]['clouds']    # Cloud cover percentage
        print(f"Solar radiation: {solar_rad} W/m², Cloud cover: {cloud_cover}%")

        # Adjust solar radiation based on cloud cover (example: reduce by 50% if completely cloudy)
        effective_solar_rad = solar_rad * (1 - cloud_cover / 100)

        # Convert irradiance to energy (Wh)
        energy = effective_solar_rad * rooftop_area * panel_efficiency
        total_energy += energy
    else:
        print("Error: Solar data not found in the response.")

    return total_energy


if __name__ == "__main__":
    lat, lng = readingCoords.reading_coords()



