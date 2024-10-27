import requests

# NASA POWER API request for solar data
def get_solar_data(lat, lon, start, end):
    url = f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=ALLSKY_SFC_SW_DWN&community=RE&longitude={lon}&latitude={lat}&format=JSON&start={start}&end={end}"
    response = requests.get(url)
    return response.json()

# Function to find available dates and hours
def get_available_dates(irradiance_data):
    available_dates = []

    if 'ALLSKY_SFC_SW_DWN' not in irradiance_data['properties']['parameter']:
        print("Error: Solar irradiance data not found in the response.")
        return available_dates

    # Iterate over the solar irradiance values
    for date, hourly_values in irradiance_data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].items():
        if isinstance(hourly_values, dict):
            # If hourly values are a dictionary (expected format)
            valid_hours = {hour: irradiance for hour, irradiance in hourly_values.items() if irradiance is not None}

            if valid_hours:
                available_dates.append({
                    'date': date,
                    'valid_hours': list(valid_hours.keys())  # Add valid hours
                })
        else:
            # If hourly values are not a dictionary, assume it's a single float value for the entire day
            if hourly_values is not None:
                available_dates.append({
                    'date': date,
                    'valid_hours': ['all']  # Use 'all' to indicate that data is available for the entire day
                })

    return available_dates


def get_available_parameters(lat, lon):
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?longitude={lon}&latitude={lat}&community=RE&format=JSON"
    response = requests.get(url)
    return response.json()

# Example usage


# Example usage
if __name__ == "__main__":
    lat = 18.5171  # Latitude of building
    lon = 73.8496  # Longitude of building
    start = '20241021'  # Start date (YYYYMMDD)
    end = '20241022'  # End date (YYYYMMDD)

    # Get solar irradiance data
    data = get_solar_data(lat, lon, start, end)
    print(data)

    # Find available dates and hours
    available_dates = get_available_dates(data)

    parameters=get_available_parameters(lat, lon)
    print(parameters)

    # Print available dates and hours
    for entry in available_dates:
        print(f"Date: {entry['date']} - Valid hours: {entry['valid_hours']}")
