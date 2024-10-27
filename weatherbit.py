import requests


def get_weatherbit_data(lat, lon, api_key):
    url = f"https://api.weatherbit.io/v2.0/current?lat={lat}&lon={lon}&key={api_key}"
    response = requests.get(url)
    return response.json()


if __name__=='__main__':
    lat = 18.5171
    lon = 73.8496
    api_key = '92e74a96cc1e4a44912c6508efe0a39d'
    weather_data = get_weatherbit_data(lat, lon, api_key)
    print(weather_data)










