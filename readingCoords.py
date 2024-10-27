import json


def reading_coords(json_file='uploads/coordinates.json'):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
            if isinstance(data, list) and len(data) > 0:
                lat = float(data[0]['lat'])
                lng = float(data[0]['lng'])
                return lat, lng
            else:
                raise ValueError('Invalid data format in the JSON file')
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error reading coordinates :{e}")
        return None, None


if __name__ == '__main__':
    lat, lng = reading_coords()
    print(lat, lng)