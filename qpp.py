from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/save-rooftop', methods=['POST'])
def save_rooftop():
    data = request.get_json()

    lat = data.get('lat')
    lng = data.get('lng')
    rooftop_coordinates = data.get('rooftopCoordinates')

    # Check if any of the required fields are missing
    if lat is None or lng is None or rooftop_coordinates is None:
        return jsonify({"error": "Missing lat, lng, or rooftopCoordinates"}), 400

    # Process your data here (e.g., save it to the database)
    # For now, we'll just print the received data
    print(f"Received data: Latitude: {lat}, Longitude: {lng}, Rooftop Coordinates: {rooftop_coordinates}")

    # Here you would normally save to a database
    # Example: save_to_database(lat, lng, rooftop_coordinates)

    return jsonify({"message": "Rooftop saved successfully!"}), 200


if __name__ == '__main__':
    app.run(debug=True)
