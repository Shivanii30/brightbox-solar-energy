from flask import Flask, render_template, send_from_directory

import solar_estimation
from solar_calculations import get_weatherbit_data,calculate_solar_energy
import readingCoords


app = Flask(__name__)


@app.route('/')
def display_results():
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
