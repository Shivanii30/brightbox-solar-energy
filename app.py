import os
import math
import json
import numpy as np
import cv2
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# No API keys needed — Open-Meteo and ESRI are fully free and open

# ── Solar calculation helpers (Open-Meteo, free, no key) ────────────────────

def get_solar_data(lat, lon):
    """
    Fetches current solar radiation and cloud cover from Open-Meteo.
    Free, no API key, no rate-limit issues for personal use.
    Docs: https://open-meteo.com/en/docs
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=shortwave_radiation,cloud_cover,direct_radiation"
        "&timezone=auto&forecast_days=1"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def calculate_solar_energy(solar_data, rooftop_area_m2, panel_efficiency=0.18):
    """
    Returns estimated energy in Wh for a single peak-sun hour,
    plus the raw solar_rad and cloud_cover values for display.
    """
    current = solar_data.get("current", {})
    solar_rad   = current.get("shortwave_radiation", 0)   # W/m²
    cloud_cover = current.get("cloud_cover", 0)            # %

    effective_rad = solar_rad * (1 - cloud_cover / 100)
    energy_wh = effective_rad * rooftop_area_m2 * panel_efficiency
    return energy_wh, solar_rad, cloud_cover


def meters_per_pixel(lat, zoom=18):
    """Real-world metres represented by one pixel at the given zoom & latitude."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


# ── Image-processing helpers ─────────────────────────────────────────────────

def detect_roof_area(image_bgr, lat, zoom=18):
    """Return roof area in m² detected from a satellite image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Bilateral filter then sharpen
    blur = cv2.bilateralFilter(gray, 5, sigmaColor=7, sigmaSpace=5)
    kernel = np.array([[-2, -2, -2], [-2, 17, -2], [-2, -2, -2]], dtype="int")
    sharp = cv2.filter2D(blur, -1, kernel)
    # Otsu threshold → find largest contour (the roof)
    _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0
    largest = max(contours, key=cv2.contourArea)
    area_px = cv2.contourArea(largest)
    mpp = meters_per_pixel(lat, zoom)
    area_m2 = area_px * (mpp ** 2)
    return area_m2, int(area_px)


def estimate_panel_count(area_m2, panel_length_m=1.65, panel_width_m=0.99, packing=0.75):
    """How many standard panels (1.65 x 0.99 m) fit on the roof."""
    panel_area = panel_length_m * panel_width_m
    return max(0, int((area_m2 * packing) / panel_area))


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/tile-proxy")
def tile_proxy():
    """
    Proxies ESRI World Imagery satellite tiles through Flask so the browser
    sees them as same-origin — required for Cropper.js canvas access.
    ESRI World Imagery is free with no API key.
    Tile format: /tile-proxy?z=18&x=123&y=456
    """
    z = request.args.get("z", 18)
    x = request.args.get("x")
    y = request.args.get("y")
    if not x or not y:
        return "Missing tile coordinates", 400

    # ESRI uses z/y/x order (row/col)
    tile_url = (
        f"https://server.arcgisonline.com/ArcGIS/rest/services"
        f"/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
    try:
        resp = requests.get(tile_url, timeout=10,
                            headers={"User-Agent": "BrightBox-SolarApp/1.0"})
        resp.raise_for_status()
    except Exception as e:
        return f"Tile fetch error: {e}", 502

    from flask import Response
    return Response(
        resp.content,
        content_type=resp.headers.get("Content-Type", "image/jpeg"),
    )


@app.route("/")
def index():
    return send_from_directory(".", "map-crop-index.html")


@app.route("/styles.css")
def styles():
    return send_from_directory(".", "styles.css")


@app.route("/result")
def result():
    return render_template("result.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Expects multipart/form-data:
      image  – PNG/JPEG cropped satellite image
      lat    – float
      lng    – float
      zoom   – int (optional, default 18)

    Returns JSON with roof area, panel count, and energy estimates.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    lat  = float(request.form.get("lat", 0))
    lng  = float(request.form.get("lng", 0))
    zoom = int(request.form.get("zoom", 18))

    # Persist the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, "rooftop.png")
    request.files["image"].save(image_path)

    # Persist coordinates
    with open(os.path.join(UPLOAD_FOLDER, "coordinates.json"), "w") as f:
        json.dump([{"lat": lat, "lng": lng}], f)

    # --- Computer-vision analysis ---
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    roof_area_m2, roof_area_px = detect_roof_area(image_bgr, lat, zoom)
    panel_count = estimate_panel_count(roof_area_m2)

    # --- Weather / solar API ---
    weather_error = None
    energy_wh = solar_rad = cloud_cover = 0
    try:
        solar_data = get_solar_data(lat, lng)
        energy_wh, solar_rad, cloud_cover = calculate_solar_energy(solar_data, roof_area_m2)
    except Exception as exc:
        weather_error = str(exc)

    # Scale to daily & yearly (assume 5 peak sun hours/day)
    energy_kwh_day  = (energy_wh * 5) / 1000
    energy_kwh_year = energy_kwh_day * 365

    return jsonify({
        "roof_area_m2":       round(roof_area_m2, 2),
        "roof_area_px":       roof_area_px,
        "panel_count":        panel_count,
        "energy_wh_peak":     round(energy_wh, 2),
        "energy_kwh_day":     round(energy_kwh_day, 2),
        "energy_kwh_year":    round(energy_kwh_year, 2),
        "solar_radiation_wm2": solar_rad,
        "cloud_cover_pct":    cloud_cover,
        "lat": lat,
        "lng": lng,
        "weather_error":      weather_error,
    })


@app.route("/save-rooftop", methods=["POST"])
def save_rooftop():
    data   = request.get_json()
    lat    = data.get("lat")
    lng    = data.get("lng")
    coords = data.get("rooftopCoordinates")
    if None in (lat, lng, coords):
        return jsonify({"error": "Missing lat, lng, or rooftopCoordinates"}), 400
    with open(os.path.join(UPLOAD_FOLDER, "coordinates.json"), "w") as f:
        json.dump([{"lat": lat, "lng": lng, "rooftop": coords}], f)
    return jsonify({"message": "Saved successfully"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
