<<<<<<< HEAD
import math
import os
import logging
import requests
import numpy as np
import cv2
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.onnx")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PANEL_LENGTH_M   = 1.65
PANEL_WIDTH_M    = 0.99
PANEL_AREA_M2    = PANEL_LENGTH_M * PANEL_WIDTH_M
PANEL_EFFICIENCY = 0.20

GRID_CO2_KG_PER_KWH = 0.82

LOSS_TEMPERATURE  = 0.92
LOSS_DUST         = 0.95
LOSS_INVERTER     = 0.96
LOSS_WIRING       = 0.98
LOSS_MISMATCH     = 0.98
LOSS_AVAILABILITY = 0.99

SYSTEM_LOSS = (
    LOSS_TEMPERATURE * LOSS_DUST * LOSS_INVERTER *
    LOSS_WIRING      * LOSS_MISMATCH * LOSS_AVAILABILITY
)

ANNUAL_DEGRADATION = 0.005
PACKING_EFFICIENCY = 0.65
MONSOON_MONTHS     = {6, 7, 8, 9}
DAYS_PER_MONTH     = [31,28,31,30,31,30,31,31,30,31,30,31]
MONTH_KEYS         = ["JAN","FEB","MAR","APR","MAY","JUN",
                      "JUL","AUG","SEP","OCT","NOV","DEC"]
MONTH_NAMES        = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]
CARDINAL           = ["N","NE","E","SE","S","SW","W","NW"]

# COCO class IDs that indicate rooftop obstacles when detected from above
# (person=0 excluded — people on roofs are temporary)
OBSTACLE_CLASSES = {
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    62: "tv/monitor",
    63: "laptop",
    66: "keyboard",
    67: "phone",
    72: "refrigerator / AC unit",
    73: "book",
    74: "clock",
    76: "scissors",
    77: "teddy bear / object",
    # Most useful for satellite:
    2:  "car / vehicle on roof",
    7:  "truck",
    9:  "traffic light / pole",
    11: "stop sign",
}

# Additional: any detected box labelled as these is always an obstacle
ROOFTOP_OBSTACLE_KEYWORDS = [
    "tank", "ac", "air", "vent", "pipe", "stair",
    "shed", "cabin", "box", "unit",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoofAnalysis:
    total_area_m2:      float
    obstacle_area_m2:   float
    shadow_area_m2:     float
    usable_area_m2:     float
    obstacle_pct:       float
    shadow_pct:         float
    orientation:        str
    orientation_deg:    float
    confidence:         float
    confidence_reasons: list = field(default_factory=list)
    obstacle_labels:    list = field(default_factory=list)
    detection_method:   str  = "hsv_fallback"

@dataclass
class SolarResource:
    annual_kwh_m2:    float
    monthly_kwh_m2:   list
    peak_sun_hours:   float
    monsoon_loss_pct: float
    data_source:      str

@dataclass
class EnergyEstimate:
    panel_count:        int
    system_kw:          float
    annual_kwh:         float
    monthly_kwh:        list
    daily_avg_kwh:      float
    best_month:         str
    best_month_kwh:     float
    worst_month:        str
    worst_month_kwh:    float
    system_loss_factor: float
    loss_breakdown:     dict
    year25_kwh:         float
    co2_kg_year:        float

@dataclass
class FullAnalysis:
    roof:   RoofAnalysis
    solar:  SolarResource
    energy: EnergyEstimate
    lat:    float
    lng:    float

    def to_dict(self):
        import dataclasses
        return dataclasses.asdict(self)

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def meters_per_pixel(lat: float, zoom: int = 18) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)

# ---------------------------------------------------------------------------
# YOLO obstacle detection (primary)
# ---------------------------------------------------------------------------

_yolo_net = None   # cached after first load

def _load_yolo():
    global _yolo_net
    if _yolo_net is not None:
        return _yolo_net
    if not os.path.exists(MODEL_PATH):
        logger.info("yolov8n.onnx not found — using HSV fallback")
        return None
    try:
        net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _yolo_net = net
        logger.info("✓ YOLOv8n ONNX loaded via cv2.dnn")
        return net
    except Exception as e:
        logger.warning(f"Failed to load YOLO model: {e}")
        return None


def detect_obstacles_yolo(image_bgr: np.ndarray,
                           roof_contour,
                           mpp: float = 0.6,
                           conf_threshold: float = 0.20,
                           iou_threshold:  float = 0.40) -> tuple:
    """
    Runs YOLOv8n inference via cv2.dnn on the cropped rooftop image.
    Returns (obstacle_mask, obstacle_area_px, labels_list).
    """
    net = _load_yolo()
    if net is None:
        return None, 0.0, []

    h, w = image_bgr.shape[:2]

    # Build roof-only mask
    roof_mask = np.zeros((h, w), dtype=np.uint8)
    if roof_contour is not None:
        cv2.drawContours(roof_mask, [roof_contour], -1, 255, -1)

    # YOLOv8 expects 640×640 normalised input
    input_size = 640
    blob = cv2.dnn.blobFromImage(
        image_bgr, scalefactor=1/255.0,
        size=(input_size, input_size),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward()                  # shape: (1, 84, 8400) for YOLOv8n

    # YOLOv8 output layout: [batch, 84, anchors]
    # 84 = 4 (box xywh) + 80 (class scores)
    predictions = outputs[0]                  # (84, 8400)
    predictions = predictions.T               # (8400, 84)

    boxes, confidences, class_ids = [], [], []
    scale_x = w / input_size
    scale_y = h / input_size

    for pred in predictions:
        scores = pred[4:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])
        if confidence < conf_threshold:
            continue

        cx, cy, bw, bh = pred[:4]
        x1 = int((cx - bw / 2) * scale_x)
        y1 = int((cy - bh / 2) * scale_y)
        x2 = int((cx + bw / 2) * scale_x)
        y2 = int((cy + bh / 2) * scale_y)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(confidence)
        class_ids.append(class_id)

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
    if len(indices) == 0:
        return np.zeros((h, w), dtype=np.uint8), 0.0, []

    obs_mask = np.zeros((h, w), dtype=np.uint8)
    labels   = []
    min_px   = int((0.4 / (mpp ** 2)))

    for i in indices.flatten():
        x, y, bw, bh = boxes[i]
        area = bw * bh
        if area < min_px:
            continue

        # Only draw inside the roof boundary
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(tmp, (x, y), (x + bw, y + bh), 255, -1)
        tmp = cv2.bitwise_and(tmp, tmp, mask=roof_mask)

        if np.sum(tmp > 0) < min_px:
            continue

        obs_mask = cv2.bitwise_or(obs_mask, tmp)
        cid   = class_ids[i]
        cname = OBSTACLE_CLASSES.get(cid, f"object (class {cid})")
        real_m2 = round(area * (mpp ** 2), 1)
        labels.append(f"{cname} (~{real_m2} m², conf {confidences[i]:.0%})")

    obs_px = float(np.sum(obs_mask > 0))
    return obs_mask, obs_px, labels


# ---------------------------------------------------------------------------
# HSV fallback obstacle detection
# ---------------------------------------------------------------------------

def detect_obstacles_hsv(image_bgr: np.ndarray,
                          roof_contour,
                          mpp: float = 0.6) -> tuple:
    """
    Colour-based obstacle detection when YOLO model is not available.
    Detects: dark tanks, metallic AC units, bright concrete staircases.
    """
    h, w  = image_bgr.shape[:2]
    hsv   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    roof_mask = np.zeros((h, w), dtype=np.uint8)
    if roof_contour is not None:
        cv2.drawContours(roof_mask, [roof_contour], -1, 255, -1)

    def masked(m):
        return cv2.bitwise_and(m, m, mask=roof_mask)

    dark     = cv2.inRange(hsv, np.array([0,  0,  0]),   np.array([180, 80,  70]))
    metal    = cv2.inRange(hsv, np.array([0,  0,  160]), np.array([180, 30, 235]))
    concrete = cv2.inRange(hsv, np.array([0,  0,  220]), np.array([180, 25, 255]))

    combined = cv2.bitwise_or(dark, cv2.bitwise_or(metal, concrete))
    combined = masked(combined)

    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  el, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, el, iterations=2)

    min_px = int(0.4 / (mpp ** 2))
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obs_mask = np.zeros((h, w), dtype=np.uint8)
    labels   = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_px:
            continue
        cv2.drawContours(obs_mask, [cnt], -1, 255, -1)
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(tmp, [cnt], -1, 255, -1)
        mean_v = cv2.mean(hsv[:,:,2], mask=tmp)[0]
        mean_s = cv2.mean(hsv[:,:,1], mask=tmp)[0]
        real_m2 = round(area * (mpp ** 2), 1)
        if mean_v < 70:
            labels.append(f"dark structure / water tank (~{real_m2} m²)")
        elif mean_s < 25 and mean_v > 200:
            labels.append(f"staircase / concrete (~{real_m2} m²)")
        else:
            labels.append(f"AC unit / metallic object (~{real_m2} m²)")

    obs_px = float(np.sum(obs_mask > 0))
    return obs_mask, obs_px, labels


# ---------------------------------------------------------------------------
# Unified obstacle entry point
# ---------------------------------------------------------------------------

def detect_obstacles(image_bgr, roof_contour, mpp=0.6):
    """
    Tries YOLO first, falls back to HSV if model not loaded.
    Returns (mask, area_px, labels, method_used).
    """
    mask, px, labels = detect_obstacles_yolo(image_bgr, roof_contour, mpp)
    if mask is not None:
        return mask, px, labels, "yolov8n"

    mask, px, labels = detect_obstacles_hsv(image_bgr, roof_contour, mpp)
    return mask, px, labels, "hsv_colour_mask"


# ---------------------------------------------------------------------------
# Roof boundary
# ---------------------------------------------------------------------------

def detect_roof_boundary(image_bgr):
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.bilateralFilter(gray, 5, 7, 5)
    k     = np.array([[-2,-2,-2],[-2,17,-2],[-2,-2,-2]], dtype="int")
    sharp = cv2.filter2D(blur, -1, k)
    _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    roof = max(cnts, key=cv2.contourArea)
    return roof, float(cv2.contourArea(roof))


# ---------------------------------------------------------------------------
# Shadow detection
# ---------------------------------------------------------------------------

def detect_shadows(image_bgr, roof_contour, obstacle_mask):
    h, w = image_bgr.shape[:2]
    roof_mask = np.zeros((h, w), dtype=np.uint8)
    if roof_contour is not None:
        cv2.drawContours(roof_mask, [roof_contour], -1, 255, -1)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L   = lab[:, :, 0]

    # Block size must be odd and scale with image — small images need smaller blocks
    block = max(11, min(31, (min(h, w) // 20) | 1))  # odd, 11–31

    thresh = cv2.adaptiveThreshold(
        L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=block, C=8)

    thresh = cv2.bitwise_and(thresh, thresh, mask=roof_mask)
    thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(obstacle_mask))

    # Kernel also scales with image
    k_size = max(3, min(7, min(h, w) // 80))
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    shadow_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, el, iterations=1)
    return shadow_mask, float(np.sum(shadow_mask > 0))


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def infer_orientation(roof_contour):
    if roof_contour is None:
        return "Unknown", 0.0
    rect    = cv2.minAreaRect(roof_contour)
    angle   = rect[2]
    bearing = (angle + 90) % 180
    idx     = int((bearing + 22.5) / 45) % 8
    return CARDINAL[idx], round(float(bearing), 1)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def compute_confidence(image_bgr, roof_px, obstacle_pct, shadow_pct):
    h, w  = image_bgr.shape[:2]
    total = h * w
    reasons = []
    score   = 1.0

    # Resolution check — after upscaling this should pass
    if total < 40_000:
        score -= 0.20
        reasons.append("Very low resolution — try zooming in on the map before cropping")
    elif total < 160_000:
        score -= 0.05
        reasons.append("Moderate resolution — tighter crop improves accuracy")

    roof_frac = (roof_px / total) if total else 0
    if roof_frac < 0.10:
        score -= 0.20
        reasons.append("Roof too small in frame — crop more tightly around just your rooftop")
    elif roof_frac < 0.30:
        score -= 0.06
        reasons.append("Crop includes area beyond the rooftop — tighter crop improves accuracy")

    if obstacle_pct > 0.45:
        score -= 0.15
        reasons.append(f"Very high obstacle coverage ({obstacle_pct:.0%}) — usable area uncertain")
    elif obstacle_pct > 0.25:
        score -= 0.06
        reasons.append(f"Significant obstacles detected ({obstacle_pct:.0%} of roof area)")

    if shadow_pct > 0.50:
        score -= 0.15
        reasons.append(f"Heavy shading ({shadow_pct:.0%}) — energy output significantly reduced")
    elif shadow_pct > 0.25:
        score -= 0.06
        reasons.append(f"Moderate shading ({shadow_pct:.0%})")

    gray      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharpness < 40:
        score -= 0.10
        reasons.append("Low image contrast — satellite tile quality may be poor here")

    if not reasons:
        reasons.append("Good image quality and clear roof boundary detected")

    return round(max(0.05, min(1.0, score)), 2), reasons


# ---------------------------------------------------------------------------
# NASA POWER solar resource
# ---------------------------------------------------------------------------

def fetch_solar_resource(lat, lon):
    url = (
        "https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?parameters=ALLSKY_SFC_SW_DWN&community=RE"
        f"&longitude={lon}&latitude={lat}&format=JSON&user=BrightBoxApp"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        vals = resp.json()["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]

        monthly_kwh = []
        monsoon_kwh = 0.0
        total_kwh   = 0.0
        for i, key in enumerate(MONTH_KEYS):
            daily     = float(vals.get(key, 4.5))
            month_tot = daily * DAYS_PER_MONTH[i]
            monthly_kwh.append(round(month_tot, 2))
            total_kwh += month_tot
            if (i + 1) in MONSOON_MONTHS:
                monsoon_kwh += month_tot

        return SolarResource(
            annual_kwh_m2    = round(total_kwh, 1),
            monthly_kwh_m2   = monthly_kwh,
            peak_sun_hours   = round(total_kwh / 365, 2),
            monsoon_loss_pct = round(monsoon_kwh / total_kwh, 3) if total_kwh else 0.22,
            data_source      = "NASA POWER (10-yr climatology)",
        )
    except Exception as e:
        logger.warning(f"NASA POWER failed ({e}) — using India regional fallback")
        return _india_fallback(lat)


def _india_fallback(lat):
    if lat < 15:
        daily = [5.8,6.2,6.8,7.0,6.5,4.2,3.8,4.0,4.8,5.5,5.6,5.5]
    elif lat < 22:
        daily = [5.4,6.0,6.6,6.8,6.2,3.8,3.5,3.7,4.5,5.2,5.3,5.1]
    else:
        daily = [4.8,5.5,6.2,6.8,6.5,4.8,4.2,4.4,5.0,5.4,5.0,4.6]
    monthly_kwh = [round(daily[i] * DAYS_PER_MONTH[i], 2) for i in range(12)]
    total   = sum(monthly_kwh)
    monsoon = sum(monthly_kwh[i] for i in range(12) if (i+1) in MONSOON_MONTHS)
    return SolarResource(
        annual_kwh_m2    = round(total, 1),
        monthly_kwh_m2   = monthly_kwh,
        peak_sun_hours   = round(total / 365, 2),
        monsoon_loss_pct = round(monsoon / total, 3) if total else 0.22,
        data_source      = "MNRE Solar Atlas (regional fallback)",
    )


# ---------------------------------------------------------------------------
# Energy model
# ---------------------------------------------------------------------------

def compute_energy(roof: RoofAnalysis, solar: SolarResource) -> EnergyEstimate:
    usable = roof.usable_area_m2
    panel_count = max(0, int((usable * PACKING_EFFICIENCY) / PANEL_AREA_M2))
    system_kw   = round(panel_count * PANEL_AREA_M2 * PANEL_EFFICIENCY, 2)

    orientation_factors = {
        "S":1.00,"SE":0.97,"SW":0.97,
        "E":0.87,"W":0.87,
        "NE":0.78,"NW":0.78,
        "N":0.70,"Unknown":0.90,
    }
    of = orientation_factors.get(roof.orientation, 0.90)

    monthly_kwh = []
    for ghi in solar.monthly_kwh_m2:
        net = ghi * usable * PANEL_EFFICIENCY * of * SYSTEM_LOSS
        monthly_kwh.append(round(net, 1))

    annual_kwh = round(sum(monthly_kwh), 1)
    best_i     = int(np.argmax(monthly_kwh))
    worst_i    = int(np.argmin(monthly_kwh))

    year25 = sum(
        annual_kwh * ((1 - ANNUAL_DEGRADATION) ** yr)
        for yr in range(25)
    )

    return EnergyEstimate(
        panel_count        = panel_count,
        system_kw          = system_kw,
        annual_kwh         = annual_kwh,
        monthly_kwh        = monthly_kwh,
        daily_avg_kwh      = round(annual_kwh / 365, 2),
        best_month         = MONTH_NAMES[best_i],
        best_month_kwh     = monthly_kwh[best_i],
        worst_month        = MONTH_NAMES[worst_i],
        worst_month_kwh    = monthly_kwh[worst_i],
        system_loss_factor = round(SYSTEM_LOSS, 3),
        loss_breakdown     = {
            "temperature_loss_pct":  round((1-LOSS_TEMPERATURE)*100, 1),
            "dust_loss_pct":         round((1-LOSS_DUST)*100,        1),
            "inverter_loss_pct":     round((1-LOSS_INVERTER)*100,    1),
            "wiring_loss_pct":       round((1-LOSS_WIRING)*100,      1),
            "mismatch_loss_pct":     round((1-LOSS_MISMATCH)*100,    1),
            "availability_loss_pct": round((1-LOSS_AVAILABILITY)*100,1),
            "total_loss_pct":        round((1-SYSTEM_LOSS)*100,      1),
            "orientation_factor":    of,
        },
        year25_kwh   = round(year25, 0),
        co2_kg_year  = round(annual_kwh * GRID_CO2_KG_PER_KWH, 0),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse(image_bgr: np.ndarray, lat: float, lng: float,
            zoom: int = 18) -> FullAnalysis:
    mpp = meters_per_pixel(lat, zoom)

    roof_cnt, roof_px = detect_roof_boundary(image_bgr)
    total_m2 = round(roof_px * (mpp ** 2), 2)

    obs_mask, obs_px, obs_labels, det_method = detect_obstacles(
        image_bgr, roof_cnt, mpp)
    obs_m2 = round(obs_px * (mpp ** 2), 2)

    shd_mask, shd_px = detect_shadows(image_bgr, roof_cnt, obs_mask)
    shd_m2 = round(shd_px * (mpp ** 2), 2)

    usable_m2 = max(0.0, round(total_m2 - obs_m2 - shd_m2, 2))
    obs_pct   = round(obs_m2 / total_m2, 3) if total_m2 else 0
    shd_pct   = round(shd_m2 / total_m2, 3) if total_m2 else 0

    orientation, orient_deg = infer_orientation(roof_cnt)
    conf, conf_reasons      = compute_confidence(image_bgr, roof_px, obs_pct, shd_pct)

    roof = RoofAnalysis(
        total_area_m2      = total_m2,
        obstacle_area_m2   = obs_m2,
        shadow_area_m2     = shd_m2,
        usable_area_m2     = usable_m2,
        obstacle_pct       = obs_pct,
        shadow_pct         = shd_pct,
        orientation        = orientation,
        orientation_deg    = orient_deg,
        confidence         = conf,
        confidence_reasons = conf_reasons,
        obstacle_labels    = obs_labels,
        detection_method   = det_method,
    )

    solar  = fetch_solar_resource(lat, lng)
    energy = compute_energy(roof, solar)

    return FullAnalysis(roof=roof, solar=solar, energy=energy, lat=lat, lng=lng)
=======
"""
solar_analysis.py — BrightBox Phase 1 + YOLO analysis engine

Obstacle detection:
  - Primary: YOLOv8n ONNX (cv2.dnn / onnxruntime) — no ultralytics needed at runtime
  - Fallback: HSV colour masking (used when yolov8n.onnx is not present)

Everything else:
  - Shadow detection    : LAB luminance adaptive thresholding
  - Solar irradiance    : NASA POWER 10-year climatology (MNRE fallback)
  - Energy model        : physics-based losses (temperature, dust, inverter, wiring, degradation)
  - Orientation         : inferred from minimum bounding rectangle
  - Confidence scoring  : image quality + detection certainty
"""

import json
import math
import os
import logging
import requests
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolov8n.onnx")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PANEL_LENGTH_M   = 1.65
PANEL_WIDTH_M    = 0.99
PANEL_AREA_M2    = PANEL_LENGTH_M * PANEL_WIDTH_M
PANEL_EFFICIENCY = 0.20

GRID_CO2_KG_PER_KWH = 0.82

LOSS_TEMPERATURE  = 0.92
LOSS_DUST         = 0.95
LOSS_INVERTER     = 0.96
LOSS_WIRING       = 0.98
LOSS_MISMATCH     = 0.98
LOSS_AVAILABILITY = 0.99

SYSTEM_LOSS = (
    LOSS_TEMPERATURE * LOSS_DUST * LOSS_INVERTER *
    LOSS_WIRING      * LOSS_MISMATCH * LOSS_AVAILABILITY
)

ANNUAL_DEGRADATION = 0.005
PACKING_EFFICIENCY = 0.65
MONSOON_MONTHS     = {6, 7, 8, 9}
DAYS_PER_MONTH     = [31,28,31,30,31,30,31,31,30,31,30,31]
MONTH_KEYS         = ["JAN","FEB","MAR","APR","MAY","JUN",
                      "JUL","AUG","SEP","OCT","NOV","DEC"]
MONTH_NAMES        = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]
CARDINAL           = ["N","NE","E","SE","S","SW","W","NW"]

# COCO class IDs that indicate rooftop obstacles when detected from above
# (person=0 excluded — people on roofs are temporary)
OBSTACLE_CLASSES = {
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    62: "tv/monitor",
    63: "laptop",
    66: "keyboard",
    67: "phone",
    72: "refrigerator / AC unit",
    73: "book",
    74: "clock",
    76: "scissors",
    77: "teddy bear / object",
    # Most useful for satellite:
    2:  "car / vehicle on roof",
    7:  "truck",
    9:  "traffic light / pole",
    11: "stop sign",
}

# Additional: any detected box labelled as these is always an obstacle
ROOFTOP_OBSTACLE_KEYWORDS = [
    "tank", "ac", "air", "vent", "pipe", "stair",
    "shed", "cabin", "box", "unit",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoofAnalysis:
    total_area_m2:      float
    obstacle_area_m2:   float
    shadow_area_m2:     float
    usable_area_m2:     float
    obstacle_pct:       float
    shadow_pct:         float
    orientation:        str
    orientation_deg:    float
    confidence:         float
    confidence_reasons: list = field(default_factory=list)
    confidence_breakdown: dict = field(default_factory=dict)
    obstacle_labels:    list = field(default_factory=list)
    detection_method:   str  = "hsv_fallback"

@dataclass
class SolarResource:
    annual_kwh_m2:    float
    monthly_kwh_m2:   list
    peak_sun_hours:   float
    monsoon_loss_pct: float
    data_source:      str

@dataclass
class EnergyEstimate:
    panel_count:        int
    system_kw:          float
    annual_kwh:         float
    monthly_kwh:        list
    daily_avg_kwh:      float
    best_month:         str
    best_month_kwh:     float
    worst_month:        str
    worst_month_kwh:    float
    system_loss_factor: float
    loss_breakdown:     dict
    year25_kwh:         float
    co2_kg_year:        float

@dataclass
class FullAnalysis:
    roof:   RoofAnalysis
    solar:  SolarResource
    energy: EnergyEstimate
    lat:    float
    lng:    float
    debug:  dict = field(default_factory=dict)

    def to_dict(self):
        import dataclasses
        return dataclasses.asdict(self)

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def meters_per_pixel(lat: float, zoom: int = 18) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def _contour_to_mask(shape, contour):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if contour is not None:
        cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def _sanitize_debug_payload(payload):
    if isinstance(payload, dict):
        return {k: _sanitize_debug_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_sanitize_debug_payload(v) for v in payload]
    if isinstance(payload, tuple):
        return [_sanitize_debug_payload(v) for v in payload]
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


def _build_debug_overlay(image_bgr, roof_mask, obstacle_mask, shadow_mask, usable_mask):
    overlay = image_bgr.copy()

    def tint(mask, color, image_weight, color_weight):
        selected = mask > 0
        if not np.any(selected):
            return
        base = overlay[selected]
        color_block = np.full_like(base, color)
        overlay[selected] = cv2.addWeighted(base, image_weight, color_block, color_weight, 0)

    tint(roof_mask, (80, 160, 255), 0.55, 0.45)
    tint(usable_mask, (60, 190, 110), 0.35, 0.65)
    tint(obstacle_mask, (40, 40, 230), 0.25, 0.75)
    tint(shadow_mask, (120, 20, 160), 0.25, 0.75)
    return overlay


def _save_debug_artifacts(debug_dir, image_bgr, debug_payload, masks):
    os.makedirs(debug_dir, exist_ok=True)

    artifacts = {
        "original_image": os.path.join(debug_dir, "original.png"),
        "roof_mask": os.path.join(debug_dir, "roof_mask.png"),
        "obstacle_mask": os.path.join(debug_dir, "obstacle_mask.png"),
        "shadow_mask": os.path.join(debug_dir, "shadow_mask.png"),
        "usable_mask": os.path.join(debug_dir, "usable_mask.png"),
        "overlay": os.path.join(debug_dir, "overlay.png"),
        "metrics": os.path.join(debug_dir, "analysis.json"),
    }

    cv2.imwrite(artifacts["original_image"], image_bgr)
    cv2.imwrite(artifacts["roof_mask"], masks["roof_mask"])
    cv2.imwrite(artifacts["obstacle_mask"], masks["obstacle_mask"])
    cv2.imwrite(artifacts["shadow_mask"], masks["shadow_mask"])
    cv2.imwrite(artifacts["usable_mask"], masks["usable_mask"])
    cv2.imwrite(
        artifacts["overlay"],
        _build_debug_overlay(
            image_bgr,
            masks["roof_mask"],
            masks["obstacle_mask"],
            masks["shadow_mask"],
            masks["usable_mask"],
        ),
    )

    with open(artifacts["metrics"], "w", encoding="utf-8") as f:
        json.dump(_sanitize_debug_payload(debug_payload), f, indent=2)

    return artifacts

# ---------------------------------------------------------------------------
# YOLO obstacle detection (primary)
# ---------------------------------------------------------------------------

_yolo_net = None   # cached after first load

def _load_yolo():
    global _yolo_net
    if _yolo_net is not None:
        return _yolo_net
    if not os.path.exists(MODEL_PATH):
        logger.info("yolov8n.onnx not found — using HSV fallback")
        return None
    try:
        net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _yolo_net = net
        logger.info("✓ YOLOv8n ONNX loaded via cv2.dnn")
        return net
    except Exception as e:
        logger.warning(f"Failed to load YOLO model: {e}")
        return None


def detect_obstacles_yolo(image_bgr: np.ndarray,
                           roof_contour,
                           mpp: float = 0.6,
                           conf_threshold: float = 0.20,
                           iou_threshold:  float = 0.40) -> tuple:
    """
    Runs YOLOv8n inference via cv2.dnn on the cropped rooftop image.
    Returns (obstacle_mask, obstacle_area_px, labels_list).
    """
    net = _load_yolo()
    if net is None:
        return None, 0.0, []

    h, w = image_bgr.shape[:2]

    # Build roof-only mask
    roof_mask = np.zeros((h, w), dtype=np.uint8)
    if roof_contour is not None:
        cv2.drawContours(roof_mask, [roof_contour], -1, 255, -1)

    # YOLOv8 expects 640×640 normalised input
    input_size = 640
    blob = cv2.dnn.blobFromImage(
        image_bgr, scalefactor=1/255.0,
        size=(input_size, input_size),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward()                  # shape: (1, 84, 8400) for YOLOv8n

    # YOLOv8 output layout: [batch, 84, anchors]
    # 84 = 4 (box xywh) + 80 (class scores)
    predictions = outputs[0]                  # (84, 8400)
    predictions = predictions.T               # (8400, 84)

    boxes, confidences, class_ids = [], [], []
    scale_x = w / input_size
    scale_y = h / input_size

    for pred in predictions:
        scores = pred[4:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])
        if confidence < conf_threshold:
            continue

        cx, cy, bw, bh = pred[:4]
        x1 = int((cx - bw / 2) * scale_x)
        y1 = int((cy - bh / 2) * scale_y)
        x2 = int((cx + bw / 2) * scale_x)
        y2 = int((cy + bh / 2) * scale_y)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(confidence)
        class_ids.append(class_id)

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
    if len(indices) == 0:
        return np.zeros((h, w), dtype=np.uint8), 0.0, []

    obs_mask = np.zeros((h, w), dtype=np.uint8)
    labels   = []
    min_px   = int((0.4 / (mpp ** 2)))

    for i in indices.flatten():
        x, y, bw, bh = boxes[i]
        area = bw * bh
        if area < min_px:
            continue

        # Only draw inside the roof boundary
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(tmp, (x, y), (x + bw, y + bh), 255, -1)
        tmp = cv2.bitwise_and(tmp, tmp, mask=roof_mask)

        if np.sum(tmp > 0) < min_px:
            continue

        obs_mask = cv2.bitwise_or(obs_mask, tmp)
        cid   = class_ids[i]
        cname = OBSTACLE_CLASSES.get(cid, f"object (class {cid})")
        real_m2 = round(area * (mpp ** 2), 1)
        labels.append(f"{cname} (~{real_m2} m², conf {confidences[i]:.0%})")

    obs_px = float(np.sum(obs_mask > 0))
    return obs_mask, obs_px, labels


# ---------------------------------------------------------------------------
# HSV fallback obstacle detection
# ---------------------------------------------------------------------------

def detect_obstacles_hsv(image_bgr: np.ndarray,
                          roof_contour,
                          mpp: float = 0.6) -> tuple:
    """
    Colour-based obstacle detection when YOLO model is not available.
    Detects: dark tanks, metallic AC units, bright concrete staircases.
    """
    h, w  = image_bgr.shape[:2]
    hsv   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    roof_mask = np.zeros((h, w), dtype=np.uint8)
    if roof_contour is not None:
        cv2.drawContours(roof_mask, [roof_contour], -1, 255, -1)

    def masked(m):
        return cv2.bitwise_and(m, m, mask=roof_mask)

    dark     = cv2.inRange(hsv, np.array([0,  0,  0]),   np.array([180, 80,  70]))
    metal    = cv2.inRange(hsv, np.array([0,  0,  160]), np.array([180, 30, 235]))
    concrete = cv2.inRange(hsv, np.array([0,  0,  220]), np.array([180, 25, 255]))

    combined = cv2.bitwise_or(dark, cv2.bitwise_or(metal, concrete))
    combined = masked(combined)

    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  el, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, el, iterations=2)

    min_px = int(0.4 / (mpp ** 2))
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obs_mask = np.zeros((h, w), dtype=np.uint8)
    labels   = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_px:
            continue
        cv2.drawContours(obs_mask, [cnt], -1, 255, -1)
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(tmp, [cnt], -1, 255, -1)
        mean_v = cv2.mean(hsv[:,:,2], mask=tmp)[0]
        mean_s = cv2.mean(hsv[:,:,1], mask=tmp)[0]
        real_m2 = round(area * (mpp ** 2), 1)
        if mean_v < 70:
            labels.append(f"dark structure / water tank (~{real_m2} m²)")
        elif mean_s < 25 and mean_v > 200:
            labels.append(f"staircase / concrete (~{real_m2} m²)")
        else:
            labels.append(f"AC unit / metallic object (~{real_m2} m²)")

    obs_px = float(np.sum(obs_mask > 0))
    return obs_mask, obs_px, labels


# ---------------------------------------------------------------------------
# Unified obstacle entry point
# ---------------------------------------------------------------------------

def detect_obstacles(image_bgr, roof_contour, mpp=0.6):
    """
    Tries YOLO first, falls back to HSV if model not loaded.
    Returns (mask, area_px, labels, method_used).
    """
    mask, px, labels = detect_obstacles_yolo(image_bgr, roof_contour, mpp)
    if mask is not None:
        return mask, px, labels, "yolov8n"

    mask, px, labels = detect_obstacles_hsv(image_bgr, roof_contour, mpp)
    return mask, px, labels, "hsv_colour_mask"


# ---------------------------------------------------------------------------
# Roof boundary
# ---------------------------------------------------------------------------

def detect_roof_boundary(image_bgr):
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.bilateralFilter(gray, 5, 7, 5)
    k     = np.array([[-2,-2,-2],[-2,17,-2],[-2,-2,-2]], dtype="int")
    sharp = cv2.filter2D(blur, -1, k)
    _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    roof = max(cnts, key=cv2.contourArea)
    return roof, float(cv2.contourArea(roof))


# ---------------------------------------------------------------------------
# Shadow detection
# ---------------------------------------------------------------------------

def detect_shadows(image_bgr, roof_contour, obstacle_mask):
    h, w = image_bgr.shape[:2]
    roof_mask = np.zeros((h, w), dtype=np.uint8)
    if roof_contour is not None:
        cv2.drawContours(roof_mask, [roof_contour], -1, 255, -1)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L   = lab[:, :, 0]

    # Block size must be odd and scale with image — small images need smaller blocks
    block = max(11, min(31, (min(h, w) // 20) | 1))  # odd, 11–31

    thresh = cv2.adaptiveThreshold(
        L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=block, C=8)

    thresh = cv2.bitwise_and(thresh, thresh, mask=roof_mask)
    thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(obstacle_mask))

    # Kernel also scales with image
    k_size = max(3, min(7, min(h, w) // 80))
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    shadow_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, el, iterations=1)
    return shadow_mask, float(np.sum(shadow_mask > 0))


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def infer_orientation(roof_contour):
    if roof_contour is None:
        return "Unknown", 0.0
    rect    = cv2.minAreaRect(roof_contour)
    angle   = rect[2]
    bearing = (angle + 90) % 180
    idx     = int((bearing + 22.5) / 45) % 8
    return CARDINAL[idx], round(float(bearing), 1)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def compute_confidence(image_bgr, roof_px, obstacle_pct, shadow_pct, detection_method):
    h, w  = image_bgr.shape[:2]
    total = h * w
    reasons = []
    penalties = []
    score   = 1.0

    # Resolution check — after upscaling this should pass
    if total < 40_000:
        score -= 0.20
        reasons.append("Very low resolution — try zooming in on the map before cropping")
        penalties.append({"metric": "image_resolution", "penalty": 0.20, "detail": "very_low"})
    elif total < 160_000:
        score -= 0.05
        reasons.append("Moderate resolution — tighter crop improves accuracy")
        penalties.append({"metric": "image_resolution", "penalty": 0.05, "detail": "moderate"})

    roof_frac = (roof_px / total) if total else 0
    if roof_frac < 0.10:
        score -= 0.20
        reasons.append("Roof too small in frame — crop more tightly around just your rooftop")
        penalties.append({"metric": "roof_fraction", "penalty": 0.20, "detail": "roof_too_small"})
    elif roof_frac < 0.30:
        score -= 0.06
        reasons.append("Crop includes area beyond the rooftop — tighter crop improves accuracy")
        penalties.append({"metric": "roof_fraction", "penalty": 0.06, "detail": "crop_too_wide"})

    if obstacle_pct > 0.45:
        score -= 0.15
        reasons.append(f"Very high obstacle coverage ({obstacle_pct:.0%}) — usable area uncertain")
        penalties.append({"metric": "obstacle_pct", "penalty": 0.15, "detail": "very_high"})
    elif obstacle_pct > 0.25:
        score -= 0.06
        reasons.append(f"Significant obstacles detected ({obstacle_pct:.0%} of roof area)")
        penalties.append({"metric": "obstacle_pct", "penalty": 0.06, "detail": "moderate"})

    if shadow_pct > 0.50:
        score -= 0.15
        reasons.append(f"Heavy shading ({shadow_pct:.0%}) — energy output significantly reduced")
        penalties.append({"metric": "shadow_pct", "penalty": 0.15, "detail": "heavy"})
    elif shadow_pct > 0.25:
        score -= 0.06
        reasons.append(f"Moderate shading ({shadow_pct:.0%})")
        penalties.append({"metric": "shadow_pct", "penalty": 0.06, "detail": "moderate"})

    gray      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharpness < 40:
        score -= 0.10
        reasons.append("Low image contrast — satellite tile quality may be poor here")
        penalties.append({"metric": "sharpness_variance", "penalty": 0.10, "detail": "low"})

    if detection_method != "yolov8n":
        score -= 0.08
        reasons.append(f"Obstacle detection used {detection_method} fallback instead of AI model")
        penalties.append({"metric": "detection_method", "penalty": 0.08, "detail": detection_method})

    if not reasons:
        reasons.append("Good image quality and clear roof boundary detected")

    score = round(max(0.05, min(1.0, score)), 2)
    breakdown = {
        "metrics": {
            "image_pixels": int(total),
            "roof_fraction": round(roof_frac, 3),
            "obstacle_pct": round(obstacle_pct, 3),
            "shadow_pct": round(shadow_pct, 3),
            "sharpness_variance": round(float(sharpness), 2),
            "detection_method": detection_method,
        },
        "penalties": penalties,
        "score_after_penalties": score,
    }
    return score, reasons, breakdown


# ---------------------------------------------------------------------------
# NASA POWER solar resource
# ---------------------------------------------------------------------------

def fetch_solar_resource(lat, lon):
    url = (
        "https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?parameters=ALLSKY_SFC_SW_DWN&community=RE"
        f"&longitude={lon}&latitude={lat}&format=JSON&user=BrightBoxApp"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        vals = resp.json()["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]

        monthly_kwh = []
        monsoon_kwh = 0.0
        total_kwh   = 0.0
        for i, key in enumerate(MONTH_KEYS):
            daily     = float(vals.get(key, 4.5))
            month_tot = daily * DAYS_PER_MONTH[i]
            monthly_kwh.append(round(month_tot, 2))
            total_kwh += month_tot
            if (i + 1) in MONSOON_MONTHS:
                monsoon_kwh += month_tot

        return SolarResource(
            annual_kwh_m2    = round(total_kwh, 1),
            monthly_kwh_m2   = monthly_kwh,
            peak_sun_hours   = round(total_kwh / 365, 2),
            monsoon_loss_pct = round(monsoon_kwh / total_kwh, 3) if total_kwh else 0.22,
            data_source      = "NASA POWER (10-yr climatology)",
        )
    except Exception as e:
        logger.warning(f"NASA POWER failed ({e}) — using India regional fallback")
        return _india_fallback(lat)


def _india_fallback(lat):
    if lat < 15:
        daily = [5.8,6.2,6.8,7.0,6.5,4.2,3.8,4.0,4.8,5.5,5.6,5.5]
    elif lat < 22:
        daily = [5.4,6.0,6.6,6.8,6.2,3.8,3.5,3.7,4.5,5.2,5.3,5.1]
    else:
        daily = [4.8,5.5,6.2,6.8,6.5,4.8,4.2,4.4,5.0,5.4,5.0,4.6]
    monthly_kwh = [round(daily[i] * DAYS_PER_MONTH[i], 2) for i in range(12)]
    total   = sum(monthly_kwh)
    monsoon = sum(monthly_kwh[i] for i in range(12) if (i+1) in MONSOON_MONTHS)
    return SolarResource(
        annual_kwh_m2    = round(total, 1),
        monthly_kwh_m2   = monthly_kwh,
        peak_sun_hours   = round(total / 365, 2),
        monsoon_loss_pct = round(monsoon / total, 3) if total else 0.22,
        data_source      = "MNRE Solar Atlas (regional fallback)",
    )


# ---------------------------------------------------------------------------
# Energy model
# ---------------------------------------------------------------------------

def compute_energy(roof: RoofAnalysis, solar: SolarResource) -> EnergyEstimate:
    usable = roof.usable_area_m2
    panel_count = max(0, int((usable * PACKING_EFFICIENCY) / PANEL_AREA_M2))
    system_kw   = round(panel_count * PANEL_AREA_M2 * PANEL_EFFICIENCY, 2)

    orientation_factors = {
        "S":1.00,"SE":0.97,"SW":0.97,
        "E":0.87,"W":0.87,
        "NE":0.78,"NW":0.78,
        "N":0.70,"Unknown":0.90,
    }
    of = orientation_factors.get(roof.orientation, 0.90)

    monthly_kwh = []
    for ghi in solar.monthly_kwh_m2:
        net = ghi * usable * PANEL_EFFICIENCY * of * SYSTEM_LOSS
        monthly_kwh.append(round(net, 1))

    annual_kwh = round(sum(monthly_kwh), 1)
    best_i     = int(np.argmax(monthly_kwh))
    worst_i    = int(np.argmin(monthly_kwh))

    year25 = sum(
        annual_kwh * ((1 - ANNUAL_DEGRADATION) ** yr)
        for yr in range(25)
    )

    return EnergyEstimate(
        panel_count        = panel_count,
        system_kw          = system_kw,
        annual_kwh         = annual_kwh,
        monthly_kwh        = monthly_kwh,
        daily_avg_kwh      = round(annual_kwh / 365, 2),
        best_month         = MONTH_NAMES[best_i],
        best_month_kwh     = monthly_kwh[best_i],
        worst_month        = MONTH_NAMES[worst_i],
        worst_month_kwh    = monthly_kwh[worst_i],
        system_loss_factor = round(SYSTEM_LOSS, 3),
        loss_breakdown     = {
            "temperature_loss_pct":  round((1-LOSS_TEMPERATURE)*100, 1),
            "dust_loss_pct":         round((1-LOSS_DUST)*100,        1),
            "inverter_loss_pct":     round((1-LOSS_INVERTER)*100,    1),
            "wiring_loss_pct":       round((1-LOSS_WIRING)*100,      1),
            "mismatch_loss_pct":     round((1-LOSS_MISMATCH)*100,    1),
            "availability_loss_pct": round((1-LOSS_AVAILABILITY)*100,1),
            "total_loss_pct":        round((1-SYSTEM_LOSS)*100,      1),
            "orientation_factor":    of,
        },
        year25_kwh   = round(year25, 0),
        co2_kg_year  = round(annual_kwh * GRID_CO2_KG_PER_KWH, 0),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse(image_bgr: np.ndarray, lat: float, lng: float,
            zoom: int = 18, debug_dir: Optional[str] = None,
            sample_id: Optional[str] = None) -> FullAnalysis:
    mpp = meters_per_pixel(lat, zoom)

    roof_cnt, roof_px = detect_roof_boundary(image_bgr)
    total_m2 = round(roof_px * (mpp ** 2), 2)
    roof_mask = _contour_to_mask(image_bgr.shape, roof_cnt)

    obs_mask, obs_px, obs_labels, det_method = detect_obstacles(
        image_bgr, roof_cnt, mpp)
    obs_m2 = round(obs_px * (mpp ** 2), 2)

    shd_mask, shd_px = detect_shadows(image_bgr, roof_cnt, obs_mask)
    shd_m2 = round(shd_px * (mpp ** 2), 2)
    usable_mask = cv2.bitwise_and(roof_mask, cv2.bitwise_not(obs_mask))
    usable_mask = cv2.bitwise_and(usable_mask, cv2.bitwise_not(shd_mask))

    usable_m2 = max(0.0, round(total_m2 - obs_m2 - shd_m2, 2))
    obs_pct   = round(obs_m2 / total_m2, 3) if total_m2 else 0
    shd_pct   = round(shd_m2 / total_m2, 3) if total_m2 else 0

    orientation, orient_deg = infer_orientation(roof_cnt)
    conf, conf_reasons, conf_breakdown = compute_confidence(
        image_bgr, roof_px, obs_pct, shd_pct, det_method
    )

    roof = RoofAnalysis(
        total_area_m2      = total_m2,
        obstacle_area_m2   = obs_m2,
        shadow_area_m2     = shd_m2,
        usable_area_m2     = usable_m2,
        obstacle_pct       = obs_pct,
        shadow_pct         = shd_pct,
        orientation        = orientation,
        orientation_deg    = orient_deg,
        confidence         = conf,
        confidence_reasons = conf_reasons,
        confidence_breakdown = conf_breakdown,
        obstacle_labels    = obs_labels,
        detection_method   = det_method,
    )

    solar  = fetch_solar_resource(lat, lng)
    energy = compute_energy(roof, solar)
    debug = {
        "sample_id": sample_id,
        "zoom": zoom,
        "meters_per_pixel": round(float(mpp), 5),
        "image_shape": {
            "height": int(image_bgr.shape[0]),
            "width": int(image_bgr.shape[1]),
        },
        "pixel_metrics": {
            "roof_area_px": round(float(roof_px), 2),
            "obstacle_area_px": round(float(obs_px), 2),
            "shadow_area_px": round(float(shd_px), 2),
            "usable_area_px": round(float(np.sum(usable_mask > 0)), 2),
        },
        "artifacts": {},
    }

    result = FullAnalysis(
        roof=roof, solar=solar, energy=energy, lat=lat, lng=lng, debug=debug
    )

    if debug_dir:
        artifact_payload = result.to_dict()
        artifact_payload["debug"]["requested_debug_dir"] = debug_dir
        artifact_paths = _save_debug_artifacts(
            debug_dir,
            image_bgr,
            artifact_payload,
            {
                "roof_mask": roof_mask,
                "obstacle_mask": obs_mask,
                "shadow_mask": shd_mask,
                "usable_mask": usable_mask,
            },
        )
        result.debug["artifacts"] = artifact_paths

    return result
>>>>>>> 66e2e02 (update struct)
