<<<<<<< HEAD
# Bright Box — Solar Energy Estimator

Analyse any rooftop from satellite imagery and get an instant solar energy potential estimate.

## How it works

1. Search for any location on the map
2. Click to drop a pin — a satellite tile loads automatically
3. Crop the rooftop area using the cropper tool
4. Click **Crop and Analyse** — the image is sent to the backend
5. Computer vision detects the roof area, calculates panel count, and fetches live solar radiation data
6. Results page shows estimated kWh/day and kWh/year

=======
# Bright Box

Bright Box is a rooftop solar pre-feasibility tool. A user selects a location, crops a rooftop from satellite imagery, and gets an estimated usable roof area, panel count, yearly energy production, and a simple savings projection.

## Current scope

- Flask backend for tile stitching and rooftop analysis
- Static map and crop UI
- Results page with confidence, energy, and savings summary
- Evaluation script for benchmark samples
- YOLO-assisted obstacle detection with a CV fallback

This is a prototype for early solar estimation, not a final installer-grade design tool.

## Run locally

### Prerequisites

- Python 3.9+
- pip

### Commands

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

## Benchmark evaluation

Add sample rooftops under `dataset/samples/<sample-id>/` and run:

```bash
python evaluate.py
```

Outputs are written to `artifacts/evaluations/`.

## Repository structure

```text
brightbox-solar-energy/
|-- app.py
|-- solar_analysis.py
|-- evaluate.py
|-- download_model.py
|-- requirements.txt
|-- README.md
|-- .env.example
|-- static/
|   |-- index.html
|   `-- styles.css
|-- templates/
|   `-- result.html
|-- models/
|   |-- yolov8n.onnx
|   `-- yolov8n.pt
|-- dataset/
|-- scripts/
|-- docs/
|   `-- endgoals.md
|-- legacy/
|   |-- assets/
|   |-- frontend-shadcn/
|   `-- experiments/
`-- uploads/
```

## Active files

- `app.py`
- `solar_analysis.py`
- `static/index.html`
- `static/styles.css`
- `templates/result.html`
- `evaluate.py`
- `dataset/`
- `scripts/`
- `models/`

## Archived files

Legacy experiments, scratch assets, and the unused shadcn component dump now live under `legacy/` so the working product code stays easy to navigate.

## Next product steps

- replace session-based result passing with stored analysis records
- add persistent storage for projects and saved runs
- expand the benchmark dataset and accuracy reporting
- move the static UI into a proper frontend app when the flow stabilizes
>>>>>>> 66e2e02 (update struct)
