# Benchmark Dataset

Store evaluation samples in `dataset/samples/<sample-id>/`.

Each sample folder should include:

- `image.png` or `image.jpg`
- `metadata.json`
- optional reviewer notes or reference masks

Example `metadata.json`:

```json
{
  "id": "pune-flat-roof-01",
  "image": "image.png",
  "lat": 18.5204,
  "lng": 73.8567,
  "zoom": 19,
  "notes": "Manual review: moderate tank clutter on southeast edge.",
  "expected": {
    "usable_area_m2": 38.5,
    "panel_count": 15,
    "annual_kwh": 8450
  }
}
```

Run the benchmark with:

```bash
python evaluate.py
```

Evaluation outputs and debug artifacts will be written to `artifacts/evaluations/`.

Quick ways to add samples:

1. Copy the template in `dataset/sample-template.metadata.json` into a new sample folder.
2. Or use the helper script:

```bash
python scripts/create_benchmark_sample.py --id pune-flat-roof-01 --image 1.jpg --lat 18.5204 --lng 73.8567 --notes "Starter sample"
```

Recommended starter workflow:

- add 5 to 10 cropped rooftop images first
- keep expected values rough at the beginning
- run `python evaluate.py`
- review overlays in `artifacts/evaluations/`
- refine the expected values after manual inspection
