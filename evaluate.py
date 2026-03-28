import json
from pathlib import Path

import cv2

import solar_analysis as sa


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset" / "samples"
EVAL_OUTPUT_DIR = ROOT / "artifacts" / "evaluations"


def _load_samples():
    if not DATASET_DIR.exists():
        return []
    return sorted(DATASET_DIR.glob("*/metadata.json"))


def _calc_error(actual, expected):
    if expected in (None, ""):
        return None
    error = round(actual - expected, 2)
    pct = round((abs(error) / expected) * 100, 2) if expected else None
    return {
        "expected": expected,
        "actual": round(actual, 2),
        "error": error,
        "abs_error": round(abs(error), 2),
        "pct_error": pct,
    }


def evaluate_sample(metadata_path: Path):
    with metadata_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    sample_dir = metadata_path.parent
    image_path = sample_dir / meta["image"]
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    sample_id = meta.get("id", sample_dir.name)
    debug_dir = EVAL_OUTPUT_DIR / sample_id
    analysis = sa.analyse(
        image_bgr,
        lat=float(meta["lat"]),
        lng=float(meta["lng"]),
        zoom=int(meta.get("zoom", 19)),
        debug_dir=str(debug_dir),
        sample_id=sample_id,
    )

    expected = meta.get("expected", {})
    return {
        "sample_id": sample_id,
        "source_image": str(image_path),
        "notes": meta.get("notes", ""),
        "detection_method": analysis.roof.detection_method,
        "confidence": analysis.roof.confidence,
        "errors": {
            "usable_area_m2": _calc_error(
                analysis.roof.usable_area_m2, expected.get("usable_area_m2")
            ),
            "panel_count": _calc_error(
                float(analysis.energy.panel_count), expected.get("panel_count")
            ),
            "annual_kwh": _calc_error(
                analysis.energy.annual_kwh, expected.get("annual_kwh")
            ),
        },
        "result": analysis.to_dict(),
    }


def main():
    samples = _load_samples()
    if not samples:
        print("No samples found in dataset/samples. Add sample folders with metadata.json first.")
        return

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for metadata_path in samples:
        result = evaluate_sample(metadata_path)
        results.append(result)
        print(
            f"{result['sample_id']}: confidence={result['confidence']:.2f}, "
            f"method={result['detection_method']}, "
            f"usable_area={result['result']['roof']['usable_area_m2']:.2f} m2, "
            f"annual_kwh={result['result']['energy']['annual_kwh']:.1f}"
        )

    summary_path = EVAL_OUTPUT_DIR / "latest-summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
