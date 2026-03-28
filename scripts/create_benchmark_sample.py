import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset" / "samples"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create a benchmark sample folder for rooftop evaluation."
    )
    parser.add_argument("--id", required=True, help="Unique sample id, e.g. pune-flat-roof-01")
    parser.add_argument("--image", required=True, help="Path to source image")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lng", type=float, required=True, help="Longitude")
    parser.add_argument("--zoom", type=int, default=19, help="Map zoom used for the image")
    parser.add_argument("--usable-area", type=float, default=0.0, help="Expected usable roof area in m2")
    parser.add_argument("--panel-count", type=int, default=0, help="Expected panel count")
    parser.add_argument("--annual-kwh", type=float, default=0.0, help="Expected annual generation")
    parser.add_argument("--notes", default="", help="Reviewer notes")
    return parser


def main():
    args = build_parser().parse_args()

    source_image = Path(args.image).resolve()
    if not source_image.exists():
        raise FileNotFoundError(f"Image not found: {source_image}")

    sample_dir = DATASET_DIR / args.id
    if sample_dir.exists():
        raise FileExistsError(f"Sample already exists: {sample_dir}")

    sample_dir.mkdir(parents=True, exist_ok=False)

    image_name = source_image.name
    destination_image = sample_dir / image_name
    shutil.copy2(source_image, destination_image)

    metadata = {
        "id": args.id,
        "image": image_name,
        "lat": args.lat,
        "lng": args.lng,
        "zoom": args.zoom,
        "notes": args.notes,
        "expected": {
            "usable_area_m2": args.usable_area,
            "panel_count": args.panel_count,
            "annual_kwh": args.annual_kwh,
        },
    }

    metadata_path = sample_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created benchmark sample: {sample_dir}")
    print(f"Image: {destination_image}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
