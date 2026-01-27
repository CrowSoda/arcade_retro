#!/usr/bin/env python3
"""
Parse creamy_chicken_dets.json log output and create a clean golden detections database.
Usage: python parse_golden_detections.py
"""

import json
import re
from pathlib import Path


def parse_detection_log(log_path: str) -> dict:
    """Parse the raw log file and extract all detections."""

    with open(log_path) as f:
        content = f.read()

    # Find all JSON objects (chunk data)
    pattern = r'\{\s*"chunk_id":\s*\d+,\s*"boxes":\s*\[.*?\],\s*"elapsed_seconds":\s*[\d.]+\s*\}'
    matches = re.findall(pattern, content, re.DOTALL)

    all_chunks = []
    all_boxes = []

    for match in matches:
        try:
            chunk = json.loads(match)
            all_chunks.append(chunk)

            # Add chunk_id to each box for reference
            for box in chunk["boxes"]:
                box["chunk_id"] = chunk["chunk_id"]
                box["elapsed_seconds"] = chunk["elapsed_seconds"]
                all_boxes.append(box)
        except json.JSONDecodeError:
            continue

    # Calculate statistics
    total_boxes = len(all_boxes)
    chunks_with_detections = len([c for c in all_chunks if c["boxes"]])

    # Group by Y position bands (frequency ranges)
    y_bands = {}
    for box in all_boxes:
        y_center = (box["y1"] + box["y2"]) / 2
        band = int(y_center / 100) * 100  # Group into 100px bands
        y_bands[band] = y_bands.get(band, 0) + 1

    return {
        "metadata": {
            "source": "creamy_chicken_tensorcade_inference",
            "total_chunks": len(all_chunks),
            "chunks_with_detections": chunks_with_detections,
            "total_boxes": total_boxes,
            "time_range_sec": [all_chunks[0]["elapsed_seconds"], all_chunks[-1]["elapsed_seconds"]]
            if all_chunks
            else [0, 0],
            "coordinate_system": {
                "x_axis": "time (0-1024 pixels)",
                "y_axis": "frequency (0-1024 pixels)",
                "normalized": False,
                "inference_size": 1024,
            },
            "y_band_distribution": dict(sorted(y_bands.items())),
        },
        "chunks": all_chunks,
        "all_boxes": all_boxes,
    }


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "docs" / "creamy_chicken_dets.json"
    output_path = base_dir / "data" / "golden_detections.json"

    # Ensure data dir exists
    output_path.parent.mkdir(exist_ok=True)

    print(f"Parsing: {input_path}")
    result = parse_detection_log(str(input_path))

    print("\nSummary:")
    print(f"  Total chunks: {result['metadata']['total_chunks']}")
    print(f"  Chunks with detections: {result['metadata']['chunks_with_detections']}")
    print(f"  Total boxes: {result['metadata']['total_boxes']}")
    print(
        f"  Time range: {result['metadata']['time_range_sec'][0]:.1f}s - {result['metadata']['time_range_sec'][1]:.1f}s"
    )
    print("\n  Y-band distribution (frequency):")
    for band, count in result["metadata"]["y_band_distribution"].items():
        print(f"    {band}-{band+100}: {count} detections")

    # Save
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to: {output_path}")
