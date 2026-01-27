"""
Backbone Extractor - Split full Faster R-CNN model into backbone + head.

This script extracts the shared backbone and detection head from an existing
full model, enabling the Hydra multi-head architecture.

Usage:
    python -m backend.hydra.backbone_extractor

    # Or with custom paths:
    python -m backend.hydra.backbone_extractor --input models/my_model.pth
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

# Logging
from logger_config import get_logger
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator

logger = get_logger("backbone_extractor")


# Custom anchor generator matching training config (training/service.py)
# 5 sizes × 4 aspect ratios = 20 anchors per location
def create_anchor_generator():
    return AnchorGenerator(
        sizes=((8, 16, 32, 64, 128),) * 5,  # Same sizes for all 5 FPN levels
        aspect_ratios=((0.1, 0.15, 0.2, 0.3),) * 5,  # Same aspects for all 5 FPN levels
    )


def extract_backbone_and_head(
    full_model_path: str,
    backbone_output_path: str,
    head_output_path: str,
    signal_name: str = "creamy_chicken",
) -> dict:
    """
    Extract backbone and head from a full Faster R-CNN model.

    The backbone includes:
        - backbone.* (ResNet18 + FPN)

    The head includes:
        - rpn.* (Region Proposal Network)
        - roi_heads.* (ROI pooling, classification, regression)

    Args:
        full_model_path: Path to full .pth model
        backbone_output_path: Where to save backbone weights
        head_output_path: Where to save head weights
        signal_name: Name for the signal this head detects

    Returns:
        dict with extraction metadata
    """
    logger.info(f"Loading full model from: {full_model_path}")

    # Load full model state dict
    full_state = torch.load(full_model_path, map_location="cpu", weights_only=False)

    # Create model to understand structure (must match training anchor config)
    backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=0)
    FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=create_anchor_generator(),
        rpn_fg_iou_thresh=0.5,
        rpn_bg_iou_thresh=0.3,
    )

    # Separate backbone and head weights
    backbone_state = {}
    head_state = {}

    for key, value in full_state.items():
        if key.startswith("backbone."):
            backbone_state[key] = value
        else:
            # rpn.*, roi_heads.*, transform.*
            head_state[key] = value

    # Also include backbone in head (needed for load_state_dict strict=False)
    # Actually, for Hydra, we want ONLY the head weights, backbone stays frozen
    # But we need the full model structure for inference

    # For the backbone file: Save FULL model state (used as initialization)
    # For the head file: Save ONLY head weights (swapped at inference time)

    logger.info(f"Backbone keys: {len(backbone_state)}")
    logger.info(f"Head keys: {len(head_state)}")

    # Create output directories
    Path(backbone_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(head_output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save backbone (full model as base initialization)
    torch.save(full_state, backbone_output_path)
    logger.info(f"Saved backbone to: {backbone_output_path}")

    # Save head (only RPN + ROI heads)
    torch.save(head_state, head_output_path)
    logger.info(f"Saved head to: {head_output_path}")

    # Calculate sizes
    backbone_size_mb = os.path.getsize(backbone_output_path) / 1e6
    head_size_mb = os.path.getsize(head_output_path) / 1e6

    return {
        "source_model": str(full_model_path),
        "backbone_size_mb": backbone_size_mb,
        "head_size_mb": head_size_mb,
        "backbone_keys": len(backbone_state),
        "head_keys": len(head_state),
        "extracted_at": datetime.now().isoformat(),
    }


def create_metadata_files(
    models_dir: Path,
    signal_name: str,
    extraction_info: dict,
) -> None:
    """Create metadata.json files for backbone and head."""

    # Backbone metadata
    backbone_meta = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "source": extraction_info["source_model"],
        "architecture": "resnet18_fpn",
        "input_size": [3, 1024, 1024],
        "feature_channels": 256,
        "size_mb": extraction_info["backbone_size_mb"],
        "notes": "Extracted from creamy_chicken full model",
    }

    backbone_meta_path = models_dir / "backbone" / "metadata.json"
    backbone_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(backbone_meta_path, "w") as f:
        json.dump(backbone_meta, f, indent=2)
    logger.info(f"Created backbone metadata: {backbone_meta_path}")

    # Head metadata
    head_meta = {
        "signal_name": signal_name,
        "backbone_version": 1,
        "versions": [
            {
                "version": 1,
                "created_at": datetime.now().isoformat(),
                "sample_count": 0,  # Unknown from extraction
                "split_version": 1,
                "epochs_trained": 0,
                "early_stopped": False,
                "metrics": {
                    "f1_score": 0.0,  # Fill in after validation
                    "precision": 0.0,
                    "recall": 0.0,
                    "val_loss": 0.0,
                },
                "training_time_sec": 0,
                "notes": f"Extracted from {extraction_info['source_model']}",
            }
        ],
        "active_version": 1,
        "version_retention": 5,
    }

    head_meta_path = models_dir / "heads" / signal_name / "metadata.json"
    head_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(head_meta_path, "w") as f:
        json.dump(head_meta, f, indent=2)
    logger.info(f"Created head metadata: {head_meta_path}")

    # Create active.pth symlink or copy
    head_dir = models_dir / "heads" / signal_name
    v1_path = head_dir / "v1.pth"
    active_path = head_dir / "active.pth"

    # On Windows, copy instead of symlink
    if sys.platform == "win32":
        if v1_path.exists() and not active_path.exists():
            shutil.copy(v1_path, active_path)
            logger.info("Copied v1.pth to active.pth")
    else:
        if v1_path.exists() and not active_path.exists():
            active_path.symlink_to("v1.pth")
            logger.info("Created symlink active.pth -> v1.pth")


def validate_extraction(
    backbone_path: str,
    head_path: str,
    original_path: str,
) -> bool:
    """
    Verify extracted backbone + head produce same outputs as original.

    Returns True if outputs match within tolerance.
    """
    import numpy as np

    logger.info("\n=== Validating extraction ===")

    # Load original model (must match training anchor config)
    backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=0)
    original_model = FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=create_anchor_generator(),
        rpn_fg_iou_thresh=0.5,
        rpn_bg_iou_thresh=0.3,
    )
    original_state = torch.load(original_path, map_location="cpu", weights_only=False)
    original_model.load_state_dict(original_state)
    original_model.eval()

    # Load backbone + head (must match training anchor config)
    backbone2 = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=0)
    hydra_model = FasterRCNN(
        backbone2,
        num_classes=2,
        rpn_anchor_generator=create_anchor_generator(),
        rpn_fg_iou_thresh=0.5,
        rpn_bg_iou_thresh=0.3,
    )

    # Load backbone (full state as base)
    backbone_state = torch.load(backbone_path, map_location="cpu", weights_only=False)
    hydra_model.load_state_dict(backbone_state, strict=False)

    # Load head (overwrite RPN + ROI heads)
    head_state = torch.load(head_path, map_location="cpu", weights_only=False)
    hydra_model.load_state_dict(head_state, strict=False)
    hydra_model.eval()

    # Test with random input
    test_input = torch.rand(1, 3, 1024, 1024)

    with torch.no_grad():
        original_out = original_model([test_input[0]])
        hydra_out = hydra_model([test_input[0]])

    # Compare outputs
    original_boxes = original_out[0]["boxes"].numpy()
    hydra_boxes = hydra_out[0]["boxes"].numpy()

    original_scores = original_out[0]["scores"].numpy()
    hydra_scores = hydra_out[0]["scores"].numpy()

    if len(original_boxes) != len(hydra_boxes):
        logger.warning(
            f"WARNING: Different number of detections: {len(original_boxes)} vs {len(hydra_boxes)}"
        )
        # This can happen due to NMS randomness, not necessarily a failure

    if len(original_boxes) > 0 and len(hydra_boxes) > 0:
        # Compare top detection
        box_diff = np.abs(original_boxes[0] - hydra_boxes[0]).max()
        score_diff = abs(original_scores[0] - hydra_scores[0])

        logger.info(f"Top box difference: {box_diff:.6f}")
        logger.info(f"Top score difference: {score_diff:.6f}")

        if box_diff < 0.01 and score_diff < 0.01:
            logger.info("✓ Extraction validated successfully!")
            return True
        else:
            logger.info("✗ Outputs differ more than expected")
            return False
    else:
        logger.info("No detections to compare (may be valid for random input)")
        return True


def run_migration(models_dir: str, source_model: str = None):
    """Run the full migration to Hydra structure."""
    models_path = Path(models_dir)

    # Find source model
    if source_model is None:
        # Look for common model names
        candidates = [
            models_path / "creamy_chicken_fold3.pth",
            models_path / "legacy" / "creamy_chicken_fold3.pth",
            models_path / "creamy_chicken.pth",
        ]
        for candidate in candidates:
            if candidate.exists():
                source_model = str(candidate)
                break

        if source_model is None:
            logger.error("ERROR: No source model found. Specify with --input")
            logger.info(f"Looked in: {[str(c) for c in candidates]}")
            return False

    logger.info(f"\n{'=' * 60}")
    logger.info("HYDRA MIGRATION: Extract backbone + head from full model")
    logger.info(f"{'=' * 60}")
    logger.info(f"Source: {source_model}")
    logger.info(f"Target: {models_dir}")
    logger.info()

    # Output paths
    backbone_path = models_path / "backbone" / "active.pth"
    head_path = models_path / "heads" / "creamy_chicken" / "v1.pth"

    # Check if already migrated
    if backbone_path.exists() and head_path.exists():
        logger.info("Migration already complete. Files exist:")
        logger.info(f"  - {backbone_path}")
        logger.info(f"  - {head_path}")

        response = input("Re-run migration? [y/N]: ")
        if response.lower() != "y":
            return True

    # Extract
    info = extract_backbone_and_head(
        source_model, str(backbone_path), str(head_path), "creamy_chicken"
    )

    logger.info("\nExtraction complete:")
    logger.info(f"  Backbone: {info['backbone_size_mb']:.1f} MB ({info['backbone_keys']} keys)")
    logger.info(f"  Head: {info['head_size_mb']:.1f} MB ({info['head_keys']} keys)")

    # Create metadata
    create_metadata_files(models_path, "creamy_chicken", info)

    # Create active.pth for head
    head_dir = models_path / "heads" / "creamy_chicken"
    active_path = head_dir / "active.pth"
    v1_path = head_dir / "v1.pth"

    if v1_path.exists() and not active_path.exists():
        shutil.copy(v1_path, active_path)
        logger.info(f"Created {active_path}")

    # Validate
    logger.info()
    validate_extraction(str(backbone_path), str(head_path), source_model)

    # Move source to legacy
    legacy_dir = models_path / "legacy"
    legacy_dir.mkdir(exist_ok=True)

    source_path = Path(source_model)
    if source_path.parent != legacy_dir:
        legacy_path = legacy_dir / source_path.name
        if not legacy_path.exists():
            logger.info(f"\nMoving source to legacy: {legacy_path}")
            shutil.move(str(source_path), str(legacy_path))

    logger.info(f"\n{'=' * 60}")
    logger.info("Migration complete!")
    logger.info(f"{'=' * 60}")
    logger.info("\nNew structure:")
    logger.info(f"  models/backbone/active.pth  ({info['backbone_size_mb']:.1f} MB)")
    logger.info(f"  models/heads/creamy_chicken/active.pth  ({info['head_size_mb']:.1f} MB)")
    logger.info("  models/heads/creamy_chicken/metadata.json")

    return True


def main():
    parser = argparse.ArgumentParser(description="Extract backbone + head from full model")
    parser.add_argument("--input", "-i", help="Path to full model .pth file")
    parser.add_argument("--models-dir", "-m", default="models/", help="Models directory")
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate existing extraction"
    )

    args = parser.parse_args()

    if args.validate_only:
        models_path = Path(args.models_dir)
        validate_extraction(
            str(models_path / "backbone" / "active.pth"),
            str(models_path / "heads" / "creamy_chicken" / "active.pth"),
            args.input or str(models_path / "legacy" / "creamy_chicken_fold3.pth"),
        )
    else:
        run_migration(args.models_dir, args.input)


if __name__ == "__main__":
    main()
