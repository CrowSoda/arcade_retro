"""
Debug script for anchor mismatch investigation.
Run from g20_demo directory: python backend/debug_anchors.py
"""

import json
import os
import sys
from pathlib import Path

import torch

# Logging
from logger_config import get_logger

logger = get_logger("debug_anchors")


# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import box_iou

logger.info("=" * 60)
logger.info("ANCHOR MISMATCH DEBUG INVESTIGATION")
logger.info("=" * 60)

# =============================================================================
# 1. Model Configuration
# =============================================================================
logger.info("\n" + "=" * 60)
logger.info("1. MODEL CONFIGURATION")
logger.info("=" * 60)

# This is the exact code from service.py
backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=0)
model = FasterRCNN(backbone, num_classes=2)

# Print RPN anchor config
logger.info("\nRPN Anchor Generator Config:")
anchor_gen = model.rpn.anchor_generator
logger.info(f"  sizes: {anchor_gen.sizes}")
logger.info(f"  aspect_ratios: {anchor_gen.aspect_ratios}")

# Print RPN config
logger.info("\nRPN Config:")
try:
    logger.info(f"  rpn_nms_thresh: {model.rpn.nms_thresh}")
    logger.info(f"  rpn_score_thresh: {model.rpn.score_thresh}")
except Exception as e:
    logger.info(f"  (Could not read some RPN config: {e})")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"\nTotal params: {total_params:,}")
logger.info(f"Trainable params: {trainable_params:,}")

# =============================================================================
# 2. Box Statistics
# =============================================================================
logger.info("\n" + "=" * 60)
logger.info("2. BOX STATISTICS FROM SAMPLES")
logger.info("=" * 60)

samples_dir = Path("training_data/signals/Creamy_Pork/samples")
if not samples_dir.exists():
    logger.error(f"ERROR: {samples_dir} does not exist!")
else:
    widths = []
    heights = []
    aspects = []

    for f in os.listdir(samples_dir):
        if f.endswith(".json"):
            meta = json.load(open(samples_dir / f))
            for box in meta.get("boxes", []):
                w = box["x_max"] - box["x_min"]
                h = box["y_max"] - box["y_min"]
                if w > 0 and h > 0:
                    widths.append(w)
                    heights.append(h)
                    aspects.append(w / h)

    if widths:
        logger.info(f"Analyzed {len(widths)} boxes:")
        logger.info(
            f"Width:  min={min(widths)}, max={max(widths)}, mean={sum(widths) / len(widths):.1f}"
        )
        logger.info(
            f"Height: min={min(heights)}, max={max(heights)}, mean={sum(heights) / len(heights):.1f}"
        )
        logger.info(
            f"Aspect: min={min(aspects):.2f}, max={max(aspects):.2f}, mean={sum(aspects) / len(aspects):.2f}"
        )
    else:
        logger.info("No valid boxes found!")

# =============================================================================
# 3. Anchor Coverage Test
# =============================================================================
logger.info("\n" + "=" * 60)
logger.info("3. ANCHOR COVERAGE TEST")
logger.info("=" * 60)

# Generate anchors for 1024x1024 image
model.eval()
dummy_image = torch.zeros(1, 3, 1024, 1024)

# Get features and anchors
with torch.no_grad():
    features = model.backbone(dummy_image)

    # Create image list
    from torchvision.models.detection.image_list import ImageList

    image_list = ImageList(dummy_image, [(1024, 1024)])

    # Get anchors
    anchors = model.rpn.anchor_generator(image_list, list(features.values()))

# Analyze anchors
all_anchors = anchors[0]
anchor_widths = (all_anchors[:, 2] - all_anchors[:, 0]).numpy()
anchor_heights = (all_anchors[:, 3] - all_anchors[:, 1]).numpy()

logger.info(f"Total anchors generated: {len(anchor_widths)}")
logger.info(f"Anchor widths:  min={anchor_widths.min():.1f}, max={anchor_widths.max():.1f}")
logger.info(f"Anchor heights: min={anchor_heights.min():.1f}, max={anchor_heights.max():.1f}")

# Check if any anchors match small boxes (~11x70)
small_enough_w = anchor_widths < 30
small_enough_h = anchor_heights < 100
small_enough = small_enough_w & small_enough_h
logger.info(f"\nAnchors with width < 30: {small_enough_w.sum()}")
logger.info(f"Anchors with height < 100: {small_enough_h.sum()}")
logger.info(f"Anchors that could match ~11x70 boxes: {small_enough.sum()}")

# =============================================================================
# 4. IoU Check
# =============================================================================
logger.info("\n" + "=" * 60)
logger.info("4. IoU CHECK WITH ACTUAL BOX")
logger.info("=" * 60)

# One of your actual boxes (from sample 0172)
gt_box = torch.tensor([[305.0, 592.0, 316.0, 662.0]])
logger.info(f"Ground truth box: {gt_box[0].tolist()}")
logger.info(f"  width={gt_box[0, 2] - gt_box[0, 0]:.0f}, height={gt_box[0, 3] - gt_box[0, 1]:.0f}")

# Compute IoU with all anchors
ious = box_iou(gt_box, all_anchors)
best_iou = ious.max().item()
best_idx = ious.argmax().item()
best_anchor = all_anchors[best_idx]

logger.info(f"\nBest matching anchor: {best_anchor.tolist()}")
logger.info(
    f"  width={best_anchor[2] - best_anchor[0]:.0f}, height={best_anchor[3] - best_anchor[1]:.0f}"
)
logger.info(f"Best IoU: {best_iou:.3f}")

# Check threshold
logger.info("\nDefault RPN thresholds:")
logger.info("  fg_iou_thresh: 0.7 (anchor assigned as positive)")
logger.info("  bg_iou_thresh: 0.3 (anchor assigned as negative)")

if best_iou < 0.3:
    logger.info("\n🚨 BEST IoU < 0.3 - NO ANCHOR WILL BE ASSIGNED AS POSITIVE!")
    logger.info("   This is why F1=0. Anchors are too big for your boxes.")
elif best_iou < 0.7:
    logger.warning(f"\n⚠️  Best IoU {best_iou:.3f} is between 0.3 and 0.7")
    logger.info("   Some anchors may be ignored (neither positive nor negative)")
else:
    logger.info(f"\n✅ Best IoU {best_iou:.3f} >= 0.7 - anchors should match")

# =============================================================================
# 5. RPN Config Details
# =============================================================================
logger.info("\n" + "=" * 60)
logger.info("5. RPN CONFIG DETAILS")
logger.info("=" * 60)

# Access internal RPN parameters
rpn = model.rpn
logger.info("RPN head conv channels: 256 (default)")
logger.info(f"RPN num_anchors: {rpn.anchor_generator.num_anchors_per_location()}")

# Show anchor generator in detail
logger.info("\nAnchor Generator Details:")
for i, (size, ar) in enumerate(zip(anchor_gen.sizes, anchor_gen.aspect_ratios, strict=False)):
    logger.info(f"  Feature level {i}: sizes={size}, aspect_ratios={ar}")

# =============================================================================
# 6. Training Data Loader Check
# =============================================================================
logger.info("\n" + "=" * 60)
logger.info("6. TRAINING DATA LOADER CHECK")
logger.info("=" * 60)

try:
    from training.dataset import create_data_loaders

    train_loader, val_loader = create_data_loaders(
        "Creamy_Pork", "training_data/signals", batch_size=4
    )

    for images, targets in train_loader:
        logger.info(f"Batch size: {len(images)}")
        logger.info(f"Image shape: {images[0].shape}")
        logger.info(f"Target boxes: {targets[0]['boxes']}")
        logger.info(f"Target labels: {targets[0]['labels']}")
        logger.info(f"Target image_id: {targets[0]['image_id']}")
        break

except Exception as e:
    logger.error(f"Error loading data: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
logger.info("\n" + "=" * 60)
logger.info("SUMMARY")
logger.info("=" * 60)

if widths:
    avg_w = sum(widths) / len(widths)
    avg_h = sum(heights) / len(heights)
    logger.info(f"\nYour boxes: avg {avg_w:.0f}x{avg_h:.0f} pixels")
    logger.info(f"Smallest anchor: {anchor_widths.min():.0f}x{anchor_heights.min():.0f} pixels")
    logger.info(f"Best IoU achievable: {best_iou:.3f}")

    if best_iou < 0.3:
        logger.info("\n🚨 DIAGNOSIS: ANCHOR SIZE MISMATCH")
        logger.info("   Your boxes are too SMALL for default Faster R-CNN anchors.")
        logger.info("   Default anchors start at 32x32 pixels.")
        logger.info("   Your boxes are ~11x70 pixels.")
        logger.info("\n   FIX: Use custom smaller anchors in FasterRCNN:")
        print("   anchor_sizes = ((8, 16, 32, 64, 128),)")
        print("   aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),)")
