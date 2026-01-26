"""
Debug script for anchor mismatch investigation.
Run from g20_demo directory: python backend/debug_anchors.py
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.ops import box_iou

print("=" * 60)
print("ANCHOR MISMATCH DEBUG INVESTIGATION")
print("=" * 60)

# =============================================================================
# 1. Model Configuration
# =============================================================================
print("\n" + "=" * 60)
print("1. MODEL CONFIGURATION")
print("=" * 60)

# This is the exact code from service.py
backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=0)
model = FasterRCNN(backbone, num_classes=2)

# Print RPN anchor config
print("\nRPN Anchor Generator Config:")
anchor_gen = model.rpn.anchor_generator
print(f"  sizes: {anchor_gen.sizes}")
print(f"  aspect_ratios: {anchor_gen.aspect_ratios}")

# Print RPN config
print("\nRPN Config:")
try:
    print(f"  rpn_nms_thresh: {model.rpn.nms_thresh}")
    print(f"  rpn_score_thresh: {model.rpn.score_thresh}")
except Exception as e:
    print(f"  (Could not read some RPN config: {e})")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

# =============================================================================
# 2. Box Statistics
# =============================================================================
print("\n" + "=" * 60)
print("2. BOX STATISTICS FROM SAMPLES")
print("=" * 60)

samples_dir = Path("training_data/signals/Creamy_Pork/samples")
if not samples_dir.exists():
    print(f"ERROR: {samples_dir} does not exist!")
else:
    widths = []
    heights = []
    aspects = []
    
    for f in os.listdir(samples_dir):
        if f.endswith('.json'):
            meta = json.load(open(samples_dir / f))
            for box in meta.get('boxes', []):
                w = box['x_max'] - box['x_min']
                h = box['y_max'] - box['y_min']
                if w > 0 and h > 0:
                    widths.append(w)
                    heights.append(h)
                    aspects.append(w / h)
    
    if widths:
        print(f"Analyzed {len(widths)} boxes:")
        print(f"Width:  min={min(widths)}, max={max(widths)}, mean={sum(widths)/len(widths):.1f}")
        print(f"Height: min={min(heights)}, max={max(heights)}, mean={sum(heights)/len(heights):.1f}")
        print(f"Aspect: min={min(aspects):.2f}, max={max(aspects):.2f}, mean={sum(aspects)/len(aspects):.2f}")
    else:
        print("No valid boxes found!")

# =============================================================================
# 3. Anchor Coverage Test
# =============================================================================
print("\n" + "=" * 60)
print("3. ANCHOR COVERAGE TEST")
print("=" * 60)

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

print(f"Total anchors generated: {len(anchor_widths)}")
print(f"Anchor widths:  min={anchor_widths.min():.1f}, max={anchor_widths.max():.1f}")
print(f"Anchor heights: min={anchor_heights.min():.1f}, max={anchor_heights.max():.1f}")

# Check if any anchors match small boxes (~11x70)
small_enough_w = anchor_widths < 30
small_enough_h = anchor_heights < 100
small_enough = small_enough_w & small_enough_h
print(f"\nAnchors with width < 30: {small_enough_w.sum()}")
print(f"Anchors with height < 100: {small_enough_h.sum()}")
print(f"Anchors that could match ~11x70 boxes: {small_enough.sum()}")

# =============================================================================
# 4. IoU Check
# =============================================================================
print("\n" + "=" * 60)
print("4. IoU CHECK WITH ACTUAL BOX")
print("=" * 60)

# One of your actual boxes (from sample 0172)
gt_box = torch.tensor([[305.0, 592.0, 316.0, 662.0]])
print(f"Ground truth box: {gt_box[0].tolist()}")
print(f"  width={gt_box[0, 2]-gt_box[0, 0]:.0f}, height={gt_box[0, 3]-gt_box[0, 1]:.0f}")

# Compute IoU with all anchors
ious = box_iou(gt_box, all_anchors)
best_iou = ious.max().item()
best_idx = ious.argmax().item()
best_anchor = all_anchors[best_idx]

print(f"\nBest matching anchor: {best_anchor.tolist()}")
print(f"  width={best_anchor[2]-best_anchor[0]:.0f}, height={best_anchor[3]-best_anchor[1]:.0f}")
print(f"Best IoU: {best_iou:.3f}")

# Check threshold
print(f"\nDefault RPN thresholds:")
print(f"  fg_iou_thresh: 0.7 (anchor assigned as positive)")
print(f"  bg_iou_thresh: 0.3 (anchor assigned as negative)")

if best_iou < 0.3:
    print("\n🚨 BEST IoU < 0.3 - NO ANCHOR WILL BE ASSIGNED AS POSITIVE!")
    print("   This is why F1=0. Anchors are too big for your boxes.")
elif best_iou < 0.7:
    print(f"\n⚠️  Best IoU {best_iou:.3f} is between 0.3 and 0.7")
    print("   Some anchors may be ignored (neither positive nor negative)")
else:
    print(f"\n✅ Best IoU {best_iou:.3f} >= 0.7 - anchors should match")

# =============================================================================
# 5. RPN Config Details
# =============================================================================
print("\n" + "=" * 60)
print("5. RPN CONFIG DETAILS")
print("=" * 60)

# Access internal RPN parameters
rpn = model.rpn
print(f"RPN head conv channels: 256 (default)")
print(f"RPN num_anchors: {rpn.anchor_generator.num_anchors_per_location()}")

# Show anchor generator in detail
print("\nAnchor Generator Details:")
for i, (size, ar) in enumerate(zip(anchor_gen.sizes, anchor_gen.aspect_ratios)):
    print(f"  Feature level {i}: sizes={size}, aspect_ratios={ar}")

# =============================================================================
# 6. Training Data Loader Check
# =============================================================================
print("\n" + "=" * 60)
print("6. TRAINING DATA LOADER CHECK")
print("=" * 60)

try:
    from training.dataset import create_data_loaders
    
    train_loader, val_loader = create_data_loaders(
        "Creamy_Pork",
        "training_data/signals",
        batch_size=4
    )
    
    for images, targets in train_loader:
        print(f"Batch size: {len(images)}")
        print(f"Image shape: {images[0].shape}")
        print(f"Target boxes: {targets[0]['boxes']}")
        print(f"Target labels: {targets[0]['labels']}")
        print(f"Target image_id: {targets[0]['image_id']}")
        break
        
except Exception as e:
    print(f"Error loading data: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if widths:
    avg_w = sum(widths)/len(widths)
    avg_h = sum(heights)/len(heights)
    print(f"\nYour boxes: avg {avg_w:.0f}x{avg_h:.0f} pixels")
    print(f"Smallest anchor: {anchor_widths.min():.0f}x{anchor_heights.min():.0f} pixels")
    print(f"Best IoU achievable: {best_iou:.3f}")
    
    if best_iou < 0.3:
        print("\n🚨 DIAGNOSIS: ANCHOR SIZE MISMATCH")
        print("   Your boxes are too SMALL for default Faster R-CNN anchors.")
        print("   Default anchors start at 32x32 pixels.")
        print("   Your boxes are ~11x70 pixels.")
        print("\n   FIX: Use custom smaller anchors in FasterRCNN:")
        print("   anchor_sizes = ((8, 16, 32, 64, 128),)")
        print("   aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),)")
