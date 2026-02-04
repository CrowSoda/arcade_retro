"""
Training Diagnostics - Run from g20_demo/backend:
    python diagnose_training.py Creamy_Shrimp
"""

import sys
from pathlib import Path

import torch
from training.dataset import create_data_loaders
from training.service import TrainingService

# Get signal name from command line
signal_name = sys.argv[1] if len(sys.argv) > 1 else "Creamy_Shrimp"
# Path relative to g20_demo (parent of backend)
training_data_dir = str(Path(__file__).parent.parent / "training_data" / "signals")

print("=" * 60)
print(f"DIAGNOSTICS FOR: {signal_name}")
print("=" * 60)

# ============================================================
# 1. CURRENT TRAIN/VAL SPLIT
# ============================================================
print("\n=== 1. TRAIN/VAL SPLIT ===")

train_loader, val_loader = create_data_loaders(signal_name, training_data_dir, batch_size=4)

train_images = len(train_loader.dataset)
val_images = len(val_loader.dataset)

train_boxes = 0
for images, targets in train_loader:
    for t in targets:
        train_boxes += len(t["boxes"])

val_boxes = 0
for images, targets in val_loader:
    for t in targets:
        val_boxes += len(t["boxes"])

print(f"Train images: {train_images}")
print(f"Train boxes: {train_boxes}")
print(f"Val images: {val_images}")
print(f"Val boxes: {val_boxes}")

# ============================================================
# 4. CONFIRM DYNAMIC ANCHORS LOADED (do this first so model is built)
# ============================================================
print("\n=== 4. DYNAMIC ANCHORS ===")

# Path to models (relative to g20_demo)
models_dir = str(Path(__file__).parent.parent / "models")

service = TrainingService(
    models_dir=models_dir,
    training_data_dir=training_data_dir,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Build model and check anchors
model = service._build_model(signal_name=signal_name)
anchor_gen = model.rpn.anchor_generator
print(f"Anchor sizes: {anchor_gen.sizes}")
print(f"Anchor aspects: {anchor_gen.aspect_ratios}")

# ============================================================
# 2. TRAINING LOSS (run 30 epochs) with confidence tracking
# ============================================================
print("\n=== 2. TRAINING LOOP (30 epochs) ===")
print("Tracking: loss, predictions at threshold=0.001, max confidence")

model.train()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.001)

device = service.device
model.to(device)

# Lower score threshold for debugging
original_thresh = model.roi_heads.score_thresh
model.roi_heads.score_thresh = 0.001  # See ALL predictions

for epoch in range(30):
    # TRAIN
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
        batch_count += 1

    avg_loss = epoch_loss / batch_count

    # EVAL with low threshold
    model.eval()
    all_scores = []
    total_preds = 0
    total_gt = 0
    with torch.inference_mode():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets, strict=False):
                scores = out["scores"].cpu().tolist()
                all_scores.extend(scores)
                total_preds += len(out["boxes"])
                total_gt += len(tgt["boxes"])

    max_conf = max(all_scores) if all_scores else 0.0
    preds_over_05 = sum(1 for s in all_scores if s > 0.05)
    preds_over_50 = sum(1 for s in all_scores if s > 0.50)

    print(
        f"Epoch {epoch + 1:2d}: loss={avg_loss:.4f}, preds={total_preds:3d}, max_conf={max_conf:.3f}, >0.05={preds_over_05}, >0.50={preds_over_50}"
    )

# Restore threshold
model.roi_heads.score_thresh = original_thresh

# ============================================================
# 3. FINAL PREDICTION DETAILS
# ============================================================
print("\n=== 3. FINAL PREDICTION DETAILS (threshold=0.001) ===")

model.eval()
model.roi_heads.score_thresh = 0.001
with torch.inference_mode():
    for batch_idx, (images, targets) in enumerate(val_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for img_idx, (out, tgt) in enumerate(zip(outputs, targets, strict=False)):
            gt_boxes = len(tgt["boxes"])
            pred_boxes = len(out["boxes"])
            scores = out["scores"].cpu().tolist()

            print(f"\nVal image {batch_idx * 4 + img_idx}:")
            print(f"  GT boxes: {gt_boxes}")
            print(f"  Pred boxes: {pred_boxes}")
            if scores:
                print(f"  Top 10 scores: {[f'{s:.3f}' for s in scores[:10]]}")
                print(f"  Scores > 0.5: {sum(1 for s in scores if s > 0.5)}")
                print(f"  Scores > 0.3: {sum(1 for s in scores if s > 0.3)}")
                print(f"  Scores > 0.05: {sum(1 for s in scores if s > 0.05)}")
            else:
                print("  NO PREDICTIONS (even at 0.001 threshold)")

        if batch_idx >= 0:  # Only show first batch
            break

print("\n" + "=" * 60)
print("DIAGNOSTICS COMPLETE")
print("=" * 60)
