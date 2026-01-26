# Phase 1: Backbone Extraction

## Overview

Extract the ResNet18-FPN backbone and detection head weights from `creamy_chicken_fold3.pth` into separate files. This is a one-time migration that enables the Hydra architecture.

## File: `backend/hydra/backbone_extractor.py`

### Complete Implementation

```python
"""
Backbone Extractor - Extract backbone and head from full FasterRCNN model.

One-time migration tool to split creamy_chicken_fold3.pth into:
  - models/backbone/v1.pth  (~55MB)
  - models/heads/creamy_chicken/v1.pth  (~10MB)

Usage:
    python -m backend.hydra.backbone_extractor \
        --input models/creamy_chicken_fold3.pth \
        --output-dir models/ \
        --validate
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN


# Weight prefixes for categorization
BACKBONE_PREFIXES = [
    "backbone.body.",      # ResNet layers
    "backbone.fpn.",       # FPN neck
]

HEAD_PREFIXES = [
    "rpn.",               # Region Proposal Network
    "roi_heads.",         # ROI detection heads
]


def hash_file(path: str) -> str:
    """Compute SHA256 hash of file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def categorize_weights(state_dict: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Split state_dict into backbone, head, and unknown categories.
    
    Returns:
        (backbone_dict, head_dict, unknown_dict)
    """
    backbone = {}
    head = {}
    unknown = {}
    
    for key, value in state_dict.items():
        is_backbone = any(key.startswith(p) for p in BACKBONE_PREFIXES)
        is_head = any(key.startswith(p) for p in HEAD_PREFIXES)
        
        if is_backbone:
            backbone[key] = value
        elif is_head:
            head[key] = value
        else:
            unknown[key] = value
    
    return backbone, head, unknown


def extract_backbone(
    full_model_path: str,
    output_dir: str,
    verbose: bool = True
) -> dict:
    """
    Extract backbone weights from full FasterRCNN model.
    
    Args:
        full_model_path: Path to creamy_chicken_fold3.pth
        output_dir: Base models directory
        verbose: Print progress info
    
    Returns:
        dict with extraction metadata
    """
    if verbose:
        print(f"Loading full model: {full_model_path}")
    
    # Load full state dict
    state_dict = torch.load(full_model_path, map_location="cpu", weights_only=False)
    
    # Categorize weights
    backbone_weights, head_weights, unknown_weights = categorize_weights(state_dict)
    
    if verbose:
        print(f"  Backbone keys: {len(backbone_weights)}")
        print(f"  Head keys: {len(head_weights)}")
        print(f"  Unknown keys: {len(unknown_weights)}")
        if unknown_weights:
            print(f"  WARNING: Unknown keys: {list(unknown_weights.keys())[:5]}...")
    
    # Create output directories
    backbone_dir = Path(output_dir) / "backbone"
    backbone_dir.mkdir(parents=True, exist_ok=True)
    
    # Save backbone
    backbone_path = backbone_dir / "v1.pth"
    torch.save(backbone_weights, backbone_path)
    
    # Calculate sizes
    backbone_size = os.path.getsize(backbone_path)
    full_size = os.path.getsize(full_model_path)
    
    if verbose:
        print(f"  Saved backbone: {backbone_path} ({backbone_size / 1e6:.1f} MB)")
    
    # Create metadata
    metadata = {
        "version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source": os.path.basename(full_model_path),
        "architecture": "resnet18_fpn",
        "input_size": [3, 1024, 1024],
        "feature_channels": 256,
        "original_model_hash": hash_file(full_model_path),
        "backbone_size_bytes": backbone_size,
        "key_count": len(backbone_weights),
        "extraction_script": "backend/hydra/backbone_extractor.py",
    }
    
    # Save metadata
    metadata_path = backbone_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"  Saved metadata: {metadata_path}")
    
    # Create active symlink (or copy on Windows)
    active_path = backbone_dir / "active.pth"
    if active_path.exists():
        active_path.unlink()
    
    try:
        active_path.symlink_to("v1.pth")
    except OSError:
        # Windows without admin - just copy
        import shutil
        shutil.copy(backbone_path, active_path)
    
    if verbose:
        print(f"  Created active link: {active_path}")
    
    return metadata


def extract_head(
    full_model_path: str,
    output_dir: str,
    signal_name: str = "creamy_chicken",
    verbose: bool = True
) -> dict:
    """
    Extract head weights from full FasterRCNN model.
    
    Args:
        full_model_path: Path to creamy_chicken_fold3.pth
        output_dir: Base models directory
        signal_name: Name for the head directory
        verbose: Print progress info
    
    Returns:
        dict with extraction metadata
    """
    if verbose:
        print(f"Extracting head for: {signal_name}")
    
    # Load full state dict
    state_dict = torch.load(full_model_path, map_location="cpu", weights_only=False)
    
    # Categorize weights
    _, head_weights, _ = categorize_weights(state_dict)
    
    # Create output directories
    head_dir = Path(output_dir) / "heads" / signal_name
    head_dir.mkdir(parents=True, exist_ok=True)
    
    # Save head
    head_path = head_dir / "v1.pth"
    torch.save(head_weights, head_path)
    
    # Calculate size
    head_size = os.path.getsize(head_path)
    
    if verbose:
        print(f"  Saved head: {head_path} ({head_size / 1e6:.1f} MB)")
    
    # Create metadata
    metadata = {
        "signal_name": signal_name,
        "backbone_version": 1,
        "active_version": 1,
        "version_retention": 5,
        "versions": [
            {
                "version": 1,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "sample_count": 127,  # Known from original training
                "split_version": 1,
                "epochs_trained": 0,  # Unknown for migrated model
                "early_stopped": False,
                "metrics": {
                    "f1_score": 0.91,  # Approximate from original
                    "precision": None,
                    "recall": None,
                    "val_loss": None,
                },
                "training_time_sec": 0,
                "is_active": True,
                "notes": "Migrated from creamy_chicken_fold3.pth",
            }
        ],
    }
    
    # Save metadata
    metadata_path = head_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"  Saved metadata: {metadata_path}")
    
    # Create active symlink
    active_path = head_dir / "active.pth"
    if active_path.exists():
        active_path.unlink()
    
    try:
        active_path.symlink_to("v1.pth")
    except OSError:
        import shutil
        shutil.copy(head_path, active_path)
    
    if verbose:
        print(f"  Created active link: {active_path}")
    
    return metadata


def build_model_from_parts(
    backbone_path: str,
    head_path: str,
    num_classes: int = 2,
    device: str = "cpu"
) -> FasterRCNN:
    """
    Reconstruct FasterRCNN from separate backbone and head files.
    
    Args:
        backbone_path: Path to backbone weights
        head_path: Path to head weights
        num_classes: Number of detection classes
        device: torch device
    
    Returns:
        Loaded FasterRCNN model
    """
    # Create model architecture
    backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=5)
    model = FasterRCNN(backbone, num_classes=num_classes)
    
    # Load weights
    backbone_weights = torch.load(backbone_path, map_location=device, weights_only=False)
    head_weights = torch.load(head_path, map_location=device, weights_only=False)
    
    # Merge and load
    combined = {**backbone_weights, **head_weights}
    model.load_state_dict(combined)
    
    return model


def validate_extraction(
    original_path: str,
    backbone_path: str,
    head_path: str,
    num_samples: int = 10,
    verbose: bool = True
) -> bool:
    """
    Verify that extracted backbone + head produce identical outputs.
    
    Args:
        original_path: Path to original full model
        backbone_path: Path to extracted backbone
        head_path: Path to extracted head
        num_samples: Number of random inputs to test
        verbose: Print progress
    
    Returns:
        True if outputs match, False otherwise
    """
    if verbose:
        print("Validating extraction...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load original model
    backbone_orig = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=5)
    model_orig = FasterRCNN(backbone_orig, num_classes=2)
    state_orig = torch.load(original_path, map_location=device, weights_only=False)
    model_orig.load_state_dict(state_orig)
    model_orig.to(device)
    model_orig.eval()
    
    # Load reconstructed model
    model_recon = build_model_from_parts(backbone_path, head_path, num_classes=2, device=device)
    model_recon.to(device)
    model_recon.eval()
    
    # Test with random inputs
    all_match = True
    
    with torch.inference_mode():
        for i in range(num_samples):
            # Random spectrogram-like input
            x = torch.randn(1, 3, 1024, 1024, device=device)
            
            # Get outputs
            out_orig = model_orig(x)
            out_recon = model_recon(x)
            
            # Compare boxes, scores, labels
            if len(out_orig) != len(out_recon):
                if verbose:
                    print(f"  Sample {i}: Output length mismatch")
                all_match = False
                continue
            
            for j, (o1, o2) in enumerate(zip(out_orig, out_recon)):
                boxes_match = torch.allclose(o1["boxes"], o2["boxes"], atol=1e-5)
                scores_match = torch.allclose(o1["scores"], o2["scores"], atol=1e-5)
                labels_match = torch.equal(o1["labels"], o2["labels"])
                
                if not (boxes_match and scores_match and labels_match):
                    if verbose:
                        print(f"  Sample {i}, output {j}: Mismatch detected")
                    all_match = False
    
    if verbose:
        if all_match:
            print("  ✓ All outputs match!")
        else:
            print("  ✗ Output mismatch detected")
    
    return all_match


def update_registry(models_dir: str, signal_name: str, metadata: dict):
    """Update the central registry with new signal info."""
    registry_path = Path(models_dir) / "registry.json"
    
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"backbone_version": None, "signals": {}}
    
    registry["backbone_version"] = 1
    registry["backbone_path"] = "backbone/active.pth"
    registry["last_updated"] = datetime.utcnow().isoformat() + "Z"
    
    if "versions" in metadata and metadata["versions"]:
        v = metadata["versions"][0]
        registry["signals"][signal_name] = {
            "active_head_version": v["version"],
            "head_path": f"heads/{signal_name}/active.pth",
            "sample_count": v.get("sample_count", 0),
            "f1_score": v["metrics"].get("f1_score"),
            "last_trained": v["created_at"],
            "is_loaded": False,
        }
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"Updated registry: {registry_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract backbone and head from FasterRCNN")
    parser.add_argument("--input", "-i", required=True, help="Path to full model .pth")
    parser.add_argument("--output-dir", "-o", default="models/", help="Output directory")
    parser.add_argument("--signal-name", "-s", default="creamy_chicken", help="Signal name for head")
    parser.add_argument("--validate", "-v", action="store_true", help="Validate extraction")
    parser.add_argument("--backup", "-b", action="store_true", help="Copy original to legacy/")
    args = parser.parse_args()
    
    # Backup original model
    if args.backup:
        legacy_dir = Path(args.output_dir) / "legacy"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        backup_path = legacy_dir / os.path.basename(args.input)
        if not backup_path.exists():
            shutil.copy(args.input, backup_path)
            print(f"Backed up original to: {backup_path}")
    
    # Extract backbone
    print("\n=== Extracting Backbone ===")
    backbone_meta = extract_backbone(args.input, args.output_dir)
    
    # Extract head
    print("\n=== Extracting Head ===")
    head_meta = extract_head(args.input, args.output_dir, args.signal_name)
    
    # Update registry
    print("\n=== Updating Registry ===")
    update_registry(args.output_dir, args.signal_name, head_meta)
    
    # Validate
    if args.validate:
        print("\n=== Validating ===")
        backbone_path = Path(args.output_dir) / "backbone" / "v1.pth"
        head_path = Path(args.output_dir) / "heads" / args.signal_name / "v1.pth"
        
        success = validate_extraction(args.input, str(backbone_path), str(head_path))
        
        if not success:
            print("\n⚠️  WARNING: Validation failed! Check extraction.")
            return 1
    
    print("\n✓ Extraction complete!")
    return 0


if __name__ == "__main__":
    exit(main())
```

---

## Execution Steps

### Step 1: Run the extraction

```bash
cd g20_demo
python -m backend.hydra.backbone_extractor \
    --input models/creamy_chicken_fold3.pth \
    --output-dir models/ \
    --signal-name creamy_chicken \
    --validate \
    --backup
```

### Step 2: Verify output structure

After extraction:
```
models/
├── backbone/
│   ├── v1.pth          # ~55MB
│   ├── active.pth      # symlink → v1.pth
│   └── metadata.json
├── heads/
│   └── creamy_chicken/
│       ├── v1.pth      # ~10MB
│       ├── active.pth  # symlink → v1.pth
│       └── metadata.json
├── legacy/
│   └── creamy_chicken_fold3.pth  # backup
└── registry.json
```

### Step 3: Verify sizes

```bash
# Check that extraction reduced size
ls -lh models/backbone/v1.pth       # Should be ~55MB
ls -lh models/heads/creamy_chicken/v1.pth  # Should be ~10MB
ls -lh models/legacy/creamy_chicken_fold3.pth  # Original ~100MB
```

---

## Validation Tests

### Test 1: Output Parity

The `--validate` flag runs 10 random inputs through both models and compares outputs. This ensures the split didn't lose any weights.

### Test 2: Manual Verification

```python
# Quick manual test
import torch
from backend.hydra.backbone_extractor import build_model_from_parts

model = build_model_from_parts(
    "models/backbone/v1.pth",
    "models/heads/creamy_chicken/v1.pth"
)
model.eval()

# Test inference
x = torch.randn(1, 3, 1024, 1024)
with torch.inference_mode():
    outputs = model(x)
    print(f"Detections: {len(outputs[0]['boxes'])}")
```

---

## Rollback Plan

If extraction fails or validation doesn't pass:

1. Delete extracted files:
   ```bash
   rm -rf models/backbone models/heads models/registry.json
   ```

2. Restore from backup:
   ```bash
   cp models/legacy/creamy_chicken_fold3.pth models/
   ```

3. Continue using original single-model inference
