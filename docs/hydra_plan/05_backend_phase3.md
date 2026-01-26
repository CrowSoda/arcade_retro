# Phase 3: Version Manager

## Overview

The VersionManager handles creating, promoting, and rolling back head versions. It implements auto-promotion logic based on F1 score improvements.

## File: `backend/hydra/version_manager.py`

```python
"""
Version Manager - Handle head versioning with auto-promotion and rollback.

Auto-promotion logic:
    - F1 >= 0.95: Promote if new >= old (no regression)
    - F1 < 0.95: Require 2% absolute improvement (new >= old + 0.02)

Version retention:
    - Keep last N versions (default 5)
    - Never delete active version
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import AUTO_PROMOTE_THRESHOLD, HIGH_F1_THRESHOLD, VERSION_RETENTION_COUNT


class VersionManager:
    """Manages model versions with auto-promotion and rollback."""
    
    def __init__(
        self, 
        models_dir: str,
        auto_promote_threshold: float = AUTO_PROMOTE_THRESHOLD
    ):
        self.models_dir = Path(models_dir)
        self.auto_promote_threshold = auto_promote_threshold
        self.heads_dir = self.models_dir / "heads"
        self.registry_path = self.models_dir / "registry.json"
    
    def _load_head_metadata(self, signal_name: str) -> dict:
        """Load metadata for a signal head."""
        path = self.heads_dir / signal_name / "metadata.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {"signal_name": signal_name, "versions": [], "active_version": None}
    
    def _save_head_metadata(self, signal_name: str, metadata: dict):
        """Save metadata for a signal head."""
        head_dir = self.heads_dir / signal_name
        head_dir.mkdir(parents=True, exist_ok=True)
        path = head_dir / "metadata.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _load_registry(self) -> dict:
        """Load the central registry."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {"backbone_version": None, "signals": {}, "last_updated": None}
    
    def _save_registry(self, registry: dict):
        """Save the central registry."""
        registry["last_updated"] = datetime.utcnow().isoformat() + "Z"
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)
    
    def create_version(
        self,
        signal_name: str,
        head_state_dict: dict,
        metrics: dict,
        sample_count: int,
        split_version: int,
        epochs_trained: int,
        early_stopped: bool,
        training_time_sec: float,
        learning_rate: float = 0.001,
        batch_size: int = 4,
        notes: str = None
    ) -> int:
        """
        Save new head version.
        
        Args:
            signal_name: Name of signal
            head_state_dict: Trained weights
            metrics: {f1_score, precision, recall, val_loss, train_loss}
            sample_count: Total samples used
            split_version: Which train/val split was used
            epochs_trained: Number of epochs
            early_stopped: Whether early stopping triggered
            training_time_sec: Training duration
            learning_rate: LR used
            batch_size: Batch size used
            notes: Optional user notes
        
        Returns:
            New version number
        """
        import torch
        
        # Load existing metadata
        metadata = self._load_head_metadata(signal_name)
        
        # Determine new version number
        if metadata["versions"]:
            new_version = max(v["version"] for v in metadata["versions"]) + 1
            parent_version = metadata.get("active_version")
        else:
            new_version = 1
            parent_version = None
        
        # Save weights
        head_dir = self.heads_dir / signal_name
        head_dir.mkdir(parents=True, exist_ok=True)
        weights_path = head_dir / f"v{new_version}.pth"
        torch.save(head_state_dict, weights_path)
        
        # Create version entry
        version_entry = {
            "version": new_version,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "sample_count": sample_count,
            "split_version": split_version,
            "epochs_trained": epochs_trained,
            "early_stopped": early_stopped,
            "metrics": {
                "f1_score": metrics.get("f1_score"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "val_loss": metrics.get("val_loss"),
                "train_loss": metrics.get("train_loss"),
            },
            "training_time_sec": training_time_sec,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "parent_version": parent_version,
            "is_active": False,
            "notes": notes,
        }
        
        metadata["versions"].append(version_entry)
        metadata["backbone_version"] = self._load_registry().get("backbone_version", 1)
        
        self._save_head_metadata(signal_name, metadata)
        
        print(f"[VersionManager] Created {signal_name} v{new_version}")
        return new_version
    
    def should_auto_promote(
        self, 
        signal_name: str, 
        new_metrics: dict
    ) -> Tuple[bool, str]:
        """
        Check if new version should be auto-promoted.
        
        Logic:
            - If new F1 >= 0.95: promote if no regression (new >= old)
            - If new F1 < 0.95: require 2% absolute improvement
        
        Returns:
            (should_promote, reason_string)
        """
        metadata = self._load_head_metadata(signal_name)
        
        new_f1 = new_metrics.get("f1_score", 0)
        
        # First version always promotes
        if not metadata["versions"] or metadata.get("active_version") is None:
            return True, "First version for signal"
        
        # Get current active version metrics
        active_version = metadata.get("active_version")
        active_entry = None
        for v in metadata["versions"]:
            if v["version"] == active_version:
                active_entry = v
                break
        
        if active_entry is None:
            return True, "No active version found"
        
        old_f1 = active_entry["metrics"].get("f1_score", 0)
        improvement = new_f1 - old_f1
        
        # High F1 threshold: just need no regression
        if new_f1 >= HIGH_F1_THRESHOLD:
            if new_f1 >= old_f1:
                return True, f"F1 {new_f1:.3f} >= {HIGH_F1_THRESHOLD} with no regression"
            else:
                return False, f"F1 regressed: {old_f1:.3f} -> {new_f1:.3f}"
        
        # Standard threshold: need 2% improvement
        if improvement >= self.auto_promote_threshold:
            pct = improvement * 100
            return True, f"F1 improved {pct:.1f}% ({old_f1:.3f} -> {new_f1:.3f})"
        
        # Not enough improvement
        pct = improvement * 100
        return False, f"F1 change {pct:+.1f}% below {self.auto_promote_threshold*100:.0f}% threshold"
    
    def promote_version(self, signal_name: str, version: int, reason: str = None) -> None:
        """Make specified version active."""
        metadata = self._load_head_metadata(signal_name)
        
        # Find and validate version exists
        version_entry = None
        for v in metadata["versions"]:
            if v["version"] == version:
                version_entry = v
                break
        
        if version_entry is None:
            raise ValueError(f"Version {version} not found for {signal_name}")
        
        # Update active flags
        for v in metadata["versions"]:
            v["is_active"] = (v["version"] == version)
        
        version_entry["promoted_at"] = datetime.utcnow().isoformat() + "Z"
        if reason:
            version_entry["promotion_reason"] = reason
        
        metadata["active_version"] = version
        self._save_head_metadata(signal_name, metadata)
        
        # Update active symlink/copy
        head_dir = self.heads_dir / signal_name
        active_path = head_dir / "active.pth"
        version_path = head_dir / f"v{version}.pth"
        
        if active_path.exists():
            active_path.unlink()
        
        try:
            active_path.symlink_to(f"v{version}.pth")
        except OSError:
            shutil.copy(version_path, active_path)
        
        # Update registry
        self.update_registry()
        
        print(f"[VersionManager] Promoted {signal_name} v{version}")
    
    def rollback(self, signal_name: str) -> int:
        """
        Rollback to previous version.
        
        Returns:
            Version number now active
        """
        metadata = self._load_head_metadata(signal_name)
        
        current_version = metadata.get("active_version")
        if current_version is None:
            raise ValueError(f"No active version for {signal_name}")
        
        # Find previous version
        versions = sorted([v["version"] for v in metadata["versions"]])
        current_idx = versions.index(current_version)
        
        if current_idx == 0:
            raise ValueError(f"Cannot rollback: {signal_name} is at first version")
        
        previous_version = versions[current_idx - 1]
        self.promote_version(signal_name, previous_version, reason="Manual rollback")
        
        return previous_version
    
    def get_version_history(self, signal_name: str) -> List[dict]:
        """Get all versions with metrics for a signal."""
        metadata = self._load_head_metadata(signal_name)
        return metadata.get("versions", [])
    
    def cleanup_old_versions(
        self, 
        signal_name: str, 
        keep_n: int = VERSION_RETENTION_COUNT
    ) -> List[int]:
        """
        Delete old versions, keeping last N.
        Never deletes active version.
        
        Returns:
            List of deleted version numbers
        """
        metadata = self._load_head_metadata(signal_name)
        versions = metadata.get("versions", [])
        active_version = metadata.get("active_version")
        
        if len(versions) <= keep_n:
            return []
        
        # Sort by version number
        sorted_versions = sorted(versions, key=lambda v: v["version"])
        
        # Keep last N and active
        to_keep = set(v["version"] for v in sorted_versions[-keep_n:])
        if active_version:
            to_keep.add(active_version)
        
        deleted = []
        head_dir = self.heads_dir / signal_name
        
        for v in sorted_versions:
            if v["version"] not in to_keep:
                # Delete weights file
                weights_path = head_dir / f"v{v['version']}.pth"
                if weights_path.exists():
                    weights_path.unlink()
                deleted.append(v["version"])
        
        # Update metadata
        metadata["versions"] = [v for v in versions if v["version"] in to_keep]
        self._save_head_metadata(signal_name, metadata)
        
        if deleted:
            print(f"[VersionManager] Cleaned up {signal_name}: deleted v{deleted}")
        
        return deleted
    
    def get_registry(self) -> dict:
        """Get full registry of all signals and versions."""
        return self._load_registry()
    
    def update_registry(self) -> None:
        """Rebuild registry.json from individual metadata files."""
        registry = self._load_registry()
        registry["signals"] = {}
        
        if not self.heads_dir.exists():
            self._save_registry(registry)
            return
        
        for signal_dir in self.heads_dir.iterdir():
            if not signal_dir.is_dir():
                continue
            
            signal_name = signal_dir.name
            metadata = self._load_head_metadata(signal_name)
            
            if not metadata.get("versions"):
                continue
            
            active_version = metadata.get("active_version")
            active_entry = None
            for v in metadata["versions"]:
                if v["version"] == active_version:
                    active_entry = v
                    break
            
            if active_entry:
                registry["signals"][signal_name] = {
                    "active_head_version": active_version,
                    "head_path": f"heads/{signal_name}/active.pth",
                    "sample_count": active_entry.get("sample_count", 0),
                    "f1_score": active_entry["metrics"].get("f1_score"),
                    "last_trained": active_entry.get("created_at"),
                    "is_loaded": False,
                }
        
        self._save_registry(registry)
    
    def get_available_signals(self) -> List[str]:
        """Get list of all signals with trained heads."""
        registry = self._load_registry()
        return list(registry.get("signals", {}).keys())
```

## File: `backend/hydra/config.py`

```python
"""
Hydra Configuration Constants
"""

# Version management
AUTO_PROMOTE_THRESHOLD = 0.02      # 2% F1 improvement required (below HIGH_F1)
HIGH_F1_THRESHOLD = 0.95           # Above this, just require no regression
VERSION_RETENTION_COUNT = 5        # Keep last N versions per signal
MIN_SAMPLES_FOR_TRAINING = 5       # Minimum labeled samples

# Training
DEFAULT_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 4
VAL_RATIO = 0.2

# Inference
DEFAULT_SCORE_THRESHOLD = 0.5
MAX_DETECTIONS_PER_HEAD = 100

# File paths (relative to g20_demo/)
BACKBONE_DIR = "models/backbone"
HEADS_DIR = "models/heads"
TRAINING_DATA_DIR = "training_data/signals"
REGISTRY_PATH = "models/registry.json"
```

---

## Usage Examples

### Create and Auto-Promote

```python
from backend.hydra.version_manager import VersionManager

vm = VersionManager("models/")

# After training completes...
version = vm.create_version(
    signal_name="creamy_chicken",
    head_state_dict=trained_weights,
    metrics={"f1_score": 0.94, "precision": 0.93, "recall": 0.95},
    sample_count=215,
    split_version=2,
    epochs_trained=18,
    early_stopped=True,
    training_time_sec=120.5,
    notes="Added urban environment samples"
)

# Check if should auto-promote
should_promote, reason = vm.should_auto_promote("creamy_chicken", {"f1_score": 0.94})
print(f"Auto-promote: {should_promote} - {reason}")

if should_promote:
    vm.promote_version("creamy_chicken", version, reason=reason)
```

### Manual Promotion

```python
# User manually promotes a specific version
vm.promote_version("creamy_chicken", version=3, reason="Manual selection by user")
```

### Rollback

```python
# User wants to go back to previous version
previous = vm.rollback("creamy_chicken")
print(f"Rolled back to v{previous}")
```

### View History

```python
history = vm.get_version_history("creamy_chicken")
for v in history:
    status = "ACTIVE" if v["is_active"] else ""
    print(f"  v{v['version']}: F1={v['metrics']['f1_score']:.3f} {status}")
```
