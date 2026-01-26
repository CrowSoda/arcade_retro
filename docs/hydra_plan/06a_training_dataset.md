# Training Dataset & Splits Implementation

## File: `backend/training/dataset.py`

```python
"""
SpectrogramDataset - Load pre-computed spectrograms for training.

Training data lives in: training_data/signals/{signal_name}/samples/
Each sample has:
  - XXXX.npz  (uint8 spectrogram, 1024x1024)
  - XXXX.json (bounding boxes, metadata)
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class SpectrogramDataset(Dataset):
    """
    Dataset for training detection heads.
    Loads pre-computed spectrograms and bounding boxes.
    """
    
    def __init__(
        self,
        samples_dir: str,
        sample_ids: List[str],
        transform: Callable = None
    ):
        """
        Args:
            samples_dir: Path to training_data/signals/{name}/samples/
            sample_ids: List of sample IDs to include (e.g., ["0001", "0002"])
            transform: Optional augmentation transforms
        """
        self.samples_dir = Path(samples_dir)
        self.sample_ids = sample_ids
        self.transform = transform
        
        # Validate samples exist
        valid_ids = []
        for sid in sample_ids:
            npz_path = self.samples_dir / f"{sid}.npz"
            json_path = self.samples_dir / f"{sid}.json"
            if npz_path.exists() and json_path.exists():
                valid_ids.append(sid)
            else:
                print(f"Warning: Sample {sid} missing files, skipping")
        
        self.sample_ids = valid_ids
        print(f"SpectrogramDataset: {len(self.sample_ids)} samples from {samples_dir}")
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Load single sample.
        
        Returns:
            (image_tensor, target_dict)
            
            image_tensor: [3, 1024, 1024] float32, normalized 0-1
            target_dict: {
                "boxes": Tensor[N, 4] in xyxy format (pixels),
                "labels": Tensor[N] (all 1s for signal class),
                "image_id": Tensor[1]
            }
        """
        sample_id = self.sample_ids[idx]
        
        # Load spectrogram
        npz_path = self.samples_dir / f"{sample_id}.npz"
        with np.load(npz_path) as data:
            spectrogram = data["spectrogram"]  # uint8, (1024, 1024)
        
        # Convert to float32 tensor, normalize to 0-1
        image = spectrogram.astype(np.float32) / 255.0
        
        # Expand to 3 channels (grayscale → RGB for ResNet)
        image = np.stack([image, image, image], axis=0)  # (3, 1024, 1024)
        image_tensor = torch.from_numpy(image)
        
        # Load boxes
        json_path = self.samples_dir / f"{sample_id}.json"
        with open(json_path) as f:
            metadata = json.load(f)
        
        boxes = []
        labels = []
        for box in metadata.get("boxes", []):
            # Boxes are stored in pixel coordinates (0-1023)
            boxes.append([
                box["x_min"],
                box["y_min"],
                box["x_max"],
                box["y_max"]
            ])
            labels.append(1)  # All are signal class
        
        if len(boxes) == 0:
            # No boxes - create dummy empty tensors
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }
        
        # Apply augmentations if provided
        if self.transform:
            image_tensor, target = self.transform(image_tensor, target)
        
        return image_tensor, target


def collate_fn(batch):
    """Custom collate for variable-length boxes."""
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets


def create_data_loaders(
    signal_name: str,
    training_data_dir: str,
    split_version: int = None,
    batch_size: int = 4,
    num_workers: int = 0  # 0 for Windows compatibility
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and val data loaders for a signal.
    
    Args:
        signal_name: Signal to load data for
        training_data_dir: Base training data directory
        split_version: Specific split version, or None for active
        batch_size: Training batch size
        num_workers: Data loading workers
    
    Returns:
        (train_loader, val_loader)
    """
    base_dir = Path(training_data_dir) / signal_name
    samples_dir = base_dir / "samples"
    splits_dir = base_dir / "splits"
    
    # Load split
    if split_version is not None:
        split_path = splits_dir / f"v{split_version}.json"
    else:
        split_path = splits_dir / "active.json"
        if not split_path.exists():
            # Find latest split
            split_files = list(splits_dir.glob("v*.json"))
            if not split_files:
                raise FileNotFoundError(f"No splits found for {signal_name}")
            split_path = max(split_files, key=lambda p: int(p.stem[1:]))
    
    with open(split_path) as f:
        split_data = json.load(f)
    
    train_ids = split_data["train_samples"]
    val_ids = split_data["val_samples"]
    
    print(f"Loading split: {len(train_ids)} train, {len(val_ids)} val")
    
    # Create datasets
    train_dataset = SpectrogramDataset(str(samples_dir), train_ids)
    val_dataset = SpectrogramDataset(str(samples_dir), val_ids)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader


# =============================================================================
# Augmentation transforms (optional, for future use)
# =============================================================================

class RandomHorizontalFlip:
    """Flip spectrogram and boxes horizontally."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: torch.Tensor, target: dict):
        if torch.rand(1) < self.p:
            image = torch.flip(image, dims=[2])  # Flip width
            
            if len(target["boxes"]) > 0:
                width = image.shape[2]
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]  # Flip x coords
                target["boxes"] = boxes
        
        return image, target


class RandomVerticalFlip:
    """Flip spectrogram and boxes vertically."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: torch.Tensor, target: dict):
        if torch.rand(1) < self.p:
            image = torch.flip(image, dims=[1])  # Flip height
            
            if len(target["boxes"]) > 0:
                height = image.shape[1]
                boxes = target["boxes"].clone()
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]  # Flip y coords
                target["boxes"] = boxes
        
        return image, target


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, image: torch.Tensor, target: dict):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
```

---

## File: `backend/training/splits.py`

```python
"""
SplitManager - Manage train/validation splits for training data.

Key principle: Validation set is FIXED once created.
New samples always go to training set.
This ensures consistent evaluation across versions.
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional


class SplitManager:
    """
    Manages train/validation splits for training data.
    Ensures validation set consistency across versions.
    """
    
    def __init__(self, training_data_dir: str):
        """Initialize with training data directory."""
        self.base_dir = Path(training_data_dir)
    
    def _get_signal_dir(self, signal_name: str) -> Path:
        return self.base_dir / signal_name
    
    def _get_splits_dir(self, signal_name: str) -> Path:
        return self._get_signal_dir(signal_name) / "splits"
    
    def _get_samples_dir(self, signal_name: str) -> Path:
        return self._get_signal_dir(signal_name) / "samples"
    
    def _list_sample_ids(self, signal_name: str) -> List[str]:
        """List all sample IDs for a signal."""
        samples_dir = self._get_samples_dir(signal_name)
        if not samples_dir.exists():
            return []
        
        # Find all .npz files
        sample_ids = []
        for npz_file in samples_dir.glob("*.npz"):
            sample_id = npz_file.stem
            json_file = samples_dir / f"{sample_id}.json"
            if json_file.exists():
                sample_ids.append(sample_id)
        
        return sorted(sample_ids)
    
    def create_initial_split(
        self,
        signal_name: str,
        val_ratio: float = 0.2,
        random_seed: int = 42
    ) -> int:
        """
        Create initial 80/20 split for new signal.
        
        Returns:
            Split version number (1)
        """
        sample_ids = self._list_sample_ids(signal_name)
        if not sample_ids:
            raise ValueError(f"No samples found for {signal_name}")
        
        # Shuffle with seed for reproducibility
        random.seed(random_seed)
        shuffled = sample_ids.copy()
        random.shuffle(shuffled)
        
        # Split
        val_count = max(1, int(len(shuffled) * val_ratio))
        val_samples = shuffled[:val_count]
        train_samples = shuffled[val_count:]
        
        # Create split data
        split_data = {
            "version": 1,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "total_samples": len(sample_ids),
            "train_samples": train_samples,
            "val_samples": val_samples,
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "val_ratio": len(val_samples) / len(sample_ids),
            "random_seed": random_seed,
            "notes": "Initial 80/20 split"
        }
        
        # Save
        splits_dir = self._get_splits_dir(signal_name)
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        split_path = splits_dir / "v1.json"
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)
        
        # Create active symlink
        active_path = splits_dir / "active.json"
        if active_path.exists():
            active_path.unlink()
        try:
            active_path.symlink_to("v1.json")
        except OSError:
            import shutil
            shutil.copy(split_path, active_path)
        
        print(f"Created split v1: {len(train_samples)} train, {len(val_samples)} val")
        return 1
    
    def extend_split(
        self,
        signal_name: str,
        new_sample_ids: List[str] = None
    ) -> int:
        """
        Extend split with new samples (added to train only).
        Validation set stays fixed.
        
        Args:
            signal_name: Signal name
            new_sample_ids: Specific new IDs, or None to auto-detect
        
        Returns:
            New split version number
        """
        # Load current split
        splits_dir = self._get_splits_dir(signal_name)
        active_path = splits_dir / "active.json"
        
        if not active_path.exists():
            # No split exists, create initial
            return self.create_initial_split(signal_name)
        
        with open(active_path) as f:
            current_split = json.load(f)
        
        current_version = current_split["version"]
        current_train = set(current_split["train_samples"])
        current_val = set(current_split["val_samples"])
        known_samples = current_train | current_val
        
        # Find new samples
        all_samples = set(self._list_sample_ids(signal_name))
        
        if new_sample_ids is not None:
            new_samples = [s for s in new_sample_ids if s not in known_samples]
        else:
            new_samples = list(all_samples - known_samples)
        
        if not new_samples:
            print("No new samples to add")
            return current_version
        
        # Add new samples to train (NOT val)
        new_train = list(current_train) + sorted(new_samples)
        new_val = list(current_val)  # Unchanged
        
        new_version = current_version + 1
        
        split_data = {
            "version": new_version,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "total_samples": len(new_train) + len(new_val),
            "train_samples": new_train,
            "val_samples": new_val,
            "train_count": len(new_train),
            "val_count": len(new_val),
            "val_ratio": len(new_val) / (len(new_train) + len(new_val)),
            "parent_version": current_version,
            "samples_added": len(new_samples),
            "notes": f"Extended from v{current_version}, added {len(new_samples)} samples"
        }
        
        # Save new version
        split_path = splits_dir / f"v{new_version}.json"
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)
        
        # Update active symlink
        active_path = splits_dir / "active.json"
        if active_path.exists():
            active_path.unlink()
        try:
            active_path.symlink_to(f"v{new_version}.json")
        except OSError:
            import shutil
            shutil.copy(split_path, active_path)
        
        print(f"Extended to v{new_version}: +{len(new_samples)} train samples")
        return new_version
    
    def get_split(
        self,
        signal_name: str,
        version: int = None
    ) -> Tuple[List[str], List[str]]:
        """
        Get train/val sample IDs for a split version.
        
        Args:
            signal_name: Signal name
            version: Split version, or None for active
        
        Returns:
            (train_sample_ids, val_sample_ids)
        """
        splits_dir = self._get_splits_dir(signal_name)
        
        if version is not None:
            split_path = splits_dir / f"v{version}.json"
        else:
            split_path = splits_dir / "active.json"
        
        if not split_path.exists():
            raise FileNotFoundError(f"Split not found: {split_path}")
        
        with open(split_path) as f:
            split_data = json.load(f)
        
        return split_data["train_samples"], split_data["val_samples"]
    
    def get_active_split_version(self, signal_name: str) -> int:
        """Get active split version number."""
        splits_dir = self._get_splits_dir(signal_name)
        active_path = splits_dir / "active.json"
        
        if not active_path.exists():
            return 0
        
        with open(active_path) as f:
            split_data = json.load(f)
        
        return split_data["version"]
    
    def get_pending_samples(self, signal_name: str) -> List[str]:
        """Get samples not yet in any split (newly added)."""
        splits_dir = self._get_splits_dir(signal_name)
        active_path = splits_dir / "active.json"
        
        all_samples = set(self._list_sample_ids(signal_name))
        
        if not active_path.exists():
            return list(all_samples)  # All are pending
        
        with open(active_path) as f:
            split_data = json.load(f)
        
        known = set(split_data["train_samples"]) | set(split_data["val_samples"])
        pending = all_samples - known
        
        return sorted(list(pending))
```
