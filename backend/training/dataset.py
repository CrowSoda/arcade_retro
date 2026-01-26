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
