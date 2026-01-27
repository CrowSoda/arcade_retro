"""
SplitManager - Manage train/validation splits for training data.

Key principle: Validation set is FIXED once created.
New samples always go to training set.
This ensures consistent evaluation across versions.
"""

import json
import random
import shutil
from datetime import datetime
from pathlib import Path

# Logging
from logger_config import get_logger

logger = get_logger("splits")


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

    def _list_sample_ids(self, signal_name: str) -> list[str]:
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
        self, signal_name: str, val_ratio: float = 0.2, random_seed: int = 42
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
            "notes": "Initial 80/20 split",
        }

        # Save
        splits_dir = self._get_splits_dir(signal_name)
        splits_dir.mkdir(parents=True, exist_ok=True)

        split_path = splits_dir / "v1.json"
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)

        # Create active symlink/copy
        self._create_active_link(splits_dir, "v1.json")

        logger.info(f"Created split v1: {len(train_samples)} train, {len(val_samples)} val")
        return 1

    def extend_split(self, signal_name: str, new_sample_ids: list[str] = None) -> int:
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
            logger.info("No new samples to add")
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
            "notes": f"Extended from v{current_version}, added {len(new_samples)} samples",
        }

        # Save new version
        split_path = splits_dir / f"v{new_version}.json"
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)

        # Update active link
        self._create_active_link(splits_dir, f"v{new_version}.json")

        logger.info(f"Extended to v{new_version}: +{len(new_samples)} train samples")
        return new_version

    def _create_active_link(self, splits_dir: Path, version_filename: str):
        """Create active.json symlink or copy."""
        active_path = splits_dir / "active.json"
        version_path = splits_dir / version_filename

        if active_path.exists():
            active_path.unlink()

        try:
            active_path.symlink_to(version_filename)
        except OSError:
            shutil.copy(version_path, active_path)

    def get_split(self, signal_name: str, version: int = None) -> tuple[list[str], list[str]]:
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

    def get_pending_samples(self, signal_name: str) -> list[str]:
        """Get samples not yet in any split (newly added)."""
        splits_dir = self._get_splits_dir(signal_name)
        active_path = splits_dir / "active.json"

        all_samples = set(self._list_sample_ids(signal_name))

        if not active_path.exists():
            return sorted(list(all_samples))  # All are pending

        with open(active_path) as f:
            split_data = json.load(f)

        known = set(split_data["train_samples"]) | set(split_data["val_samples"])
        pending = all_samples - known

        return sorted(list(pending))

    def get_split_summary(self, signal_name: str) -> dict:
        """Get summary of current split state."""
        splits_dir = self._get_splits_dir(signal_name)
        active_path = splits_dir / "active.json"

        all_samples = self._list_sample_ids(signal_name)
        pending = self.get_pending_samples(signal_name)

        if not active_path.exists():
            return {
                "total_samples": len(all_samples),
                "pending_samples": len(pending),
                "train_count": 0,
                "val_count": 0,
                "active_version": None,
            }

        with open(active_path) as f:
            split_data = json.load(f)

        return {
            "total_samples": len(all_samples),
            "pending_samples": len(pending),
            "train_count": split_data["train_count"],
            "val_count": split_data["val_count"],
            "active_version": split_data["version"],
        }
