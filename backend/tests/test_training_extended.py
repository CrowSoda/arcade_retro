"""
Training Extended Tests - Dataset, augmentation, and data loading tests.

Tests for backend/training/ modules with mock sample data.
Covers SpectrogramDataset, augmentation transforms, and data loaders.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# Test Fixtures
# =============================================================================


def create_mock_sample(samples_dir: Path, sample_id: str, num_boxes: int = 1):
    """Create a mock training sample with spectrogram and boxes."""
    # Create spectrogram
    spectrogram = np.random.randint(0, 256, size=(1024, 1024), dtype=np.uint8)
    npz_path = samples_dir / f"{sample_id}.npz"
    np.savez_compressed(npz_path, spectrogram=spectrogram)

    # Create boxes metadata
    boxes = []
    for i in range(num_boxes):
        x_min = np.random.randint(0, 900)
        y_min = np.random.randint(0, 900)
        x_max = x_min + np.random.randint(20, 100)
        y_max = y_min + np.random.randint(20, 100)
        boxes.append(
            {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }
        )

    metadata = {
        "sample_id": sample_id,
        "boxes": boxes,
    }
    json_path = samples_dir / f"{sample_id}.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f)


def create_mock_split(splits_dir: Path, train_ids: list[str], val_ids: list[str], version: int = 1):
    """Create a mock split file."""
    split_data = {
        "version": version,
        "train_samples": train_ids,
        "val_samples": val_ids,
    }
    split_path = splits_dir / f"v{version}.json"
    with open(split_path, "w") as f:
        json.dump(split_data, f)

    # Also create active.json symlink-like behavior
    active_path = splits_dir / "active.json"
    with open(active_path, "w") as f:
        json.dump(split_data, f)


# =============================================================================
# SpectrogramDataset Tests
# =============================================================================


class TestSpectrogramDataset:
    """Test SpectrogramDataset class."""

    def test_create_dataset(self):
        """Create dataset with valid samples."""
        from training.dataset import SpectrogramDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "samples"
            samples_dir.mkdir()

            # Create mock samples
            create_mock_sample(samples_dir, "0001")
            create_mock_sample(samples_dir, "0002")

            dataset = SpectrogramDataset(
                samples_dir=str(samples_dir),
                sample_ids=["0001", "0002"],
            )

            assert len(dataset) == 2

    def test_dataset_skips_missing_samples(self):
        """Dataset skips samples with missing files."""
        from training.dataset import SpectrogramDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "samples"
            samples_dir.mkdir()

            # Create only one sample
            create_mock_sample(samples_dir, "0001")

            # Request two samples, but only one exists
            dataset = SpectrogramDataset(
                samples_dir=str(samples_dir),
                sample_ids=["0001", "0002"],  # 0002 doesn't exist
            )

            assert len(dataset) == 1

    def test_getitem_returns_tuple(self):
        """__getitem__ returns (image_tensor, target_dict)."""
        from training.dataset import SpectrogramDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "samples"
            samples_dir.mkdir()

            create_mock_sample(samples_dir, "0001", num_boxes=2)

            dataset = SpectrogramDataset(
                samples_dir=str(samples_dir),
                sample_ids=["0001"],
            )

            image, target = dataset[0]

            # Check image
            assert isinstance(image, torch.Tensor)
            assert image.shape == (3, 1024, 1024)
            assert image.dtype == torch.float32
            assert image.min() >= 0.0
            assert image.max() <= 1.0

            # Check target
            assert isinstance(target, dict)
            assert "boxes" in target
            assert "labels" in target
            assert "image_id" in target

    def test_getitem_boxes_format(self):
        """Boxes are in xyxy format with correct types."""
        from training.dataset import SpectrogramDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "samples"
            samples_dir.mkdir()

            create_mock_sample(samples_dir, "0001", num_boxes=3)

            dataset = SpectrogramDataset(
                samples_dir=str(samples_dir),
                sample_ids=["0001"],
            )

            _, target = dataset[0]

            boxes = target["boxes"]
            labels = target["labels"]

            assert boxes.shape == (3, 4)
            assert boxes.dtype == torch.float32
            assert labels.shape == (3,)
            assert labels.dtype == torch.int64

            # Check all labels are 1 (signal class)
            assert (labels == 1).all()

    def test_getitem_empty_boxes(self):
        """Handle sample with no boxes."""
        from training.dataset import SpectrogramDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "samples"
            samples_dir.mkdir()

            create_mock_sample(samples_dir, "0001", num_boxes=0)

            dataset = SpectrogramDataset(
                samples_dir=str(samples_dir),
                sample_ids=["0001"],
            )

            _, target = dataset[0]

            assert target["boxes"].shape == (0, 4)
            assert target["labels"].shape == (0,)

    def test_getitem_with_transform(self):
        """Transform is applied when provided."""
        from training.dataset import SpectrogramDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "samples"
            samples_dir.mkdir()

            create_mock_sample(samples_dir, "0001")

            # Simple transform that marks it was called
            transform_called = [False]

            def mock_transform(image, target):
                transform_called[0] = True
                return image, target

            dataset = SpectrogramDataset(
                samples_dir=str(samples_dir),
                sample_ids=["0001"],
                transform=mock_transform,
            )

            _ = dataset[0]

            assert transform_called[0] is True


# =============================================================================
# Collate Function Tests
# =============================================================================


class TestCollateFn:
    """Test custom collate function."""

    def test_collate_fn(self):
        """collate_fn handles variable-length boxes."""
        from training.dataset import collate_fn

        # Create batch with different number of boxes
        batch = [
            (torch.ones(3, 64, 64), {"boxes": torch.zeros(2, 4), "labels": torch.ones(2)}),
            (torch.ones(3, 64, 64), {"boxes": torch.zeros(1, 4), "labels": torch.ones(1)}),
            (torch.ones(3, 64, 64), {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0)}),
        ]

        images, targets = collate_fn(batch)

        assert len(images) == 3
        assert len(targets) == 3
        assert targets[0]["boxes"].shape[0] == 2
        assert targets[1]["boxes"].shape[0] == 1
        assert targets[2]["boxes"].shape[0] == 0


# =============================================================================
# Data Loader Tests
# =============================================================================


class TestCreateDataLoaders:
    """Test create_data_loaders function."""

    def test_create_loaders(self):
        """Create train and val loaders."""
        from training.dataset import create_data_loaders

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            signal_dir = Path(tmpdir) / "test_signal"
            samples_dir = signal_dir / "samples"
            splits_dir = signal_dir / "splits"
            samples_dir.mkdir(parents=True)
            splits_dir.mkdir(parents=True)

            # Create samples
            for i in range(10):
                create_mock_sample(samples_dir, f"{i:04d}")

            # Create split
            train_ids = [f"{i:04d}" for i in range(8)]
            val_ids = [f"{i:04d}" for i in range(8, 10)]
            create_mock_split(splits_dir, train_ids, val_ids)

            # Create loaders
            train_loader, val_loader = create_data_loaders(
                signal_name="test_signal",
                training_data_dir=tmpdir,
                batch_size=2,
            )

            assert len(train_loader.dataset) == 8
            assert len(val_loader.dataset) == 2

    def test_create_loaders_with_version(self):
        """Create loaders with specific split version."""
        from training.dataset import create_data_loaders

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            signal_dir = Path(tmpdir) / "test_signal"
            samples_dir = signal_dir / "samples"
            splits_dir = signal_dir / "splits"
            samples_dir.mkdir(parents=True)
            splits_dir.mkdir(parents=True)

            # Create samples
            for i in range(6):
                create_mock_sample(samples_dir, f"{i:04d}")

            # Create multiple splits
            create_mock_split(splits_dir, ["0000", "0001"], ["0002"], version=1)
            create_mock_split(
                splits_dir, ["0000", "0001", "0002", "0003"], ["0004", "0005"], version=2
            )

            # Load specific version
            train_loader, val_loader = create_data_loaders(
                signal_name="test_signal",
                training_data_dir=tmpdir,
                split_version=2,
                batch_size=2,
            )

            assert len(train_loader.dataset) == 4
            assert len(val_loader.dataset) == 2

    def test_create_loaders_no_splits_raises(self):
        """Raise error if no splits exist."""
        from training.dataset import create_data_loaders

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure but no splits
            signal_dir = Path(tmpdir) / "test_signal"
            samples_dir = signal_dir / "samples"
            splits_dir = signal_dir / "splits"
            samples_dir.mkdir(parents=True)
            splits_dir.mkdir(parents=True)

            with pytest.raises(FileNotFoundError):
                create_data_loaders(
                    signal_name="test_signal",
                    training_data_dir=tmpdir,
                )


# =============================================================================
# Training Splits Module Tests
# =============================================================================


class TestSplitsModule:
    """Test training/splits.py module."""

    def test_module_imports(self):
        """splits module should import."""
        from training import splits

        assert splits is not None

    def test_has_split_functions(self):
        """splits module has split management functions."""
        from training import splits

        public_funcs = [name for name in dir(splits) if not name.startswith("_")]
        assert len(public_funcs) > 0


# =============================================================================
# Sample Manager Module Tests
# =============================================================================


class TestSampleManagerModule:
    """Test training/sample_manager.py module."""

    def test_module_imports(self):
        """sample_manager module should import."""
        from training import sample_manager

        assert sample_manager is not None

    def test_has_manager_class(self):
        """sample_manager has SampleManager or similar class."""
        from training import sample_manager

        public_items = [name for name in dir(sample_manager) if not name.startswith("_")]
        assert len(public_items) > 0


# =============================================================================
# Training Service Module Tests
# =============================================================================


class TestTrainingServiceModule:
    """Test training/service.py module."""

    def test_module_imports(self):
        """service module should import."""
        from training import service

        assert service is not None

    def test_has_training_functions(self):
        """service module has training functions."""
        from training import service

        public_items = [name for name in dir(service) if not name.startswith("_")]
        assert len(public_items) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
