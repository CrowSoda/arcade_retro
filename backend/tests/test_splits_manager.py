"""
SplitManager Tests - Train/validation split management.

Tests for backend/training/splits.py module.
Covers SplitManager class methods and split file handling.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


def create_mock_sample(samples_dir: Path, sample_id: str):
    """Create a mock training sample with spectrogram and boxes."""
    spectrogram = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
    np.savez_compressed(samples_dir / f"{sample_id}.npz", spectrogram=spectrogram)
    with open(samples_dir / f"{sample_id}.json", "w") as f:
        json.dump({"sample_id": sample_id, "boxes": []}, f)


# =============================================================================
# SplitManager Initialization Tests
# =============================================================================


class TestSplitManagerInit:
    """Test SplitManager initialization."""

    def test_create_manager(self):
        """Create split manager with base directory."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SplitManager(tmpdir)

            assert manager.base_dir == Path(tmpdir)

    def test_get_signal_dir(self):
        """_get_signal_dir returns correct path."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SplitManager(tmpdir)
            signal_dir = manager._get_signal_dir("test_signal")

            assert signal_dir == Path(tmpdir) / "test_signal"

    def test_get_splits_dir(self):
        """_get_splits_dir returns correct path."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SplitManager(tmpdir)
            splits_dir = manager._get_splits_dir("test_signal")

            assert splits_dir == Path(tmpdir) / "test_signal" / "splits"

    def test_get_samples_dir(self):
        """_get_samples_dir returns correct path."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SplitManager(tmpdir)
            samples_dir = manager._get_samples_dir("test_signal")

            assert samples_dir == Path(tmpdir) / "test_signal" / "samples"


# =============================================================================
# Sample Listing Tests
# =============================================================================


class TestListSampleIds:
    """Test _list_sample_ids method."""

    def test_list_empty_dir(self):
        """Returns empty list for empty/missing samples dir."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SplitManager(tmpdir)
            sample_ids = manager._list_sample_ids("test_signal")

            assert sample_ids == []

    def test_list_samples_with_data(self):
        """Returns sorted list of sample IDs."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal" / "samples"
            samples_dir.mkdir(parents=True)

            create_mock_sample(samples_dir, "0003")
            create_mock_sample(samples_dir, "0001")
            create_mock_sample(samples_dir, "0002")

            manager = SplitManager(tmpdir)
            sample_ids = manager._list_sample_ids("test_signal")

            assert sample_ids == ["0001", "0002", "0003"]

    def test_list_requires_both_files(self):
        """Sample requires both .npz and .json files."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal" / "samples"
            samples_dir.mkdir(parents=True)

            # Only npz file
            np.savez_compressed(samples_dir / "0001.npz", spectrogram=np.zeros((64, 64)))

            # Only json file
            with open(samples_dir / "0002.json", "w") as f:
                json.dump({}, f)

            # Both files
            create_mock_sample(samples_dir, "0003")

            manager = SplitManager(tmpdir)
            sample_ids = manager._list_sample_ids("test_signal")

            assert sample_ids == ["0003"]


# =============================================================================
# Initial Split Creation Tests
# =============================================================================


class TestCreateInitialSplit:
    """Test create_initial_split method."""

    def test_create_initial_split(self):
        """Create 80/20 split for new signal."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal" / "samples"
            samples_dir.mkdir(parents=True)

            for i in range(10):
                create_mock_sample(samples_dir, f"{i:04d}")

            manager = SplitManager(tmpdir)
            version = manager.create_initial_split("test_signal")

            assert version == 1

            # Check split file was created
            split_path = Path(tmpdir) / "test_signal" / "splits" / "v1.json"
            assert split_path.exists()

            with open(split_path) as f:
                split_data = json.load(f)

            assert split_data["version"] == 1
            assert split_data["train_count"] + split_data["val_count"] == 10
            assert len(split_data["train_samples"]) == 8  # 80%
            assert len(split_data["val_samples"]) == 2  # 20%

    def test_create_split_no_samples_raises(self):
        """Raises ValueError if no samples found."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SplitManager(tmpdir)

            with pytest.raises(ValueError, match="No samples found"):
                manager.create_initial_split("empty_signal")

    def test_create_split_creates_active_link(self):
        """Creates active.json link/copy."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal" / "samples"
            samples_dir.mkdir(parents=True)

            for i in range(5):
                create_mock_sample(samples_dir, f"{i:04d}")

            manager = SplitManager(tmpdir)
            manager.create_initial_split("test_signal")

            active_path = Path(tmpdir) / "test_signal" / "splits" / "active.json"
            assert active_path.exists()

    def test_create_split_reproducible(self):
        """Same seed produces same split."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal" / "samples"
            samples_dir.mkdir(parents=True)

            for i in range(10):
                create_mock_sample(samples_dir, f"{i:04d}")

            manager = SplitManager(tmpdir)
            manager.create_initial_split("test_signal", random_seed=42)

            with open(Path(tmpdir) / "test_signal" / "splits" / "v1.json") as f:
                split1 = json.load(f)

        # Create again with same seed
        with tempfile.TemporaryDirectory() as tmpdir2:
            samples_dir2 = Path(tmpdir2) / "test_signal" / "samples"
            samples_dir2.mkdir(parents=True)

            for i in range(10):
                create_mock_sample(samples_dir2, f"{i:04d}")

            manager2 = SplitManager(tmpdir2)
            manager2.create_initial_split("test_signal", random_seed=42)

            with open(Path(tmpdir2) / "test_signal" / "splits" / "v1.json") as f:
                split2 = json.load(f)

        assert split1["train_samples"] == split2["train_samples"]
        assert split1["val_samples"] == split2["val_samples"]


# =============================================================================
# Extend Split Tests
# =============================================================================


class TestExtendSplit:
    """Test extend_split method."""

    def test_extend_creates_initial_if_missing(self):
        """Creates initial split if none exists."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal" / "samples"
            samples_dir.mkdir(parents=True)

            for i in range(5):
                create_mock_sample(samples_dir, f"{i:04d}")

            manager = SplitManager(tmpdir)
            version = manager.extend_split("test_signal")

            # Should create v1
            assert version == 1

    def test_extend_adds_to_train(self):
        """New samples go to training set only."""
        import random

        from training.splits import SplitManager

        # Isolate random seed for this test
        random.seed(12345)

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal_extend" / "samples"
            samples_dir.mkdir(parents=True)

            # Initial samples
            for i in range(10):
                create_mock_sample(samples_dir, f"{i:04d}")

            manager = SplitManager(tmpdir)
            manager.create_initial_split("test_signal_extend", random_seed=12345)

            # Get initial val set
            _, initial_val = manager.get_split("test_signal_extend")
            initial_val_set = set(initial_val)

            # Add more samples
            for i in range(10, 15):
                create_mock_sample(samples_dir, f"{i:04d}")

            version = manager.extend_split("test_signal_extend")

            assert version == 2

            # Val set should be unchanged (same samples)
            _, new_val = manager.get_split("test_signal_extend")
            assert set(new_val) == initial_val_set

    def test_extend_no_new_samples(self):
        """Returns current version if no new samples."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal" / "samples"
            samples_dir.mkdir(parents=True)

            for i in range(5):
                create_mock_sample(samples_dir, f"{i:04d}")

            manager = SplitManager(tmpdir)
            manager.create_initial_split("test_signal")

            # Try to extend without adding samples
            version = manager.extend_split("test_signal")

            assert version == 1  # No change


# =============================================================================
# Get Split Tests
# =============================================================================


class TestGetSplit:
    """Test get_split method."""

    def test_get_active_split(self):
        """Get active split returns train/val lists."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal" / "samples"
            samples_dir.mkdir(parents=True)

            for i in range(10):
                create_mock_sample(samples_dir, f"{i:04d}")

            manager = SplitManager(tmpdir)
            manager.create_initial_split("test_signal")

            train, val = manager.get_split("test_signal")

            assert len(train) == 8
            assert len(val) == 2
            assert isinstance(train, list)
            assert isinstance(val, list)

    def test_get_specific_version(self):
        """Get specific split version."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal" / "samples"
            samples_dir.mkdir(parents=True)

            for i in range(10):
                create_mock_sample(samples_dir, f"{i:04d}")

            manager = SplitManager(tmpdir)
            manager.create_initial_split("test_signal")

            train, val = manager.get_split("test_signal", version=1)

            assert len(train) + len(val) == 10

    def test_get_missing_split_raises(self):
        """Raises FileNotFoundError for missing split."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SplitManager(tmpdir)

            with pytest.raises(FileNotFoundError):
                manager.get_split("nonexistent_signal")


# =============================================================================
# Split Summary Tests
# =============================================================================


class TestGetSplitSummary:
    """Test get_split_summary and related methods."""

    def test_get_active_version_no_split(self):
        """get_active_split_version returns 0 if no split."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SplitManager(tmpdir)
            version = manager.get_active_split_version("test_signal")

            assert version == 0

    def test_get_active_version_with_split(self):
        """get_active_split_version returns version number."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal" / "samples"
            samples_dir.mkdir(parents=True)

            for i in range(5):
                create_mock_sample(samples_dir, f"{i:04d}")

            manager = SplitManager(tmpdir)
            manager.create_initial_split("test_signal")

            version = manager.get_active_split_version("test_signal")

            assert version == 1

    def test_get_pending_samples_no_split(self):
        """get_pending_samples returns all samples if no split."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal" / "samples"
            samples_dir.mkdir(parents=True)

            for i in range(5):
                create_mock_sample(samples_dir, f"{i:04d}")

            manager = SplitManager(tmpdir)
            pending = manager.get_pending_samples("test_signal")

            assert len(pending) == 5

    def test_get_pending_samples_with_split(self):
        """get_pending_samples returns new samples not in split."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal" / "samples"
            samples_dir.mkdir(parents=True)

            for i in range(5):
                create_mock_sample(samples_dir, f"{i:04d}")

            manager = SplitManager(tmpdir)
            manager.create_initial_split("test_signal")

            # Add new sample
            create_mock_sample(samples_dir, "0010")

            pending = manager.get_pending_samples("test_signal")

            assert pending == ["0010"]

    def test_get_split_summary_no_split(self):
        """get_split_summary returns defaults with no split."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal" / "samples"
            samples_dir.mkdir(parents=True)

            for i in range(5):
                create_mock_sample(samples_dir, f"{i:04d}")

            manager = SplitManager(tmpdir)
            summary = manager.get_split_summary("test_signal")

            assert summary["total_samples"] == 5
            assert summary["pending_samples"] == 5
            assert summary["train_count"] == 0
            assert summary["val_count"] == 0
            assert summary["active_version"] is None

    def test_get_split_summary_with_split(self):
        """get_split_summary returns correct counts."""
        from training.splits import SplitManager

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "test_signal" / "samples"
            samples_dir.mkdir(parents=True)

            for i in range(10):
                create_mock_sample(samples_dir, f"{i:04d}")

            manager = SplitManager(tmpdir)
            manager.create_initial_split("test_signal")

            summary = manager.get_split_summary("test_signal")

            assert summary["total_samples"] == 10
            assert summary["pending_samples"] == 0
            assert summary["train_count"] == 8
            assert summary["val_count"] == 2
            assert summary["active_version"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
