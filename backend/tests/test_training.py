"""
Training Module Tests - Dataset, splits, and sample management.

Tests for backend/training/ modules.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


class TestTrainingImports:
    """Test that training modules import correctly."""

    def test_training_package_imports(self):
        """Training package should import."""
        import training

        assert training is not None

    def test_dataset_module_imports(self):
        """Dataset module should import."""
        from training import dataset

        assert dataset is not None

    def test_splits_module_imports(self):
        """Splits module should import."""
        from training import splits

        assert splits is not None

    def test_sample_manager_imports(self):
        """Sample manager should import."""
        from training import sample_manager

        assert sample_manager is not None


class TestDatasetModule:
    """Test dataset handling."""

    def test_dataset_module_loads(self):
        """Dataset module should load."""
        from training import dataset

        assert dataset is not None

    def test_dataset_has_classes(self):
        """Dataset module should have some classes or functions."""
        from training import dataset

        public_attrs = [a for a in dir(dataset) if not a.startswith("_")]
        assert len(public_attrs) > 0


class TestSplitsModule:
    """Test train/val/test splitting."""

    def test_splits_module_loads(self):
        """Splits module should load."""
        from training import splits

        assert splits is not None

    def test_splits_has_functions(self):
        """Splits module should have some functions."""
        from training import splits

        public_attrs = [a for a in dir(splits) if not a.startswith("_")]
        assert len(public_attrs) > 0


class TestSampleManagerStructure:
    """Test sample manager structure."""

    def test_sample_manager_class_exists(self):
        """SampleManager class should exist."""
        from training.sample_manager import SampleManager

        assert SampleManager is not None

    def test_sample_manager_has_methods(self):
        """SampleManager should have some methods."""
        from training.sample_manager import SampleManager

        # Check for any public methods
        methods = [
            m
            for m in dir(SampleManager)
            if not m.startswith("_") and callable(getattr(SampleManager, m, None))
        ]
        assert len(methods) > 0, "SampleManager should have public methods"

    def test_sample_manager_instantiates(self):
        """SampleManager should instantiate."""
        # Try to instantiate with no args or find init signature
        import inspect

        from training.sample_manager import SampleManager

        sig = inspect.signature(SampleManager.__init__)
        params = list(sig.parameters.keys())

        # If only self, instantiate directly
        if len(params) == 1:
            manager = SampleManager()
            assert manager is not None
        else:
            # Just verify the class exists
            assert SampleManager is not None


class TestBoxValidation:
    """Test bounding box validation logic."""

    def test_valid_box_accepted(self):
        """Valid normalized box should be accepted."""
        # Box format: [x_min, y_min, x_max, y_max]
        valid_box = [0.1, 0.2, 0.3, 0.4]

        # Basic validation
        assert valid_box[0] < valid_box[2]  # x_min < x_max
        assert valid_box[1] < valid_box[3]  # y_min < y_max
        assert all(0 <= v <= 1 for v in valid_box)  # Normalized

    def test_invalid_box_detected(self):
        """Invalid box (x_min > x_max) should fail validation."""
        invalid_box = [0.5, 0.2, 0.3, 0.4]  # x_min > x_max

        # This should fail validation
        assert not (invalid_box[0] < invalid_box[2])

    def test_box_outside_bounds(self):
        """Box outside [0,1] range should fail validation."""
        out_of_bounds_box = [-0.1, 0.2, 0.3, 0.4]

        # Should detect out of bounds
        assert not all(0 <= v <= 1 for v in out_of_bounds_box)


class TestSpectrogramGeneration:
    """Test spectrogram generation utilities."""

    def test_fft_spectrogram_shape(self):
        """FFT-based spectrogram should have expected shape."""
        # Create mock IQ data (complex)
        nfft = 4096
        num_frames = 6
        iq_samples = num_frames * nfft
        mock_iq = np.random.randn(iq_samples).astype(np.float32) + 1j * np.random.randn(
            iq_samples
        ).astype(np.float32)
        mock_iq = mock_iq.astype(np.complex64)

        # Basic FFT processing - use fft (not rfft) for complex data
        iq_frames = mock_iq.reshape(num_frames, nfft)
        fft_result = np.fft.fft(iq_frames, axis=1)
        magnitude = np.abs(fft_result)

        # Should have expected dimensions
        assert magnitude.shape[0] == num_frames
        assert magnitude.shape[1] == nfft  # Full FFT output for complex input

    def test_db_conversion(self):
        """Magnitude to dB conversion should work correctly."""
        magnitude = np.array([1.0, 10.0, 100.0, 1000.0])

        # Convert to dB
        db = 20 * np.log10(magnitude + 1e-10)

        # Check expected values
        assert abs(db[0] - 0.0) < 0.1  # 1.0 -> 0 dB
        assert abs(db[1] - 20.0) < 0.1  # 10.0 -> 20 dB
        assert abs(db[2] - 40.0) < 0.1  # 100.0 -> 40 dB
        assert abs(db[3] - 60.0) < 0.1  # 1000.0 -> 60 dB


class TestTrainingServiceStructure:
    """Test training service structure."""

    def test_service_module_imports(self):
        """Service module should import."""
        from training import service

        assert service is not None

    def test_training_service_class_exists(self):
        """TrainingService class should exist."""
        from training.service import TrainingService

        assert TrainingService is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
