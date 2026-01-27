"""
Hydra Extended Tests - Detector, config, and version manager tests.

Tests for backend/hydra/ modules with mocks for GPU/model operations.
Covers Detection dataclass, HydraDetector class methods, config, and version_manager.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# Detection Dataclass Tests
# =============================================================================


class TestDetectionDataclass:
    """Test Detection dataclass."""

    def test_create_detection(self):
        """Create Detection with all fields."""
        from hydra.detector import Detection

        det = Detection(
            box_id=0,
            x1=0.1,
            y1=0.2,
            x2=0.3,
            y2=0.4,
            confidence=0.95,
            class_id=1,
            class_name="signal_a",
            signal_name="signal_a",
        )

        assert det.box_id == 0
        assert det.x1 == 0.1
        assert det.y1 == 0.2
        assert det.x2 == 0.3
        assert det.y2 == 0.4
        assert det.confidence == 0.95
        assert det.class_id == 1
        assert det.class_name == "signal_a"
        assert det.signal_name == "signal_a"

    def test_detection_box_coords_normalized(self):
        """Detection coordinates should be in 0-1 range for normalized."""
        from hydra.detector import Detection

        det = Detection(
            box_id=0,
            x1=0.0,
            y1=0.0,
            x2=1.0,
            y2=1.0,
            confidence=0.5,
            class_id=1,
            class_name="test",
            signal_name="test",
        )

        assert 0 <= det.x1 <= 1
        assert 0 <= det.y1 <= 1
        assert 0 <= det.x2 <= 1
        assert 0 <= det.y2 <= 1


# =============================================================================
# Hydra Config Tests
# =============================================================================


class TestHydraConfigModule:
    """Test hydra/config.py module."""

    def test_config_imports(self):
        """hydra.config module should import."""
        from hydra import config

        assert config is not None

    def test_has_default_score_threshold(self):
        """Config should have DEFAULT_SCORE_THRESHOLD."""
        from hydra.config import DEFAULT_SCORE_THRESHOLD

        assert isinstance(DEFAULT_SCORE_THRESHOLD, float)
        assert 0 < DEFAULT_SCORE_THRESHOLD < 1

    def test_has_max_detections(self):
        """Config should have MAX_DETECTIONS_PER_HEAD."""
        from hydra.config import MAX_DETECTIONS_PER_HEAD

        assert isinstance(MAX_DETECTIONS_PER_HEAD, int)
        assert MAX_DETECTIONS_PER_HEAD > 0


# =============================================================================
# HydraDetector Init Tests (Mocked)
# =============================================================================


class TestHydraDetectorInit:
    """Test HydraDetector initialization."""

    def test_detector_init_no_models(self):
        """Initialize detector with empty models directory."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            detector = HydraDetector(models_dir=tmpdir, device="cpu")

            assert detector.models_dir == Path(tmpdir)
            assert detector.model is None
            assert detector.heads == {}
            assert detector._backbone_loaded is False
            assert detector._current_head is None

    def test_detector_init_loads_registry(self):
        """Detector loads registry.json if present."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a registry file
            registry = {
                "backbone_version": 1,
                "signals": {"test_signal": {"active_head_version": 1}},
                "last_updated": "2024-01-01T00:00:00Z",
            }
            registry_path = Path(tmpdir) / "registry.json"
            with open(registry_path, "w") as f:
                json.dump(registry, f)

            detector = HydraDetector(models_dir=tmpdir, device="cpu")

            assert "test_signal" in detector.registry.get("signals", {})

    def test_detector_scans_for_heads(self):
        """Detector scans heads directory if no registry."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create heads directory with a signal
            heads_dir = Path(tmpdir) / "heads" / "signal_a"
            heads_dir.mkdir(parents=True)

            # Create a fake head file
            (heads_dir / "active.pth").touch()

            detector = HydraDetector(models_dir=tmpdir, device="cpu")

            # Should have scanned and found signal_a
            assert "signal_a" in detector.registry.get("signals", {})


class TestHydraDetectorRegistry:
    """Test registry management."""

    def test_get_registry(self):
        """get_registry returns registry dict."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            detector = HydraDetector(models_dir=tmpdir, device="cpu")
            registry = detector.get_registry()

            assert isinstance(registry, dict)
            assert "signals" in registry

    def test_get_available_signals_empty(self):
        """get_available_signals returns empty list when no heads."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            detector = HydraDetector(models_dir=tmpdir, device="cpu")
            signals = detector.get_available_signals()

            assert isinstance(signals, list)
            assert len(signals) == 0

    def test_get_available_signals_with_heads(self):
        """get_available_signals returns signal names."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create registry with signals
            registry = {
                "backbone_version": 1,
                "signals": {
                    "signal_a": {"active_head_version": 1},
                    "signal_b": {"active_head_version": 1},
                },
            }
            registry_path = Path(tmpdir) / "registry.json"
            with open(registry_path, "w") as f:
                json.dump(registry, f)

            detector = HydraDetector(models_dir=tmpdir, device="cpu")
            signals = detector.get_available_signals()

            assert "signal_a" in signals
            assert "signal_b" in signals

    def test_save_registry(self):
        """_save_registry writes registry to disk."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            detector = HydraDetector(models_dir=tmpdir, device="cpu")
            detector.registry["signals"]["new_signal"] = {"version": 1}
            detector._save_registry()

            # Check file was written
            registry_path = Path(tmpdir) / "registry.json"
            assert registry_path.exists()

            with open(registry_path) as f:
                saved = json.load(f)
            assert "new_signal" in saved.get("signals", {})


class TestHydraDetectorHeadManagement:
    """Test head loading/unloading (mocked)."""

    def test_get_loaded_heads_empty(self):
        """get_loaded_heads returns empty list initially."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            detector = HydraDetector(models_dir=tmpdir, device="cpu")
            loaded = detector.get_loaded_heads()

            assert isinstance(loaded, list)
            assert len(loaded) == 0

    def test_unload_heads_clears_heads(self):
        """unload_heads removes heads from memory."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            detector = HydraDetector(models_dir=tmpdir, device="cpu")
            # Manually add a head
            detector.heads["test_signal"] = {"state": "fake"}

            detector.unload_heads(["test_signal"])

            assert "test_signal" not in detector.heads

    def test_unload_all_heads(self):
        """unload_heads with None unloads all."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            detector = HydraDetector(models_dir=tmpdir, device="cpu")
            # Manually add heads
            detector.heads["signal_a"] = {"state": "fake"}
            detector.heads["signal_b"] = {"state": "fake"}

            detector.unload_heads(None)

            assert len(detector.heads) == 0


class TestHydraDetectorTiming:
    """Test timing and memory stats."""

    def test_get_timing_stats(self):
        """get_timing_stats returns timing dict."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            detector = HydraDetector(models_dir=tmpdir, device="cpu")
            stats = detector.get_timing_stats()

            assert isinstance(stats, dict)
            assert "backbone_ms" in stats
            assert "heads_ms" in stats
            assert "total_ms" in stats

    def test_get_memory_usage_cpu(self):
        """get_memory_usage returns zeros on CPU."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            detector = HydraDetector(models_dir=tmpdir, device="cpu")
            memory = detector.get_memory_usage()

            assert isinstance(memory, dict)
            assert "allocated_mb" in memory
            assert "cached_mb" in memory
            # On CPU, should be 0
            assert memory["allocated_mb"] == 0


class TestHydraDetectorHeadInfo:
    """Test head metadata retrieval."""

    def test_get_head_info_no_metadata(self):
        """get_head_info returns empty dict if no metadata."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            detector = HydraDetector(models_dir=tmpdir, device="cpu")
            info = detector.get_head_info("nonexistent_signal")

            assert isinstance(info, dict)
            assert len(info) == 0

    def test_get_head_info_with_metadata(self):
        """get_head_info returns metadata if present."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create metadata file
            head_dir = Path(tmpdir) / "heads" / "signal_a"
            head_dir.mkdir(parents=True)

            metadata = {
                "signal_name": "signal_a",
                "active_version": 2,
                "versions": [{"version": 1}, {"version": 2}],
            }
            with open(head_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            detector = HydraDetector(models_dir=tmpdir, device="cpu")
            info = detector.get_head_info("signal_a")

            assert info["signal_name"] == "signal_a"
            assert info["active_version"] == 2


# =============================================================================
# Version Manager Tests
# =============================================================================


class TestVersionManager:
    """Test hydra/version_manager.py module."""

    def test_module_imports(self):
        """version_manager module should import."""
        from hydra import version_manager

        assert version_manager is not None

    def test_has_version_functions(self):
        """version_manager has version management functions."""
        from hydra import version_manager

        public_funcs = [name for name in dir(version_manager) if not name.startswith("_")]
        assert len(public_funcs) > 0


# =============================================================================
# Backbone Extractor Tests
# =============================================================================


class TestBackboneExtractor:
    """Test hydra/backbone_extractor.py module."""

    def test_module_imports(self):
        """backbone_extractor module should import."""
        try:
            from hydra import backbone_extractor

            assert backbone_extractor is not None
        except ImportError as e:
            if "torch" in str(e):
                pytest.skip("PyTorch not available")
            raise

    def test_has_extractor_classes(self):
        """backbone_extractor has extraction classes/functions."""
        try:
            from hydra import backbone_extractor

            public_items = [name for name in dir(backbone_extractor) if not name.startswith("_")]
            assert len(public_items) > 0
        except ImportError as e:
            if "torch" in str(e):
                pytest.skip("PyTorch not available")
            raise


# =============================================================================
# Integration: Detector without GPU
# =============================================================================


class TestDetectorNoGPU:
    """Test detector behavior without GPU (CPU fallback)."""

    def test_cpu_device_selected_without_cuda(self):
        """Detector falls back to CPU when CUDA unavailable."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            # Request CUDA but it may not be available
            detector = HydraDetector(models_dir=tmpdir, device="cuda")

            # Should have either cuda or cpu
            assert detector.device.type in ["cuda", "cpu"]

    def test_load_backbone_requires_file(self):
        """load_backbone raises if backbone file missing."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            detector = HydraDetector(models_dir=tmpdir, device="cpu")

            with pytest.raises(FileNotFoundError):
                detector.load_backbone()

    def test_detect_requires_backbone(self):
        """detect raises if backbone not loaded."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            detector = HydraDetector(models_dir=tmpdir, device="cpu")

            with pytest.raises(RuntimeError, match="Backbone not loaded"):
                import torch

                fake_tensor = torch.zeros(1, 3, 1024, 1024)
                detector.detect(fake_tensor)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestDetectorErrorHandling:
    """Test error handling in detector."""

    def test_detect_single_requires_loaded_head(self):
        """detect_single raises if head not loaded."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal structure
            (Path(tmpdir) / "backbone").mkdir()
            (Path(tmpdir) / "backbone" / "active.pth").touch()

            detector = HydraDetector(models_dir=tmpdir, device="cpu")

            with pytest.raises(ValueError, match="Head not loaded"):
                import torch

                fake_tensor = torch.zeros(1, 3, 1024, 1024)
                detector.detect_single(fake_tensor, "nonexistent_signal")

    def test_load_heads_warns_on_missing_head(self):
        """load_heads warns (doesn't crash) for missing head files."""
        from hydra.detector import HydraDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            detector = HydraDetector(models_dir=tmpdir, device="cpu")
            detector._backbone_loaded = True  # Fake backbone loaded

            # Try to load nonexistent head - should not raise
            detector.load_heads(["nonexistent_signal"])

            # Head should not be in loaded list
            assert "nonexistent_signal" not in detector.heads


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
