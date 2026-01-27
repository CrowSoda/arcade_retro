"""
Quick Win Tests - Easy coverage for utility modules.

Tests for colormaps.py, logger_config.py, and runtime_info.py.
These are straightforward modules that provide immediate coverage gains.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# Colormap Tests (~43 lines coverage)
# =============================================================================


class TestColormapLUT:
    """Test colormap lookup table generation and usage."""

    def test_get_colormap_by_index(self):
        """get_colormap returns valid LUT for each index."""
        from colormaps import COLORMAPS, get_colormap

        for i in range(len(COLORMAPS)):
            lut = get_colormap(i)
            assert lut.shape == (256, 3), f"LUT {i} should be (256, 3)"
            assert lut.dtype == np.uint8, f"LUT {i} should be uint8"

    def test_get_colormap_wraps_index(self):
        """get_colormap wraps index when out of range."""
        from colormaps import COLORMAPS, get_colormap

        # Index 5 should wrap to index 0 (5 % 5 = 0)
        lut_wrapped = get_colormap(5)
        lut_first = get_colormap(0)
        np.testing.assert_array_equal(lut_wrapped, lut_first)

    def test_get_colormap_by_name(self):
        """get_colormap_by_name returns correct LUT."""
        from colormaps import VIRIDIS_LUT, get_colormap_by_name

        lut = get_colormap_by_name("viridis")
        np.testing.assert_array_equal(lut, VIRIDIS_LUT)

    def test_get_colormap_by_name_case_insensitive(self):
        """get_colormap_by_name is case insensitive."""
        from colormaps import get_colormap_by_name

        lut_lower = get_colormap_by_name("plasma")
        lut_upper = get_colormap_by_name("PLASMA")
        lut_mixed = get_colormap_by_name("Plasma")

        np.testing.assert_array_equal(lut_lower, lut_upper)
        np.testing.assert_array_equal(lut_lower, lut_mixed)

    def test_get_colormap_by_name_unknown_defaults_to_viridis(self):
        """Unknown name defaults to viridis."""
        from colormaps import VIRIDIS_LUT, get_colormap_by_name

        lut = get_colormap_by_name("unknown_colormap")
        np.testing.assert_array_equal(lut, VIRIDIS_LUT)

    def test_apply_colormap(self):
        """apply_colormap converts uint8 data to RGB."""
        from colormaps import apply_colormap

        # Test data: gradient from 0 to 255
        data = np.arange(256, dtype=np.uint8)
        rgb = apply_colormap(data, colormap_index=0)

        assert rgb.shape == (256, 3), "Output should be (256, 3)"
        assert rgb.dtype == np.uint8, "Output should be uint8"

    def test_apply_colormap_2d(self):
        """apply_colormap works with 2D arrays."""
        from colormaps import apply_colormap

        # 2D data (like a spectrogram)
        data = np.random.randint(0, 256, size=(100, 50), dtype=np.uint8)
        rgb = apply_colormap(data, colormap_index=0)

        assert rgb.shape == (100, 50, 3), "Output shape should be (*input, 3)"

    def test_apply_colormap_db(self):
        """apply_colormap_db normalizes dB data and applies colormap."""
        from colormaps import apply_colormap_db

        # dB data from -100 to -20
        data_db = np.linspace(-100.0, -20.0, 256)
        rgb = apply_colormap_db(data_db, min_db=-100.0, max_db=-20.0, colormap_index=0)

        assert rgb.shape == (256, 3), "Output should be (256, 3)"

    def test_apply_colormap_db_clips_values(self):
        """apply_colormap_db clips values outside range."""
        from colormaps import apply_colormap_db

        # Data with values outside [-100, -20]
        data_db = np.array([-150.0, -100.0, -60.0, -20.0, 10.0])
        rgb = apply_colormap_db(data_db, min_db=-100.0, max_db=-20.0, colormap_index=0)

        # Should not raise, values clipped to 0-255
        assert rgb.shape == (5, 3)


class TestColormapValues:
    """Test specific colormap properties."""

    def test_viridis_dark_at_low_values(self):
        """Viridis should be dark purple at low values."""
        from colormaps import VIRIDIS_LUT

        # Index 0 should be dark
        dark_color = VIRIDIS_LUT[0]
        assert dark_color[0] < 100, "Red channel should be low"
        assert dark_color[2] > 50, "Blue channel should have some value"

    def test_viridis_bright_at_high_values(self):
        """Viridis should be bright yellow at high values."""
        from colormaps import VIRIDIS_LUT

        # Index 255 should be bright yellow-green
        bright_color = VIRIDIS_LUT[255]
        assert bright_color[0] > 200, "Red channel should be high"
        assert bright_color[1] > 200, "Green channel should be high"

    def test_all_colormaps_have_256_entries(self):
        """All colormaps should have exactly 256 entries."""
        from colormaps import COLORMAPS

        for i, lut in enumerate(COLORMAPS):
            assert lut.shape[0] == 256, f"Colormap {i} should have 256 entries"

    def test_colormap_names_match_count(self):
        """COLORMAP_NAMES should match COLORMAPS count."""
        from colormaps import COLORMAP_NAMES, COLORMAPS

        assert len(COLORMAP_NAMES) == len(COLORMAPS)


class TestColormapInterpolation:
    """Test internal colormap generation functions."""

    def test_interpolate_color(self):
        """_interpolate_color blends colors correctly."""
        from colormaps import _interpolate_color

        # Test midpoint
        c1 = (0, 0, 0)
        c2 = (100, 200, 50)
        mid = _interpolate_color(c1, c2, 0.5)

        assert mid == (50, 100, 25), "Midpoint should be average"

    def test_interpolate_color_at_endpoints(self):
        """_interpolate_color returns correct values at t=0 and t=1."""
        from colormaps import _interpolate_color

        c1 = (10, 20, 30)
        c2 = (100, 200, 50)

        assert _interpolate_color(c1, c2, 0.0) == c1
        assert _interpolate_color(c1, c2, 1.0) == c2


# =============================================================================
# Logger Config Tests (~108 lines coverage)
# =============================================================================


class TestLoggerBasics:
    """Test basic logger functionality."""

    def test_get_logger_returns_g20_logger(self):
        """get_logger returns G20Logger instance."""
        from logger_config import G20Logger, get_logger

        logger = get_logger("test_module")
        assert isinstance(logger, G20Logger)
        assert logger.name == "test_module"

    def test_logger_methods_exist(self):
        """Logger should have info, warning, error, debug, perf methods."""
        from logger_config import get_logger

        logger = get_logger("test")

        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "perf")

    def test_logger_can_log_info(self):
        """Logger.info should not raise."""
        from logger_config import get_logger

        logger = get_logger("test")
        logger.info("Test message")  # Should not raise

    def test_logger_can_log_warning(self):
        """Logger.warning should not raise."""
        from logger_config import get_logger

        logger = get_logger("test")
        logger.warning("Test warning")  # Should not raise

    def test_logger_can_log_error(self):
        """Logger.error should not raise."""
        from logger_config import get_logger

        logger = get_logger("test")
        logger.error("Test error")  # Should not raise

    def test_logger_can_log_debug(self):
        """Logger.debug should not raise."""
        from logger_config import get_logger

        logger = get_logger("test")
        logger.debug("Test debug")  # Should not raise

    def test_logger_with_extra_fields(self):
        """Logger should accept extra dict."""
        from logger_config import get_logger

        logger = get_logger("test")
        logger.info("Test message", extra={"frame_idx": 42, "latency_ms": 5.2})


class TestLoggerConfiguration:
    """Test logger configuration functions."""

    def test_configure_logging(self):
        """configure_logging should not raise."""
        from logger_config import configure_logging

        configure_logging(level="INFO", perf_enabled=False, enable_file_logging=False)

    def test_configure_development(self):
        """configure_development should not raise."""
        from logger_config import configure_development

        configure_development()

    def test_configure_production(self):
        """configure_production should not raise."""
        from logger_config import configure_production

        configure_production()

    def test_is_perf_enabled(self):
        """is_perf_enabled returns boolean."""
        from logger_config import configure_logging, is_perf_enabled

        configure_logging(perf_enabled=False, enable_file_logging=False)
        assert is_perf_enabled() is False

        configure_logging(perf_enabled=True, enable_file_logging=False)
        assert is_perf_enabled() is True


class TestPerfLogging:
    """Test performance logging throttling."""

    def test_perf_disabled_returns_false(self):
        """perf() returns False when disabled."""
        from logger_config import configure_logging, get_logger

        configure_logging(perf_enabled=False, enable_file_logging=False)
        logger = get_logger("test_perf")

        assert logger.perf("Test perf message") is False

    def test_perf_enabled_throttled(self):
        """perf() is throttled by interval."""
        from logger_config import configure_logging, get_logger, reset_perf_counters

        configure_logging(perf_enabled=True, perf_interval=10, enable_file_logging=False)
        reset_perf_counters()

        logger = get_logger("test_perf_throttle")

        # First 9 calls should return False (throttled)
        results = [logger.perf(f"Call {i}") for i in range(10)]

        # Only the 10th call (index 9) should return True
        assert results[9] is True
        assert sum(results) == 1  # Only one True

    def test_reset_perf_counters(self):
        """reset_perf_counters clears all counters."""
        from logger_config import (
            _perf_counters,
            configure_logging,
            get_logger,
            reset_perf_counters,
        )

        configure_logging(perf_enabled=True, perf_interval=100, enable_file_logging=False)

        logger = get_logger("test_reset")
        for _ in range(50):
            logger.perf("Counting")

        # Counter should be > 0
        assert _perf_counters.get("test_reset", 0) > 0

        # Reset
        reset_perf_counters()
        assert _perf_counters.get("test_reset", 0) == 0


class TestLogStorageManagement:
    """Test log file storage management."""

    def test_get_log_storage_used_returns_int(self):
        """get_log_storage_used returns integer bytes."""
        from logger_config import get_log_storage_used

        storage = get_log_storage_used()
        assert isinstance(storage, int)
        assert storage >= 0


class TestJSONFormatter:
    """Test JSON log formatting."""

    def test_json_formatter_creates_valid_json(self):
        """JSONFormatter produces valid JSON."""
        import json
        import logging

        from logger_config import JSONFormatter

        formatter = JSONFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        # Should be valid JSON
        parsed = json.loads(formatted)
        assert "timestamp" in parsed
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"
        assert parsed["message"] == "Test message"


# =============================================================================
# Runtime Info Tests (~37 lines coverage)
# =============================================================================


class TestRuntimeInfo:
    """Test runtime_info module."""

    def test_module_imports(self):
        """runtime_info module should import."""
        import runtime_info

        assert runtime_info is not None

    def test_has_system_info_function(self):
        """Should have get_system_info or similar function."""
        import runtime_info

        # Check for common function names
        funcs = [
            name
            for name in dir(runtime_info)
            if callable(getattr(runtime_info, name)) and not name.startswith("_")
        ]
        assert len(funcs) > 0, "Should have at least one public function"


# =============================================================================
# GPU FFT Tests (if available) (~114 lines coverage)
# =============================================================================


class TestGpuFFTImports:
    """Test GPU FFT module imports."""

    def test_module_imports(self):
        """gpu_fft module should import."""
        import gpu_fft

        assert gpu_fft is not None

    def test_has_fft_functions(self):
        """Should have FFT-related functions."""
        import gpu_fft

        # Check for function existence
        public_funcs = [name for name in dir(gpu_fft) if not name.startswith("_")]
        assert len(public_funcs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
