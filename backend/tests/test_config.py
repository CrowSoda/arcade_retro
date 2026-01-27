"""
Configuration Tests - Validate settings and capabilities.

Tests for pydantic-settings configuration system.
"""

import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


class TestSettingsValidation:
    """Test configuration validation rules."""

    def test_fft_size_must_be_power_of_two(self):
        """FFT sizes should be powers of 2."""
        from config.settings import FFTSettings

        settings = FFTSettings()
        # Check inference_nfft is power of 2
        assert settings.inference_nfft & (settings.inference_nfft - 1) == 0
        # Check waterfall_nfft is power of 2
        assert settings.waterfall_nfft & (settings.waterfall_nfft - 1) == 0

    def test_frequency_range_valid(self):
        """SDR frequency range should be sensible."""
        from config.settings import SDRSettings

        settings = SDRSettings()
        assert settings.min_freq_mhz > 0
        assert settings.max_freq_mhz > settings.min_freq_mhz
        assert settings.default_center_freq_mhz >= settings.min_freq_mhz
        assert settings.default_center_freq_mhz <= settings.max_freq_mhz

    def test_score_threshold_in_range(self):
        """Inference score threshold should be 0-1."""
        from config.settings import InferenceSettings

        settings = InferenceSettings()
        assert 0.0 <= settings.score_threshold <= 1.0

    def test_port_numbers_valid(self):
        """Port numbers should be in valid range (0 = dynamic)."""
        from config.settings import ServerSettings

        settings = ServerSettings()
        assert 1024 <= settings.grpc_port <= 65535
        # ws_port=0 means "dynamically assigned" which is valid
        assert settings.ws_port == 0 or 1024 <= settings.ws_port <= 65535

    def test_paths_are_pathlib(self):
        """Path settings should be Path objects."""
        from config.settings import PathSettings

        settings = PathSettings()
        assert isinstance(settings.data_dir, Path)
        assert isinstance(settings.models_dir, Path)
        assert isinstance(settings.logs_dir, Path)


class TestEnvironmentOverrides:
    """Test environment variable overrides."""

    def test_override_grpc_port(self):
        """Environment variable should override default port."""
        from config.settings import reload_settings

        original = os.environ.get("G20_SERVER_GRPC_PORT")
        os.environ["G20_SERVER_GRPC_PORT"] = "50099"

        try:
            settings = reload_settings()
            assert settings.server.grpc_port == 50099
        finally:
            if original is None:
                os.environ.pop("G20_SERVER_GRPC_PORT", None)
            else:
                os.environ["G20_SERVER_GRPC_PORT"] = original

    def test_override_score_threshold(self):
        """Environment variable should override score threshold."""
        from config.settings import reload_settings

        original = os.environ.get("G20_INFERENCE_SCORE_THRESHOLD")
        os.environ["G20_INFERENCE_SCORE_THRESHOLD"] = "0.75"

        try:
            settings = reload_settings()
            assert settings.inference.score_threshold == 0.75
        finally:
            if original is None:
                os.environ.pop("G20_INFERENCE_SCORE_THRESHOLD", None)
            else:
                os.environ["G20_INFERENCE_SCORE_THRESHOLD"] = original


class TestCapabilities:
    """Test capabilities API for frontend."""

    def test_capabilities_structure(self):
        """Capabilities should have all required sections."""
        from config.capabilities import get_capabilities

        caps = get_capabilities()
        required_keys = ["fft", "display", "sdr", "inference", "power", "version"]
        for key in required_keys:
            assert key in caps, f"Missing key: {key}"

    def test_fft_capabilities(self):
        """FFT capabilities should include options."""
        from config.capabilities import get_capabilities

        caps = get_capabilities()
        fft = caps["fft"]
        assert "inference_nfft" in fft
        assert "waterfall_nfft_options" in fft
        assert isinstance(fft["waterfall_nfft_options"], list)
        assert len(fft["waterfall_nfft_options"]) > 0

    def test_display_capabilities(self):
        """Display capabilities should include colormaps."""
        from config.capabilities import get_capabilities

        caps = get_capabilities()
        display = caps["display"]
        assert "colormap_options" in display
        assert "viridis" in display["colormap_options"]
        assert "fps_options" in display
        assert 30 in display["fps_options"]

    def test_sdr_capabilities(self):
        """SDR capabilities should have valid range."""
        from config.capabilities import get_capabilities

        caps = get_capabilities()
        sdr = caps["sdr"]
        assert sdr["min_freq_mhz"] < sdr["max_freq_mhz"]
        assert "bandwidth_options_mhz" in sdr

    def test_capabilities_json_serializable(self):
        """Capabilities should serialize to JSON without error."""
        import json

        from config.capabilities import get_capabilities_json

        json_str = get_capabilities_json()
        # Should not raise
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


class TestSettingsSingleton:
    """Test settings singleton behavior."""

    def test_same_instance_returned(self):
        """get_settings() should return cached instance."""
        from config.settings import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reload_creates_new_instance(self):
        """reload_settings() should create fresh instance."""
        from config.settings import get_settings, reload_settings

        s1 = get_settings()
        s2 = reload_settings()
        # After reload, get_settings returns new instance
        s3 = get_settings()
        assert s1 is not s2
        assert s2 is s3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
