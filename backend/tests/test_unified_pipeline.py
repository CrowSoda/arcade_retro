"""
UnifiedPipeline Tests - Pipeline orchestration tests.

Tests for backend/unified_pipeline.py module.
Covers LUT generation, helper classes, and utility functions.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# WaterfallSource Enum Tests
# =============================================================================


class TestWaterfallSource:
    """Test WaterfallSource enum."""

    def test_enum_exists(self):
        """WaterfallSource class exists."""
        from unified_pipeline import WaterfallSource

        assert WaterfallSource is not None

    def test_enum_has_values(self):
        """WaterfallSource has expected values."""
        from unified_pipeline import WaterfallSource

        # Check actual values
        assert hasattr(WaterfallSource, "MANUAL")
        assert hasattr(WaterfallSource, "RX1_SCANNING")
        assert hasattr(WaterfallSource, "RX1_RECORDING")


# =============================================================================
# LUT Generation Tests
# =============================================================================


class TestLUTGeneration:
    """Test colormap LUT generation functions."""

    def test_viridis_lut(self):
        """Viridis LUT has correct shape."""
        from unified_pipeline import _generate_viridis_lut

        lut = _generate_viridis_lut()

        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8
        assert lut.min() >= 0
        assert lut.max() <= 255

    def test_plasma_lut(self):
        """Plasma LUT has correct shape."""
        from unified_pipeline import _generate_plasma_lut

        lut = _generate_plasma_lut()

        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8

    def test_inferno_lut(self):
        """Inferno LUT has correct shape."""
        from unified_pipeline import _generate_inferno_lut

        lut = _generate_inferno_lut()

        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8

    def test_magma_lut(self):
        """Magma LUT has correct shape."""
        from unified_pipeline import _generate_magma_lut

        lut = _generate_magma_lut()

        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8

    def test_turbo_lut(self):
        """Turbo LUT has correct shape."""
        from unified_pipeline import _generate_turbo_lut

        lut = _generate_turbo_lut()

        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8


# =============================================================================
# Module Constants Tests
# =============================================================================


class TestModuleConstants:
    """Test module-level constants and LUTs."""

    def test_viridis_lut_constant(self):
        """VIRIDIS_LUT constant exists."""
        from unified_pipeline import VIRIDIS_LUT

        assert VIRIDIS_LUT is not None
        assert VIRIDIS_LUT.shape == (256, 3)

    def test_plasma_lut_constant(self):
        """PLASMA_LUT constant exists."""
        from unified_pipeline import PLASMA_LUT

        assert PLASMA_LUT is not None

    def test_inferno_lut_constant(self):
        """INFERNO_LUT constant exists."""
        from unified_pipeline import INFERNO_LUT

        assert INFERNO_LUT is not None

    def test_colormap_luts_dict(self):
        """COLORMAP_LUTS dict exists."""
        from unified_pipeline import COLORMAP_LUTS

        assert isinstance(COLORMAP_LUTS, dict)
        assert len(COLORMAP_LUTS) >= 4

    def test_colormap_names_list(self):
        """COLORMAP_NAMES list exists."""
        from unified_pipeline import COLORMAP_NAMES

        assert isinstance(COLORMAP_NAMES, list)
        assert "viridis" in COLORMAP_NAMES or len(COLORMAP_NAMES) > 0

    def test_valid_fft_sizes(self):
        """VALID_FFT_SIZES exists."""
        from unified_pipeline import VALID_FFT_SIZES

        # VALID_FFT_SIZES is a dict mapping sizes to descriptions
        assert isinstance(VALID_FFT_SIZES, dict)
        assert len(VALID_FFT_SIZES) > 0
        # Keys are FFT sizes (powers of 2)
        for size in VALID_FFT_SIZES.keys():
            assert size > 0 and (size & (size - 1)) == 0


# =============================================================================
# Capture Detection Tests
# =============================================================================


class TestCaptureDetection:
    """Test capture_detection function."""

    def test_capture_detection_callable(self):
        """capture_detection is callable."""
        from unified_pipeline import capture_detection

        assert callable(capture_detection)


# =============================================================================
# Module-Level Functions Tests
# =============================================================================


class TestModuleLevelFunctions:
    """Test module-level functions."""

    def test_cleanup_function_exists(self):
        """_cleanup function exists."""
        from unified_pipeline import _cleanup

        assert callable(_cleanup)

    def test_signal_handler_exists(self):
        """_signal_handler function exists."""
        from unified_pipeline import _signal_handler

        assert callable(_signal_handler)


# =============================================================================
# Class Existence Tests
# =============================================================================


class TestClassExistence:
    """Test that expected classes exist."""

    def test_unified_iq_source_exists(self):
        """UnifiedIQSource class exists."""
        from unified_pipeline import UnifiedIQSource

        assert UnifiedIQSource is not None

    def test_triple_buffered_pipeline_exists(self):
        """TripleBufferedPipeline class exists."""
        from unified_pipeline import TripleBufferedPipeline

        assert TripleBufferedPipeline is not None

    def test_video_stream_server_exists(self):
        """VideoStreamServer class exists."""
        from unified_pipeline import VideoStreamServer

        assert VideoStreamServer is not None

    def test_stream_source_selector_exists(self):
        """StreamSourceSelector class exists."""
        from unified_pipeline import StreamSourceSelector

        assert StreamSourceSelector is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
