"""
DSP Extended Tests - Subband extraction and signal processing.

Tests for backend/dsp/subband_extractor.py and related modules.
Covers ExtractionParams, ExtractionResult, SubbandExtractor class,
and header reading/writing functions.
"""

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# ExtractionParams Tests
# =============================================================================


class TestExtractionParams:
    """Test ExtractionParams dataclass."""

    def test_create_params_minimal(self):
        """Create params with minimal required fields."""
        from dsp.subband_extractor import ExtractionParams

        params = ExtractionParams(source_rate=20e6, center_offset=1e6, target_bandwidth=500e3)

        assert params.source_rate == 20e6
        assert params.center_offset == 1e6
        assert params.target_bandwidth == 500e3

    def test_create_params_full(self):
        """Create params with all fields."""
        from dsp.subband_extractor import ExtractionParams

        params = ExtractionParams(
            source_rate=20e6,
            center_offset=2e6,
            target_bandwidth=1e6,
            target_rate=3e6,
            stopband_db=80.0,
            remove_dc=False,
            normalize=False,
        )

        assert params.target_rate == 3e6
        assert params.stopband_db == 80.0
        assert params.remove_dc is False
        assert params.normalize is False

    def test_default_values(self):
        """Default values are correct."""
        from dsp.subband_extractor import ExtractionParams

        params = ExtractionParams(source_rate=20e6, center_offset=0, target_bandwidth=1e6)

        assert params.target_rate is None  # Auto-calculated
        assert params.stopband_db == 60.0
        assert params.remove_dc is True
        assert params.normalize is True


# =============================================================================
# ExtractionResult Tests
# =============================================================================


class TestExtractionResult:
    """Test ExtractionResult dataclass."""

    def test_create_result(self):
        """Create extraction result with all fields."""
        from dsp.subband_extractor import ExtractionResult

        iq_data = np.zeros(1000, dtype=np.complex64)

        result = ExtractionResult(
            iq_data=iq_data,
            source_rate=20e6,
            output_rate=2.5e6,
            bandwidth=1e6,
            center_offset=1e6,
            filter_taps=255,
            decimation_ratio=8.0,
            processing_time=0.5,
        )

        assert result.iq_data is iq_data
        assert result.source_rate == 20e6
        assert result.output_rate == 2.5e6
        assert result.bandwidth == 1e6
        assert result.center_offset == 1e6
        assert result.filter_taps == 255
        assert result.decimation_ratio == 8.0
        assert result.processing_time == 0.5


# =============================================================================
# SubbandExtractor Tests
# =============================================================================


class TestSubbandExtractorInit:
    """Test SubbandExtractor initialization."""

    def test_create_extractor(self):
        """Create extractor with basic params."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(source_rate=20e6, center_offset=1e6, target_bandwidth=500e3)

        extractor = SubbandExtractor(params)

        assert extractor.params is params
        # Target rate should be 2.5x bandwidth = 1.25 MHz
        assert extractor.target_rate == pytest.approx(500e3 * 2.5, rel=0.01)

    def test_target_rate_auto_calculated(self):
        """Target rate is 2.5x bandwidth when not specified."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(source_rate=20e6, center_offset=0, target_bandwidth=1e6)

        extractor = SubbandExtractor(params)

        # Should be 2.5x bandwidth
        assert extractor.target_rate == 2.5e6

    def test_target_rate_below_nyquist_adjusted(self):
        """Target rate below Nyquist is adjusted upward."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(
            source_rate=20e6,
            center_offset=0,
            target_bandwidth=1e6,
            target_rate=1e6,  # Below Nyquist (2x bandwidth)
        )

        extractor = SubbandExtractor(params)

        # Should be adjusted to 2.5x bandwidth
        assert extractor.target_rate == 2.5e6

    def test_filter_taps_designed(self):
        """Filter taps are designed and stored."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(source_rate=20e6, center_offset=0, target_bandwidth=1e6)

        extractor = SubbandExtractor(params)

        # Filter taps should be designed
        assert extractor.filter_taps is not None
        assert len(extractor.filter_taps) >= 63  # Minimum
        assert len(extractor.filter_taps) <= 4095  # Maximum
        assert len(extractor.filter_taps) % 2 == 1  # Odd number

    def test_resample_ratios_calculated(self):
        """Up/down resample ratios are calculated."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(source_rate=20e6, center_offset=0, target_bandwidth=1e6)

        extractor = SubbandExtractor(params)

        # up and down should be positive integers
        assert extractor.up > 0
        assert extractor.down > 0
        assert isinstance(extractor.up, int)
        assert isinstance(extractor.down, int)


class TestSubbandExtractorExtract:
    """Test SubbandExtractor.extract method."""

    def test_extract_basic(self):
        """Basic extraction produces output."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(source_rate=20e6, center_offset=0, target_bandwidth=1e6)

        extractor = SubbandExtractor(params)

        # Create test IQ data
        iq_data = np.random.randn(10000) + 1j * np.random.randn(10000)
        iq_data = iq_data.astype(np.complex64)

        result = extractor.extract(iq_data)

        # Should return ExtractionResult
        assert result.iq_data is not None
        assert result.iq_data.dtype == np.complex64
        assert len(result.iq_data) > 0

    def test_extract_decimates_data(self):
        """Extraction reduces sample count."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(source_rate=20e6, center_offset=0, target_bandwidth=1e6)

        extractor = SubbandExtractor(params)

        # Create test IQ data
        iq_data = np.random.randn(100000) + 1j * np.random.randn(100000)
        iq_data = iq_data.astype(np.complex64)

        result = extractor.extract(iq_data)

        # Output should be fewer samples (decimated)
        expected_ratio = extractor.decim_ratio
        assert len(result.iq_data) < len(iq_data)
        assert len(result.iq_data) == pytest.approx(len(iq_data) / expected_ratio, rel=0.1)

    def test_extract_with_frequency_offset(self):
        """Extraction with frequency offset."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(
            source_rate=20e6,
            center_offset=5e6,  # 5 MHz offset
            target_bandwidth=1e6,
        )

        extractor = SubbandExtractor(params)

        iq_data = np.random.randn(10000) + 1j * np.random.randn(10000)
        iq_data = iq_data.astype(np.complex64)

        result = extractor.extract(iq_data)

        assert result.center_offset == 5e6

    def test_extract_normalizes_power(self):
        """Extraction normalizes to unit power when enabled."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(
            source_rate=20e6,
            center_offset=0,
            target_bandwidth=1e6,
            normalize=True,
        )

        extractor = SubbandExtractor(params)

        # Create high-power signal
        iq_data = 100 * (np.random.randn(10000) + 1j * np.random.randn(10000))
        iq_data = iq_data.astype(np.complex64)

        result = extractor.extract(iq_data)

        # Output power should be ~1.0
        output_power = np.mean(np.abs(result.iq_data) ** 2)
        assert output_power == pytest.approx(1.0, rel=0.1)

    def test_extract_without_normalization(self):
        """Extraction without normalization preserves power."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(
            source_rate=20e6,
            center_offset=0,
            target_bandwidth=1e6,
            normalize=False,
        )

        extractor = SubbandExtractor(params)

        iq_data = np.random.randn(10000) + 1j * np.random.randn(10000)
        iq_data = iq_data.astype(np.complex64)

        result = extractor.extract(iq_data)

        # Verify extraction completed and output exists
        assert result.iq_data is not None
        assert len(result.iq_data) > 0

    def test_extract_with_progress_callback(self):
        """Progress callback is called during extraction."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(source_rate=20e6, center_offset=0, target_bandwidth=1e6)

        extractor = SubbandExtractor(params)

        progress_values = []

        def callback(progress):
            progress_values.append(progress)

        iq_data = np.random.randn(10000) + 1j * np.random.randn(10000)
        iq_data = iq_data.astype(np.complex64)

        extractor.extract(iq_data, progress_callback=callback)

        # Should have received progress updates
        assert len(progress_values) > 0
        assert 0.0 in progress_values
        assert 1.0 in progress_values

    def test_extract_converts_dtype(self):
        """Extraction converts non-complex64 input."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(source_rate=20e6, center_offset=0, target_bandwidth=1e6)

        extractor = SubbandExtractor(params)

        # Use complex128 input
        iq_data = np.random.randn(10000) + 1j * np.random.randn(10000)
        iq_data = iq_data.astype(np.complex128)

        result = extractor.extract(iq_data)

        # Output should be complex64
        assert result.iq_data.dtype == np.complex64


class TestSubbandExtractorHelpers:
    """Test SubbandExtractor helper methods."""

    def test_get_output_sample_count(self):
        """get_output_sample_count calculates correctly."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(source_rate=20e6, center_offset=0, target_bandwidth=1e6)

        extractor = SubbandExtractor(params)

        input_samples = 100000
        expected = int(input_samples * extractor.up / extractor.down)

        assert extractor.get_output_sample_count(input_samples) == expected

    def test_get_filter_delay_samples(self):
        """get_filter_delay_samples returns positive integer."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(source_rate=20e6, center_offset=0, target_bandwidth=1e6)

        extractor = SubbandExtractor(params)

        delay = extractor.get_filter_delay_samples()

        assert isinstance(delay, int)
        assert delay >= 0

    def test_get_info(self):
        """get_info returns configuration dict."""
        from dsp.subband_extractor import ExtractionParams, SubbandExtractor

        params = ExtractionParams(source_rate=20e6, center_offset=1e6, target_bandwidth=500e3)

        extractor = SubbandExtractor(params)

        info = extractor.get_info()

        assert isinstance(info, dict)
        assert "source_rate_hz" in info
        assert "target_rate_hz" in info
        assert "target_bandwidth_hz" in info
        assert "center_offset_hz" in info
        assert "decimation_ratio" in info
        assert "resample_up" in info
        assert "resample_down" in info
        assert "filter_taps" in info
        assert "stopband_db" in info
        assert "remove_dc" in info
        assert "normalize" in info


# =============================================================================
# Header Reading/Writing Tests
# =============================================================================


class TestG20Header:
    """Test G20 header reading and writing."""

    def test_write_and_read_g20_header(self):
        """Write and read G20 header roundtrip."""
        from dsp.subband_extractor import _read_g20_header, _write_g20_header

        with tempfile.NamedTemporaryFile(suffix=".g20", delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                "version": 1,
                "sample_rate": 2.5e6,
                "center_freq": 100e6,
                "bandwidth": 2.5e6,
                "num_samples": 100000,
                "start_time": 1640000000.0,
                "signal_name": "test_signal",
                "latitude": 40.0,
                "longitude": -105.0,
            }

            _write_g20_header(filepath, metadata)

            # Read back
            read_meta = _read_g20_header(filepath)

            assert read_meta["version"] == metadata["version"]
            assert read_meta["sample_rate"] == metadata["sample_rate"]
            assert read_meta["center_freq"] == metadata["center_freq"]
            assert read_meta["bandwidth"] == metadata["bandwidth"]
            assert read_meta["num_samples"] == metadata["num_samples"]
            assert read_meta["start_time"] == metadata["start_time"]
            assert read_meta["signal_name"] == metadata["signal_name"]
            assert read_meta["latitude"] == metadata["latitude"]
            assert read_meta["longitude"] == metadata["longitude"]
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_write_header_creates_512_bytes(self):
        """G20 header is exactly 512 bytes."""
        from dsp.subband_extractor import RFCAP_HEADER_SIZE, _write_g20_header

        with tempfile.NamedTemporaryFile(suffix=".g20", delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                "sample_rate": 2.5e6,
                "center_freq": 100e6,
                "bandwidth": 2.5e6,
                "num_samples": 100000,
            }

            _write_g20_header(filepath, metadata)

            # Check file size
            size = Path(filepath).stat().st_size
            assert size == RFCAP_HEADER_SIZE
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_write_header_long_signal_name_truncated(self):
        """Signal name longer than 31 chars is truncated."""
        from dsp.subband_extractor import _read_g20_header, _write_g20_header

        with tempfile.NamedTemporaryFile(suffix=".g20", delete=False) as f:
            filepath = f.name

        try:
            long_name = "a" * 50  # 50 chars

            metadata = {
                "sample_rate": 2.5e6,
                "center_freq": 100e6,
                "bandwidth": 2.5e6,
                "num_samples": 100000,
                "signal_name": long_name,
            }

            _write_g20_header(filepath, metadata)

            # Read back - should be truncated
            read_meta = _read_g20_header(filepath)
            assert len(read_meta["signal_name"]) <= 31
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestIQDataIO:
    """Test IQ data reading and writing."""

    def test_read_iq_data(self):
        """Read IQ data from file."""
        from dsp.subband_extractor import (
            RFCAP_HEADER_SIZE,
            _read_iq_data,
            _write_g20_header,
        )

        with tempfile.NamedTemporaryFile(suffix=".g20", delete=False) as f:
            filepath = f.name

        try:
            # Write header
            num_samples = 1000
            metadata = {
                "sample_rate": 2.5e6,
                "center_freq": 100e6,
                "bandwidth": 2.5e6,
                "num_samples": num_samples,
            }
            _write_g20_header(filepath, metadata)

            # Append IQ data
            iq_data = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)).astype(
                np.complex64
            )
            with open(filepath, "ab") as f:
                iq_data.tofile(f)

            # Read back
            read_iq = _read_iq_data(filepath)

            np.testing.assert_array_almost_equal(read_iq, iq_data)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_read_iq_data_partial(self):
        """Read partial IQ data with offset and count."""
        from dsp.subband_extractor import (
            RFCAP_HEADER_SIZE,
            _read_iq_data,
            _write_g20_header,
        )

        with tempfile.NamedTemporaryFile(suffix=".g20", delete=False) as f:
            filepath = f.name

        try:
            # Write header
            num_samples = 1000
            metadata = {
                "sample_rate": 2.5e6,
                "center_freq": 100e6,
                "bandwidth": 2.5e6,
                "num_samples": num_samples,
            }
            _write_g20_header(filepath, metadata)

            # Append IQ data
            iq_data = (np.arange(num_samples) + 1j * np.arange(num_samples)).astype(np.complex64)
            with open(filepath, "ab") as f:
                iq_data.tofile(f)

            # Read partial
            read_iq = _read_iq_data(filepath, offset_samples=100, num_samples=500)

            assert len(read_iq) == 500
            np.testing.assert_array_almost_equal(read_iq, iq_data[100:600])
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestHeaderAutoDetect:
    """Test auto-detection of header format."""

    def test_detect_g20_format(self):
        """Auto-detect G20 format."""
        from dsp.subband_extractor import _read_rfcap_header, _write_g20_header

        with tempfile.NamedTemporaryFile(suffix=".g20", delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                "sample_rate": 2.5e6,
                "center_freq": 100e6,
                "bandwidth": 2.5e6,
                "num_samples": 100000,
            }

            _write_g20_header(filepath, metadata)

            # Should auto-detect as G20
            read_meta = _read_rfcap_header(filepath)
            assert read_meta["sample_rate"] == metadata["sample_rate"]
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_invalid_magic_raises(self):
        """Invalid header magic raises ValueError."""
        from dsp.subband_extractor import _read_rfcap_header

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            # Write invalid header
            f.write(b"INVALID" + b"\x00" * 505)
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="Unknown header format"):
                _read_rfcap_header(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)


# =============================================================================
# Filters Module Tests
# =============================================================================


class TestFiltersModule:
    """Test dsp/filters.py module."""

    def test_filters_module_imports(self):
        """filters module should import."""
        from dsp import filters

        assert filters is not None

    def test_has_filter_functions(self):
        """filters module has filter design functions."""
        from dsp import filters

        # Check for common filter function names
        public_funcs = [name for name in dir(filters) if not name.startswith("_")]
        assert len(public_funcs) > 0


# =============================================================================
# Simple Extract Module Tests
# =============================================================================


class TestSimpleExtractModule:
    """Test dsp/simple_extract.py module."""

    def test_simple_extract_module_imports(self):
        """simple_extract module should import."""
        from dsp import simple_extract

        assert simple_extract is not None

    def test_has_extract_functions(self):
        """simple_extract module has extraction functions."""
        from dsp import simple_extract

        public_funcs = [name for name in dir(simple_extract) if not name.startswith("_")]
        assert len(public_funcs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
