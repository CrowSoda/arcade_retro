"""
DSP Module Tests - Signal processing functions.

Tests for backend/dsp/ modules.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


class TestDSPImports:
    """Test that DSP modules import correctly."""

    def test_dsp_package_imports(self):
        """DSP package should import."""
        import dsp

        assert dsp is not None

    def test_filters_module_imports(self):
        """Filters module should import."""
        from dsp import filters

        assert filters is not None

    def test_simple_extract_imports(self):
        """Simple extract module should import."""
        from dsp import simple_extract

        assert simple_extract is not None

    def test_subband_extractor_imports(self):
        """Subband extractor should import."""
        from dsp import subband_extractor

        assert subband_extractor is not None


class TestFFTProperties:
    """Test FFT mathematical properties."""

    def test_parseval_theorem(self):
        """Energy in time domain equals energy in frequency domain."""
        # Create test signal
        n_samples = 4096
        iq_data = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        iq_data = iq_data.astype(np.complex64)

        # Time-domain energy
        time_energy = np.sum(np.abs(iq_data) ** 2)

        # Frequency-domain energy
        fft_result = np.fft.fft(iq_data)
        freq_energy = np.sum(np.abs(fft_result) ** 2) / n_samples

        # Should be equal within numerical precision
        np.testing.assert_allclose(time_energy, freq_energy, rtol=1e-5)

    def test_fft_linearity(self):
        """FFT is linear: FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)."""
        n_samples = 1024
        x = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        y = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        a, b = 2.0, 3.0

        # Combined transform
        combined_fft = np.fft.fft(a * x + b * y)

        # Sum of individual transforms
        sum_fft = a * np.fft.fft(x) + b * np.fft.fft(y)

        np.testing.assert_allclose(combined_fft, sum_fft, rtol=1e-5)

    def test_ifft_inverts_fft(self):
        """IFFT(FFT(x)) = x within numerical precision."""
        n_samples = 2048
        iq_data = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        iq_data = iq_data.astype(np.complex64)

        # Round-trip
        reconstructed = np.fft.ifft(np.fft.fft(iq_data))

        # Use looser tolerance for complex64 (single precision)
        np.testing.assert_allclose(reconstructed, iq_data, rtol=1e-4)

    def test_fft_shift_symmetry(self):
        """fftshift moves DC component to center."""
        n_samples = 256
        signal = np.zeros(n_samples, dtype=np.complex64)
        signal[0] = 1.0  # DC component

        fft_result = np.fft.fft(signal)
        shifted = np.fft.fftshift(fft_result)

        # DC should now be at center
        center = n_samples // 2
        assert abs(shifted[center]) == pytest.approx(1.0, rel=1e-5)


class TestWindowFunctions:
    """Test window function properties."""

    def test_hann_window_symmetry(self):
        """Hann window should be symmetric."""
        n = 1024
        window = np.hanning(n)

        # Check symmetry
        np.testing.assert_allclose(window, window[::-1], rtol=1e-10)

    def test_hann_window_endpoints(self):
        """Hann window endpoints should be near zero."""
        n = 1024
        window = np.hanning(n)

        # Endpoints should be small
        assert window[0] < 0.01
        assert window[-1] < 0.01

    def test_hann_window_peak(self):
        """Hann window peak should be 1.0 at center."""
        n = 1024
        window = np.hanning(n)

        # Peak should be at center and approximately 1.0
        center = n // 2
        assert window[center] > 0.95

    def test_window_reduces_spectral_leakage(self):
        """Windowing should reduce spectral leakage."""
        n = 1024
        # Pure tone at non-FFT-bin frequency
        freq = 10.5  # Not an integer bin
        t = np.arange(n) / n
        signal = np.exp(2j * np.pi * freq * t * n / n)

        # FFT without window (rectangular)
        fft_no_window = np.abs(np.fft.fft(signal))

        # FFT with Hann window
        window = np.hanning(n)
        fft_windowed = np.abs(np.fft.fft(signal * window))

        # Windowed should have lower sidelobes
        # Check average of sidelobes (bins far from main lobe)
        far_bins = list(range(50, 100)) + list(range(n - 100, n - 50))

        avg_sidelobe_no_window = np.mean(fft_no_window[far_bins])
        avg_sidelobe_windowed = np.mean(fft_windowed[far_bins])

        # Windowed sidelobes should be significantly lower
        assert avg_sidelobe_windowed < avg_sidelobe_no_window


class TestMagnitudeConversion:
    """Test magnitude and dB conversions."""

    def test_db_from_magnitude(self):
        """dB conversion should follow 20*log10(magnitude)."""
        magnitudes = np.array([1.0, 10.0, 100.0, 0.1, 0.01])
        expected_db = np.array([0.0, 20.0, 40.0, -20.0, -40.0])

        computed_db = 20 * np.log10(magnitudes)

        np.testing.assert_allclose(computed_db, expected_db, rtol=1e-5)

    def test_power_db_from_magnitude(self):
        """Power dB should follow 10*log10(magnitude^2) = 20*log10(magnitude)."""
        magnitude = 10.0

        # Two equivalent formulations
        power_db = 10 * np.log10(magnitude**2)
        amplitude_db = 20 * np.log10(magnitude)

        assert power_db == pytest.approx(amplitude_db, rel=1e-5)

    def test_db_floor_prevents_log_zero(self):
        """dB conversion should handle zero/small values gracefully."""
        magnitudes = np.array([0.0, 1e-20, 1e-10, 1.0])

        # With floor
        floor = 1e-10
        safe_magnitudes = np.maximum(magnitudes, floor)
        db = 20 * np.log10(safe_magnitudes)

        # Should not have -inf
        assert not np.any(np.isinf(db))

        # Minimum should be 20*log10(floor)
        expected_min = 20 * np.log10(floor)
        assert np.min(db) == pytest.approx(expected_min, rel=1e-5)


class TestFrequencyCalculations:
    """Test frequency axis calculations."""

    def test_fft_frequency_axis(self):
        """FFT frequency axis should span [-fs/2, fs/2]."""
        n = 1024
        fs = 20e6  # 20 MHz sample rate

        freqs = np.fft.fftfreq(n, 1 / fs)
        freqs_shifted = np.fft.fftshift(freqs)

        # Check range
        assert np.min(freqs_shifted) >= -fs / 2
        assert np.max(freqs_shifted) <= fs / 2

        # Check DC is at center after shift
        center_idx = n // 2
        assert abs(freqs_shifted[center_idx]) < fs / n

    def test_rfft_frequency_axis(self):
        """RFFT frequency axis should span [0, fs/2]."""
        n = 1024
        fs = 20e6

        freqs = np.fft.rfftfreq(n, 1 / fs)

        # Should only have positive frequencies
        assert np.min(freqs) >= 0
        assert np.max(freqs) == pytest.approx(fs / 2, rel=1e-5)

        # Length should be n//2 + 1
        assert len(freqs) == n // 2 + 1


class TestDecimation:
    """Test decimation/downsampling."""

    def test_decimation_reduces_samples(self):
        """Decimation should reduce sample count by factor."""
        n_samples = 10000
        decimation_factor = 4

        signal = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)

        # Simple decimation (without anti-alias filter for this test)
        decimated = signal[::decimation_factor]

        expected_length = n_samples // decimation_factor
        assert len(decimated) == expected_length

    def test_decimation_preserves_low_frequencies(self):
        """Decimation should preserve frequencies below Nyquist/decimation_factor."""
        n = 4096
        fs = 20e6
        decimation = 4

        # Create low-frequency tone (well below new Nyquist)
        new_nyquist = fs / (2 * decimation)
        tone_freq = new_nyquist * 0.1  # 10% of new Nyquist

        t = np.arange(n) / fs
        signal = np.exp(2j * np.pi * tone_freq * t)

        # Decimate (simple, no filter for this test)
        decimated = signal[::decimation]

        # Check tone is preserved in FFT
        fft_result = np.abs(np.fft.fft(decimated))
        peak_bin = np.argmax(fft_result[: len(decimated) // 2])

        # Calculate expected bin
        expected_bin = int(tone_freq / (fs / decimation) * len(decimated))

        # Should be close
        assert abs(peak_bin - expected_bin) <= 1


class TestComplexSignals:
    """Test complex signal operations."""

    def test_complex_magnitude(self):
        """Complex magnitude should be sqrt(re^2 + im^2)."""
        re = np.array([3.0, 0.0, -4.0])
        im = np.array([4.0, 5.0, 3.0])
        z = re + 1j * im

        expected_mag = np.array([5.0, 5.0, 5.0])
        computed_mag = np.abs(z)

        np.testing.assert_allclose(computed_mag, expected_mag, rtol=1e-5)

    def test_complex_phase(self):
        """Complex phase should be arctan2(im, re)."""
        z = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])
        expected_phase = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
        computed_phase = np.angle(z)

        np.testing.assert_allclose(computed_phase, expected_phase, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
