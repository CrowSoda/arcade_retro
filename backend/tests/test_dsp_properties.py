"""
Property-Based DSP Tests using Hypothesis.

These tests verify mathematical invariants that must ALWAYS hold,
regardless of input values. Hypothesis generates many random test cases
to find edge cases that might break our assumptions.

Per roadmap: Week 7-8 advanced testing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# Hypothesis Strategies for RF/DSP Data
# =============================================================================


# IQ samples (complex64) - bounded magnitude to avoid overflow
def complex_samples(min_size=64, max_size=8192):
    """Strategy for complex IQ samples."""
    return arrays(
        dtype=np.complex64,
        shape=st.integers(min_value=min_size, max_value=max_size),
        elements=st.complex_numbers(
            min_magnitude=0,
            max_magnitude=10.0,  # Bounded to avoid overflow
            allow_nan=False,
            allow_infinity=False,
        ),
    ).map(lambda x: x.astype(np.complex64))


# Power-of-2 sizes for FFT (FFT is most efficient at power-of-2)
fft_sizes = st.sampled_from([64, 128, 256, 512, 1024, 2048, 4096])

# Frequency values (normalized 0-1)
normalized_freqs = st.floats(min_value=0.01, max_value=0.49, allow_nan=False)

# dB values (typical dynamic range)
db_values = st.floats(min_value=-120.0, max_value=20.0, allow_nan=False)


# =============================================================================
# FFT Mathematical Properties
# =============================================================================


class TestFFTProperties:
    """Property-based tests for FFT mathematical invariants."""

    @given(complex_samples(min_size=64, max_size=4096))
    @settings(max_examples=100, deadline=None)
    def test_parseval_theorem(self, iq_data):
        """
        Parseval's theorem: Energy is preserved between time and frequency domains.

        ∑|x[n]|² = (1/N) ∑|X[k]|²

        This MUST hold for any valid signal - if it doesn't, our FFT is broken.
        """
        # Skip zero-length arrays
        assume(len(iq_data) > 0)

        # Time-domain energy
        time_energy = np.sum(np.abs(iq_data) ** 2)

        # Frequency-domain energy (scaled by N for Parseval's)
        fft_result = np.fft.fft(iq_data)
        freq_energy = np.sum(np.abs(fft_result) ** 2) / len(iq_data)

        # Should be equal within floating point tolerance
        np.testing.assert_allclose(
            time_energy,
            freq_energy,
            rtol=1e-4,
            err_msg="Parseval's theorem violated - energy not conserved",
        )

    @given(complex_samples(min_size=64, max_size=2048))
    @settings(max_examples=100, deadline=None)
    def test_fft_ifft_roundtrip(self, iq_data):
        """
        IFFT(FFT(x)) = x

        The inverse FFT must perfectly reconstruct the original signal.
        """
        assume(len(iq_data) > 0)

        # Forward then inverse
        fft_result = np.fft.fft(iq_data)
        reconstructed = np.fft.ifft(fft_result)

        # Should match original (with absolute tolerance for float32 precision)
        # float32 has ~7 decimal digits precision; FFT involves N multiplications
        # so cumulative error can reach ~sqrt(N)*eps ≈ 3e-6 for N=4096
        np.testing.assert_allclose(
            reconstructed,
            iq_data,
            rtol=1e-4,
            atol=5e-6,  # atol for float32 complex64 cumulative precision
            err_msg="FFT/IFFT roundtrip failed",
        )

    @given(
        complex_samples(min_size=64, max_size=1024),
        st.floats(min_value=0.5, max_value=5.0),  # Smaller range to avoid precision issues
        st.floats(min_value=0.5, max_value=5.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_fft_linearity(self, iq_data, scale_a, scale_b):
        """
        FFT is linear: FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)

        Linearity is fundamental - must hold for any signals and scalars.
        """
        assume(len(iq_data) > 0)

        # Convert to float64 for better precision in linearity test
        iq_data_64 = iq_data.astype(np.complex128)

        # Create a second signal (shifted version)
        y = np.roll(iq_data_64, len(iq_data) // 4)

        # Combined transform
        combined = scale_a * iq_data_64 + scale_b * y
        combined_fft = np.fft.fft(combined)

        # Sum of scaled individual transforms
        sum_fft = scale_a * np.fft.fft(iq_data_64) + scale_b * np.fft.fft(y)

        np.testing.assert_allclose(
            combined_fft,
            sum_fft,
            rtol=1e-8,  # Higher precision with float64
            atol=1e-10,
            err_msg="FFT linearity violated",
        )

    @given(complex_samples(min_size=64, max_size=2048), st.integers(min_value=1, max_value=100))
    @settings(max_examples=50, deadline=None)
    def test_fft_shift_theorem(self, iq_data, shift_samples):
        """
        Time shift ↔ Phase rotation in frequency domain.

        A circular shift in time corresponds to multiplying by complex exponential
        in frequency.
        """
        assume(len(iq_data) > 0)
        n = len(iq_data)
        shift = shift_samples % n  # Keep shift within bounds

        # Original FFT
        original_fft = np.fft.fft(iq_data)

        # Shifted signal
        shifted = np.roll(iq_data, shift)
        shifted_fft = np.fft.fft(shifted)

        # Expected: original FFT * exp(-j * 2π * k * shift / N)
        k = np.arange(n)
        phase_factor = np.exp(-2j * np.pi * k * shift / n)
        expected_fft = original_fft * phase_factor

        np.testing.assert_allclose(
            shifted_fft,
            expected_fft,
            rtol=1e-4,
            atol=1e-6,  # atol for near-zero frequency components
            err_msg="FFT shift theorem violated",
        )


# =============================================================================
# Magnitude and dB Conversion Properties
# =============================================================================


class TestMagnitudeProperties:
    """Property-based tests for magnitude and dB conversions."""

    @given(
        arrays(
            dtype=np.float32,
            shape=st.integers(min_value=1, max_value=1000),
            elements=st.floats(
                min_value=1e-10, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_db_conversion_invertible(self, magnitudes):
        """
        dB conversion is invertible: 10^(dB/20) = original magnitude.
        """
        # Convert to dB
        db = 20 * np.log10(magnitudes)

        # Convert back
        reconstructed = 10 ** (db / 20)

        np.testing.assert_allclose(
            reconstructed, magnitudes, rtol=1e-5, err_msg="dB conversion not invertible"
        )

    @given(
        arrays(
            dtype=np.float64,  # Use float64 for better precision in log operations
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(
                min_value=1e-10, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
        ),
        st.floats(min_value=0.1, max_value=100.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_db_scaling(self, magnitudes, scale_factor):
        """
        Scaling by factor k adds 20*log10(k) dB.
        """
        # Use float64 for log calculations to avoid float32 precision issues
        original_db = 20 * np.log10(magnitudes.astype(np.float64))
        scaled_db = 20 * np.log10((magnitudes * scale_factor).astype(np.float64))

        expected_diff = 20 * np.log10(scale_factor)
        actual_diff = scaled_db - original_db

        np.testing.assert_allclose(
            actual_diff,
            np.full_like(actual_diff, expected_diff),
            rtol=1e-10,  # float64 precision
            err_msg="dB scaling property violated",
        )

    @given(complex_samples(min_size=10, max_size=1000))
    @settings(max_examples=100, deadline=None)
    def test_magnitude_non_negative(self, iq_data):
        """
        Complex magnitude is always non-negative.
        """
        magnitudes = np.abs(iq_data)
        assert np.all(magnitudes >= 0), "Magnitude should never be negative"

    @given(complex_samples(min_size=10, max_size=1000))
    @settings(max_examples=100, deadline=None)
    def test_magnitude_triangle_inequality(self, iq_data):
        """
        |a + b| ≤ |a| + |b| (triangle inequality for complex numbers)
        """
        assume(len(iq_data) > 1)

        a = iq_data[:-1]
        b = iq_data[1:]

        lhs = np.abs(a + b)
        rhs = np.abs(a) + np.abs(b)

        # LHS should be <= RHS (with small tolerance for floating point)
        assert np.all(lhs <= rhs + 1e-5), "Triangle inequality violated"


# =============================================================================
# Window Function Properties
# =============================================================================


class TestWindowProperties:
    """Property-based tests for window functions."""

    @given(fft_sizes)
    @settings(max_examples=20, deadline=None)
    def test_hann_window_symmetry(self, n):
        """
        Hann window is symmetric: w[k] = w[N-1-k]
        """
        window = np.hanning(n)

        # Should be symmetric
        np.testing.assert_allclose(
            window, window[::-1], rtol=1e-10, err_msg="Hann window not symmetric"
        )

    @given(fft_sizes)
    @settings(max_examples=20, deadline=None)
    def test_hann_window_bounded(self, n):
        """
        Hann window values are in [0, 1].
        """
        window = np.hanning(n)

        assert np.all(window >= 0), "Window has negative values"
        assert np.all(window <= 1), "Window exceeds 1"

    @given(fft_sizes)
    @settings(max_examples=20, deadline=None)
    def test_hann_window_endpoints_small(self, n):
        """
        Hann window endpoints should be small (near zero).
        """
        window = np.hanning(n)

        # Endpoints should be < 0.01
        assert window[0] < 0.01, "Left endpoint too large"
        assert window[-1] < 0.01, "Right endpoint too large"

    @given(fft_sizes)
    @settings(max_examples=20, deadline=None)
    def test_window_reduces_energy(self, n):
        """
        Windowed signal has less or equal energy than original.
        """
        # Create test signal
        signal = np.random.randn(n).astype(np.float32)
        window = np.hanning(n)

        original_energy = np.sum(signal**2)
        windowed_energy = np.sum((signal * window) ** 2)

        assert windowed_energy <= original_energy + 1e-6, "Windowing increased energy"


# =============================================================================
# Frequency Axis Properties
# =============================================================================


class TestFrequencyAxisProperties:
    """Property-based tests for frequency calculations."""

    @given(
        fft_sizes,
        st.floats(min_value=1e6, max_value=100e6, allow_nan=False),  # Sample rates 1-100 MHz
    )
    @settings(max_examples=50, deadline=None)
    def test_fft_freq_range(self, n, sample_rate):
        """
        FFT frequency axis should span [-fs/2, fs/2).
        """
        freqs = np.fft.fftfreq(n, 1 / sample_rate)

        # After fftshift, should span from -fs/2 to fs/2
        shifted_freqs = np.fft.fftshift(freqs)

        # Min should be approximately -fs/2
        assert shifted_freqs[0] >= -sample_rate / 2 - 1e-6
        # Max should be less than fs/2
        assert shifted_freqs[-1] < sample_rate / 2 + 1e-6

    @given(fft_sizes, st.floats(min_value=1e6, max_value=100e6, allow_nan=False))
    @settings(max_examples=50, deadline=None)
    def test_fft_freq_spacing(self, n, sample_rate):
        """
        FFT frequency bins should be evenly spaced by fs/N.
        """
        freqs = np.fft.fftfreq(n, 1 / sample_rate)
        shifted_freqs = np.fft.fftshift(freqs)

        expected_spacing = sample_rate / n

        # Check spacing (except at wrap-around)
        diffs = np.diff(shifted_freqs)

        np.testing.assert_allclose(
            diffs,
            np.full_like(diffs, expected_spacing),
            rtol=1e-5,
            err_msg=f"Frequency spacing not uniform (expected {expected_spacing})",
        )

    @given(fft_sizes, st.floats(min_value=1e6, max_value=100e6, allow_nan=False))
    @settings(max_examples=50, deadline=None)
    def test_rfft_freq_positive_only(self, n, sample_rate):
        """
        rfftfreq should return only positive frequencies [0, fs/2].
        """
        freqs = np.fft.rfftfreq(n, 1 / sample_rate)

        assert np.all(freqs >= 0), "rfftfreq has negative frequencies"
        assert freqs[-1] <= sample_rate / 2 + 1e-6, "rfftfreq exceeds Nyquist"


# =============================================================================
# Decimation Properties
# =============================================================================


class TestDecimationProperties:
    """Property-based tests for decimation/downsampling."""

    @given(st.integers(min_value=100, max_value=10000), st.integers(min_value=2, max_value=16))
    @settings(max_examples=50, deadline=None)
    def test_decimation_length(self, num_samples, decimation_factor):
        """
        Decimation reduces sample count by ceil(n/k) for arr[::k].
        """
        signal = np.random.randn(num_samples).astype(np.float32)
        decimated = signal[::decimation_factor]

        # Python slicing arr[::k] gives ceil(len(arr)/k) elements
        expected_length = (num_samples + decimation_factor - 1) // decimation_factor
        assert len(decimated) == expected_length

    @given(st.integers(min_value=1000, max_value=5000), st.integers(min_value=2, max_value=8))
    @settings(max_examples=30, deadline=None)
    def test_decimation_preserves_dc(self, num_samples, decimation_factor):
        """
        DC component is preserved through decimation (for constant signal).
        """
        dc_value = 42.0
        signal = np.full(num_samples, dc_value, dtype=np.float32)
        decimated = signal[::decimation_factor]

        # DC should be preserved
        np.testing.assert_allclose(decimated, np.full_like(decimated, dc_value), rtol=1e-5)


# =============================================================================
# Complex Signal Properties
# =============================================================================


class TestComplexSignalProperties:
    """Property-based tests for complex signal operations."""

    @given(complex_samples(min_size=10, max_size=1000))
    @settings(max_examples=100, deadline=None)
    def test_magnitude_squared_identity(self, iq_data):
        """
        |z|² = Re(z)² + Im(z)² = z * conj(z)
        """
        # Method 1: direct
        mag_sq_1 = np.abs(iq_data) ** 2

        # Method 2: real + imag
        mag_sq_2 = iq_data.real**2 + iq_data.imag**2

        # Method 3: z * conj(z)
        mag_sq_3 = np.real(iq_data * np.conj(iq_data))

        np.testing.assert_allclose(mag_sq_1, mag_sq_2, rtol=1e-5)
        np.testing.assert_allclose(mag_sq_1, mag_sq_3, rtol=1e-5)

    @given(complex_samples(min_size=10, max_size=1000))
    @settings(max_examples=100, deadline=None)
    def test_conjugate_properties(self, iq_data):
        """
        conj(conj(z)) = z
        """
        double_conj = np.conj(np.conj(iq_data))

        np.testing.assert_allclose(
            double_conj, iq_data, rtol=1e-6, err_msg="Double conjugate should equal original"
        )

    @given(complex_samples(min_size=10, max_size=1000))
    @settings(max_examples=100, deadline=None)
    def test_euler_identity(self, iq_data):
        """
        For unit-magnitude complex numbers, |e^(jθ)| = 1.
        """
        # Get phase
        phases = np.angle(iq_data)

        # Create unit complex from phase
        unit_complex = np.exp(1j * phases)

        # Magnitude should be 1
        np.testing.assert_allclose(
            np.abs(unit_complex), np.ones_like(np.abs(unit_complex)), rtol=1e-6
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
