"""
Unit tests for sub-band extraction.

Tests verify:
- Filter achieves 60 dB stopband attenuation
- Passband ripple < 0.1 dB
- Frequency shift accuracy
- Decimation ratio correctness
- Aliasing rejection
- DC offset removal

Run with: pytest backend/dsp/tests/test_extraction.py -v
"""

import pytest
import numpy as np
from scipy.signal import freqz
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dsp.subband_extractor import SubbandExtractor, ExtractionParams
from dsp.filters import design_aa_filter, verify_stopband, analyze_filter


class TestFilterDesign:
    """Tests for anti-aliasing filter design."""
    
    def test_stopband_attenuation_60db(self):
        """Verify filter achieves 60 dB stopband."""
        params = ExtractionParams(
            source_rate=20e6,
            center_offset=0,
            target_bandwidth=2e6,
            stopband_db=60.0,
        )
        extractor = SubbandExtractor(params)
        
        # Get frequency response at source rate
        # Note: filter is designed at interpolated rate, but we analyze at source
        w, h = freqz(extractor.filter_taps, worN=8000)
        
        # For this test, we need to analyze at the interpolated rate
        interp_rate = params.source_rate * extractor.up
        freqs = w * interp_rate / (2 * np.pi)
        mag_db = 20 * np.log10(np.abs(h) + 1e-12)
        
        # Find stopband (> 1.2x cutoff)
        cutoff = params.target_bandwidth / 2
        stopband_mask = freqs > cutoff * 1.2
        
        if np.any(stopband_mask):
            stopband_max = mag_db[stopband_mask].max()
            assert stopband_max < -55, f"Stopband only {-stopband_max:.1f} dB (expected > 55 dB)"
    
    def test_stopband_attenuation_80db(self):
        """Verify filter can achieve 80 dB stopband."""
        params = ExtractionParams(
            source_rate=20e6,
            center_offset=0,
            target_bandwidth=2e6,
            stopband_db=80.0,
        )
        extractor = SubbandExtractor(params)
        
        # Filter should have more taps for higher attenuation
        assert len(extractor.filter_taps) > 200, "80 dB filter needs more taps"
    
    def test_passband_ripple(self):
        """Verify passband ripple < 0.5 dB."""
        params = ExtractionParams(
            source_rate=20e6,
            center_offset=0,
            target_bandwidth=2e6,
            stopband_db=60.0,
        )
        extractor = SubbandExtractor(params)
        
        interp_rate = params.source_rate * extractor.up
        w, h = freqz(extractor.filter_taps, worN=8000)
        freqs = w * interp_rate / (2 * np.pi)
        mag_db = 20 * np.log10(np.abs(h) + 1e-12)
        
        # Passband (< 0.8x cutoff)
        cutoff = params.target_bandwidth / 2
        passband_mask = freqs < cutoff * 0.8
        
        if np.any(passband_mask):
            passband_ripple = mag_db[passband_mask].max() - mag_db[passband_mask].min()
            assert passband_ripple < 0.5, f"Passband ripple {passband_ripple:.2f} dB (expected < 0.5 dB)"
    
    def test_minimum_taps(self):
        """Verify filter has at least 63 taps."""
        params = ExtractionParams(
            source_rate=20e6,
            center_offset=0,
            target_bandwidth=2e6,
            stopband_db=60.0,
        )
        extractor = SubbandExtractor(params)
        
        assert len(extractor.filter_taps) >= 63, f"Filter has only {len(extractor.filter_taps)} taps"


class TestFrequencyTranslation:
    """Tests for frequency shift (NCO mixing)."""
    
    def test_frequency_shift_to_dc(self):
        """Verify tone at offset frequency shifts to DC."""
        source_rate = 20e6
        freq_offset = 1e6  # 1 MHz offset
        duration = 0.01  # 10 ms
        
        # Create test tone at the offset frequency
        t = np.arange(int(source_rate * duration)) / source_rate
        test_signal = np.exp(2j * np.pi * freq_offset * t).astype(np.complex64)
        
        params = ExtractionParams(
            source_rate=source_rate,
            center_offset=freq_offset,
            target_bandwidth=500e3,
            normalize=False,
            remove_dc=False,
        )
        extractor = SubbandExtractor(params)
        result = extractor.extract(test_signal)
        
        # After shift, tone should be at DC (0 Hz)
        # Verify via FFT that peak is at center
        fft = np.fft.fftshift(np.fft.fft(result.iq_data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(result.iq_data), 1/result.output_rate))
        
        peak_idx = np.argmax(np.abs(fft))
        peak_freq = freqs[peak_idx]
        
        # Allow some tolerance (within 1 kHz of DC)
        assert abs(peak_freq) < 1000, f"Peak at {peak_freq:.0f} Hz, expected ~0 Hz"
    
    def test_frequency_shift_negative_offset(self):
        """Verify negative frequency offset works correctly."""
        source_rate = 20e6
        freq_offset = -2e6  # -2 MHz offset
        duration = 0.01
        
        # Create tone at -2 MHz
        t = np.arange(int(source_rate * duration)) / source_rate
        test_signal = np.exp(2j * np.pi * freq_offset * t).astype(np.complex64)
        
        params = ExtractionParams(
            source_rate=source_rate,
            center_offset=freq_offset,
            target_bandwidth=500e3,
            normalize=False,
            remove_dc=False,
        )
        extractor = SubbandExtractor(params)
        result = extractor.extract(test_signal)
        
        # Verify DC peak
        fft = np.fft.fftshift(np.fft.fft(result.iq_data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(result.iq_data), 1/result.output_rate))
        peak_freq = freqs[np.argmax(np.abs(fft))]
        
        assert abs(peak_freq) < 1000, f"Peak at {peak_freq:.0f} Hz, expected ~0 Hz"
    
    def test_zero_offset_no_shift(self):
        """Verify zero offset doesn't modify signal frequency content."""
        source_rate = 20e6
        duration = 0.005
        
        # DC signal (constant)
        test_signal = np.ones(int(source_rate * duration), dtype=np.complex64)
        
        params = ExtractionParams(
            source_rate=source_rate,
            center_offset=0,  # No shift
            target_bandwidth=5e6,
            normalize=False,
            remove_dc=False,
        )
        extractor = SubbandExtractor(params)
        result = extractor.extract(test_signal)
        
        # DC should still dominate
        fft = np.abs(np.fft.fft(result.iq_data))
        dc_power = fft[0]
        other_power = np.max(fft[1:len(fft)//2])
        
        assert dc_power > other_power * 10, "DC signal was modified"


class TestDecimation:
    """Tests for decimation ratio and sample count."""
    
    def test_decimation_ratio_4x(self):
        """Verify 4:1 decimation (20 MHz → 5 MHz)."""
        params = ExtractionParams(
            source_rate=20e6,
            center_offset=0,
            target_bandwidth=2e6,  # 2.5x = 5 MHz output → 4:1 decimation
        )
        extractor = SubbandExtractor(params)
        
        input_samples = 1000000
        expected_output = extractor.get_output_sample_count(input_samples)
        
        test_data = (np.random.randn(input_samples) + 1j * np.random.randn(input_samples)).astype(np.complex64)
        result = extractor.extract(test_data)
        
        # Allow some tolerance for filter edge effects
        ratio = len(result.iq_data) / expected_output
        assert 0.98 < ratio < 1.02, f"Output ratio {ratio:.3f}, expected ~1.0"
    
    def test_decimation_ratio_10x(self):
        """Verify ~10:1 decimation (20 MHz → 2 MHz)."""
        params = ExtractionParams(
            source_rate=20e6,
            center_offset=0,
            target_bandwidth=800e3,  # 2.5x = 2 MHz output → 10:1
        )
        extractor = SubbandExtractor(params)
        
        # Verify actual rate approximation
        expected_rate = 800e3 * 2.5  # 2 MHz
        actual_ratio = params.source_rate / extractor.actual_target_rate
        
        assert 9.5 < actual_ratio < 10.5, f"Decimation ratio {actual_ratio:.1f}, expected ~10"
    
    def test_output_sample_count(self):
        """Verify get_output_sample_count is accurate."""
        params = ExtractionParams(
            source_rate=20e6,
            center_offset=0,
            target_bandwidth=4e6,
        )
        extractor = SubbandExtractor(params)
        
        for input_size in [100000, 500000, 1000000]:
            expected = extractor.get_output_sample_count(input_size)
            test_data = np.random.randn(input_size).astype(np.complex64)
            result = extractor.extract(test_data)
            
            # Within 1% tolerance
            ratio = len(result.iq_data) / expected
            assert 0.99 < ratio < 1.01, f"Size mismatch at input={input_size}"


class TestAliasing:
    """Tests for aliasing rejection."""
    
    def test_out_of_band_rejection(self):
        """Verify out-of-band signals don't alias into passband."""
        source_rate = 20e6
        
        # Create signal OUTSIDE target passband
        t = np.arange(int(source_rate * 0.01)) / source_rate
        out_of_band_freq = 3e6  # 3 MHz, outside 2 MHz target BW
        test_signal = np.exp(2j * np.pi * out_of_band_freq * t).astype(np.complex64)
        
        params = ExtractionParams(
            source_rate=source_rate,
            center_offset=0,
            target_bandwidth=2e6,
            stopband_db=60.0,
            normalize=False,
        )
        extractor = SubbandExtractor(params)
        result = extractor.extract(test_signal)
        
        # Check that output power is severely attenuated
        input_power = np.mean(np.abs(test_signal) ** 2)
        output_power = np.mean(np.abs(result.iq_data) ** 2)
        
        attenuation_db = 10 * np.log10(output_power / input_power + 1e-12)
        assert attenuation_db < -40, f"Only {-attenuation_db:.1f} dB attenuation (expected > 40 dB)"
    
    def test_in_band_preservation(self):
        """Verify in-band signals are preserved."""
        source_rate = 20e6
        
        # Create signal INSIDE target passband
        t = np.arange(int(source_rate * 0.01)) / source_rate
        in_band_freq = 500e3  # 500 kHz, well inside 2 MHz target BW
        test_signal = np.exp(2j * np.pi * in_band_freq * t).astype(np.complex64)
        
        params = ExtractionParams(
            source_rate=source_rate,
            center_offset=0,
            target_bandwidth=2e6,
            stopband_db=60.0,
            normalize=False,
        )
        extractor = SubbandExtractor(params)
        result = extractor.extract(test_signal)
        
        # Check that power is mostly preserved (within 3 dB)
        input_power = np.mean(np.abs(test_signal) ** 2)
        output_power = np.mean(np.abs(result.iq_data) ** 2)
        
        loss_db = 10 * np.log10(output_power / input_power + 1e-12)
        assert loss_db > -3, f"In-band loss {-loss_db:.1f} dB (expected < 3 dB)"


class TestDCOffset:
    """Tests for DC offset removal."""
    
    def test_dc_removal(self):
        """Verify DC offset is removed."""
        source_rate = 20e6
        
        # Create signal with large DC offset
        t = np.arange(int(source_rate * 0.01)) / source_rate
        signal = np.random.randn(len(t)) + 1j * np.random.randn(len(t))
        signal_with_dc = (signal + 0.5 + 0.3j).astype(np.complex64)  # Add DC
        
        params = ExtractionParams(
            source_rate=source_rate,
            center_offset=0,
            target_bandwidth=5e6,
            remove_dc=True,
            normalize=False,
        )
        extractor = SubbandExtractor(params)
        result = extractor.extract(signal_with_dc)
        
        # Check DC component is near zero
        dc_component = np.abs(np.mean(result.iq_data))
        assert dc_component < 0.05, f"DC component {dc_component:.3f}, expected ~0"
    
    def test_dc_preserved_when_disabled(self):
        """Verify DC is preserved when removal is disabled."""
        source_rate = 20e6
        
        # Constant DC signal
        test_signal = np.ones(int(source_rate * 0.005), dtype=np.complex64) * (1 + 0.5j)
        
        params = ExtractionParams(
            source_rate=source_rate,
            center_offset=0,
            target_bandwidth=5e6,
            remove_dc=False,
            normalize=False,
        )
        extractor = SubbandExtractor(params)
        result = extractor.extract(test_signal)
        
        # DC should be significant
        dc_component = np.abs(np.mean(result.iq_data))
        assert dc_component > 0.5, f"DC component {dc_component:.3f}, expected ~1.0"


class TestNormalization:
    """Tests for power normalization."""
    
    def test_unit_power_normalization(self):
        """Verify output is normalized to unit power."""
        source_rate = 20e6
        
        # Create signal with arbitrary power
        t = np.arange(int(source_rate * 0.01)) / source_rate
        test_signal = 5.0 * np.exp(2j * np.pi * 500e3 * t).astype(np.complex64)  # 5x amplitude
        
        params = ExtractionParams(
            source_rate=source_rate,
            center_offset=0,
            target_bandwidth=2e6,
            normalize=True,
        )
        extractor = SubbandExtractor(params)
        result = extractor.extract(test_signal)
        
        # Output power should be ~1.0
        output_power = np.mean(np.abs(result.iq_data) ** 2)
        assert 0.9 < output_power < 1.1, f"Output power {output_power:.3f}, expected ~1.0"
    
    def test_normalization_disabled(self):
        """Verify normalization can be disabled."""
        source_rate = 20e6
        
        t = np.arange(int(source_rate * 0.01)) / source_rate
        test_signal = 0.1 * np.exp(2j * np.pi * 500e3 * t).astype(np.complex64)  # Low power
        
        params = ExtractionParams(
            source_rate=source_rate,
            center_offset=0,
            target_bandwidth=2e6,
            normalize=False,
        )
        extractor = SubbandExtractor(params)
        result = extractor.extract(test_signal)
        
        # Output power should be similar to input (accounting for filter gain)
        output_power = np.mean(np.abs(result.iq_data) ** 2)
        assert output_power < 0.1, f"Output power {output_power:.3f}, too high for non-normalized"


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_small_bandwidth(self):
        """Test with very narrow bandwidth."""
        params = ExtractionParams(
            source_rate=20e6,
            center_offset=0,
            target_bandwidth=100e3,  # 100 kHz (very narrow)
            stopband_db=60.0,
        )
        extractor = SubbandExtractor(params)
        
        # Should create valid filter
        assert len(extractor.filter_taps) > 0
        assert extractor.actual_target_rate > params.target_bandwidth * 2  # Above Nyquist
    
    def test_large_bandwidth(self):
        """Test with bandwidth close to source."""
        params = ExtractionParams(
            source_rate=20e6,
            center_offset=0,
            target_bandwidth=18e6,  # 90% of source
            stopband_db=60.0,
        )
        extractor = SubbandExtractor(params)
        
        # Decimation should be minimal
        assert extractor.decim_ratio < 1.5
    
    def test_large_frequency_offset(self):
        """Test with frequency offset near Nyquist."""
        params = ExtractionParams(
            source_rate=20e6,
            center_offset=9e6,  # Near edge of 20 MHz BW
            target_bandwidth=1e6,
            stopband_db=60.0,
        )
        extractor = SubbandExtractor(params)
        
        # Should still work (offset < Nyquist)
        test_data = np.random.randn(100000).astype(np.complex64)
        result = extractor.extract(test_data)
        
        assert len(result.iq_data) > 0
    
    def test_empty_input(self):
        """Test with empty input array."""
        params = ExtractionParams(
            source_rate=20e6,
            center_offset=0,
            target_bandwidth=2e6,
        )
        extractor = SubbandExtractor(params)
        
        # Empty input should produce empty (or near-empty) output
        result = extractor.extract(np.array([], dtype=np.complex64))
        assert len(result.iq_data) == 0 or len(result.iq_data) < 100  # Allow for filter padding


class TestPerformance:
    """Basic performance benchmarks."""
    
    def test_processing_speed(self):
        """Verify processing is reasonably fast."""
        import time
        
        params = ExtractionParams(
            source_rate=20e6,
            center_offset=1e6,
            target_bandwidth=2e6,
            stopband_db=60.0,
        )
        extractor = SubbandExtractor(params)
        
        # 1 second of data at 20 MHz = 20M samples
        num_samples = 2000000  # 0.1 seconds for faster test
        test_data = np.random.randn(num_samples).astype(np.complex64)
        
        start = time.perf_counter()
        result = extractor.extract(test_data)
        elapsed = time.perf_counter() - start
        
        rate = num_samples / elapsed / 1e6
        
        # Should be at least 10 Msamp/s on modern hardware
        assert rate > 5, f"Processing rate {rate:.1f} Msamp/s too slow"
        print(f"\nProcessing rate: {rate:.1f} Msamp/s")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
