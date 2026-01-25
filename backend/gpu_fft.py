"""
GPU-Accelerated Spectrogram Processor

Batched FFT with full pipeline on GPU.
No CPU round-trips during processing.

Performance targets on Jetson Orin:
  - 8K FFT:  1-3ms
  - 16K FFT: 2-5ms
  - 32K FFT: 4-7ms
  - 64K FFT: 6-12ms

Math for 33ms chunk at 20MHz sample rate:
  - Samples per chunk: 660,000
  - With 65536 FFT, 32768 hop: (660,000 - 65536) // 32768 + 1 = 19 FFTs
"""

import time
import numpy as np
import torch
from numpy.lib.stride_tricks import as_strided

# Valid FFT sizes (must be power of 2)
VALID_FFT_SIZES = {
    8192: "8K (2441 Hz/bin, fastest)",
    16384: "16K (1221 Hz/bin, fast)",
    32768: "32K (610 Hz/bin, balanced)",
    65536: "64K (305 Hz/bin, detailed)",
}

DEFAULT_FFT_SIZE = 65536  # Match current behavior


class GPUSpectrogramProcessor:
    """
    Batched FFT with full pipeline on GPU.
    No CPU round-trips during processing.
    
    Usage:
        proc = GPUSpectrogramProcessor(fft_size=32768)
        db_rows = proc.process(iq_data)  # Returns numpy array (num_ffts, fft_size)
        
    To change FFT size at runtime:
        proc.update_fft_size(16384)  # Includes warmup
    """
    
    def __init__(self, fft_size: int = DEFAULT_FFT_SIZE, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.fft_size = fft_size
        self.hop_size = fft_size // 2  # 50% overlap
        
        # Pre-compute window on GPU
        self.window = torch.hann_window(fft_size, dtype=torch.float32, device=self.device)
        
        # Timing stats
        self.last_prep_ms = 0.0
        self.last_gpu_ms = 0.0
        
        # Flag to indicate warmup in progress (for UI feedback)
        self.is_warming_up = False
        
        # Warmup cuFFT kernels (CRITICAL - first call is slow)
        if self.device.type == 'cuda':
            self._warmup()
        
        print(f"[GPUSpectrogramProcessor] Initialized: FFT={fft_size}, device={self.device}")
    
    def _warmup(self):
        """
        Pre-compile cuFFT kernels to avoid first-frame latency.
        
        CRITICAL: cuFFT compiles kernels on first use for each FFT size.
        Without warmup, first frame after size change will spike 50-200ms.
        """
        self.is_warming_up = True
        print(f"[GPUSpectrogramProcessor] Warming up cuFFT kernels for FFT size {self.fft_size}...")
        
        warmup_start = time.perf_counter()
        
        # Create dummy data matching expected batch size (~19 FFTs for 65536)
        # Use slightly larger batch to ensure kernel is compiled for typical use
        dummy_batch_size = 25
        dummy = torch.randn(dummy_batch_size, self.fft_size, dtype=torch.complex64, device=self.device)
        
        # Run multiple iterations to ensure kernel is fully compiled and cached
        for i in range(5):
            _ = torch.fft.fft(dummy, dim=-1)
            _ = torch.fft.fftshift(_, dim=-1)
        
        torch.cuda.synchronize()
        
        warmup_ms = (time.perf_counter() - warmup_start) * 1000
        self.is_warming_up = False
        print(f"[GPUSpectrogramProcessor] Warmup complete ({warmup_ms:.1f}ms)")
    
    def update_fft_size(self, new_size: int) -> dict:
        """
        Called when user changes FFT size in settings.
        
        IMPORTANT: This includes warmup which may take 100-500ms.
        UI should show "Reconfiguring..." during this time.
        
        Returns:
            dict with 'success', 'old_size', 'new_size', 'warmup_ms'
        """
        if new_size not in VALID_FFT_SIZES:
            raise ValueError(f"Invalid FFT size: {new_size}. Valid: {list(VALID_FFT_SIZES.keys())}")
        
        if new_size == self.fft_size:
            return {
                'success': True,
                'old_size': self.fft_size,
                'new_size': new_size,
                'warmup_ms': 0,
                'message': 'No change needed'
            }
        
        old_size = self.fft_size
        print(f"[GPUSpectrogramProcessor] Updating FFT size: {old_size} -> {new_size}")
        
        warmup_start = time.perf_counter()
        
        # Update parameters
        self.fft_size = new_size
        self.hop_size = new_size // 2
        self.window = torch.hann_window(new_size, dtype=torch.float32, device=self.device)
        
        # CRITICAL: Warmup for new size - don't skip!
        if self.device.type == 'cuda':
            self._warmup()
        
        warmup_ms = (time.perf_counter() - warmup_start) * 1000
        
        return {
            'success': True,
            'old_size': old_size,
            'new_size': new_size,
            'warmup_ms': warmup_ms,
            'message': f'FFT size changed, warmup took {warmup_ms:.1f}ms'
        }
    
    def decimate_rows(self, db_tensor: torch.Tensor, target_rows: int = 20) -> torch.Tensor:
        """
        Reduce N FFT rows to fixed target_rows for display.
        Uses max-pooling to preserve signal peaks.
        
        This decouples FFT resolution from display bandwidth:
        - 8K FFT → 160 rows → decimated to 20 rows
        - 16K FFT → 79 rows → decimated to 20 rows
        - 32K FFT → 39 rows → decimated to 20 rows
        - 64K FFT → 19 rows → kept as-is
        
        Args:
            db_tensor: GPU tensor of shape (num_ffts, fft_size)
            target_rows: Fixed number of rows to output (default 20)
        
        Returns:
            Decimated tensor of shape (target_rows, fft_size)
        """
        num_rows = db_tensor.shape[0]
        
        if num_rows <= target_rows:
            return db_tensor  # Nothing to do, already small enough
        
        # Calculate how many rows to pool together
        pool_size = num_rows // target_rows
        
        # Trim to exact multiple
        trim_to = pool_size * target_rows
        trimmed = db_tensor[:trim_to]
        
        # Reshape and take max along pool dimension
        # Shape: (target_rows, pool_size, fft_size) → (target_rows, fft_size)
        fft_size = trimmed.shape[1]
        reshaped = trimmed.view(target_rows, pool_size, fft_size)
        decimated = reshaped.max(dim=1).values  # Max preserves signal peaks
        
        return decimated
    
    def process(self, iq_data: np.ndarray, decimate_to: int = 0) -> np.ndarray:
        """
        Full pipeline on GPU:
        segment → window → FFT → fftshift → abs → log10 → (optional) decimate
        
        Input: numpy complex64 array (raw IQ samples)
        Output: numpy array, shape (num_ffts or decimate_to, fft_size), dtype float32 (dB values)
        
        Args:
            iq_data: Raw IQ samples
            decimate_to: If > 0, decimate output to this many rows (max-pooling)
        
        Performance:
            - CPU prep (segmentation): ~1-2ms
            - GPU compute: ~2-8ms depending on FFT size
            - Total: ~3-10ms (target <15ms)
        """
        t0 = time.perf_counter()
        
        # Handle dtype conversion if needed
        if iq_data.dtype == np.complex128:
            iq_data = iq_data.astype(np.complex64)
        
        # === Step A: Segment on CPU (unavoidable, data arrives as numpy) ===
        num_ffts = (len(iq_data) - self.fft_size) // self.hop_size + 1
        
        if num_ffts < 1:
            self.last_prep_ms = 0
            self.last_gpu_ms = 0
            return np.array([], dtype=np.float32).reshape(0, self.fft_size)
        
        # Create batched segments via stride tricks (zero-copy view)
        itemsize = iq_data.strides[0]
        segments = as_strided(
            iq_data,
            shape=(num_ffts, self.fft_size),
            strides=(self.hop_size * itemsize, itemsize)
        )
        # Must copy because strided view isn't contiguous for GPU transfer
        segments = np.ascontiguousarray(segments)
        
        t1 = time.perf_counter()
        self.last_prep_ms = (t1 - t0) * 1000
        
        # === Step B: Transfer to GPU (ONE transfer, not 19) ===
        gpu_data = torch.from_numpy(segments).to(self.device, non_blocking=True)
        
        # === Step C: Full pipeline on GPU (NO ROUND TRIPS) ===
        # Window (broadcasts across batch dimension)
        windowed = gpu_data * self.window
        
        # FFT (batched, along last axis)
        fft_result = torch.fft.fft(windowed, dim=-1)
        
        # fftshift (batched) - center DC component
        fft_shifted = torch.fft.fftshift(fft_result, dim=-1)
        
        # Magnitude to dB
        mag = torch.abs(fft_shifted)
        db = 20.0 * torch.log10(mag + 1e-10)
        
        # Sync GPU and transfer back to CPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        t2 = time.perf_counter()
        self.last_gpu_ms = (t2 - t1) * 1000
        
        return db.cpu().numpy().astype(np.float32)
    
    def get_timing_stats(self) -> dict:
        """Return last timing measurements for performance monitoring."""
        return {
            'prep_ms': self.last_prep_ms,
            'gpu_ms': self.last_gpu_ms,
            'total_ms': self.last_prep_ms + self.last_gpu_ms,
            'fft_size': self.fft_size,
            'is_warming_up': self.is_warming_up,
        }
    
    def get_info(self) -> dict:
        """Get processor configuration info."""
        return {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'device': str(self.device),
            'description': VALID_FFT_SIZES.get(self.fft_size, 'Unknown'),
            'valid_sizes': list(VALID_FFT_SIZES.keys()),
        }


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("GPU FFT Processor Test")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, using CPU")
    
    # Create processor
    proc = GPUSpectrogramProcessor(fft_size=32768)
    print(f"Processor info: {proc.get_info()}")
    
    # Generate test data (660,000 samples = 33ms at 20MHz)
    print("\nGenerating test data (660,000 complex samples)...")
    iq_data = (np.random.randn(660000) + 1j * np.random.randn(660000)).astype(np.complex64)
    
    # Test each FFT size
    print("\nBenchmarking all FFT sizes:")
    print("-" * 50)
    
    for fft_size in [8192, 16384, 32768, 65536]:
        proc.update_fft_size(fft_size)
        
        # Run multiple times to get stable timing
        times = []
        for _ in range(10):
            result = proc.process(iq_data)
            stats = proc.get_timing_stats()
            times.append(stats['total_ms'])
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        
        num_ffts = (len(iq_data) - fft_size) // (fft_size // 2) + 1
        print(f"  FFT {fft_size:5d}: {num_ffts:2d} FFTs, avg={avg_time:.1f}ms, min={min_time:.1f}ms, shape={result.shape}")
    
    print("-" * 50)
    print("Test complete!")
