"""
Circular buffer for waterfall data storage.

NOTE: Rendering has moved to unified_pipeline.py (row-strip mode).
This module is kept for potential future use (inference context, etc.)
but most methods are now unused.
"""

import numpy as np
from typing import Tuple


def _generate_viridis_lut() -> np.ndarray:
    """Generate viridis colormap lookup table (256 entries, RGB)."""
    viridis_data = [
        (0.267004, 0.004874, 0.329415),  # 0
        (0.282327, 0.140926, 0.457517),  # 32
        (0.253935, 0.265254, 0.529983),  # 64
        (0.206756, 0.371758, 0.553117),  # 96
        (0.163625, 0.471133, 0.558148),  # 128
        (0.127568, 0.566949, 0.550556),  # 160
        (0.134692, 0.658636, 0.517649),  # 192
        (0.477504, 0.821444, 0.318195),  # 224
        (0.993248, 0.906157, 0.143936),  # 255
    ]
    
    lut = np.zeros((256, 3), dtype=np.uint8)
    positions = [0, 32, 64, 96, 128, 160, 192, 224, 255]
    
    for i in range(256):
        for j in range(len(positions) - 1):
            if positions[j] <= i <= positions[j + 1]:
                t = (i - positions[j]) / (positions[j + 1] - positions[j])
                r = viridis_data[j][0] * (1 - t) + viridis_data[j + 1][0] * t
                g = viridis_data[j][1] * (1 - t) + viridis_data[j + 1][1] * t
                b = viridis_data[j][2] * (1 - t) + viridis_data[j + 1][2] * t
                lut[i] = [int(r * 255), int(g * 255), int(b * 255)]
                break
    
    return lut


# Pre-compute colormap
VIRIDIS_LUT = _generate_viridis_lut()


class WaterfallBuffer:
    """
    Circular buffer that stores raw dB values.
    
    In row-strip mode, this class is NOT used for rendering.
    Rendering happens in unified_pipeline.py's VideoStreamServer._db_to_rgba().
    
    This class may still be useful for:
    - Storing context for inference
    - Future replay/export features
    """
    
    def __init__(
        self,
        width: int = 2048,
        height: int = 1024,
        dynamic_range_db: float = 80.0,
    ):
        """
        Args:
            width: Number of frequency bins per row
            height: Number of time rows (history depth)
            dynamic_range_db: dB range for normalization
        """
        self.width = width
        self.height = height
        self.dynamic_range_db = dynamic_range_db
        self.lut = VIRIDIS_LUT
        
        # Circular buffer storing raw dB values
        self.db_buffer = np.full((height, width), -120.0, dtype=np.float32)
        self.write_row = 0
        self.rows_written = 0
        
        # Track noise floor for adaptive normalization
        self.noise_floor_db = -80.0
        self.noise_alpha = 0.02  # Slow tracking
        
        # PSD data for the most recent row
        self.latest_psd = np.zeros(width, dtype=np.float32)
        
        print(f"[WaterfallBuffer] Initialized: {width}x{height}, {dynamic_range_db}dB range")
    
    def add_row(self, db_row: np.ndarray) -> None:
        """
        Add a new row of dB values to the buffer.
        
        Args:
            db_row: 1D numpy array of dB values, length must equal self.width or will be resampled
        """
        # Resample if needed using max-pooling (preserves signal peaks)
        if len(db_row) != self.width:
            stride = len(db_row) // self.width
            if stride > 1:
                # Max-pooling: take max of each group
                truncated_len = self.width * stride
                db_row = db_row[:truncated_len].reshape(self.width, stride).max(axis=1)
            else:
                # Upsampling case - use interpolation
                indices = np.linspace(0, len(db_row) - 1, self.width).astype(int)
                db_row = db_row[indices]
        
        # Update noise floor estimate (exponential moving average of median)
        median_db = np.median(db_row)
        self.noise_floor_db = self.noise_alpha * median_db + (1 - self.noise_alpha) * self.noise_floor_db
        
        # Store PSD for this row
        self.latest_psd = db_row.astype(np.float32)
        
        # Write to circular buffer
        self.db_buffer[self.write_row] = db_row
        self.write_row = (self.write_row + 1) % self.height
        self.rows_written += 1
    
    def get_psd(self) -> np.ndarray:
        """Get the most recent PSD row."""
        return self.latest_psd.copy()
    
    def clear(self) -> None:
        """Clear the buffer to initial state."""
        self.db_buffer.fill(-120.0)
        self.write_row = 0
        self.rows_written = 0
        self.noise_floor_db = -80.0
    
    def get_fill_percentage(self) -> float:
        """Get how much of the buffer has been filled (0-100%)."""
        return min(100.0, (self.rows_written / self.height) * 100)


if __name__ == "__main__":
    # Test the waterfall buffer
    print("Testing WaterfallBuffer...")
    
    buffer = WaterfallBuffer(width=2048, height=1024, dynamic_range_db=80.0)
    
    # Add some test rows
    for i in range(100):
        # Simulate FFT output with a signal
        db_row = np.random.randn(2048) * 10 - 80  # Noise floor around -80dB
        db_row[1000:1050] += 40  # Signal at bin 1000-1050
        buffer.add_row(db_row)
    
    print(f"Noise floor: {buffer.noise_floor_db:.1f} dB")
    print(f"Fill: {buffer.get_fill_percentage():.1f}%")
    print(f"Latest PSD shape: {buffer.get_psd().shape}")
    
    print("Test passed!")
