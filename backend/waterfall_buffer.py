"""
Circular buffer for waterfall display frames.
Accumulates FFT rows and renders complete frames for video encoding.
"""

import numpy as np
from typing import Tuple, List, Dict

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WaterfallBuffer] Warning: cv2 not available, detection boxes will be simple rectangles")


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


def _generate_turbo_lut() -> np.ndarray:
    """Generate turbo colormap lookup table (256 entries, RGB).
    
    Turbo is a perceptually uniform colormap that's good for scientific visualization.
    """
    # Turbo colormap key points (Google's turbo colormap)
    turbo_data = [
        (0.18995, 0.07176, 0.23217),  # 0
        (0.25107, 0.25237, 0.63374),  # 32
        (0.20094, 0.45064, 0.88891),  # 64
        (0.12204, 0.62222, 0.81697),  # 96
        (0.19228, 0.77018, 0.56984),  # 128
        (0.47528, 0.87962, 0.30549),  # 160
        (0.78875, 0.92025, 0.16914),  # 192
        (0.97612, 0.74913, 0.14574),  # 224
        (0.98880, 0.30549, 0.20602),  # 255
    ]
    
    lut = np.zeros((256, 3), dtype=np.uint8)
    positions = [0, 32, 64, 96, 128, 160, 192, 224, 255]
    
    for i in range(256):
        for j in range(len(positions) - 1):
            if positions[j] <= i <= positions[j + 1]:
                t = (i - positions[j]) / (positions[j + 1] - positions[j])
                r = turbo_data[j][0] * (1 - t) + turbo_data[j + 1][0] * t
                g = turbo_data[j][1] * (1 - t) + turbo_data[j + 1][1] * t
                b = turbo_data[j][2] * (1 - t) + turbo_data[j + 1][2] * t
                lut[i] = [int(r * 255), int(g * 255), int(b * 255)]
                break
    
    return lut


# Pre-compute colormaps
VIRIDIS_LUT = _generate_viridis_lut()
TURBO_LUT = _generate_turbo_lut()


class WaterfallBuffer:
    """
    Circular buffer that accumulates waterfall rows and renders video frames.
    
    The buffer stores raw dB values. When a frame is requested, it:
    1. Rolls the buffer so newest row is at bottom
    2. Normalizes dB values to 0-255
    3. Applies colormap
    4. Returns RGB frame ready for video encoding
    """
    
    def __init__(
        self,
        width: int = 2048,
        height: int = 1024,
        dynamic_range_db: float = 80.0,
        colormap: str = 'viridis',
    ):
        """
        Args:
            width: Number of frequency bins per row
            height: Number of time rows (history depth)
            dynamic_range_db: dB range for normalization
            colormap: 'viridis', 'turbo', or 'grayscale'
        """
        self.width = width
        self.height = height
        self.dynamic_range_db = dynamic_range_db
        self.colormap = colormap
        
        # Select colormap LUT
        if colormap == 'viridis':
            self.lut = VIRIDIS_LUT
        elif colormap == 'turbo':
            self.lut = TURBO_LUT
        else:
            self.lut = None  # Grayscale
        
        # Circular buffer storing raw dB values
        self.db_buffer = np.full((height, width), -120.0, dtype=np.float32)
        self.write_row = 0
        self.rows_written = 0
        
        # Track noise floor for adaptive normalization
        self.noise_floor_db = -80.0
        self.noise_alpha = 0.02  # Slow tracking
        
        # PSD data for the most recent row
        self.latest_psd = np.zeros(width, dtype=np.float32)
        
        print(f"[WaterfallBuffer] Initialized: {width}x{height}, {dynamic_range_db}dB range, colormap={colormap}")
    
    def add_row(self, db_row: np.ndarray) -> None:
        """
        Add a new row of dB values to the buffer.
        
        Args:
            db_row: 1D numpy array of dB values, length must equal self.width or will be resampled
        """
        # Resample if needed
        if len(db_row) != self.width:
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
    
    def get_frame(self) -> np.ndarray:
        """
        Render the current buffer state as an RGB frame.
        
        Returns:
            numpy array, shape (height, width, 3), dtype uint8, RGB format
            Newest data at bottom, oldest at top
        """
        # Roll buffer so newest row is at bottom (row 0 = oldest, row H-1 = newest)
        rolled = np.roll(self.db_buffer, -self.write_row, axis=0)
        
        # Normalize using tracked noise floor
        min_db = self.noise_floor_db - 5  # Slightly below noise
        max_db = self.noise_floor_db + self.dynamic_range_db
        
        normalized = np.clip((rolled - min_db) / (max_db - min_db + 1e-6), 0, 1)
        indices = (normalized * 255).astype(np.uint8)
        
        # Apply colormap
        if self.lut is not None:
            rgb = self.lut[indices]  # Shape: (height, width, 3)
        else:
            # Grayscale
            rgb = np.stack([indices, indices, indices], axis=-1)
        
        return rgb
    
    def get_frame_with_psd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get both the frame and the latest PSD row.
        
        Returns:
            Tuple of (rgb_frame, psd_data)
        """
        return self.get_frame(), self.latest_psd.copy()
    
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
    
    def get_frame_with_detections(self, detections: list) -> np.ndarray:
        """Return frame without boxes - Flutter handles overlay."""
        return self.get_frame()


class MultiResolutionWaterfallBuffer:
    """
    Waterfall buffer that maintains multiple resolution levels.
    Useful for zoom functionality.
    """
    
    def __init__(
        self,
        base_width: int = 4096,
        base_height: int = 2048,
        num_levels: int = 3,
        dynamic_range_db: float = 80.0,
    ):
        """
        Args:
            base_width: Full resolution width
            base_height: Full resolution height (time history)
            num_levels: Number of mipmap levels (each level is 2x smaller)
            dynamic_range_db: dB range for normalization
        """
        self.num_levels = num_levels
        self.buffers = []
        
        for level in range(num_levels):
            scale = 2 ** level
            w = base_width // scale
            h = base_height // scale
            self.buffers.append(WaterfallBuffer(w, h, dynamic_range_db))
        
        print(f"[MultiResWaterfall] Initialized {num_levels} levels")
    
    def add_row(self, db_row: np.ndarray) -> None:
        """Add a row to all resolution levels."""
        for buffer in self.buffers:
            buffer.add_row(db_row)
    
    def get_frame(self, level: int = 0) -> np.ndarray:
        """Get frame at specified resolution level (0 = full res)."""
        level = max(0, min(level, self.num_levels - 1))
        return self.buffers[level].get_frame()


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
    
    # Get frame
    frame = buffer.get_frame()
    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"Noise floor: {buffer.noise_floor_db:.1f} dB")
    print(f"Fill: {buffer.get_fill_percentage():.1f}%")
    
    # Save test image
    try:
        from PIL import Image
        img = Image.fromarray(frame, mode='RGB')
        img.save('waterfall_test.png')
        print("Saved waterfall_test.png")
    except ImportError:
        print("PIL not available, skipping image save")
    
    print("Test passed!")
