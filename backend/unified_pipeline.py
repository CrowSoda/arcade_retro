"""
Unified IQ Pipeline - ROW-STRIP STREAMING VERSION

ARCHITECTURE:
- Row-strip streaming: backend sends FFT row strips, Flutter stitches locally
- Raw RGBA pixels (no video encoding needed)
- Detection boxes tracked by absolute row index for perfect sync

PERFORMANCE:
- ~310KB per frame (38 rows × 2048 width × 4 bytes RGBA)
- 30fps = ~9.3 MB/s bandwidth
- <60ms end-to-end latency

INFERENCE: 4096 FFT, 2048 hop, 80dB (MUST MATCH TENSORCADE!)
WATERFALL: 32768 FFT, 16384 hop, 60dB (high resolution display)
"""

import os
import sys
import time
import asyncio
import logging
import struct
import json
import atexit
import signal
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

import torch
import torch.nn.functional as F

# Handle both module import (from server.py) and direct execution
try:
    from .gpu_fft import GPUSpectrogramProcessor, VALID_FFT_SIZES, DEFAULT_FFT_SIZE
except ImportError:
    from gpu_fft import GPUSpectrogramProcessor, VALID_FFT_SIZES, DEFAULT_FFT_SIZE

def _cleanup():
    print("[Cleanup] Shutting down...", flush=True)
    print("[Cleanup] Done", flush=True)

atexit.register(_cleanup)

def _signal_handler(sig, frame):
    print(f"[Signal] Received {sig}, exiting...", flush=True)
    _cleanup()
    sys.exit(0)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified_pipeline")

BASE_DIR = Path(__file__).parent.parent


# =============================================================================
# WATERFALL SOURCE SELECTION - Priority: Manual > RX2 Recording > RX1 Recording > RX1 Scanning
# =============================================================================

class WaterfallSource:
    """Which RX stream is feeding the waterfall display."""
    RX1_SCANNING = 0   # Default: RX1 is scanning, no detection/recording
    RX1_RECORDING = 1  # RX1 detected something and is recording
    RX2_RECORDING = 2  # RX2 is collecting (RX1 handed off detection)
    MANUAL = 3         # Manual collection on any RX


class StreamSourceSelector:
    """
    Determines which IQ stream should feed the waterfall display.
    
    Priority (highest to lowest):
    1. Manual collection (user explicitly recording)
    2. RX2 recording (detector handed off to collector)
    3. RX1 recording (single RX mode, detector is also collecting)
    4. RX1 scanning (default: just hunting for signals)
    
    The waterfall always shows the "most important" stream:
    - If anything is recording to disk, show that stream
    - Otherwise, show the scanner stream
    """
    
    def __init__(self):
        self.current_source = WaterfallSource.RX1_SCANNING
        self.rx1_recording = False
        self.rx2_recording = False
        self.manual_active = False
        self._manual_rx = None  # Which RX is doing manual collection
    
    def update(self, rx1_recording: bool = None, rx2_recording: bool = None, 
               manual_active: bool = None, manual_rx: int = None) -> int:
        """
        Update source state and return which source should feed the waterfall.
        
        Returns: WaterfallSource enum value (0-3)
        """
        if rx1_recording is not None:
            self.rx1_recording = rx1_recording
        if rx2_recording is not None:
            self.rx2_recording = rx2_recording
        if manual_active is not None:
            self.manual_active = manual_active
        if manual_rx is not None:
            self._manual_rx = manual_rx
        
        # Priority selection
        if self.manual_active:
            self.current_source = WaterfallSource.MANUAL
        elif self.rx2_recording:
            self.current_source = WaterfallSource.RX2_RECORDING
        elif self.rx1_recording:
            self.current_source = WaterfallSource.RX1_RECORDING
        else:
            self.current_source = WaterfallSource.RX1_SCANNING
        
        return self.current_source
    
    @property
    def is_recording(self) -> bool:
        """True if any recording is active (not just scanning)."""
        return self.current_source != WaterfallSource.RX1_SCANNING
    
    @property
    def source_name(self) -> str:
        """Human-readable name for current source."""
        names = {
            WaterfallSource.RX1_SCANNING: "RX1_SCANNING",
            WaterfallSource.RX1_RECORDING: "RX1_RECORDING",
            WaterfallSource.RX2_RECORDING: "RX2_RECORDING",
            WaterfallSource.MANUAL: "MANUAL",
        }
        return names.get(self.current_source, "UNKNOWN")
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# Pre-compute viridis colormap LUT (256 entries, RGB)
def _generate_viridis_lut():
    """Generate viridis colormap lookup table."""
    # Viridis colormap values (simplified - key points)
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
        # Find surrounding control points
        for j in range(len(positions) - 1):
            if positions[j] <= i <= positions[j + 1]:
                t = (i - positions[j]) / (positions[j + 1] - positions[j])
                r = viridis_data[j][0] * (1 - t) + viridis_data[j + 1][0] * t
                g = viridis_data[j][1] * (1 - t) + viridis_data[j + 1][1] * t
                b = viridis_data[j][2] * (1 - t) + viridis_data[j + 1][2] * t
                lut[i] = [int(r * 255), int(g * 255), int(b * 255)]
                break
    
    return lut

VIRIDIS_LUT = _generate_viridis_lut()


# =============================================================================
# FFT DEBUG OUTPUT - Save high-res PNGs with detection boxes overlaid
# =============================================================================
DEBUG_DIR = Path('/tmp/fft_debug') if sys.platform != 'win32' else BASE_DIR / 'fft_debug'
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_ENABLED = False  # Set to True to save PNG captures

def capture_detection(fft_magnitude: np.ndarray, detection_boxes: list, chunk_index: int, label: str = ""):
    """
    Save high-res PNG with detection boxes drawn.
    Call this when a detection happens.
    
    fft_magnitude: raw FFT data (2D numpy array, shape: time × freq)
    detection_boxes: list of dicts with x1,y1,x2,y2 (normalized 0-1) and class_name
    chunk_index: monotonic counter
    """
    if not DEBUG_ENABLED:
        return
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        timestamp = datetime.now().strftime('%H%M%S_%f')
        
        # Big figure for high resolution
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Plot spectrogram (transpose so freq is Y axis)
        im = ax.imshow(fft_magnitude.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax, label='Magnitude (dB)')
        
        time_bins, freq_bins = fft_magnitude.shape
        
        # Draw detection boxes
        for det in detection_boxes:
            # Convert normalized coords (0-1) to pixel coords
            # NOTE: In model output, x1/x2 = time axis, y1/y2 = freq axis
            time_start = det['x1'] * time_bins
            time_end = det['x2'] * time_bins
            freq_start = det['y1'] * freq_bins
            freq_end = det['y2'] * freq_bins
            
            rect = patches.Rectangle(
                (time_start, freq_start),
                time_end - time_start,
                freq_end - freq_start,
                fill=False, 
                edgecolor='red', 
                linewidth=2
            )
            ax.add_patch(rect)
            
            # Add label
            class_name = det.get('class_name', 'unknown')
            conf = det.get('confidence', 0)
            ax.text(time_start, freq_end + 5, f'{class_name} {conf:.0%}', 
                    color='red', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Time (FFT index)')
        ax.set_ylabel('Frequency bin')
        label_str = f" ({label})" if label else ""
        ax.set_title(f'Detection #{chunk_index}{label_str} @ {timestamp}\nShape: {fft_magnitude.shape}, Range: [{fft_magnitude.min():.1f}, {fft_magnitude.max():.1f}] dB')
        
        # Save high-res PNG
        filename = f'detection_{chunk_index:04d}_{timestamp}.png'
        filepath = DEBUG_DIR / filename
        plt.savefig(str(filepath), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'[DEBUG] Saved {filename} ({len(detection_boxes)} boxes)', flush=True)
        
    except Exception as e:
        print(f'[DEBUG] Failed to save detection capture: {e}', flush=True)
        import traceback
        traceback.print_exc()


@dataclass
class TimestampedChunk:
    sequence_id: int
    pts: float
    sample_offset: int
    data: np.ndarray


@dataclass
class Detection:
    box_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    parent_pts: float


class UnifiedIQSource:
    def __init__(self, file_path: str, sample_rate: float = 20e6, start_offset_sec: float = 0.0):
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.bytes_per_sample = 8
        
        self.file = open(file_path, 'rb')
        self.file_size = os.path.getsize(file_path)
        
        self.start_offset = int(start_offset_sec * sample_rate * self.bytes_per_sample)
        self.position = self.start_offset
        self.sequence_id = 0
        
        logger.info(f"UnifiedIQSource: {file_path}, {self.file_size / 1e9:.2f} GB")
    
    def read_chunk(self, duration_ms: float = 33.0) -> Optional[TimestampedChunk]:
        samples = int(self.sample_rate * duration_ms / 1000)
        bytes_needed = samples * self.bytes_per_sample
        
        if self.position + bytes_needed > self.file_size:
            self.position = self.start_offset
        
        pts = (self.position - self.start_offset) / self.bytes_per_sample / self.sample_rate
        
        self.file.seek(self.position)
        raw = self.file.read(bytes_needed)
        
        if len(raw) < bytes_needed:
            return None
        
        self.position += bytes_needed
        self.sequence_id += 1
        
        iq_data = np.frombuffer(raw, dtype=np.complex64)
        
        return TimestampedChunk(
            sequence_id=self.sequence_id,
            pts=pts,
            sample_offset=self.position,
            data=iq_data,
        )
    
    def close(self):
        self.file.close()


class TripleBufferedPipeline:
    """
    SEPARATE params for INFERENCE vs WATERFALL.
    """
    
    def __init__(self, model_path: str, num_classes: int = 2):
        # =====================================================
        # INFERENCE PARAMS - MUST MATCH TENSORCADE EXACTLY!
        # =====================================================
        self.inference_fft_size = 4096
        self.inference_noverlap = 2048  # 50% overlap
        self.inference_hop_length = 2048  # 4096 - 2048
        self.inference_dynamic_range = 80.0
        
        # =====================================================
        # WATERFALL PARAMS - GPU-ACCELERATED
        # =====================================================
        self.waterfall_fft_size = DEFAULT_FFT_SIZE  # 65536 (305 Hz/bin)
        self.waterfall_hop = self.waterfall_fft_size // 2  # 50% overlap
        self.waterfall_dynamic_range = 60.0  # Better contrast for display
        
        # GPU FFT Processor (replaces numpy FFT loop)
        self.gpu_fft = GPUSpectrogramProcessor(
            fft_size=self.waterfall_fft_size,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Noise floor tracking (exponential moving average)
        self.noise_floor_db = -60.0
        self.noise_alpha = 0.02  # Slow tracking
        
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # CUDA streams
        self.num_streams = 3
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)] if self.device.type == 'cuda' else [None] * 3
        self.current_stream = 0
        
        self._load_model(model_path)
        
        # TWO SEPARATE WINDOWS
        self.inference_window = torch.hann_window(self.inference_fft_size, device=self.device)
        self.waterfall_window = np.hanning(65536).astype(np.float32)
        
        self.class_names = ['background', 'creamy_chicken']
        
        logger.info(f"Pipeline: {self.device}")
        logger.info(f"  INFERENCE: FFT={self.inference_fft_size}, hop={self.inference_hop_length}, dyn={self.inference_dynamic_range}dB")
        logger.info(f"  WATERFALL: FFT={self.waterfall_fft_size}, hop={self.waterfall_hop}, dyn={self.waterfall_dynamic_range}dB")
    
    def _load_model(self, model_path: str):
        import torchvision
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        
        logger.info(f"Loading: {model_path}")
        
        backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=5)
        self.model = torchvision.models.detection.FasterRCNN(backbone, num_classes=self.num_classes)
        
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        
        if self.device.type == 'cuda':
            self.model.half()
        
        logger.info(f"Model loaded on {self.device}")
    
    def compute_spectrogram(self, iq_data: np.ndarray) -> torch.Tensor:
        """INFERENCE spectrogram - MUST MATCH TENSORCADE."""
        if iq_data.dtype == np.complex128:
            iq_data = iq_data.astype(np.complex64)
        
        chunk = torch.from_numpy(iq_data).to(self.device)
        
        # TENSORCADE PARAMS (4096 FFT, 2048 hop, 80dB)
        Zxx = torch.stft(
            chunk,
            n_fft=self.inference_fft_size,  # 4096
            hop_length=self.inference_hop_length,  # 2048
            win_length=self.inference_fft_size,
            window=self.inference_window,
            center=False,
            return_complex=True
        )
        
        Zxx = torch.fft.fftshift(Zxx, dim=0)
        power = Zxx.abs().square()
        sxx_db = 10 * torch.log10(power + 1e-12)
        
        vmax = sxx_db.max()
        vmin = vmax - self.inference_dynamic_range  # 80dB
        sxx_norm = ((sxx_db - vmin) / (vmax - vmin + 1e-12)).clamp_(0, 1)
        
        sxx_norm = sxx_norm.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(sxx_norm, size=(1024, 1024), mode='bilinear', align_corners=False)
        resized = torch.flip(resized, dims=[2])
        
        return resized.expand(-1, 3, -1, -1)
    
    def process_chunk(self, iq_data: np.ndarray, pts: float, score_threshold: float = 0.5) -> Dict[str, Any]:
        """Run inference."""
        stream_idx = self.current_stream
        stream = self.streams[stream_idx]
        
        start_time = time.perf_counter()
        
        if self.device.type == 'cuda' and stream is not None:
            with torch.cuda.stream(stream):
                spec = self.compute_spectrogram(iq_data)
                with torch.inference_mode():
                    outputs = self.model(spec.half())
                torch.cuda.synchronize()
        else:
            spec = self.compute_spectrogram(iq_data)
            with torch.inference_mode():
                outputs = self.model(spec)
        
        detections = []
        if outputs and len(outputs) > 0:
            out = outputs[0]
            mask = out['scores'] >= score_threshold
            boxes = out['boxes'][mask].cpu().numpy()
            scores = out['scores'][mask].cpu().numpy()
            labels = out['labels'][mask].cpu().numpy()
            
            for i in range(len(boxes)):
                box = boxes[i]
                det = Detection(
                    box_id=i,
                    x1=float(box[0]) / 1024,  # Normalized 0-1
                    y1=1.0 - (float(box[3]) / 1024),  # Flip Y axis and swap (y2->y1)
                    x2=float(box[2]) / 1024,
                    y2=1.0 - (float(box[1]) / 1024),  # Flip Y axis and swap (y1->y2)
                    confidence=float(scores[i]),
                    class_id=int(labels[i]),
                    class_name=self.class_names[int(labels[i])] if int(labels[i]) < len(self.class_names) else 'unknown',
                    parent_pts=pts,
                )
                detections.append(det)
        
        inference_ms = (time.perf_counter() - start_time) * 1000
        self.current_stream = (self.current_stream + 1) % self.num_streams
        
        return {
            'detections': detections,
            'pts': pts,
            'inference_ms': inference_ms,
            'stream_idx': stream_idx,
        }
    
    # Target display rows per frame - decouples FFT resolution from display bandwidth
    # ~20 rows gives smooth scrolling without overwhelming Flutter
    TARGET_DISPLAY_ROWS = 20
    
    def compute_waterfall_rows(self, iq_data: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated batched FFT for waterfall display.
        
        Uses GPUSpectrogramProcessor for ~5-10x speedup over CPU numpy.
        ALWAYS outputs ~20 rows regardless of FFT size (via max-pooling decimation).
        
        This decouples FFT resolution from display bandwidth:
          - 8K FFT → 160 FFTs → decimated to 20 rows
          - 16K FFT → 79 FFTs → decimated to 20 rows
          - 32K FFT → 39 FFTs → decimated to 20 rows
          - 64K FFT → 19 FFTs → kept as-is (already ≤20)
        
        Returns dB values: shape (~20, fft_size)
        """
        import torch
        
        # GPU FFT (fast, 4-5ms regardless of FFT size)
        db_rows = self.gpu_fft.process(iq_data)
        
        if len(db_rows) == 0:
            return db_rows
        
        # Decimate to fixed row count (keeps Flutter happy)
        num_rows = db_rows.shape[0]
        target_rows = self.TARGET_DISPLAY_ROWS
        
        if num_rows <= target_rows:
            return db_rows  # Already small enough (64K FFT case)
        
        # Max-pooling decimation (preserves signal peaks)
        pool_size = num_rows // target_rows
        trim_to = pool_size * target_rows
        trimmed = db_rows[:trim_to]
        
        # Reshape and take max along pool dimension
        fft_size = trimmed.shape[1]
        reshaped = trimmed.reshape(target_rows, pool_size, fft_size)
        decimated = reshaped.max(axis=1)  # Max preserves signal peaks
        
        return decimated
    
    def update_waterfall_fft_size(self, new_size: int) -> dict:
        """
        Called when user changes FFT size via settings.
        
        IMPORTANT: Includes cuFFT warmup which may take 100-500ms.
        Returns dict with success status and timing info.
        """
        result = self.gpu_fft.update_fft_size(new_size)
        
        if result['success']:
            # Update our local tracking vars
            self.waterfall_fft_size = new_size
            self.waterfall_hop = new_size // 2
            
            # Note: waterfall_window is no longer used (GPU processor has its own)
            logger.info(f"[Pipeline] Waterfall FFT size updated: {result['old_size']} -> {new_size} "
                       f"(warmup: {result['warmup_ms']:.1f}ms)")
        
        return result
    
    def get_fft_timing_stats(self) -> dict:
        """Get FFT timing stats for performance monitoring."""
        return self.gpu_fft.get_timing_stats()
    
    def compute_waterfall_row_rgba(self, iq_data: np.ndarray, target_width: int = 1024) -> tuple:
        """
        OPTIMIZED: Compute waterfall row and return pre-rendered RGBA pixels + raw dB values.
        Returns (rgba_bytes, db_bytes, noise_floor_db)
        - rgba_bytes: Pre-rendered RGBA for waterfall display
        - db_bytes: Raw Float32 dB values for PSD chart
        """
        fft_size = self.waterfall_fft_size  # 4096
        
        # Take last chunk
        if len(iq_data) >= fft_size:
            segment = iq_data[-fft_size:]
        else:
            segment = np.pad(iq_data, (0, fft_size - len(iq_data)))
        
        # Window + FFT (vectorized)
        segment = segment * self.waterfall_window
        fft_data = np.fft.fftshift(np.fft.fft(segment))
        
        # Magnitude to dB
        db = 20 * np.log10(np.abs(fft_data) + 1e-6)
        
        # Max-pooling instead of decimation (preserves signal peaks)
        stride = fft_size // target_width
        if stride > 1:
            # Reshape to (target_width, stride) and take max of each group
            truncated_len = target_width * stride
            db_downsampled = db[:truncated_len].reshape(target_width, stride).max(axis=1).astype(np.float32)
        else:
            db_downsampled = db[:target_width].astype(np.float32)
        
        # Update noise floor (median, tracked over time)
        current_median = np.median(db_downsampled)
        self.noise_floor_db = self.noise_alpha * current_median + (1 - self.noise_alpha) * self.noise_floor_db
        
        # Normalize to 0-255 using tracked noise floor
        min_db = self.noise_floor_db - 5  # Slightly below noise
        max_db = self.noise_floor_db + self.waterfall_dynamic_range
        
        normalized = np.clip((db_downsampled - min_db) / (max_db - min_db), 0, 1)
        indices = (normalized * 255).astype(np.uint8)
        
        # Apply colormap (vectorized lookup)
        rgb = VIRIDIS_LUT[indices]  # Shape: (1024, 3)
        
        # Build RGBA (add alpha=255)
        rgba = np.zeros((target_width, 4), dtype=np.uint8)
        rgba[:, :3] = rgb
        rgba[:, 3] = 255
        
        return rgba.tobytes(), db_downsampled.tobytes(), self.noise_floor_db


# =============================================================================
# ROW-STRIP STREAMING SERVER - Lightweight row-based waterfall streaming
# =============================================================================

class VideoStreamServer:
    """
    Row-strip streaming server - sends FFT row strips instead of full frames.
    
    Flutter client maintains the pixel buffer and stitches strips together.
    This is more efficient and allows perfect row-index based box tracking.
    
    Protocol:
    - 0x01 + bytes: Row strip (zlib compressed RGB pixels)
    - 0x02 + json:  Detection data with absolute row indices
    - 0x03 + json:  Metadata (strip dimensions, total rows, etc.)
    
    Detection boxes tracked by absolute row index - Flutter handles positioning.
    """
    
    # Message type constants
    MSG_STRIP = 0x01      # Row strip data
    MSG_DETECTION = 0x02  # Detection JSON
    MSG_METADATA = 0x03   # Stream metadata
    
    # Keep old name for compatibility
    MSG_VIDEO = MSG_STRIP
    
    def __init__(
        self,
        iq_file: str,
        model_path: str,
        video_width: int = 2048,
        time_span_seconds: float = 5.0,  # Used for Flutter buffer sizing hint
        video_fps: int = 30,
    ):
        self.iq_source = UnifiedIQSource(iq_file)
        self.pipeline = TripleBufferedPipeline(model_path)
        
        # Waterfall source selector - determines which RX stream feeds the display
        self.source_selector = StreamSourceSelector()
        
        # Strip parameters
        self.video_width = video_width
        self.video_fps = video_fps
        # Reduced from 5s to 2.5s for better rendering performance (~30fps target)
        self.time_span_seconds = 2.5
        
        # Calculate rows_per_frame - FIXED at TARGET_DISPLAY_ROWS due to decimation
        # The GPU FFT processor now always outputs ~20 rows regardless of FFT size
        # (larger FFT outputs are max-pooled down to 20 rows)
        self.rows_per_frame = self.pipeline.TARGET_DISPLAY_ROWS  # Always ~20
        
        # ROW-STRIP MODE: No large buffer needed! Just colormap for encoding strips
        # Flutter maintains its own pixel buffer
        self.video_height = self.rows_per_frame  # Strip height = rows per frame (~38)
        
        # Store latest detections for JSON sending
        self.latest_detections: List[Dict] = []
        
        # NO VIDEO ENCODER NEEDED - we send raw/zlib compressed strips
        self.encoder = None
        
        # Inference buffer
        self.inference_buffer: List[TimestampedChunk] = []
        self.inference_chunk_count = 6
        
        # State
        self.is_running = False
        self.current_pts = 0.0
        
        # Score threshold - user configurable
        self.score_threshold = 0.5  # Default 50%, can be changed via command
        
        # Row tracking for detection synchronization
        self.total_rows_written = 0  # Monotonic counter, never resets
        self.rows_this_frame = 0     # Rows added in current frame
        
        # Suggested buffer height for Flutter client (uses 2.5s, not the parameter)
        self.suggested_buffer_height = int(self.time_span_seconds * video_fps * self.rows_per_frame)
        
        logger.info(f"VideoStreamServer initialized (ROW-STRIP MODE):")
        logger.info(f"  Strip size: {self.video_width}x{self.rows_per_frame} (~38 rows per frame)")
        logger.info(f"  Suggested client buffer: {self.video_width}x{self.suggested_buffer_height}")
        logger.info(f"  Time span hint: {self.time_span_seconds}s")
    
    async def send_metadata(self, websocket):
        """Send stream metadata to client - ROW-STRIP MODE."""
        metadata = {
            'type': 'metadata',
            'mode': 'row_strip',  # Tell Flutter we're in strip mode
            'strip_width': self.video_width,
            'rows_per_strip': self.rows_per_frame,  # ~38 rows per message
            'video_fps': self.video_fps,
            'suggested_buffer_height': self.suggested_buffer_height,  # Client creates buffer this size
            'time_span_seconds': self.time_span_seconds,
            'encoder': 'rgba_raw',  # Raw RGBA pixels (no compression for simplicity)
        }
        await websocket.send(bytes([self.MSG_METADATA]) + json.dumps(metadata).encode())
        logger.info(f"Sent metadata (ROW-STRIP): {metadata}")
    
    def _db_to_rgba(self, db_rows: np.ndarray, target_width: int = 2048) -> np.ndarray:
        """Convert dB values to RGBA pixels using viridis colormap."""
        num_rows = db_rows.shape[0]
        fft_size = db_rows.shape[1]
        
        # Resample to target width (max-pooling preserves peaks)
        stride = fft_size // target_width
        if stride > 1:
            truncated_len = target_width * stride
            # Take max of each group for all rows at once
            db_resampled = db_rows[:, :truncated_len].reshape(num_rows, target_width, stride).max(axis=2)
        else:
            db_resampled = db_rows[:, :target_width]
        
        # Update noise floor tracking
        current_median = np.median(db_resampled)
        self.pipeline.noise_floor_db = (self.pipeline.noise_alpha * current_median + 
                                         (1 - self.pipeline.noise_alpha) * self.pipeline.noise_floor_db)
        
        # Normalize to 0-255
        min_db = self.pipeline.noise_floor_db - 5
        max_db = self.pipeline.noise_floor_db + self.pipeline.waterfall_dynamic_range
        normalized = np.clip((db_resampled - min_db) / (max_db - min_db + 1e-6), 0, 1)
        indices = (normalized * 255).astype(np.uint8)
        
        # Apply colormap - shape (num_rows, target_width, 3)
        rgb = VIRIDIS_LUT[indices]
        
        # Add alpha channel - shape (num_rows, target_width, 4)
        rgba = np.zeros((num_rows, target_width, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255
        
        return rgba
    
    async def run_pipeline(self, websocket):
        """Main row-strip streaming loop - sends strips instead of full frames."""
        import zlib
        
        self.is_running = True
        frame_count = 0
        
        # Send metadata first
        await self.send_metadata(websocket)
        
        logger.info(f"Row-strip pipeline started ({self.video_fps}fps, ~{self.rows_per_frame} rows/frame)")
        
        while self.is_running:
            try:
                frame_start = time.perf_counter()
                
                # Dynamic FPS - read from self.video_fps each iteration
                frame_interval = 1.0 / self.video_fps
                
                # === PERF TIMING: IQ Read ===
                t0 = time.perf_counter()
                chunk = self.iq_source.read_chunk(duration_ms=33)
                t_iq_read = (time.perf_counter() - t0) * 1000
                
                if chunk is None:
                    await asyncio.sleep(0.01)
                    continue
                
                self.current_pts = chunk.pts
                
                # === PERF TIMING: FFT/Waterfall ===
                t0 = time.perf_counter()
                db_rows = self.pipeline.compute_waterfall_rows(chunk.data)
                t_fft = (time.perf_counter() - t0) * 1000
                
                self.rows_this_frame = len(db_rows)
                
                if self.rows_this_frame > 0:
                    # === PERF TIMING: RGBA Conversion ===
                    t0 = time.perf_counter()
                    rgba_strip = self._db_to_rgba(db_rows, target_width=self.video_width)
                    t_rgba = (time.perf_counter() - t0) * 1000
                    
                    # Get the LAST row's dB values for PSD chart
                    last_row_db = db_rows[-1]  # Most recent row
                    # Resample to target width (same as _db_to_rgba does)
                    stride = last_row_db.shape[0] // self.video_width
                    if stride > 1:
                        truncated_len = self.video_width * stride
                        psd_db = last_row_db[:truncated_len].reshape(self.video_width, stride).max(axis=1).astype(np.float32)
                    else:
                        psd_db = last_row_db[:self.video_width].astype(np.float32)
                    psd_bytes = psd_db.tobytes()
                    
                    # Pack strip message: header + raw RGBA bytes + PSD bytes
                    # Header: 17 bytes (was 16, added source_id)
                    #   - frame_id: uint32 (4 bytes)
                    #   - total_rows: uint32 (4 bytes, monotonic counter)
                    #   - rows_in_strip: uint16 (2 bytes)
                    #   - strip_width: uint16 (2 bytes)
                    #   - pts: float32 (4 bytes)
                    #   - source_id: uint8 (1 byte) - 0=SCAN, 1=RX1_REC, 2=RX2_REC, 3=MANUAL
                    header = struct.pack('<I I H H f B',
                        frame_count,
                        self.total_rows_written,
                        self.rows_this_frame,
                        self.video_width,
                        self.current_pts,
                        self.source_selector.current_source,  # Waterfall source indicator
                    )
                    
                    # === PERF TIMING: WebSocket Send ===
                    t0 = time.perf_counter()
                    strip_bytes = rgba_strip.tobytes()
                    await websocket.send(bytes([self.MSG_STRIP]) + header + strip_bytes + psd_bytes)
                    t_send = (time.perf_counter() - t0) * 1000
                    
                    # Update row counter AFTER sending
                    self.total_rows_written += self.rows_this_frame
                    
                    # === PERF TIMING: Total frame time ===
                    t_total = (time.perf_counter() - frame_start) * 1000
                    
                    # Print timing every 30 frames (with detailed GPU FFT breakdown)
                    if frame_count % 30 == 0:
                        fft_stats = self.pipeline.get_fft_timing_stats()
                        print(f"[PERF-PY] Frame {frame_count}: iq={t_iq_read:.1f}ms "
                              f"fft={t_fft:.1f}ms (prep={fft_stats['prep_ms']:.1f}+gpu={fft_stats['gpu_ms']:.1f}) "
                              f"rgba={t_rgba:.1f}ms send={t_send:.1f}ms TOTAL={t_total:.1f}ms "
                              f"(target={frame_interval*1000:.1f}ms) [FFT={fft_stats['fft_size']}]", flush=True)
                
                # Run inference every N frames
                self.inference_buffer.append(chunk)
                if len(self.inference_buffer) >= self.inference_chunk_count:
                    combined = np.concatenate([c.data for c in self.inference_buffer])
                    pts = self.inference_buffer[0].pts
                    frame_id = frame_count
                    self.inference_buffer.clear()
                    
                    # Fire-and-forget inference
                    asyncio.create_task(
                        self._run_inference_async(websocket, combined, pts, frame_id)
                    )
                
                frame_count += 1
                
                # Rate limit
                elapsed = time.perf_counter() - frame_start
                sleep_time = max(0.001, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Strip pipeline error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        logger.info(f"Strip pipeline stopped after {frame_count} frames, {self.total_rows_written} total rows")
    
    async def _run_inference_sync(self, websocket, iq_data, pts, frame_id):
        """Run inference SYNCHRONOUSLY (CUDA is not thread-safe with asyncio.to_thread)."""
        try:
            # Run directly - no threading
            result = self.pipeline.process_chunk(iq_data, pts)
            
            det_list = [{
                'detection_id': d.box_id,
                'x1': d.x1, 'y1': d.y1, 'x2': d.x2, 'y2': d.y2,
                'confidence': d.confidence,
                'class_id': d.class_id,
                'class_name': d.class_name,
            } for d in result['detections']]
            
            # STORE DETECTIONS for rendering on video frame
            self.latest_detections = det_list
            
            msg = json.dumps({
                'type': 'detection_frame',
                'frame_id': frame_id,
                'pts': result['pts'],
                'inference_ms': result['inference_ms'],
                'detections': det_list,
            })
            
            # Use MSG_DETECTION prefix
            await websocket.send(bytes([self.MSG_DETECTION]) + msg.encode())
            
        except Exception as e:
            logger.error(f"[INFERENCE ERROR] Frame {frame_id}: {e}")
    
    async def _run_inference_async(self, websocket, iq_data, pts, frame_id):
        """Run inference in background thread with timeout."""
        try:
            # Capture row state BEFORE inference (inference may take time)
            base_row = self.total_rows_written
            rows_per_single_frame = self.rows_this_frame if self.rows_this_frame > 0 else 38  # Default estimate
            
            # IMPORTANT: Inference runs on inference_chunk_count frames combined (6 frames)
            # So the total rows in the inference window is rows_per_frame × chunk_count
            total_inference_rows = rows_per_single_frame * self.inference_chunk_count
            
            # Add 30 second timeout to detect if inference is hanging
            # Use server's score_threshold (user configurable)
            result = await asyncio.wait_for(
                asyncio.to_thread(self.pipeline.process_chunk, iq_data, pts, self.score_threshold),
                timeout=30.0
            )
            
            # Build detection list WITH ROW OFFSET for perfect sync
            # Use the model's ACTUAL bounding box output directly
            det_list = [{
                'detection_id': d.box_id,
                'x1': d.x1, 'y1': d.y1, 'x2': d.x2, 'y2': d.y2,
                'confidence': d.confidence,
                'class_id': d.class_id,
                'class_name': d.class_name,
                # ROW SYNC: Direct conversion from model's x coordinates
                'row_offset': int(d.x1 * total_inference_rows),
                'row_span': max(1, int((d.x2 - d.x1) * total_inference_rows)),
            } for d in result['detections']]
            
            # STORE DETECTIONS for rendering on video frame
            self.latest_detections = det_list
            
            msg = json.dumps({
                'type': 'detection_frame',
                'frame_id': frame_id,
                'pts': result['pts'],
                'inference_ms': result['inference_ms'],
                # ROW SYNC: Send row tracking info for Flutter positioning
                'base_row': base_row,
                'rows_in_frame': total_inference_rows,  # Now includes all 6 frames worth of rows
                'detections': det_list,
            })
            
            # Use MSG_DETECTION prefix
            await websocket.send(bytes([self.MSG_DETECTION]) + msg.encode())
            
        except Exception as e:
            logger.error(f"[INFERENCE ERROR] Frame {frame_id}: {e}")
    
    def stop(self):
        """Stop the pipeline."""
        self.is_running = False
        self.iq_source.close()
        # No encoder in row-strip mode


async def video_ws_handler(websocket, iq_file: str, model_path: str):
    """WebSocket handler for video streaming."""
    server = VideoStreamServer(iq_file, model_path)
    pipeline_task = asyncio.create_task(server.run_pipeline(websocket))
    
    try:
        async for message in websocket:
            try:
                # Handle both bytes and string
                if isinstance(message, bytes):
                    message = message.decode('utf-8')
                
                if isinstance(message, str):
                    data = json.loads(message)
                    cmd = data.get('command')
            except Exception as e:
                logger.error(f"Failed to process message: {e}")
                continue
            
            if isinstance(message, str):
                data = json.loads(message)
                cmd = data.get('command')
                
                if cmd == 'stop':
                    server.stop()
                    break
                elif cmd == 'status':
                    await websocket.send(json.dumps({
                        'type': 'status',
                        'is_running': server.is_running,
                        'pts': server.current_pts,
                    }))
                elif cmd == 'set_time_span':
                    try:
                        seconds = float(data.get('seconds', 5.0))
                        # ROW-STRIP MODE: Flutter manages buffer, just update suggested size
                        new_suggested_height = int(seconds * server.video_fps * server.rows_per_frame)
                        
                        print(f"[Pipeline] Time span changing: {server.time_span_seconds}s -> {seconds}s (suggested buffer: {new_suggested_height} rows)", flush=True)
                        
                        # Update server state
                        server.time_span_seconds = seconds
                        server.suggested_buffer_height = new_suggested_height
                        
                        # Send updated metadata to client - Flutter will resize its buffer
                        metadata = {
                            'type': 'metadata',
                            'mode': 'row_strip',
                            'strip_width': server.video_width,
                            'rows_per_strip': server.rows_per_frame,
                            'video_fps': server.video_fps,
                            'suggested_buffer_height': new_suggested_height,
                            'time_span_seconds': seconds,
                            'encoder': 'rgba_raw',
                        }
                        await websocket.send(bytes([server.MSG_METADATA]) + json.dumps(metadata).encode())
                        print(f"[Pipeline] Metadata sent - Flutter should resize buffer to {new_suggested_height} rows", flush=True)
                        
                        # Send acknowledgment
                        await websocket.send(json.dumps({
                            'type': 'time_span_ack',
                            'seconds': seconds,
                            'suggested_buffer_height': new_suggested_height,
                        }))
                        print(f"[Pipeline] Ack sent - COMPLETE!", flush=True)
                        
                    except Exception as e:
                        print(f"[Pipeline] ERROR in set_time_span: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                elif cmd == 'set_fps':
                    try:
                        new_fps = int(data.get('fps', 30))
                        new_fps = max(1, min(60, new_fps))  # Clamp to 1-60
                        
                        old_fps = server.video_fps
                        server.video_fps = new_fps
                        
                        # DON'T recalculate buffer - keep it based on 30fps so display stays full
                        # Buffer stays the same, we just send slower
                        
                        print(f"[Pipeline] Send rate changing: {old_fps} -> {new_fps}fps (buffer unchanged at {server.suggested_buffer_height} rows)", flush=True)
                        
                        # Send acknowledgment only - no metadata update to avoid buffer resize
                        await websocket.send(json.dumps({
                            'type': 'fps_ack',
                            'fps': new_fps,
                        }))
                        print(f"[Pipeline] FPS change complete!", flush=True)
                        
                    except Exception as e:
                        print(f"[Pipeline] ERROR in set_fps: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                elif cmd == 'set_score_threshold':
                    try:
                        threshold = float(data.get('threshold', 0.5))
                        threshold = max(0.0, min(1.0, threshold))  # Clamp to 0-1
                        
                        old_threshold = server.score_threshold
                        server.score_threshold = threshold
                        
                        print(f"[Pipeline] Score threshold changing: {old_threshold:.2f} -> {threshold:.2f}", flush=True)
                        
                        # Send acknowledgment
                        await websocket.send(json.dumps({
                            'type': 'score_threshold_ack',
                            'threshold': threshold,
                        }))
                        print(f"[Pipeline] Score threshold change complete!", flush=True)
                        
                    except Exception as e:
                        print(f"[Pipeline] ERROR in set_score_threshold: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                
                elif cmd == 'set_fft_size':
                    try:
                        new_size = int(data.get('size', DEFAULT_FFT_SIZE))
                        
                        # Validate FFT size
                        if new_size not in VALID_FFT_SIZES:
                            print(f"[Pipeline] ERROR: Invalid FFT size {new_size}", flush=True)
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'command': 'set_fft_size',
                                'message': f'Invalid FFT size: {new_size}. Valid: {list(VALID_FFT_SIZES.keys())}'
                            }))
                            continue
                        
                        old_size = server.pipeline.waterfall_fft_size
                        print(f"[Pipeline] FFT size changing: {old_size} -> {new_size} (warmup in progress...)", flush=True)
                        
                        # Update FFT size (includes cuFFT warmup - may take 100-500ms!)
                        result = server.pipeline.update_waterfall_fft_size(new_size)
                        
                        if result['success']:
                            # rows_per_frame stays FIXED at TARGET_DISPLAY_ROWS (20)
                            # because we always decimate to ~20 rows regardless of FFT size
                            # This decouples FFT resolution from display bandwidth!
                            server.rows_per_frame = server.pipeline.TARGET_DISPLAY_ROWS
                            
                            # Buffer height also stays consistent (always 20 rows/frame)
                            server.suggested_buffer_height = int(server.time_span_seconds * server.video_fps * server.rows_per_frame)
                            
                            print(f"[Pipeline] FFT size changed: {old_size} -> {new_size} "
                                  f"(rows_per_frame={server.rows_per_frame}, warmup={result['warmup_ms']:.1f}ms)", flush=True)
                            
                            # Send updated metadata to client
                            metadata = {
                                'type': 'metadata',
                                'mode': 'row_strip',
                                'strip_width': server.video_width,
                                'rows_per_strip': server.rows_per_frame,
                                'video_fps': server.video_fps,
                                'suggested_buffer_height': server.suggested_buffer_height,
                                'time_span_seconds': server.time_span_seconds,
                                'encoder': 'rgba_raw',
                                'fft_size': new_size,
                            }
                            await websocket.send(bytes([server.MSG_METADATA]) + json.dumps(metadata).encode())
                            
                            # Send acknowledgment with timing info
                            await websocket.send(json.dumps({
                                'type': 'fft_size_ack',
                                'old_size': old_size,
                                'new_size': new_size,
                                'rows_per_frame': server.rows_per_frame,
                                'warmup_ms': result['warmup_ms'],
                                'description': VALID_FFT_SIZES[new_size],
                            }))
                            print(f"[Pipeline] FFT size change complete!", flush=True)
                        else:
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'command': 'set_fft_size',
                                'message': result.get('message', 'FFT size change failed')
                            }))
                        
                    except Exception as e:
                        print(f"[Pipeline] ERROR in set_fft_size: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'command': 'set_fft_size',
                            'message': str(e)
                        }))
    except Exception as e:
        logger.error(f"Video handler error: {e}")
    finally:
        server.stop()
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    import sys
    
    iq_file = str(DATA_DIR / "825MHz.sigmf-data")
    model_path = str(MODELS_DIR / "creamy_chicken_fold3.pth")
    
    if not os.path.exists(iq_file):
        print(f"IQ file not found: {iq_file}")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)
    
    print("Testing unified pipeline...")
    source = UnifiedIQSource(iq_file)
    pipeline = TripleBufferedPipeline(model_path)
    
    for i in range(10):
        chunk = source.read_chunk(duration_ms=200)
        if chunk:
            result = pipeline.process_chunk(chunk.data, chunk.pts)
            rgba, db, nf = pipeline.compute_waterfall_row_rgba(chunk.data)
            print(f"Frame {i}: PTS={chunk.pts:.3f}s, {len(result['detections'])} detections, {result['inference_ms']:.1f}ms, RGBA={len(rgba)} bytes, dB={len(db)} bytes")
    
    source.close()
    print("Done!")


# Alias for backward compatibility with server.py
# server.py imports UnifiedServer, but the class is actually VideoStreamServer
UnifiedServer = VideoStreamServer
