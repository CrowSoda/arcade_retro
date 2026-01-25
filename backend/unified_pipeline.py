"""
Unified IQ Pipeline - VIDEO STREAMING VERSION

ARCHITECTURE:
- Full frame waterfall buffer accumulated on backend
- H.264/NVENC encoding (or JPEG fallback)
- Single video stream + detection JSON over WebSocket

PERFORMANCE:
- Hardware video encoding: 2-6 Mbps vs 120+ Mbps raw pixels
- <60ms end-to-end latency
- High resolution 2048x1024 waterfall

INFERENCE: 4096 FFT, 2048 hop, 80dB (MUST MATCH TENSORCADE!)
WATERFALL: 8192 FFT, 4096 hop, 80dB (high resolution display)
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

import torch
import torch.nn.functional as F

# Global encoder instance for cleanup
_encoder_instance = None

def _cleanup():
    print("[Cleanup] Shutting down...", flush=True)
    if _encoder_instance:
        _encoder_instance.close()
    print("[Cleanup] Done", flush=True)

atexit.register(_cleanup)

def _signal_handler(sig, frame):
    print(f"[Signal] Received {sig}, exiting...", flush=True)
    _cleanup()
    sys.exit(0)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

# Import video streaming components
from video_encoder import create_encoder, BaseEncoder
from waterfall_buffer import WaterfallBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified_pipeline")

BASE_DIR = Path(__file__).parent.parent
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
        # WATERFALL PARAMS - FOR DISPLAY ONLY (fast)
        # =====================================================
        self.waterfall_fft_size = 32768  # 4× better frequency resolution (610 Hz/bin)
        self.waterfall_hop = 16384  # 50% overlap
        self.waterfall_dynamic_range = 60.0  # Better contrast for display
        
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
        self.waterfall_window = np.hanning(32768).astype(np.float32)
        
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
                    y1=float(box[1]) / 1024,
                    x2=float(box[2]) / 1024,
                    y2=float(box[3]) / 1024,
                    confidence=float(scores[i]),
                    class_id=int(labels[i]),
                    class_name=self.class_names[int(labels[i])] if int(labels[i]) < len(self.class_names) else 'unknown',
                    parent_pts=pts,
                )
                detections.append(det)
                # DEBUG: Print raw detection coordinates
                print(f"[DET RAW] box_id={det.box_id} x1={det.x1:.3f} y1={det.y1:.3f} x2={det.x2:.3f} y2={det.y2:.3f} class={det.class_name} conf={det.confidence:.2f} pts={det.parent_pts:.3f}", flush=True)
        
        inference_ms = (time.perf_counter() - start_time) * 1000
        self.current_stream = (self.current_stream + 1) % self.num_streams
        
        return {
            'detections': detections,
            'pts': pts,
            'inference_ms': inference_ms,
            'stream_idx': stream_idx,
        }
    
    def compute_waterfall_rows(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Compute STFT spectrogram - returns MULTIPLE rows, one per time slice.
        Each row is a full FFT, preserving both time and frequency resolution.
        """
        fft_size = self.waterfall_fft_size  # 8192
        hop_size = fft_size // 2  # 50% overlap = 4096
        
        # Calculate number of complete FFTs we can do
        num_ffts = (len(iq_data) - fft_size) // hop_size + 1
        
        if num_ffts < 1:
            # Not enough data for even one FFT
            return np.array([])
        
        # Pre-allocate output
        rows = np.zeros((num_ffts, fft_size), dtype=np.float32)
        
        for i in range(num_ffts):
            start = i * hop_size
            segment = iq_data[start:start + fft_size]
            
            # Window
            segment = segment * self.waterfall_window
            
            # FFT
            fft_data = np.fft.fft(segment)
            fft_data = np.fft.fftshift(fft_data)
            
            # Magnitude to dB
            mag = np.abs(fft_data)
            rows[i] = 20 * np.log10(mag + 1e-10)
        
        return rows  # Shape: (num_ffts, fft_size)
    
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


class WaterfallOnlyPipeline:
    """Lightweight pipeline for waterfall display only - NO model loading."""
    
    def __init__(self):
        self.waterfall_fft_size = 8192
        self.waterfall_hop = 4096
        self.waterfall_dynamic_range = 60.0
        self.noise_floor_db = -60.0
        self.noise_alpha = 0.02
        self.waterfall_window = np.hanning(8192).astype(np.float32)
    
    def compute_waterfall_row_rgba(self, iq_data: np.ndarray, target_width: int = 1024) -> tuple:
        """Compute waterfall row - returns (rgba_bytes, db_bytes, noise_floor_db)."""
        fft_size = self.waterfall_fft_size
        
        if len(iq_data) >= fft_size:
            segment = iq_data[-fft_size:]
        else:
            segment = np.pad(iq_data, (0, fft_size - len(iq_data)))
        
        segment = segment * self.waterfall_window
        fft_data = np.fft.fftshift(np.fft.fft(segment))
        db = 20 * np.log10(np.abs(fft_data) + 1e-6)
        
        # Max-pooling instead of decimation (preserves signal peaks)
        stride = fft_size // target_width
        if stride > 1:
            truncated_len = target_width * stride
            db_downsampled = db[:truncated_len].reshape(target_width, stride).max(axis=1).astype(np.float32)
        else:
            db_downsampled = db[:target_width].astype(np.float32)
        
        current_median = np.median(db_downsampled)
        self.noise_floor_db = self.noise_alpha * current_median + (1 - self.noise_alpha) * self.noise_floor_db
        
        min_db = self.noise_floor_db - 5
        max_db = self.noise_floor_db + self.waterfall_dynamic_range
        
        normalized = np.clip((db_downsampled - min_db) / (max_db - min_db), 0, 1)
        indices = (normalized * 255).astype(np.uint8)
        
        rgb = VIRIDIS_LUT[indices]
        rgba = np.zeros((target_width, 4), dtype=np.uint8)
        rgba[:, :3] = rgb
        rgba[:, 3] = 255
        
        return rgba.tobytes(), db_downsampled.tobytes(), self.noise_floor_db


class UnifiedServer:
    def __init__(self, iq_file: str, model_path: str):
        self.iq_source = UnifiedIQSource(iq_file)
        # Use lightweight waterfall-only pipeline - NO MODEL, NO GPU
        self.pipeline = WaterfallOnlyPipeline()
        
        self.is_running = False
        
        logger.info("UnifiedServer: WATERFALL ONLY mode (no inference)")
    
    async def run_pipeline(self, websocket):
        """
        Main loop - OPTIMIZED BINARY PROTOCOL
        
        Binary waterfall message format:
        - Byte 0: Message type (0x01 = waterfall)
        - Bytes 1-4: Sequence ID (uint32, little-endian)
        - Bytes 5-12: PTS (float64, little-endian)
        - Bytes 13-16: Width (uint32, little-endian)
        - Bytes 17 - (17 + width*4): RGBA pixel data (width * 4 bytes)
        - Remaining: Float32 dB values for PSD (width * 4 bytes)
        
        Detections still use JSON (infrequent, small)
        """
        import json
        
        self.is_running = True
        frame_count = 0
        
        logger.info("Pipeline started (30fps rate-limited)")
        
        while self.is_running:
            try:
                frame_start = time.perf_counter()  # For 30fps rate limiting
                
                chunk = self.iq_source.read_chunk(duration_ms=33)
                
                if chunk is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # Compute waterfall RGBA + dB (2048 pixels wide for high resolution)
                rgba_bytes, db_bytes, _ = self.pipeline.compute_waterfall_row_rgba(chunk.data, target_width=2048)
                
                # Pack binary header
                header = struct.pack('<BBBBI d I', 
                    0x01, 0x00, 0x00, 0x00,
                    chunk.sequence_id,
                    chunk.pts,
                    2048
                )
                
                # Send over WebSocket
                await websocket.send(header + rgba_bytes + db_bytes)
                
                # NO INFERENCE - UnifiedServer only does waterfall (video pipeline handles inference)
                
                frame_count += 1
                
                # 30fps rate limiting - sleep to hit 33ms per frame
                elapsed = time.perf_counter() - frame_start
                sleep_time = max(0.001, 0.033 - elapsed)  # Target 33ms = 30fps
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        logger.info(f"Pipeline stopped after {frame_count} frames")
    
    async def _run_inference_async(self, websocket, iq_data, pts, frame_id):
        """Run inference in background thread - truly doesn't block waterfall loop."""
        import json
        
        try:
            # MUST use to_thread() - otherwise process_chunk() blocks the event loop!
            result = await asyncio.to_thread(self.pipeline.process_chunk, iq_data, pts)
            
            det_list = [{
                'detection_id': d.box_id,
                'x1': d.x1, 'y1': d.y1, 'x2': d.x2, 'y2': d.y2,
                'confidence': d.confidence,
                'class_id': d.class_id,
                'class_name': d.class_name,
            } for d in result['detections']]
            
            msg = json.dumps({
                'type': 'detection_frame',
                'frame_id': frame_id,
                'pts': result['pts'],
                'inference_ms': result['inference_ms'],
                'detections': det_list,
            })
            
            await websocket.send(msg)
            
        except Exception as e:
            logger.error(f"[INFERENCE ERROR] Frame {frame_id}: {e}")
    
    def stop(self):
        self.is_running = False
        self.iq_source.close()


async def unified_ws_handler(websocket, iq_file: str, model_path: str):
    """Legacy handler - uses row-by-row streaming."""
    server = UnifiedServer(iq_file, model_path)
    pipeline_task = asyncio.create_task(server.run_pipeline(websocket))
    
    try:
        async for message in websocket:
            # Only handle text messages (commands)
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
                    }))
    except Exception as e:
        logger.error(f"Handler error: {e}")
    finally:
        server.stop()
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass


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
        
        # Strip parameters
        self.video_width = video_width
        self.video_fps = video_fps
        self.time_span_seconds = time_span_seconds
        
        # Calculate rows_per_frame from FFT settings
        # At 20MHz sample rate, 33ms = 660,000 samples
        # With FFT=32768, hop=16384: num_ffts = (660000 - 32768) / 16384 + 1 ≈ 38
        sample_rate = 20e6
        chunk_duration_ms = 33.0
        samples_per_chunk = int(sample_rate * chunk_duration_ms / 1000)  # 660,000
        fft_size = self.pipeline.waterfall_fft_size  # 32768
        hop_size = fft_size // 2  # 16384
        self.rows_per_frame = max(1, (samples_per_chunk - fft_size) // hop_size + 1)  # ~38
        
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
        
        # Row tracking for detection synchronization
        self.total_rows_written = 0  # Monotonic counter, never resets
        self.rows_this_frame = 0     # Rows added in current frame
        
        # Suggested buffer height for Flutter client
        self.suggested_buffer_height = int(time_span_seconds * video_fps * self.rows_per_frame)
        
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
        
        frame_interval = 1.0 / self.video_fps
        
        while self.is_running:
            try:
                frame_start = time.perf_counter()
                
                # Read IQ chunk
                chunk = self.iq_source.read_chunk(duration_ms=33)
                if chunk is None:
                    await asyncio.sleep(0.01)
                    continue
                
                self.current_pts = chunk.pts
                
                # Compute FFT rows (multiple rows per chunk)
                db_rows = self.pipeline.compute_waterfall_rows(chunk.data)
                self.rows_this_frame = len(db_rows)
                
                if self.rows_this_frame > 0:
                    # Convert dB to RGBA pixels
                    rgba_strip = self._db_to_rgba(db_rows, target_width=self.video_width)
                    
                    # Pack strip message: header + raw RGBA bytes
                    # Header: 16 bytes
                    #   - frame_id: uint32
                    #   - total_rows: uint32 (monotonic counter)
                    #   - rows_in_strip: uint16
                    #   - strip_width: uint16
                    #   - pts: float32
                    header = struct.pack('<I I H H f',
                        frame_count,
                        self.total_rows_written,
                        self.rows_this_frame,
                        self.video_width,
                        self.current_pts
                    )
                    
                    # Send strip: MSG_STRIP + header + RGBA bytes
                    strip_bytes = rgba_strip.tobytes()
                    await websocket.send(bytes([self.MSG_STRIP]) + header + strip_bytes)
                    
                    # Update row counter AFTER sending
                    self.total_rows_written += self.rows_this_frame
                    
                    if frame_count % 30 == 0:
                        logger.info(f"[STRIP] Frame {frame_count}: {self.rows_this_frame} rows, {len(strip_bytes)} bytes, total_rows={self.total_rows_written}")
                
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
            rows_in_frame = self.rows_this_frame if self.rows_this_frame > 0 else 38  # Default estimate
            
            # Add 30 second timeout to detect if inference is hanging
            result = await asyncio.wait_for(
                asyncio.to_thread(self.pipeline.process_chunk, iq_data, pts),
                timeout=30.0
            )
            
            # Build detection list WITH ROW OFFSET for perfect sync
            # Detection x1/x2 is time axis (0-1), maps to row_offset within frame
            det_list = [{
                'detection_id': d.box_id,
                'x1': d.x1, 'y1': d.y1, 'x2': d.x2, 'y2': d.y2,
                'confidence': d.confidence,
                'class_id': d.class_id,
                'class_name': d.class_name,
                # ROW SYNC: x1 is time position (0-1), maps to row within frame
                'row_offset': int(d.x1 * rows_in_frame),
                'row_span': max(1, int((d.x2 - d.x1) * rows_in_frame)),
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
                'rows_in_frame': rows_in_frame,
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
            print(f"[WS RECV] *** MESSAGE RECEIVED ***", flush=True)
            print(f"[WS RECV] Type: {type(message)}", flush=True)
            
            try:
                # Handle both bytes and string
                if isinstance(message, bytes):
                    print(f"[WS RECV] Got BYTES ({len(message)} bytes), decoding...", flush=True)
                    message = message.decode('utf-8')
                
                if isinstance(message, str):
                    print(f"[WS RECV] Text message: {message[:200]}", flush=True)
                    data = json.loads(message)
                    cmd = data.get('command')
                    print(f"[WS RECV] Command: {cmd}", flush=True)
            except Exception as e:
                print(f"[WS RECV ERROR] Failed to process message: {e}", flush=True)
                import traceback
                traceback.print_exc()
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
