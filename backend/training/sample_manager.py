"""
SampleManager - Save and manage training samples.

Handles the flow: Real-world box coords → centered IQ extraction → spectrogram → saved files

CRITICAL FIX (Jan 2026): 
- Flutter now sends REAL UNITS (seconds, MHz) instead of normalized 0-1 coords
- Python extracts IQ CENTERED on each box and computes its own spectrogram
- This ensures training data matches what the model sees during inference

The key insight is that Flutter's spectrogram differs from Python's inference spectrogram
(different FFT params, different time windows). By using real-world units as the source
of truth, Python can generate the CORRECT spectrogram for training.
"""

import os
import sys
import json
import base64
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import torch

# Force unbuffered stdout so prints show immediately
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

def debug_log(msg: str):
    """Write to both stdout and debug log file."""
    print(f"[DEBUG] {msg}", flush=True)
    log_path = Path("training_data/COORD_DEBUG.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")

from hydra.config import (
    INFERENCE_FFT_SIZE,
    INFERENCE_HOP_LENGTH,
    INFERENCE_DYNAMIC_RANGE_DB,
    INFERENCE_OUTPUT_SIZE,
)

# Fixed training window duration (100ms is typical for signal detection)
TRAINING_WINDOW_SEC = 0.1


class SampleManager:
    """Manages training sample storage with real-world coordinate conversion."""
    
    def __init__(
        self, 
        training_data_dir: str = "training_data/signals", 
        device: str = "cuda"
    ):
        self.base_dir = Path(training_data_dir)
        self.device = device
        self._spec_pipeline = None
    
    def compute_spectrogram(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Compute spectrogram using INFERENCE FFT parameters.
        
        Args:
            iq_data: Complex64 IQ samples
        
        Returns:
            uint8 spectrogram array (1024, 1024)
            Y=0 is HIGH frequency (top of image)
        """
        # FFT parameters - MUST match inference.py
        nfft = INFERENCE_FFT_SIZE
        hop = INFERENCE_HOP_LENGTH
        dynamic_range = INFERENCE_DYNAMIC_RANGE_DB
        
        # Compute STFT
        num_frames = (len(iq_data) - nfft) // hop + 1
        if num_frames < 1:
            # Pad if needed
            iq_data = np.pad(iq_data, (0, nfft - len(iq_data) + 1), mode='constant')
            num_frames = 1
        
        # Window function
        window = np.hanning(nfft).astype(np.float32)
        
        # Compute frames
        frames = []
        for i in range(num_frames):
            start = i * hop
            frame = iq_data[start:start + nfft] * window
            spectrum = np.fft.fftshift(np.fft.fft(frame))
            power = np.abs(spectrum) ** 2 + 1e-10
            power_db = 10 * np.log10(power)
            frames.append(power_db)
        
        spectrogram = np.stack(frames, axis=1)  # (nfft, num_frames)
        
        # Normalize to dynamic range
        max_val = spectrogram.max()
        min_val = max_val - dynamic_range
        spectrogram = np.clip(spectrogram, min_val, max_val)
        spectrogram = (spectrogram - min_val) / dynamic_range  # 0-1
        
        # Resize to output size
        from scipy.ndimage import zoom
        target_h, target_w = INFERENCE_OUTPUT_SIZE
        zoom_factors = (target_h / spectrogram.shape[0], target_w / spectrogram.shape[1])
        spectrogram = zoom(spectrogram, zoom_factors, order=1)
        
        # CRITICAL: Flip Y axis so Y=0 is HIGH frequency (matches TensorCade convention)
        spectrogram = np.flipud(spectrogram)
        
        # Convert to uint8
        spectrogram = (spectrogram * 255).astype(np.uint8)
        
        return spectrogram
    
    def _box_to_pixels(
        self,
        box: Dict,
        window_start_sec: float,
        window_duration_sec: float,
        bandwidth_hz: float,
        center_freq_hz: float,
        output_size: int = 1024,
    ) -> Dict:
        """
        Convert box from real units (seconds, MHz) to pixel coordinates.
        
        This is the CRITICAL function that ensures boxes land on the correct
        location in Python's spectrogram (not Flutter's).
        
        Args:
            box: Dict with time_start_sec, time_end_sec, freq_start_mhz, freq_end_mhz
            window_start_sec: Start time of the IQ window we extracted
            window_duration_sec: Duration of the IQ window
            bandwidth_hz: Actual RF bandwidth in Hz (from header, NOT sample_rate!)
            center_freq_hz: Center frequency in Hz
            output_size: Spectrogram size (1024)
        
        Returns:
            Dict with x_min, y_min, x_max, y_max in pixels
        """
        # Calculate frequency bounds
        freq_min_hz = center_freq_hz - bandwidth_hz / 2
        freq_max_hz = center_freq_hz + bandwidth_hz / 2
        
        # DEBUG: Write to log file for debugging coordinate issues
        debug_log = Path("training_data/coord_debug.log")
        debug_log.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_log, 'a') as f:
            f.write(f"[_box_to_pixels] INPUT: box_freq={box['freq_start_mhz']:.2f}-{box['freq_end_mhz']:.2f} MHz\n")
            f.write(f"[_box_to_pixels] PARAMS: center={center_freq_hz/1e6:.2f} MHz, bw={bandwidth_hz/1e6:.2f} MHz\n")
            f.write(f"[_box_to_pixels] RANGE: freq_min={freq_min_hz/1e6:.2f}, freq_max={freq_max_hz/1e6:.2f} MHz\n")
        
        # === TIME → X pixels ===
        # Box time relative to our extracted window
        rel_time_start = box['time_start_sec'] - window_start_sec
        rel_time_end = box['time_end_sec'] - window_start_sec
        
        # Clamp to window bounds
        rel_time_start = max(0, min(rel_time_start, window_duration_sec))
        rel_time_end = max(0, min(rel_time_end, window_duration_sec))
        
        # Convert to pixels
        x_min = (rel_time_start / window_duration_sec) * output_size
        x_max = (rel_time_end / window_duration_sec) * output_size
        
        # === FREQUENCY → Y pixels ===
        freq_min_hz = center_freq_hz - bandwidth_hz / 2
        freq_max_hz = center_freq_hz + bandwidth_hz / 2
        
        # Box frequencies in Hz
        box_freq_low_hz = box['freq_start_mhz'] * 1e6
        box_freq_high_hz = box['freq_end_mhz'] * 1e6
        
        # Clamp to valid range
        box_freq_low_hz = max(freq_min_hz, min(box_freq_low_hz, freq_max_hz))
        box_freq_high_hz = max(freq_min_hz, min(box_freq_high_hz, freq_max_hz))
        
        # Normalize to 0-1 (0 = lowest freq, 1 = highest freq)
        freq_frac_low = (box_freq_low_hz - freq_min_hz) / bandwidth_hz
        freq_frac_high = (box_freq_high_hz - freq_min_hz) / bandwidth_hz
        
        # After flipud: Y=0 is HIGH freq, Y=output_size is LOW freq
        # So invert: high freq → low Y value
        y_min = (1 - freq_frac_high) * output_size
        y_max = (1 - freq_frac_low) * output_size
        
        # Ensure min < max and convert to int
        return {
            'x_min': int(min(x_min, x_max)),
            'x_max': int(max(x_min, x_max)),
            'y_min': int(min(y_min, y_max)),
            'y_max': int(max(y_min, y_max)),
        }
    
    def _generate_sample_id(
        self,
        signal_name: str,
        source_file: str,
        time_offset_sec: float
    ) -> str:
        """
        Generate deterministic sample ID from content hash.
        
        Same segment = same ID, always. This means:
        - Labeling same segment twice → overwrites (not appends)
        - Different segment → new file
        - Re-run training on same file → same IDs → no duplicates
        """
        # Use milliseconds for time offset to avoid float precision issues
        time_offset_ms = int(time_offset_sec * 1000)
        
        # Get just the filename (not full path) for consistency
        source_basename = os.path.basename(source_file) if source_file else "unknown"
        
        # Create deterministic key
        key = f"{signal_name}:{source_basename}:{time_offset_ms}"
        
        # Hash to 16 hex chars
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def save_sample(
        self,
        signal_name: str,
        iq_data_b64: str,
        boxes: List[Dict],
        metadata: Dict
    ) -> Tuple[str, bool]:
        """
        Save a training sample with REAL-WORLD coordinate conversion.
        
        NEW BEHAVIOR (Jan 2026):
        - Boxes contain time_start_sec, time_end_sec, freq_start_mhz, freq_end_mhz
        - We extract IQ centered on the box (if rfcap_path provided)
        - We compute our own spectrogram with inference FFT params
        - We convert real units to pixels for OUR spectrogram
        
        Args:
            signal_name: Signal class name
            iq_data_b64: Base64-encoded IQ data (complex64) - fallback if no rfcap_path
            boxes: List of box dicts with REAL UNITS (time_start_sec, freq_start_mhz, etc.)
            metadata: Source file info, frequencies, etc.
        
        Returns:
            (sample_id, is_new) - is_new=False if file already existed (overwritten)
        """
        # Create directories
        signal_dir = self.base_dir / signal_name
        samples_dir = signal_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Get metadata
        source_file = metadata.get("source_file", metadata.get("rfcap_path", ""))
        time_offset = metadata.get("time_offset_sec", 0.0)
        center_freq_hz = metadata.get("center_freq_hz", metadata.get("center_freq_mhz", 825.0) * 1e6)
        sample_rate = metadata.get("sample_rate", metadata.get("sample_rate_mhz", 20.0) * 1e6)
        # CRITICAL FIX: The spectrogram ACTUAL bandwidth is determined by sample_rate (Nyquist),
        # NOT by the header's "bandwidth" field (which may be wrong or represent something else).
        # The FFT computes frequencies from -sample_rate/2 to +sample_rate/2.
        # So we MUST use sample_rate as the bandwidth for coordinate conversion!
        bandwidth_hz = sample_rate  # FFT spans -fs/2 to +fs/2, so bandwidth = sample_rate
        
        # Log what the file claimed vs what we're using
        file_claimed_bw = metadata.get("bandwidth_hz", metadata.get("bandwidth_mhz", 0) * 1e6)
        debug_log(f"NOTE: File header claimed bandwidth={file_claimed_bw/1e6:.2f} MHz, using sample_rate={bandwidth_hz/1e6:.2f} MHz for spectrogram")
        
        # Generate deterministic sample ID
        sample_id = self._generate_sample_id(signal_name, source_file, time_offset)
        
        # Check if this exact sample already exists
        npz_path = samples_dir / f"{sample_id}.npz"
        is_new = not npz_path.exists()
        
        # Determine IQ extraction strategy
        # Check if boxes have real-world units (new format)
        has_real_units = boxes and 'time_start_sec' in boxes[0]
        
        if has_real_units and boxes:
            # NEW FORMAT: Extract IQ centered on box, compute fresh pixel coords
            box = boxes[0]  # Use first box for centering
            
            # Calculate centered window
            box_time_start = box['time_start_sec']
            box_time_end = box['time_end_sec']
            box_center = (box_time_start + box_time_end) / 2
            
            window_start = box_center - TRAINING_WINDOW_SEC / 2
            window_end = box_center + TRAINING_WINDOW_SEC / 2
            
            # Clamp to file bounds
            if window_start < 0:
                window_start = 0
                window_end = TRAINING_WINDOW_SEC
            
            window_duration = window_end - window_start
            
            # Try to read IQ from file if path provided
            rfcap_path = metadata.get("rfcap_path")
            if rfcap_path and os.path.exists(rfcap_path):
                try:
                    iq_data = self._read_iq_from_rfcap(
                        rfcap_path, 
                        int(window_start * sample_rate),
                        int(window_duration * sample_rate)
                    )
                    print(f"[SampleManager] Extracted {len(iq_data)} samples centered on box")
                except Exception as e:
                    print(f"[SampleManager] Failed to read rfcap, using provided IQ: {e}")
                    iq_bytes = base64.b64decode(iq_data_b64)
                    iq_data = np.frombuffer(iq_bytes, dtype=np.complex64)
                    window_start = time_offset
                    window_duration = len(iq_data) / sample_rate
            else:
                # Use provided IQ data
                iq_bytes = base64.b64decode(iq_data_b64)
                iq_data = np.frombuffer(iq_bytes, dtype=np.complex64)
                window_start = time_offset
                window_duration = len(iq_data) / sample_rate
            
            # Compute spectrogram
            spec_uint8 = self.compute_spectrogram(iq_data)
            
            # Convert real-unit boxes to pixel coordinates for OUR spectrogram
            pixel_boxes = []
            h, w = INFERENCE_OUTPUT_SIZE
            
            # CRITICAL: Flutter only shows POSITIVE frequencies (DC to +fs/2)
            # because _spectrogramHeight = fftSize ~/ 2.
            # The displayed frequency range is: center_freq (DC) to center_freq + sample_rate/2
            # (NOT center_freq - sample_rate/2 to center_freq + sample_rate/2!)
            freq_min_hz = center_freq_hz  # DC (bottom of Flutter display)
            freq_range_hz = bandwidth_hz / 2  # Only positive frequencies displayed
            freq_max_hz = center_freq_hz + freq_range_hz
            
            debug_log("=" * 60)
            debug_log(f"COORDINATE CONVERSION for {signal_name}")
            debug_log(f"  POSITIVE FREQ ONLY: center={center_freq_hz/1e6:.2f} MHz, half_bw={freq_range_hz/1e6:.2f} MHz")
            debug_log(f"  Displayed range: {freq_min_hz/1e6:.2f} - {freq_max_hz/1e6:.2f} MHz")
            debug_log(f"  Time window: {window_start:.4f}s - {window_start+window_duration:.4f}s ({window_duration*1000:.1f}ms)")
            debug_log(f"  Output size: {w}x{h}")
            
            for i, box in enumerate(boxes):
                debug_log(f"--- BOX {i} ---")
                debug_log(f"  FLUTTER SENT:")
                debug_log(f"    time: {box['time_start_sec']:.4f}s - {box['time_end_sec']:.4f}s")
                debug_log(f"    freq: {box['freq_start_mhz']:.4f} MHz - {box['freq_end_mhz']:.4f} MHz")
                
                # === TIME → X pixels ===
                rel_time_start = box['time_start_sec'] - window_start
                rel_time_end = box['time_end_sec'] - window_start
                
                debug_log(f"  TIME CONVERSION:")
                debug_log(f"    relative to window: {rel_time_start:.4f}s - {rel_time_end:.4f}s")
                
                # Clamp to window bounds
                rel_time_start = max(0, min(rel_time_start, window_duration))
                rel_time_end = max(0, min(rel_time_end, window_duration))
                
                # Convert to pixels: x = (relative_time / window_duration) * width
                x_min = (rel_time_start / window_duration) * w
                x_max = (rel_time_end / window_duration) * w
                debug_log(f"    X pixels: {x_min:.0f} - {x_max:.0f}")
                
                # === FREQUENCY → Y pixels ===
                box_freq_low_hz = box['freq_start_mhz'] * 1e6
                box_freq_high_hz = box['freq_end_mhz'] * 1e6
                
                debug_log(f"  FREQ CONVERSION:")
                debug_log(f"    box freq Hz: {box_freq_low_hz/1e6:.4f} - {box_freq_high_hz/1e6:.4f} MHz")
                
                # Normalize to 0-1 within the band: freq_frac = (freq - freq_min) / bandwidth
                freq_frac_low = (box_freq_low_hz - freq_min_hz) / bandwidth_hz
                freq_frac_high = (box_freq_high_hz - freq_min_hz) / bandwidth_hz
                
                debug_log(f"    normalized (0=low freq, 1=high freq): {freq_frac_low:.4f} - {freq_frac_high:.4f}")
                
                # Clamp to valid range
                freq_frac_low = max(0, min(1, freq_frac_low))
                freq_frac_high = max(0, min(1, freq_frac_high))
                
                # Y=0 is HIGH freq (after flipud), Y=1023 is LOW freq
                # So: y = (1 - freq_frac) * height
                y_from_low = (1 - freq_frac_low) * h
                y_from_high = (1 - freq_frac_high) * h
                
                debug_log(f"    Y from low freq: (1-{freq_frac_low:.4f})*{h} = {y_from_low:.0f}")
                debug_log(f"    Y from high freq: (1-{freq_frac_high:.4f})*{h} = {y_from_high:.0f}")
                
                y_min = min(y_from_low, y_from_high)
                y_max = max(y_from_low, y_from_high)
                
                debug_log(f"    Y pixels: {y_min:.0f} - {y_max:.0f}")
                
                pixel_box = {
                    'x_min': int(min(x_min, x_max)),
                    'x_max': int(max(x_min, x_max)),
                    'y_min': int(y_min),
                    'y_max': int(y_max),
                }
                
                debug_log(f"  FINAL PIXEL BOX: x={pixel_box['x_min']}-{pixel_box['x_max']}, y={pixel_box['y_min']}-{pixel_box['y_max']}")
                
                # Ensure positive width/height
                if pixel_box['x_max'] <= pixel_box['x_min']:
                    pixel_box['x_max'] = pixel_box['x_min'] + 1
                if pixel_box['y_max'] <= pixel_box['y_min']:
                    pixel_box['y_max'] = pixel_box['y_min'] + 1
                
                # Clamp to valid range
                pixel_box['x_min'] = max(0, pixel_box['x_min'])
                pixel_box['y_min'] = max(0, pixel_box['y_min'])
                pixel_box['x_max'] = min(w - 1, pixel_box['x_max'])
                pixel_box['y_max'] = min(h - 1, pixel_box['y_max'])
                
                pixel_box['label'] = 'signal'
                pixel_box['confidence'] = None
                pixel_boxes.append(pixel_box)
                
                print(f"[SampleManager] Box real: t={box['time_start_sec']:.3f}-{box['time_end_sec']:.3f}s, "
                      f"f={box['freq_start_mhz']:.2f}-{box['freq_end_mhz']:.2f}MHz")
                print(f"[SampleManager] Box pixel: x={pixel_box['x_min']}-{pixel_box['x_max']}, "
                      f"y={pixel_box['y_min']}-{pixel_box['y_max']}")
        else:
            # OLD FORMAT: Use provided IQ and normalized coords (backwards compat)
            print("[SampleManager] WARNING: Using old normalized coord format - results may be incorrect!")
            
            iq_bytes = base64.b64decode(iq_data_b64)
            iq_data = np.frombuffer(iq_bytes, dtype=np.complex64)
            
            spec_uint8 = self.compute_spectrogram(iq_data)
            
            window_start = time_offset
            window_duration = len(iq_data) / sample_rate
            
            # Convert normalized boxes to pixel coordinates (old method)
            pixel_boxes = []
            h, w = INFERENCE_OUTPUT_SIZE
            for box in boxes:
                x1_px = int(box.get("x1", 0) * w)
                y1_px = int(box.get("y1", 0) * h)
                x2_px = int(box.get("x2", 1) * w)
                y2_px = int(box.get("y2", 1) * h)
                
                x_min = min(x1_px, x2_px)
                x_max = max(x1_px, x2_px)
                y_min = min(y1_px, y2_px)
                y_max = max(y1_px, y2_px)
                
                if x_max <= x_min:
                    x_max = x_min + 1
                if y_max <= y_min:
                    y_max = y_min + 1
                
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w - 1, x_max)
                y_max = min(h - 1, y_max)
                
                pixel_boxes.append({
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "label": "signal",
                    "confidence": None,
                })
        
        # DEBUG OFF: Spectrogram debug PNG generation disabled
        # Uncomment the block below to re-enable debug output
        # debug_dir = self.base_dir.parent / "debug_spectrograms"
        # debug_dir.mkdir(parents=True, exist_ok=True)
        # ... (debug code removed for performance)
        
        # Save spectrogram
        npz_path = samples_dir / f"{sample_id}.npz"
        np.savez_compressed(npz_path, spectrogram=spec_uint8)
        
        # Save metadata
        sample_metadata = {
            "sample_id": sample_id,
            "source_capture": metadata.get("source_file"),
            "time_offset_sec": metadata.get("time_offset_sec"),
            "duration_sec": metadata.get("duration_sec"),
            "window_start_sec": window_start,
            "window_duration_sec": window_duration,
            "center_freq_mhz": center_freq_hz / 1e6,
            "center_freq_hz": center_freq_hz,
            "sample_rate_mhz": sample_rate / 1e6,
            "sample_rate": sample_rate,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "created_by": "user_label",
            "coordinate_format": "real_units" if has_real_units else "normalized_legacy",
            "spectrogram_params": {
                "fft_size": INFERENCE_FFT_SIZE,
                "hop_length": INFERENCE_HOP_LENGTH,
                "dynamic_range_db": INFERENCE_DYNAMIC_RANGE_DB,
                "output_size": list(INFERENCE_OUTPUT_SIZE),
            },
            "boxes": pixel_boxes,
            "original_boxes": boxes,  # Keep original for debugging
        }
        
        json_path = samples_dir / f"{sample_id}.json"
        with open(json_path, "w") as f:
            json.dump(sample_metadata, f, indent=2)
        
        # Update manifest
        self._update_manifest(signal_name, sample_id, metadata)
        
        return (sample_id, is_new)
    
    def _read_iq_from_rfcap(
        self, 
        rfcap_path: str, 
        start_sample: int, 
        num_samples: int
    ) -> np.ndarray:
        """
        Read IQ data from RFCAP file.
        
        Args:
            rfcap_path: Path to .rfcap file
            start_sample: Starting sample index
            num_samples: Number of complex samples to read
        
        Returns:
            Complex64 numpy array
        """
        # RFCAP format: 512-byte header, then interleaved I/Q float32
        HEADER_SIZE = 512
        BYTES_PER_SAMPLE = 8  # 4 bytes I + 4 bytes Q
        
        with open(rfcap_path, 'rb') as f:
            # Skip header and seek to start position
            offset = HEADER_SIZE + start_sample * BYTES_PER_SAMPLE
            f.seek(offset)
            
            # Read raw bytes
            raw = f.read(num_samples * BYTES_PER_SAMPLE)
            
            # Parse as complex64
            iq_data = np.frombuffer(raw, dtype=np.complex64)
        
        return iq_data
    
    def _update_manifest(self, signal_name: str, sample_id: str, metadata: Dict):
        """Update the signal's manifest file."""
        signal_dir = self.base_dir / signal_name
        manifest_path = signal_dir / "manifest.json"
        
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        else:
            manifest = {
                "signal_name": signal_name,
                "total_samples": 0,
                "sources": [],
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
        
        manifest["total_samples"] += 1
        manifest["last_updated"] = datetime.utcnow().isoformat() + "Z"
        
        # Track source file
        source_file = metadata.get("source_file")
        if source_file:
            source_entry = next(
                (s for s in manifest["sources"] if s["capture_file"] == source_file), 
                None
            )
            if source_entry:
                source_entry["sample_count"] += 1
            else:
                manifest["sources"].append({
                    "capture_file": source_file,
                    "sample_count": 1,
                    "added_at": datetime.utcnow().isoformat() + "Z",
                })
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    
    def get_sample_count(self, signal_name: str) -> int:
        """Get total sample count for a signal."""
        samples_dir = self.base_dir / signal_name / "samples"
        if not samples_dir.exists():
            return 0
        return len(list(samples_dir.glob("*.npz")))
    
    def get_sample_info(self, signal_name: str, sample_id: str) -> Optional[Dict]:
        """Get metadata for a specific sample."""
        json_path = self.base_dir / signal_name / "samples" / f"{sample_id}.json"
        if not json_path.exists():
            return None
        with open(json_path) as f:
            return json.load(f)
    
    def list_samples(self, signal_name: str) -> List[Dict]:
        """List all samples for a signal with basic info."""
        samples_dir = self.base_dir / signal_name / "samples"
        if not samples_dir.exists():
            return []
        
        samples = []
        for json_file in sorted(samples_dir.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            samples.append({
                "sample_id": data["sample_id"],
                "source_capture": data.get("source_capture"),
                "time_offset_sec": data.get("time_offset_sec"),
                "created_at": data.get("created_at"),
                "box_count": len(data.get("boxes", [])),
            })
        
        return samples
    
    def delete_sample(self, signal_name: str, sample_id: str) -> bool:
        """Delete a sample."""
        samples_dir = self.base_dir / signal_name / "samples"
        npz_path = samples_dir / f"{sample_id}.npz"
        json_path = samples_dir / f"{sample_id}.json"
        
        deleted = False
        if npz_path.exists():
            npz_path.unlink()
            deleted = True
        if json_path.exists():
            json_path.unlink()
            deleted = True
        
        # Update manifest count
        if deleted:
            manifest_path = self.base_dir / signal_name / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                manifest["total_samples"] = max(0, manifest["total_samples"] - 1)
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
        
        return deleted
    
    def get_manifest(self, signal_name: str) -> Optional[Dict]:
        """Get manifest for a signal."""
        manifest_path = self.base_dir / signal_name / "manifest.json"
        if not manifest_path.exists():
            return None
        with open(manifest_path) as f:
            return json.load(f)
