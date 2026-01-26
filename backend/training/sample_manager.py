"""
SampleManager - Save and manage training samples.

Handles the flow: IQ data + boxes → computed spectrogram → saved files

CRITICAL: Spectrograms are computed by the BACKEND using INFERENCE FFT params
to ensure training data matches what the model sees during inference.
"""

import os
import json
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import torch

from ..hydra.config import (
    INFERENCE_FFT_SIZE,
    INFERENCE_HOP_LENGTH,
    INFERENCE_DYNAMIC_RANGE_DB,
    INFERENCE_OUTPUT_SIZE,
)


class SampleManager:
    """Manages training sample storage."""
    
    def __init__(
        self, 
        training_data_dir: str = "training_data/signals", 
        device: str = "cuda"
    ):
        self.base_dir = Path(training_data_dir)
        self.device = device
        self._spec_pipeline = None
    
    def _get_spec_pipeline(self):
        """Lazy-load the spectrogram pipeline."""
        if self._spec_pipeline is None:
            # Import here to avoid circular dependency
            from ..unified_pipeline import TripleBufferedPipeline
            # We just need the spectrogram computation, not the full pipeline
            # For now, use a simple implementation
            pass
        return self._spec_pipeline
    
    def compute_spectrogram(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Compute spectrogram using INFERENCE FFT parameters.
        
        Args:
            iq_data: Complex64 IQ samples
        
        Returns:
            uint8 spectrogram array (1024, 1024)
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
        
        # Convert to uint8
        spectrogram = (spectrogram * 255).astype(np.uint8)
        
        return spectrogram
    
    def save_sample(
        self,
        signal_name: str,
        iq_data_b64: str,
        boxes: List[Dict],
        metadata: Dict
    ) -> str:
        """
        Save a training sample.
        
        Args:
            signal_name: Signal class name
            iq_data_b64: Base64-encoded IQ data (complex64)
            boxes: List of normalized box dicts [{x1, y1, x2, y2}, ...]
            metadata: Source file info, frequencies, etc.
        
        Returns:
            Sample ID (e.g., "0042")
        """
        # Create directories
        signal_dir = self.base_dir / signal_name
        samples_dir = signal_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Decode IQ data
        iq_bytes = base64.b64decode(iq_data_b64)
        iq_data = np.frombuffer(iq_bytes, dtype=np.complex64)
        
        # Compute spectrogram using INFERENCE params
        spec_uint8 = self.compute_spectrogram(iq_data)
        
        # Generate sample ID
        existing = list(samples_dir.glob("*.npz"))
        next_id = len(existing) + 1
        sample_id = f"{next_id:04d}"
        
        # Convert normalized boxes to pixel coordinates
        pixel_boxes = []
        h, w = INFERENCE_OUTPUT_SIZE
        for box in boxes:
            pixel_boxes.append({
                "x_min": int(box["x1"] * w),
                "y_min": int(box["y1"] * h),
                "x_max": int(box["x2"] * w),
                "y_max": int(box["y2"] * h),
                "label": "signal",
                "confidence": None,
            })
        
        # Save spectrogram
        npz_path = samples_dir / f"{sample_id}.npz"
        np.savez_compressed(npz_path, spectrogram=spec_uint8)
        
        # Save metadata
        sample_metadata = {
            "sample_id": sample_id,
            "source_capture": metadata.get("source_file"),
            "time_offset_sec": metadata.get("time_offset_sec"),
            "duration_sec": metadata.get("duration_sec"),
            "center_freq_mhz": metadata.get("center_freq_mhz"),
            "sample_rate_mhz": metadata.get("sample_rate_mhz"),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "created_by": "user_label",
            "spectrogram_params": {
                "fft_size": INFERENCE_FFT_SIZE,
                "hop_length": INFERENCE_HOP_LENGTH,
                "dynamic_range_db": INFERENCE_DYNAMIC_RANGE_DB,
                "output_size": list(INFERENCE_OUTPUT_SIZE),
            },
            "boxes": pixel_boxes,
        }
        
        json_path = samples_dir / f"{sample_id}.json"
        with open(json_path, "w") as f:
            json.dump(sample_metadata, f, indent=2)
        
        # Update manifest
        self._update_manifest(signal_name, sample_id, metadata)
        
        print(f"Saved sample {sample_id} for {signal_name}")
        return sample_id
    
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
