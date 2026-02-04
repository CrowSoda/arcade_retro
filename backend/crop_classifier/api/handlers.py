"""
WebSocket API handlers for crop classifier and DCM signal expansion.

Commands:
- expand_labels: DCM-based signal expansion (NEW - replaces bootstrap)
- crop_detect: Run blob detection on spectrogram → return crops
- crop_detect_file: Run blob detection on RFCAP file
- crop_infer: Run inference on spectrogram → return detections
- crop_status: Get current status

NOTE: Siamese/few-shot training has been ARCHIVED.
Use the existing Hydra Faster R-CNN training with k-fold validation instead.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

# Debug configuration
BASE_DIR = Path(__file__).parent.parent.parent.parent
DEBUG_DIR = Path("/tmp/crop_debug") if sys.platform != "win32" else BASE_DIR / "crop_debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_BLOB_DETECTION = True

# Import blob detection (still useful for crop extraction)
from ..inference.blob_detector import BlobDetector, DetectionResult
from ..inference.preprocessor import CropPreprocessor

logger = logging.getLogger(__name__)


class CropClassifierHandler:
    """
    Handles crop classifier WebSocket commands.

    Key commands:
    - expand_labels: DCM-based signal expansion (replaces bootstrap)
    - crop_detect: Blob detection on spectrogram
    - crop_detect_file: Blob detection on RFCAP chunks
    """

    def __init__(
        self,
        models_dir: Path = Path("models/crop_classifier"),
        training_data_dir: Path = Path("training_data/crop_classifier"),
        device: str = "cuda",
    ):
        self.models_dir = models_dir
        self.training_data_dir = training_data_dir
        self.device = device

        # Blob detector (still useful for crop extraction)
        self.detector = BlobDetector()
        self.preprocessor = CropPreprocessor()

        # DCM Signal Expander - lazy initialized (requires GPU)
        self._expander = None
        self._expander_error = None

        # Pending data for labeling workflow
        self._pending_crops: dict[str, np.ndarray] = {}
        self._pending_boxes: dict[str, dict] = {}
        self._expansion_candidates: list[dict] = []

    def _get_expander(self):
        """Lazy-initialize the DCM signal expander."""
        if self._expander is not None:
            return self._expander

        if self._expander_error is not None:
            raise RuntimeError(self._expander_error)

        try:
            from ..detection.expander import ExpanderConfig, SignalExpander

            self._expander = SignalExpander(ExpanderConfig())
            logger.info("[Handler] DCM SignalExpander initialized")
            return self._expander
        except Exception as e:
            self._expander_error = f"Failed to initialize DCM expander: {e}"
            logger.error(f"[Handler] {self._expander_error}")
            raise RuntimeError(self._expander_error) from e

    async def handle_command(
        self,
        command: str,
        data: dict[str, Any],
        send_response: callable,
    ) -> None:
        """Route command to appropriate handler."""
        handlers = {
            # NEW: DCM-based expansion
            "expand_labels": self._handle_expand_labels,
            # Blob detection (still useful)
            "crop_detect": self._handle_detect,
            "crop_detect_file": self._handle_detect_file,
            # Status and utility
            "crop_status": self._handle_status,
            "crop_label": self._handle_label,
            # LEGACY (deprecated but kept for compatibility)
            "bootstrap": self._handle_expand_labels,  # Redirect to new
            "bootstrap_file": self._handle_expand_labels,  # Redirect to new
            "confirm": self._handle_confirm,
        }

        handler = handlers.get(command)
        if not handler:
            await send_response(
                {
                    "command": command,
                    "status": "error",
                    "error": f"Unknown command: {command}",
                }
            )
            return

        try:
            await handler(data, send_response)
        except Exception as e:
            logger.exception(f"Error handling {command}")
            await send_response(
                {
                    "command": command,
                    "status": "error",
                    "error": str(e),
                }
            )

    async def _handle_expand_labels(
        self,
        data: dict[str, Any],
        send_response: callable,
    ) -> None:
        """
        NEW: DCM-based signal expansion.

        Takes user's ~20 labeled signals and finds all similar instances.
        Uses GPU-accelerated DCM correlation + CFAR detection.

        Input:
            rfcap_path: str - Path to RFCAP file
            seed_boxes: list[{t_start, t_end}, ...] - User's labeled time bounds
            pfa: float (default 0.001) - Probability of false alarm
            top_k: int (default 200) - Max candidates to return
            diversity_sampling: bool (default True) - Enable diversity sampling
            auto_accept_threshold: float (default 0.85) - Auto-accept above this
            auto_reject_threshold: float (default 0.15) - Auto-reject below this

        Output:
            status: "ok"
            detections: list[{t_start, t_end, score, auto_decision}, ...]
            stats: {templates, detections, processing_time_ms, auto_accepted, ...}
        """
        start_time = time.time()

        rfcap_path = data.get("rfcap_path")
        seed_boxes = data.get("seed_boxes", [])
        pfa = data.get("pfa", 0.001)
        top_k = data.get("top_k", 200)
        diversity_sampling = data.get("diversity_sampling", True)
        auto_accept = data.get("auto_accept_threshold", 0.85)
        auto_reject = data.get("auto_reject_threshold", 0.15)

        if not rfcap_path:
            raise ValueError("Missing 'rfcap_path' in request")
        if not seed_boxes:
            raise ValueError("Missing 'seed_boxes' - label some signals first!")

        if len(seed_boxes) < 2:
            logger.warning(f"[expand_labels] Only {len(seed_boxes)} seeds - results may be poor")

        logger.info(
            f"[expand_labels] Starting with {len(seed_boxes)} seeds, pfa={pfa}, top_k={top_k}"
        )

        # Get expander (will fail fast if no GPU)
        try:
            expander = self._get_expander()
        except RuntimeError as e:
            await send_response(
                {
                    "command": "expand_labels",
                    "status": "error",
                    "error": str(e),
                    "gpu_required": True,
                }
            )
            return

        # Update expander config
        from ..detection.expander import ExpanderConfig

        expander.config = ExpanderConfig(
            pfa=pfa,
            top_k=top_k,
            diversity_sampling=diversity_sampling,
            auto_accept_threshold=auto_accept,
            auto_reject_threshold=auto_reject,
        )

        # Progress callback
        async def progress_cb(progress: float, msg: str):
            await send_response(
                {
                    "command": "expand_labels_progress",
                    "progress": progress,
                    "message": msg,
                }
            )

        # Run expansion (blocking, use executor)
        def run_expansion():
            return expander.expand(
                rfcap_path=rfcap_path,
                seed_boxes=seed_boxes,
            )

        try:
            result = await asyncio.get_event_loop().run_in_executor(None, run_expansion)
        except Exception as e:
            logger.exception("[expand_labels] Expansion failed")
            await send_response(
                {
                    "command": "expand_labels",
                    "status": "error",
                    "error": f"Expansion failed: {e}",
                }
            )
            return

        # Store candidates for confirm flow
        self._expansion_candidates = [c.to_dict() for c in result.candidates]

        # Build response
        detections = []
        for c in result.candidates:
            detections.append(
                {
                    "t_start": c.t_start,
                    "t_end": c.t_end,
                    "score": c.score,
                    "snr_db": c.snr_db,
                    "auto_decision": c.auto_decision,
                    "time_bin": c.time_bin,
                }
            )

        processing_time = (time.time() - start_time) * 1000

        await send_response(
            {
                "command": "expand_labels",
                "status": "ok",
                "detections": detections,
                "stats": {
                    "templates": result.templates_used,
                    "detections": result.total_detections,
                    "returned": result.count,
                    "processing_time_ms": processing_time,
                    "auto_accepted": result.auto_accepted,
                    "auto_rejected": result.auto_rejected,
                    "need_review": result.need_review,
                },
            }
        )

    async def _handle_confirm(
        self,
        data: dict[str, Any],
        send_response: callable,
    ) -> None:
        """
        Record user's confirmations from swipe UI.

        This is called after expand_labels - user has swiped through candidates.

        Input:
            confirmed: list[int] - indices that user accepted (swipe right)
            rejected: list[int] - indices that user rejected (swipe left)
            seed_boxes: list[{t_start, t_end}] - original user labels

        Output:
            positives: list[{t_start, t_end}] - seeds + confirmed
            negatives: list[{t_start, t_end}] - rejected
            stats: {total_positives, total_negatives, ready_for_training}
        """
        confirmed_indices = data.get("confirmed", [])
        rejected_indices = data.get("rejected", [])
        seed_boxes = data.get("seed_boxes", [])

        logger.info(
            f"[confirm] {len(confirmed_indices)} confirmed, {len(rejected_indices)} rejected"
        )

        # Build positive set: seeds + confirmed candidates
        positives = list(seed_boxes)  # Start with seeds
        negatives = []

        for idx in confirmed_indices:
            if 0 <= idx < len(self._expansion_candidates):
                c = self._expansion_candidates[idx]
                positives.append(
                    {
                        "t_start": c["t_start"],
                        "t_end": c["t_end"],
                        "score": c["score"],
                        "source": "expansion",
                    }
                )

        for idx in rejected_indices:
            if 0 <= idx < len(self._expansion_candidates):
                c = self._expansion_candidates[idx]
                negatives.append(
                    {
                        "t_start": c["t_start"],
                        "t_end": c["t_end"],
                        "score": c["score"],
                        "source": "expansion_rejected",
                    }
                )

        # Determine if ready for training
        min_positives = 50
        min_negatives = 50
        ready = len(positives) >= min_positives and len(negatives) >= min_negatives

        await send_response(
            {
                "command": "confirm",
                "status": "ok",
                "positives": positives,
                "negatives": negatives,
                "stats": {
                    "total_positives": len(positives),
                    "total_negatives": len(negatives),
                    "from_seeds": len(seed_boxes),
                    "from_expansion": len(confirmed_indices),
                    "ready_for_training": ready,
                    "min_required": min_positives,
                },
            }
        )

    async def _handle_detect(
        self,
        data: dict[str, Any],
        send_response: callable,
    ) -> None:
        """
        Run blob detection on spectrogram.

        Input:
            spectrogram: base64 PNG or raw bytes
            include_crops: bool (default True)

        Output:
            crops: List of {id, image (base64), bbox}
        """
        spec_data = data.get("spectrogram")
        if not spec_data:
            raise ValueError("Missing 'spectrogram' in request")

        spectrogram = self._decode_image(spec_data)
        result = self.detector.detect(spectrogram)

        logger.info(f"[crop_detect] Found {result.count} blobs")

        crops_response = []
        for i, box in enumerate(result.boxes):
            crop_id = f"crop_{i:04d}"
            padded = box.with_padding(spectrogram.shape)
            crop_region = spectrogram[padded.y_min : padded.y_max, padded.x_min : padded.x_max]

            self._pending_crops[crop_id] = crop_region
            self._pending_boxes[crop_id] = padded.to_dict()

            crop_info = {
                "id": crop_id,
                "bbox": padded.to_dict(),
            }

            if data.get("include_crops", True):
                crop_info["image"] = self._encode_crop(crop_region)

            crops_response.append(crop_info)

        await send_response(
            {
                "command": "crop_detect",
                "status": "success",
                "crops": crops_response,
                "total": len(crops_response),
                "processing_time_ms": result.processing_time_ms,
            }
        )

    async def _handle_detect_file(
        self,
        data: dict[str, Any],
        send_response: callable,
    ) -> None:
        """
        Run blob detection on RFCAP file chunks.

        Input:
            rfcap_path: str
            scan_duration_sec: float (default: full file)
            chunk_sec: float (default: 0.5)

        Output:
            crops: List of detected crops with time info
        """
        start_time = time.time()
        self._pending_crops.clear()
        self._pending_boxes.clear()

        rfcap_path = data.get("rfcap_path")
        scan_duration_sec = data.get("scan_duration_sec", -1)
        chunk_sec = data.get("chunk_sec", 0.5)

        if not rfcap_path:
            raise ValueError("Missing 'rfcap_path' in request")

        header = self._read_rfcap_header(rfcap_path)
        sample_rate = header.get("sample_rate", 20e6)
        total_samples = header.get("num_samples", 0)
        total_duration = total_samples / sample_rate

        if scan_duration_sec <= 0 or scan_duration_sec > total_duration:
            scan_duration_sec = total_duration

        chunk_samples = int(chunk_sec * sample_rate)
        num_chunks = int(scan_duration_sec / chunk_sec)

        logger.info(
            f"[crop_detect_file] Processing {scan_duration_sec:.1f}s in {num_chunks} chunks"
        )

        all_crops = []

        with open(rfcap_path, "rb") as f:
            f.seek(512)

            for chunk_idx in range(num_chunks):
                chunk_time_sec = chunk_idx * chunk_sec

                raw = f.read(chunk_samples * 8)
                if len(raw) < chunk_samples * 8:
                    if len(raw) < 1024:
                        break
                    raw = raw + b"\x00" * (chunk_samples * 8 - len(raw))

                iq_data = np.frombuffer(raw, dtype=np.complex64)
                spectrogram = self._iq_to_spectrogram(iq_data, sample_rate)
                result = self.detector.detect(spectrogram, return_mask=True)

                if DEBUG_BLOB_DETECTION and chunk_idx < 3:
                    self._save_blob_debug(spectrogram, result, chunk_idx)

                for i, box in enumerate(result.boxes):
                    crop_id = f"crop_{chunk_idx:04d}_{i:03d}"
                    padded = box.with_padding(spectrogram.shape)
                    crop_region = spectrogram[
                        padded.y_min : padded.y_max, padded.x_min : padded.x_max
                    ]

                    self._pending_crops[crop_id] = crop_region
                    self._pending_boxes[crop_id] = padded.to_dict()

                    all_crops.append(
                        {
                            "id": crop_id,
                            "image": self._encode_crop(crop_region),
                            "bbox": padded.to_dict(),
                            "time_sec": chunk_time_sec,
                            "chunk_idx": chunk_idx,
                        }
                    )

                if chunk_idx % 10 == 0:
                    await send_response(
                        {
                            "command": "crop_detect_progress",
                            "progress": (chunk_idx + 1) / num_chunks,
                            "chunks_processed": chunk_idx + 1,
                            "crops_found": len(all_crops),
                        }
                    )

        processing_time_ms = (time.time() - start_time) * 1000

        await send_response(
            {
                "command": "crop_detect_file",
                "status": "success",
                "crops": all_crops,
                "total": len(all_crops),
                "chunks_processed": num_chunks,
                "scan_duration_sec": scan_duration_sec,
                "processing_time_ms": processing_time_ms,
            }
        )

    async def _handle_label(
        self,
        data: dict[str, Any],
        send_response: callable,
    ) -> None:
        """Store labels for crops (without training)."""
        labels = data.get("labels", {})
        signal_name = data.get("signal_name", "signal")

        labels_dir = self.training_data_dir / signal_name
        labels_dir.mkdir(parents=True, exist_ok=True)

        crops_dir = labels_dir / "crops"
        crops_dir.mkdir(exist_ok=True)

        saved_labels = {}
        for crop_id, is_signal in labels.items():
            if crop_id not in self._pending_crops:
                continue

            crop = self._pending_crops[crop_id]
            crop_path = crops_dir / f"{crop_id}.png"
            cv2.imwrite(str(crop_path), crop)
            saved_labels[crop_id] = 1 if is_signal else 0

        labels_path = labels_dir / "labels.json"
        existing_labels = {}
        if labels_path.exists():
            with open(labels_path) as f:
                existing_labels = json.load(f)

        existing_labels.update(saved_labels)
        with open(labels_path, "w") as f:
            json.dump(existing_labels, f, indent=2)

        await send_response(
            {
                "command": "crop_label",
                "status": "success",
                "stored_count": len(saved_labels),
                "total_labels": len(existing_labels),
            }
        )

    async def _handle_status(
        self,
        data: dict[str, Any],
        send_response: callable,
    ) -> None:
        """Get current status."""
        # Check GPU availability
        gpu_available = False
        gpu_error = None
        try:
            from ..detection.dcm import _GPU_AVAILABLE, _GPU_ERROR_MSG

            gpu_available = _GPU_AVAILABLE
            gpu_error = _GPU_ERROR_MSG
        except ImportError:
            gpu_error = "DCM module not available"

        await send_response(
            {
                "command": "crop_status",
                "status": "success",
                "gpu_available": gpu_available,
                "gpu_error": gpu_error,
                "pending_crops": len(self._pending_crops),
                "expansion_candidates": len(self._expansion_candidates),
                "device": self.device,
            }
        )

    # Helper methods

    def _read_rfcap_header(self, path: str) -> dict:
        """Read RFCAP file header (G20 format)."""
        import struct

        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"G20\x00":
                raise ValueError(f"Invalid RFCAP magic: {magic!r}")

            version = struct.unpack("<I", f.read(4))[0]
            sample_rate = struct.unpack("<d", f.read(8))[0]
            center_freq = struct.unpack("<d", f.read(8))[0]
            bandwidth = struct.unpack("<d", f.read(8))[0]
            num_samples = struct.unpack("<Q", f.read(8))[0]
            start_time = struct.unpack("<d", f.read(8))[0]

            signal_name_bytes = f.read(32)
            null_idx = signal_name_bytes.find(b"\x00")
            if null_idx >= 0:
                signal_name_bytes = signal_name_bytes[:null_idx]
            signal_name = signal_name_bytes.decode("utf-8", errors="ignore")

            return {
                "version": version,
                "sample_rate": sample_rate,
                "center_freq": center_freq,
                "bandwidth": bandwidth,
                "num_samples": num_samples,
                "start_time": start_time,
                "signal_name": signal_name,
            }

    def _iq_to_spectrogram(self, iq_data: np.ndarray, sample_rate: float) -> np.ndarray:
        """Convert IQ data to spectrogram (matches training pipeline)."""
        from scipy import signal as scipy_signal

        nfft = 4096
        noverlap = 2048
        dynamic_range_db = 80.0

        f, t, Sxx = scipy_signal.spectrogram(
            iq_data,
            fs=sample_rate,
            nperseg=nfft,
            noverlap=noverlap,
            mode="psd",
            return_onesided=False,
        )

        Sxx = np.fft.fftshift(Sxx, axes=0)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        db_max = Sxx_db.max()
        db_min = db_max - dynamic_range_db

        Sxx_norm = np.clip(Sxx_db, db_min, db_max)
        Sxx_norm = (Sxx_norm - db_min) / dynamic_range_db * 255

        return np.flipud(Sxx_norm.astype(np.uint8))

    def _decode_image(self, data: Any) -> np.ndarray:
        """Decode image from base64 or bytes."""
        if isinstance(data, str):
            img_bytes = base64.b64decode(data)
            img = Image.open(io.BytesIO(img_bytes))
            return np.array(img)
        elif isinstance(data, bytes | bytearray):
            img = Image.open(io.BytesIO(data))
            return np.array(img)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Invalid image data type: {type(data)}")

    def _encode_crop(self, crop: np.ndarray) -> str:
        """Encode crop to base64 PNG."""
        if crop.dtype != np.uint8:
            if crop.max() <= 1.0:
                crop = (crop * 255).astype(np.uint8)
            else:
                crop = crop.astype(np.uint8)

        _, buffer = cv2.imencode(".png", crop)
        return base64.b64encode(buffer).decode("utf-8")

    def _save_blob_debug(
        self, spectrogram: np.ndarray, result: DetectionResult, chunk_idx: int
    ) -> None:
        """Save debug visualization."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(spectrogram, cmap="viridis", aspect="auto")
        axes[0].set_title(f"Spectrogram (chunk {chunk_idx})")

        if result.binary_mask is not None:
            axes[1].imshow(result.binary_mask, cmap="gray", aspect="auto")
            axes[1].set_title("Binary Mask")
        else:
            axes[1].text(0.5, 0.5, "No mask", ha="center", va="center")

        axes[2].imshow(spectrogram, cmap="viridis", aspect="auto")
        axes[2].set_title(f"Detections: {result.count}")

        for box in result.boxes:
            rect = plt.Rectangle(
                (box.x_min, box.y_min),
                box.x_max - box.x_min,
                box.y_max - box.y_min,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            axes[2].add_patch(rect)

        plt.tight_layout()

        filepath = DEBUG_DIR / f"blob_debug_{chunk_idx:04d}.png"
        plt.savefig(str(filepath), dpi=100)
        plt.close()

        logger.info(f"[DEBUG] Saved {filepath}")


# Global handler instance
_handler: CropClassifierHandler | None = None


def get_handler() -> CropClassifierHandler:
    """Get or create global handler instance."""
    global _handler
    if _handler is None:
        _handler = CropClassifierHandler()
    return _handler


async def handle_crop_command(
    command: str,
    data: dict[str, Any],
    send_response: callable,
) -> None:
    """
    Entry point for crop classifier commands.

    Add to server.py message router:
        if command.startswith("crop_") or command == "expand_labels":
            await handle_crop_command(command, data, send_response)
    """
    handler = get_handler()
    await handler.handle_command(command, data, send_response)
