"""
Hydra Detector - Shared backbone with multiple detection heads.

Loads backbone once (~55MB GPU), adds heads as needed (~10MB each).
All heads share backbone features, dramatically reducing memory and latency.

Performance:
    - 1 head:  backbone + head = ~65MB, ~10ms inference
    - 6 heads: backbone + 6 heads = ~115MB, ~35ms inference
    - 12 heads: backbone + 12 heads = ~175MB, ~45ms inference

Usage:
    detector = HydraDetector("models/")
    detector.load_backbone()
    detector.load_heads(["creamy_chicken", "lte_uplink"])
    results = detector.detect(spectrogram_tensor)
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

# Logging
from logger_config import get_logger
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator

from .config import DEFAULT_SCORE_THRESHOLD, MAX_DETECTIONS_PER_HEAD

logger = get_logger("detector")


logger = logging.getLogger("hydra.detector")


@dataclass
class Detection:
    """Single detection result."""

    box_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    signal_name: str


class HydraDetector:
    """
    Multi-head detector with shared backbone.

    Memory efficient: One backbone in GPU memory, heads loaded/unloaded as needed.
    """

    def __init__(self, models_dir: str, device: str = "cuda"):
        self.models_dir = Path(models_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load or create registry
        self.registry = self._load_or_scan_registry()

        # State
        self.model: FasterRCNN | None = None
        self.heads: dict[str, dict] = {}  # signal_name -> head state dict
        self._backbone_loaded = False
        self._current_head: str | None = None

        # Timing stats
        self._last_backbone_ms = 0.0
        self._last_heads_ms = {}

        logger.info(f"HydraDetector initialized: {self.device}")
        logger.info(f"Available signals: {list(self.registry.get('signals', {}).keys())}")

    def _load_or_scan_registry(self) -> dict:
        """Load registry.json or scan heads directory to build it."""
        registry_path = self.models_dir / "registry.json"

        # If registry exists, load it
        if registry_path.exists():
            try:
                with open(registry_path) as f:
                    registry = json.load(f)
                logger.info(f"Loaded registry with {len(registry.get('signals', {}))} signals")
                return registry
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")

        # Scan for heads
        logger.info("Scanning for available heads...")
        return self.scan_and_build_registry()

    def scan_and_build_registry(self) -> dict:
        """Scan heads directory and build registry from metadata files."""
        registry = {
            "backbone_version": None,
            "signals": {},
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        # Check backbone
        backbone_meta = self.models_dir / "backbone" / "metadata.json"
        if backbone_meta.exists():
            with open(backbone_meta) as f:
                meta = json.load(f)
                registry["backbone_version"] = meta.get("version", 1)

        # Scan heads directory
        heads_dir = self.models_dir / "heads"
        if not heads_dir.exists():
            logger.warning(f"Heads directory not found: {heads_dir}")
            return registry

        for signal_dir in heads_dir.iterdir():
            if not signal_dir.is_dir():
                continue

            signal_name = signal_dir.name

            # Check for active.pth or v*.pth
            active_path = signal_dir / "active.pth"
            if not active_path.exists():
                # Look for any version file
                versions = list(signal_dir.glob("v*.pth"))
                if not versions:
                    continue
                # Use latest version
                versions.sort(key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0)
                active_path = versions[-1]

            # Load metadata if available
            metadata_path = signal_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    meta = json.load(f)

                active_version = meta.get("active_version", 1)
                versions = meta.get("versions", [])

                # Find active version metrics
                active_metrics = {}
                for v in versions:
                    if v.get("version") == active_version:
                        active_metrics = v.get("metrics", {})
                        break

                registry["signals"][signal_name] = {
                    "active_head_version": active_version,
                    "sample_count": meta.get("versions", [{}])[-1].get("sample_count", 0),
                    "f1_score": active_metrics.get("f1_score", 0.0),
                    "precision": active_metrics.get("precision", 0.0),
                    "recall": active_metrics.get("recall", 0.0),
                    "last_trained": meta.get("versions", [{}])[-1].get("created_at", ""),
                    "head_path": str(active_path.relative_to(self.models_dir)),
                }
            else:
                # No metadata, just track the head exists
                registry["signals"][signal_name] = {
                    "active_head_version": 1,
                    "sample_count": 0,
                    "f1_score": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "last_trained": "",
                    "head_path": str(active_path.relative_to(self.models_dir)),
                }

            logger.info(f"Found head: {signal_name}")

        # Save registry
        self._save_registry(registry)

        return registry

    def _save_registry(self, registry: dict = None) -> None:
        """Save registry to disk."""
        if registry is None:
            registry = self.registry

        registry["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        registry_path = self.models_dir / "registry.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        logger.info(f"Saved registry to {registry_path}")

    def load_backbone(self) -> None:
        """Load the shared backbone and create base model."""
        if self._backbone_loaded:
            logger.info("Backbone already loaded")
            return

        backbone_path = self.models_dir / "backbone" / "active.pth"
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone not found: {backbone_path}")

        logger.info(f"Loading backbone from {backbone_path}")

        # Create FasterRCNN model with ResNet18-FPN backbone
        backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=0)

        # Wide-coverage default anchors - covers aspect ratios from 0.1 (wide) to 4.0 (tall)
        # NOTE: Per-head anchors are loaded dynamically when switching heads
        # These defaults cover the common signal shapes for initial model creation
        default_sizes = (31, 44, 54, 70, 77, 63, 40, 70, 70)  # From get_default_anchors()
        default_aspects = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 0.5, 2.0)

        anchor_generator = AnchorGenerator(
            sizes=(default_sizes,) * 5,  # Same anchors for all 5 FPN levels
            aspect_ratios=(default_aspects,) * 5,
        )

        self.model = FasterRCNN(
            backbone,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            # Lower RPN thresholds for small objects (matches training)
            rpn_fg_iou_thresh=0.5,
            rpn_bg_iou_thresh=0.3,
        )

        # Load backbone weights
        backbone_state = torch.load(backbone_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(backbone_state, strict=False)

        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        # Freeze backbone
        for name, param in self.model.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False

        self._backbone_loaded = True
        logger.info("Backbone loaded")

    def load_heads(self, signal_names: list[str]) -> None:
        """
        Load detection heads for specified signals.

        This performs a FULL SWAP: unloads heads not in the list, loads new ones.
        This ensures the detector only runs the requested signals.

        Args:
            signal_names: List of signal names to load
        """
        if not self._backbone_loaded:
            self.load_backbone()

        # FULL SWAP: Unload heads NOT in the requested list
        requested_set = set(signal_names)
        current_heads = list(self.heads.keys())
        heads_to_unload = [h for h in current_heads if h not in requested_set]

        if heads_to_unload:
            logger.info(f"Unloading heads not in request: {heads_to_unload}")
            self.unload_heads(heads_to_unload)

        # Load new heads
        for name in signal_names:
            if name in self.heads:
                logger.info(f"Head already loaded: {name}")
                continue

            head_path = self.models_dir / "heads" / name / "active.pth"
            if not head_path.exists():
                logger.warning(f"Head not found: {head_path}")
                continue

            logger.info(f"Loading head: {name}")

            # Load head state dict
            head_state = torch.load(head_path, map_location=self.device, weights_only=False)
            self.heads[name] = head_state

            # Update registry
            if name in self.registry.get("signals", {}):
                self.registry["signals"][name]["is_loaded"] = True

        logger.info(f"Loaded heads: {list(self.heads.keys())}")

    def _switch_head(self, signal_name: str) -> None:
        """Switch the active head on the model."""
        if signal_name == self._current_head:
            return

        if signal_name not in self.heads:
            raise ValueError(f"Head not loaded: {signal_name}")

        # Load head weights into model
        head_state = self.heads[signal_name]
        self.model.load_state_dict(head_state, strict=False)
        self._current_head = signal_name

    def unload_heads(self, signal_names: list[str] = None) -> None:
        """
        Unload heads to free memory.

        Args:
            signal_names: Specific heads to unload, or None for all
        """
        if signal_names is None:
            signal_names = list(self.heads.keys())

        for name in signal_names:
            if name in self.heads:
                del self.heads[name]

                if name in self.registry.get("signals", {}):
                    self.registry["signals"][name]["is_loaded"] = False

                if name == self._current_head:
                    self._current_head = None

                logger.info(f"Unloaded head: {name}")

        # Free GPU memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_loaded_heads(self) -> list[str]:
        """Return list of currently loaded head names."""
        return list(self.heads.keys())

    @torch.inference_mode()
    def detect(
        self, spectrogram: torch.Tensor, score_threshold: float = DEFAULT_SCORE_THRESHOLD
    ) -> dict[str, list[Detection]]:
        """
        Run all loaded heads on spectrogram EFFICIENTLY.

        OPTIMIZED: Compute backbone features ONCE, then run all heads on shared features.
        This is 2-3x faster than switching heads sequentially.

        Args:
            spectrogram: [1, 3, 1024, 1024] normalized tensor (0-1 float)
            score_threshold: Minimum confidence for detections

        Returns:
            Dict mapping signal_name → list of Detection objects
        """
        if not self._backbone_loaded:
            raise RuntimeError("Backbone not loaded")

        if not self.heads:
            raise RuntimeError("No heads loaded")

        # Ensure correct device
        spectrogram = spectrogram.to(self.device)

        results = {}

        # OPTIMIZATION: Compute backbone features ONCE
        t0 = time.perf_counter()
        images = [spectrogram[0]]

        # Get original image sizes for FasterRCNN transform
        original_image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]

        # Apply transform (normalization, resize)
        images_transformed, targets = self.model.transform(images, None)

        # Compute backbone features ONCE (the expensive part!)
        features = self.model.backbone(images_transformed.tensors)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._last_backbone_ms = (time.perf_counter() - t0) * 1000

        # Now run each head on the SHARED features (fast - just RPN + ROI)
        for signal_name in self.heads:
            t0 = time.perf_counter()

            # Load this head's RPN + ROI weights
            head_state = self.heads[signal_name]

            # Only load RPN and ROI head weights (not backbone!)
            for key, value in head_state.items():
                if key.startswith("rpn.") or key.startswith("roi_heads."):
                    # Get the parameter in the model and copy in-place
                    param = self.model
                    parts = key.split(".")
                    for part in parts[:-1]:
                        param = getattr(param, part)
                    getattr(param, parts[-1]).copy_(value.to(self.device))

            # Run RPN on shared features
            proposals, proposal_losses = self.model.rpn(images_transformed, features, None)

            # Run ROI heads on shared features
            detections_list, detector_losses = self.model.roi_heads(
                features, proposals, images_transformed.image_sizes, None
            )

            # Post-process detections back to original image size
            detections_out = self.model.transform.postprocess(
                detections_list, images_transformed.image_sizes, original_image_sizes
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._last_heads_ms[signal_name] = (time.perf_counter() - t0) * 1000

            # Convert to Detection objects
            detections = []
            out = detections_out[0]
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()

            for i in range(min(len(boxes), MAX_DETECTIONS_PER_HEAD)):
                if scores[i] < score_threshold:
                    continue

                # Raw normalized coordinates - matches model's coordinate system
                # Model outputs boxes in [x1, y1, x2, y2] pixel coords for 1024x1024 image
                det = Detection(
                    box_id=i,
                    x1=float(boxes[i][0]) / 1024,
                    y1=float(boxes[i][1]) / 1024,
                    x2=float(boxes[i][2]) / 1024,
                    y2=float(boxes[i][3]) / 1024,
                    confidence=float(scores[i]),
                    class_id=int(labels[i]),
                    class_name=signal_name if labels[i] == 1 else "background",
                    signal_name=signal_name,
                )
                detections.append(det)

            results[signal_name] = detections

        return results

    @torch.inference_mode()
    def detect_single(
        self,
        spectrogram: torch.Tensor,
        signal_name: str,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> list[Detection]:
        """Run single head for testing/validation."""
        if signal_name not in self.heads:
            raise ValueError(f"Head not loaded: {signal_name}")

        results = self.detect(spectrogram, score_threshold)
        return results.get(signal_name, [])

    def get_head_info(self, signal_name: str) -> dict:
        """Get metadata for a head."""
        metadata_path = self.models_dir / "heads" / signal_name / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}

    def get_timing_stats(self) -> dict:
        """Get timing stats from last inference."""
        return {
            "backbone_ms": self._last_backbone_ms,
            "heads_ms": self._last_heads_ms.copy(),
            "total_ms": sum(self._last_heads_ms.values()),
        }

    def get_memory_usage(self) -> dict:
        """Get GPU memory usage."""
        if self.device.type != "cuda":
            return {"allocated_mb": 0, "cached_mb": 0}

        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1e6,
            "cached_mb": torch.cuda.memory_reserved() / 1e6,
        }

    def get_registry(self) -> dict:
        """Get the model registry."""
        return self.registry

    def get_available_signals(self) -> list[str]:
        """Get list of all signals with trained heads."""
        return list(self.registry.get("signals", {}).keys())
