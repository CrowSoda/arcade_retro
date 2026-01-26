# Phase 2: Hydra Detector

## Overview

The HydraDetector is the core inference class that loads one backbone and multiple signal-specific heads, running them in parallel for efficient multi-signal detection.

## File: `backend/hydra/detector.py`

### Class Interface

```python
class HydraDetector:
    """
    Shared-backbone multi-head detector.
    
    Usage:
        detector = HydraDetector("models/")
        detector.load_backbone()
        detector.load_heads(["creamy_chicken", "lte_uplink", "wifi_24"])
        
        # Run inference on spectrogram
        results = detector.detect(spectrogram_tensor)
        # Returns: {"creamy_chicken": [...], "lte_uplink": [...], ...}
    """
    
    def __init__(self, models_dir: str, device: str = "cuda")
    def load_backbone(self) -> None
    def load_heads(self, signal_names: List[str]) -> None
    def unload_heads(self, signal_names: List[str] = None) -> None
    def get_loaded_heads(self) -> List[str]
    def detect(self, spectrogram: torch.Tensor) -> Dict[str, List[Detection]]
    def detect_single(self, spectrogram: torch.Tensor, signal_name: str) -> List[Detection]
    def get_head_info(self, signal_name: str) -> dict
```

### Complete Implementation

```python
"""
Hydra Detector - Shared backbone with multiple detection heads.

Loads backbone once (~55MB GPU), adds heads as needed (~10MB each).
All heads share backbone features, dramatically reducing memory and latency.

Performance:
    - 1 head:  backbone + head = ~65MB, ~10ms inference
    - 6 heads: backbone + 6 heads = ~115MB, ~35ms inference
    - 12 heads: backbone + 12 heads = ~175MB, ~45ms inference
"""

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead, AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads

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
    signal_name: str  # Which head made this detection


class HydraBackbone(nn.Module):
    """
    Frozen backbone that outputs FPN features.
    
    This wraps the ResNet18-FPN backbone and is shared across all heads.
    """
    
    def __init__(self):
        super().__init__()
        self.backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=0)
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract FPN features from images."""
        return self.backbone(images)
    
    def load_weights(self, state_dict: dict):
        """Load backbone weights from state dict."""
        self.load_state_dict(state_dict, strict=False)


class HydraHead(nn.Module):
    """
    Detection head (RPN + ROI) for a single signal type.
    
    Takes FPN features from backbone and produces detections.
    """
    
    def __init__(self, signal_name: str, num_classes: int = 2):
        super().__init__()
        self.signal_name = signal_name
        self.num_classes = num_classes
        
        # Anchor generator (same as FasterRCNN default)
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        # RPN head
        out_channels = 256  # FPN output channels
        self.rpn_head = RPNHead(out_channels, self.anchor_generator.num_anchors_per_location()[0])
        
        # RPN
        self.rpn = RegionProposalNetwork(
            self.anchor_generator,
            self.rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7,
        )
        
        # ROI heads
        from torchvision.ops import MultiScaleRoIAlign
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )
        
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size
        )
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            representation_size,
            num_classes
        )
        
        self.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
        )
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        images: torch.Tensor,
        image_sizes: List[Tuple[int, int]],
        targets: Optional[List[Dict]] = None
    ):
        """
        Run detection head on backbone features.
        
        Args:
            features: FPN features from backbone
            images: Original images (for RPN)
            image_sizes: List of (H, W) for each image
            targets: Optional training targets
        
        Returns:
            List of detection dicts: [{"boxes": ..., "scores": ..., "labels": ...}]
        """
        from torchvision.models.detection.image_list import ImageList
        
        # Create ImageList for RPN
        image_list = ImageList(images, image_sizes)
        
        # RPN forward
        proposals, proposal_losses = self.rpn(image_list, features, targets)
        
        # ROI heads forward
        detections, detector_losses = self.roi_heads(features, proposals, image_sizes, targets)
        
        if self.training:
            return {**proposal_losses, **detector_losses}
        
        return detections
    
    def load_weights(self, state_dict: dict):
        """Load head weights from state dict."""
        # Map keys from full model format to head format
        head_dict = {}
        for key, value in state_dict.items():
            if key.startswith("rpn."):
                # rpn.head.conv.weight -> rpn_head.conv.weight
                new_key = key.replace("rpn.head.", "rpn_head.")
                new_key = new_key.replace("rpn.anchor_generator.", "anchor_generator.")
                head_dict[new_key] = value
            elif key.startswith("roi_heads."):
                new_key = key.replace("roi_heads.", "roi_heads.")
                head_dict[new_key] = value
        
        self.load_state_dict(head_dict, strict=False)


class HydraDetector:
    """
    Multi-head detector with shared backbone.
    
    Memory efficient: One backbone in GPU memory, heads loaded/unloaded as needed.
    """
    
    def __init__(self, models_dir: str, device: str = "cuda"):
        self.models_dir = Path(models_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load registry
        registry_path = self.models_dir / "registry.json"
        if registry_path.exists():
            with open(registry_path) as f:
                self.registry = json.load(f)
        else:
            self.registry = {"backbone_version": None, "signals": {}}
        
        # State
        self.backbone: Optional[HydraBackbone] = None
        self.heads: Dict[str, HydraHead] = {}
        self._backbone_loaded = False
        
        # Timing stats
        self._last_backbone_ms = 0.0
        self._last_heads_ms = {}
        
        logger.info(f"HydraDetector initialized: {self.device}")
    
    def load_backbone(self) -> None:
        """Load the shared backbone to GPU."""
        if self._backbone_loaded:
            logger.info("Backbone already loaded")
            return
        
        backbone_path = self.models_dir / "backbone" / "active.pth"
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone not found: {backbone_path}")
        
        logger.info(f"Loading backbone from {backbone_path}")
        
        # Create and load backbone
        self.backbone = HydraBackbone()
        state_dict = torch.load(backbone_path, map_location=self.device, weights_only=False)
        self.backbone.load_weights(state_dict)
        
        # Move to device and set eval mode
        self.backbone.to(self.device)
        self.backbone.eval()
        
        # Use FP16 on CUDA
        if self.device.type == "cuda":
            self.backbone.half()
        
        self._backbone_loaded = True
        logger.info("Backbone loaded")
    
    def load_heads(self, signal_names: List[str]) -> None:
        """
        Load detection heads for specified signals.
        
        Args:
            signal_names: List of signal names to load
        """
        if not self._backbone_loaded:
            self.load_backbone()
        
        for name in signal_names:
            if name in self.heads:
                logger.info(f"Head already loaded: {name}")
                continue
            
            head_path = self.models_dir / "heads" / name / "active.pth"
            if not head_path.exists():
                logger.warning(f"Head not found: {head_path}")
                continue
            
            logger.info(f"Loading head: {name}")
            
            # Create and load head
            head = HydraHead(signal_name=name, num_classes=2)
            state_dict = torch.load(head_path, map_location=self.device, weights_only=False)
            head.load_weights(state_dict)
            
            # Move to device
            head.to(self.device)
            head.eval()
            
            if self.device.type == "cuda":
                head.half()
            
            self.heads[name] = head
            
            # Update registry
            if name in self.registry.get("signals", {}):
                self.registry["signals"][name]["is_loaded"] = True
        
        logger.info(f"Loaded heads: {list(self.heads.keys())}")
    
    def unload_heads(self, signal_names: List[str] = None) -> None:
        """
        Unload heads to free GPU memory.
        
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
                
                logger.info(f"Unloaded head: {name}")
        
        # Free GPU memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def get_loaded_heads(self) -> List[str]:
        """Return list of currently loaded head names."""
        return list(self.heads.keys())
    
    @torch.inference_mode()
    def detect(
        self,
        spectrogram: torch.Tensor,
        score_threshold: float = 0.5
    ) -> Dict[str, List[Detection]]:
        """
        Run all loaded heads on spectrogram.
        
        Args:
            spectrogram: [1, 3, 1024, 1024] normalized tensor
            score_threshold: Minimum confidence for detections
        
        Returns:
            Dict mapping signal_name → list of Detection objects
        """
        if not self._backbone_loaded:
            raise RuntimeError("Backbone not loaded")
        
        if not self.heads:
            raise RuntimeError("No heads loaded")
        
        # Ensure correct device and dtype
        spectrogram = spectrogram.to(self.device)
        if self.device.type == "cuda":
            spectrogram = spectrogram.half()
        
        # Get image sizes
        batch_size = spectrogram.shape[0]
        image_sizes = [(spectrogram.shape[2], spectrogram.shape[3])] * batch_size
        
        # Run backbone (shared across all heads)
        t0 = time.perf_counter()
        features = self.backbone(spectrogram)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._last_backbone_ms = (time.perf_counter() - t0) * 1000
        
        # Run each head
        results = {}
        for signal_name, head in self.heads.items():
            t0 = time.perf_counter()
            
            outputs = head(features, spectrogram, image_sizes)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._last_heads_ms[signal_name] = (time.perf_counter() - t0) * 1000
            
            # Convert to Detection objects
            detections = []
            for batch_idx, out in enumerate(outputs):
                boxes = out["boxes"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                
                for i in range(len(boxes)):
                    if scores[i] < score_threshold:
                        continue
                    
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
        score_threshold: float = 0.5
    ) -> List[Detection]:
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
            "total_ms": self._last_backbone_ms + sum(self._last_heads_ms.values()),
        }
    
    def get_memory_usage(self) -> dict:
        """Get GPU memory usage."""
        if self.device.type != "cuda":
            return {"allocated_mb": 0, "cached_mb": 0}
        
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1e6,
            "cached_mb": torch.cuda.memory_reserved() / 1e6,
        }
```

---

## Integration with Unified Pipeline

### Modify `backend/unified_pipeline.py`

```python
# Add to imports
from hydra.detector import HydraDetector, Detection

# In TripleBufferedPipeline.__init__():

# REMOVE old model loading:
# self._load_model(model_path)

# ADD Hydra detector:
self.detector = HydraDetector(models_dir="models/", device=self.device)
self.detector.load_backbone()
# Note: Heads loaded later when mission starts

# ADD method for mission loading:
def load_mission_signals(self, signal_names: List[str]) -> None:
    """Load detection heads for mission signals."""
    self.detector.load_heads(signal_names)

def unload_mission_signals(self) -> None:
    """Unload all heads when mission ends."""
    self.detector.unload_heads()

# MODIFY process_chunk():
def process_chunk(self, iq_data: np.ndarray, pts: float, score_threshold: float = 0.5):
    """Run inference using Hydra detector."""
    spec = self.compute_spectrogram(iq_data)
    
    # Run all loaded heads
    all_detections = self.detector.detect(spec, score_threshold)
    
    # Flatten to single list
    detections = []
    for signal_name, signal_dets in all_detections.items():
        for det in signal_dets:
            detections.append(det)
    
    return {
        "detections": detections,
        "pts": pts,
        "inference_ms": self.detector.get_timing_stats()["total_ms"],
    }
```

---

## ⚠️ Important: Weight Loading Validation

The `HydraHead.load_weights()` method remaps keys from extracted head state dict to the fresh RPN/ROI modules. This mapping may be fragile if torchvision's internal structure changes.

**Critical testing required in Phase 1:**

```python
def validate_weight_loading():
    """Run this after extraction to verify weights load correctly."""
    
    # Load extracted head weights
    head_state = torch.load("models/heads/creamy_chicken/v1.pth")
    
    # Create fresh HydraHead
    head = HydraHead(256)  # 256 = FPN channels
    
    # Load weights with verbose logging
    head.load_weights(head_state, verbose=True)
    
    # Verify all parameters were loaded
    expected_keys = set(head_state.keys())
    loaded_keys = set()  # Track which keys were actually mapped
    
    # If any keys weren't loaded, that's a problem
    missing = expected_keys - loaded_keys
    if missing:
        print(f"WARNING: {len(missing)} keys not loaded: {missing}")
        
    # Test inference
    dummy_features = {
        '0': torch.randn(1, 256, 256, 256),
        '1': torch.randn(1, 256, 128, 128),
        '2': torch.randn(1, 256, 64, 64),
        '3': torch.randn(1, 256, 32, 32),
        'pool': torch.randn(1, 256, 16, 16),
    }
    
    try:
        output = head(dummy_features)
        print(f"✓ Head inference works, output has {len(output[0]['boxes'])} boxes")
    except Exception as e:
        print(f"✗ Head inference failed: {e}")
```

If weight loading fails, the fallback is to keep heads as full FasterRCNN models (slower) until the key mapping is fixed.

---

## Usage Example

```python
# Create detector
detector = HydraDetector("models/")

# Load backbone (one time)
detector.load_backbone()

# Load heads for mission
detector.load_heads(["creamy_chicken", "lte_uplink", "wifi_24"])

# Run inference
spec = torch.randn(1, 3, 1024, 1024)
results = detector.detect(spec, score_threshold=0.5)

for signal_name, detections in results.items():
    print(f"{signal_name}: {len(detections)} detections")

# Check timing
print(detector.get_timing_stats())
# {'backbone_ms': 5.2, 'heads_ms': {'creamy_chicken': 3.1, ...}, 'total_ms': 15.8}

# Unload heads when done
detector.unload_heads()
```
