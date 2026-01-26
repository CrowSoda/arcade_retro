# Phase 4-5: Training Service & WebSocket API

## Training Service Overview

The TrainingService handles head training with frozen backbone. It receives training data from Flutter, computes spectrograms using the locked inference FFT params, and trains only the head layers.

## File: `backend/training/service.py`

```python
"""
Training Service - Train detection heads with frozen backbone.

Key features:
    - Frozen backbone (no gradients)
    - Early stopping with patience=5
    - Progress callbacks for UI updates
    - Auto-promotion check after training
"""

import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN

from ..hydra.config import *
from ..hydra.version_manager import VersionManager
from .dataset import SpectrogramDataset, create_data_loaders
from .splits import SplitManager


@dataclass
class TrainingProgress:
    """Progress update during training."""
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    f1_score: float
    precision: float
    recall: float
    is_best: bool
    elapsed_sec: float


@dataclass
class TrainingResult:
    """Final training result."""
    signal_name: str
    version: int
    sample_count: int
    epochs_trained: int
    early_stopped: bool
    metrics: dict
    training_time_sec: float
    notes: str = None
    previous_version: int = None
    previous_metrics: dict = None
    auto_promoted: bool = False
    promotion_reason: str = None


class TrainingService:
    """Trains detection heads against frozen backbone."""
    
    def __init__(
        self,
        models_dir: str = "models",
        training_data_dir: str = "training_data/signals",
        device: str = "cuda"
    ):
        self.models_dir = Path(models_dir)
        self.training_data_dir = Path(training_data_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.version_manager = VersionManager(models_dir)
        self.split_manager = SplitManager(training_data_dir)
        
        # Training state
        self._is_training = False
        self._cancel_requested = False
    
    def train_new_signal(
        self,
        signal_name: str,
        notes: str = None,
        callback: Callable[[TrainingProgress], None] = None
    ) -> TrainingResult:
        """Train head for brand new signal."""
        
        # Create initial split
        self.split_manager.create_initial_split(signal_name)
        
        return self._train(signal_name, notes, callback, is_new=True)
    
    def extend_signal(
        self,
        signal_name: str,
        notes: str = None,
        callback: Callable[[TrainingProgress], None] = None
    ) -> TrainingResult:
        """Continue training with additional samples."""
        
        # Extend split with new samples
        self.split_manager.extend_split(signal_name)
        
        return self._train(signal_name, notes, callback, is_new=False)
    
    def cancel_training(self):
        """Request training cancellation."""
        self._cancel_requested = True
    
    def _train(
        self,
        signal_name: str,
        notes: str,
        callback: Callable,
        is_new: bool
    ) -> TrainingResult:
        """Internal training implementation."""
        
        self._is_training = True
        self._cancel_requested = False
        start_time = time.time()
        
        try:
            # Build model with frozen backbone
            model = self._build_model()
            
            # Load existing head weights if extending
            if not is_new:
                self._load_head_weights(model, signal_name)
            
            # Create data loaders
            train_loader, val_loader = create_data_loaders(
                signal_name,
                str(self.training_data_dir),
                batch_size=DEFAULT_BATCH_SIZE
            )
            
            sample_count = len(train_loader.dataset) + len(val_loader.dataset)
            
            if sample_count < MIN_SAMPLES_FOR_TRAINING:
                raise ValueError(f"Need at least {MIN_SAMPLES_FOR_TRAINING} samples")
            
            # Get previous metrics for comparison
            previous_version = None
            previous_metrics = None
            if not is_new:
                history = self.version_manager.get_version_history(signal_name)
                if history:
                    active = [v for v in history if v.get("is_active")]
                    if active:
                        previous_version = active[0]["version"]
                        previous_metrics = active[0]["metrics"]
            
            # Training loop
            best_state, best_metrics, epochs_trained, early_stopped = self._train_loop(
                model, train_loader, val_loader, callback
            )
            
            training_time = time.time() - start_time
            
            # Create new version
            version = self.version_manager.create_version(
                signal_name=signal_name,
                head_state_dict=best_state,
                metrics=best_metrics,
                sample_count=sample_count,
                split_version=self.split_manager.get_active_split_version(signal_name),
                epochs_trained=epochs_trained,
                early_stopped=early_stopped,
                training_time_sec=training_time,
                notes=notes
            )
            
            # Check auto-promotion
            should_promote, reason = self.version_manager.should_auto_promote(
                signal_name, best_metrics
            )
            
            if should_promote:
                self.version_manager.promote_version(signal_name, version, reason)
            
            # Cleanup old versions
            self.version_manager.cleanup_old_versions(signal_name)
            
            return TrainingResult(
                signal_name=signal_name,
                version=version,
                sample_count=sample_count,
                epochs_trained=epochs_trained,
                early_stopped=early_stopped,
                metrics=best_metrics,
                training_time_sec=training_time,
                notes=notes,
                previous_version=previous_version,
                previous_metrics=previous_metrics,
                auto_promoted=should_promote,
                promotion_reason=reason
            )
            
        finally:
            self._is_training = False
    
    def _build_model(self) -> FasterRCNN:
        """Build FasterRCNN with frozen backbone."""
        
        # Load backbone weights
        backbone_path = self.models_dir / "backbone" / "active.pth"
        backbone_state = torch.load(backbone_path, map_location=self.device)
        
        # Create model
        backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=0)
        model = FasterRCNN(backbone, num_classes=2)
        
        # Load backbone weights
        model.load_state_dict(backbone_state, strict=False)
        
        # Freeze backbone
        for name, param in model.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
        
        model.to(self.device)
        return model
    
    def _load_head_weights(self, model: FasterRCNN, signal_name: str):
        """Load existing head weights."""
        head_path = self.models_dir / "heads" / signal_name / "active.pth"
        if head_path.exists():
            head_state = torch.load(head_path, map_location=self.device)
            model.load_state_dict(head_state, strict=False)
    
    def _train_loop(
        self,
        model: FasterRCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callback: Callable
    ) -> Tuple[dict, dict, int, bool]:
        """Training loop with early stopping."""
        
        # Only optimize non-frozen params
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=DEFAULT_LEARNING_RATE)
        
        best_f1 = 0.0
        best_state = None
        best_metrics = None
        patience_counter = 0
        
        for epoch in range(DEFAULT_EPOCHS):
            if self._cancel_requested:
                break
            
            # Train
            model.train()
            train_loss = 0.0
            for images, targets in train_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                train_loss += losses.item()
            
            train_loss /= len(train_loader)
            
            # Evaluate
            metrics = self._evaluate(model, val_loader)
            
            # Check for improvement
            is_best = metrics["f1_score"] > best_f1
            if is_best:
                best_f1 = metrics["f1_score"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()
                             if not k.startswith("backbone.")}
                best_metrics = metrics.copy()
                best_metrics["train_loss"] = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Callback
            if callback:
                progress = TrainingProgress(
                    epoch=epoch + 1,
                    total_epochs=DEFAULT_EPOCHS,
                    train_loss=train_loss,
                    val_loss=metrics["val_loss"],
                    f1_score=metrics["f1_score"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    is_best=is_best,
                    elapsed_sec=time.time()
                )
                callback(progress)
            
            # Early stopping
            if patience_counter >= EARLY_STOP_PATIENCE:
                return best_state, best_metrics, epoch + 1, True
        
        return best_state, best_metrics, DEFAULT_EPOCHS, False
    
    def _evaluate(self, model: FasterRCNN, val_loader: DataLoader) -> dict:
        """Evaluate model on validation set."""
        model.eval()
        
        all_preds = []
        all_targets = []
        val_loss = 0.0
        
        with torch.inference_mode():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                
                outputs = model(images)
                
                for out, tgt in zip(outputs, targets):
                    # Simplified F1 calculation
                    pred_boxes = out["boxes"].cpu()
                    pred_scores = out["scores"].cpu()
                    true_boxes = tgt["boxes"]
                    
                    # Count TP, FP, FN based on IoU threshold
                    tp, fp, fn = self._count_matches(pred_boxes, pred_scores, true_boxes)
                    all_preds.append((tp, fp))
                    all_targets.append(fn)
        
        # Aggregate metrics
        total_tp = sum(p[0] for p in all_preds)
        total_fp = sum(p[1] for p in all_preds)
        total_fn = sum(all_targets)
        
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        return {
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "val_loss": val_loss / len(val_loader) if val_loader else 0,
        }
    
    def _count_matches(self, pred_boxes, pred_scores, true_boxes, iou_thresh=0.5, score_thresh=0.5):
        """Count TP, FP, FN for F1 calculation."""
        # Filter by score
        mask = pred_scores >= score_thresh
        pred_boxes = pred_boxes[mask]
        
        if len(pred_boxes) == 0:
            return 0, 0, len(true_boxes)
        
        if len(true_boxes) == 0:
            return 0, len(pred_boxes), 0
        
        # Compute IoU matrix
        from torchvision.ops import box_iou
        iou_matrix = box_iou(pred_boxes, true_boxes)
        
        # Greedy matching
        matched_gt = set()
        tp = 0
        for i in range(len(pred_boxes)):
            max_iou, max_j = iou_matrix[i].max(0)
            if max_iou >= iou_thresh and max_j.item() not in matched_gt:
                tp += 1
                matched_gt.add(max_j.item())
        
        fp = len(pred_boxes) - tp
        fn = len(true_boxes) - len(matched_gt)
        
        return tp, fp, fn
```

---

## WebSocket API Additions

### New Commands (Flutter → Backend)

```json
// Start training for new signal
{"command": "train_signal", "signal_name": "wifi_24", "notes": "initial training"}

// Extend training with new samples
{"command": "extend_signal", "signal_name": "creamy_chicken", "notes": "added low SNR samples"}

// Cancel ongoing training
{"command": "cancel_training"}

// Version management
{"command": "promote_version", "signal_name": "creamy_chicken", "version": 3}
{"command": "rollback_signal", "signal_name": "creamy_chicken"}
{"command": "get_version_history", "signal_name": "creamy_chicken"}
{"command": "get_registry"}

// Head loading (for mission)
{"command": "load_heads", "signal_names": ["creamy_chicken", "lte_uplink"]}
{"command": "unload_heads", "signal_names": ["wifi_24"]}
{"command": "get_loaded_heads"}

// Training data management
{"command": "add_training_sample", "signal_name": "creamy_chicken", 
 "spectrogram_b64": "...", "boxes": [...], "metadata": {...}}
{"command": "get_training_samples", "signal_name": "creamy_chicken"}
```

### Response Messages (Backend → Flutter)

```json
// Training progress (every epoch)
{
    "type": "training_progress",
    "signal_name": "creamy_chicken",
    "epoch": 15,
    "total_epochs": 50,
    "train_loss": 0.0312,
    "val_loss": 0.0287,
    "f1_score": 0.92,
    "precision": 0.91,
    "recall": 0.93,
    "is_best": true,
    "elapsed_sec": 45.2
}

// Training complete
{
    "type": "training_complete",
    "signal_name": "creamy_chicken",
    "version": 3,
    "sample_count": 215,
    "epochs_trained": 18,
    "early_stopped": true,
    "metrics": {"f1_score": 0.95, "precision": 0.94, "recall": 0.96},
    "previous_version": 2,
    "previous_metrics": {"f1_score": 0.93, "precision": 0.92, "recall": 0.94},
    "auto_promoted": true,
    "promotion_reason": "F1 improved 2.2%",
    "training_time_sec": 120.5
}

// Training failed
{
    "type": "training_failed",
    "signal_name": "creamy_chicken",
    "error": "Not enough samples (minimum 5 required)"
}

// Version history
{
    "type": "version_history",
    "signal_name": "creamy_chicken",
    "active_version": 2,
    "versions": [
        {"version": 1, "f1_score": 0.91, "sample_count": 127, "created_at": "...", "notes": "..."},
        {"version": 2, "f1_score": 0.93, "sample_count": 200, "created_at": "...", "is_active": true}
    ]
}

// Registry
{
    "type": "registry",
    "backbone_version": 1,
    "signals": {
        "creamy_chicken": {"active_version": 2, "f1_score": 0.93, "sample_count": 200},
        "lte_uplink": {"active_version": 1, "f1_score": 0.87, "sample_count": 89}
    }
}

// Loaded heads
{
    "type": "loaded_heads",
    "heads": ["creamy_chicken", "lte_uplink", "wifi_24"]
}
```

---

## Server Integration

Add to `backend/server.py`:

```python
from training.service import TrainingService

class Server:
    def __init__(self):
        # ... existing init ...
        self.training_service = TrainingService()
    
    async def handle_command(self, websocket, data):
        command = data.get("command")
        
        if command == "train_signal":
            await self._handle_train(websocket, data, is_new=True)
        
        elif command == "extend_signal":
            await self._handle_train(websocket, data, is_new=False)
        
        elif command == "cancel_training":
            self.training_service.cancel_training()
        
        elif command == "get_registry":
            registry = self.training_service.version_manager.get_registry()
            await websocket.send(json.dumps({"type": "registry", **registry}))
        
        # ... other commands ...
    
    async def _handle_train(self, websocket, data, is_new):
        signal_name = data["signal_name"]
        notes = data.get("notes")
        
        def progress_callback(progress):
            asyncio.create_task(websocket.send(json.dumps({
                "type": "training_progress",
                "signal_name": signal_name,
                "epoch": progress.epoch,
                "total_epochs": progress.total_epochs,
                "train_loss": progress.train_loss,
                "val_loss": progress.val_loss,
                "f1_score": progress.f1_score,
                "is_best": progress.is_best,
                "elapsed_sec": progress.elapsed_sec,
            })))
        
        try:
            if is_new:
                result = self.training_service.train_new_signal(signal_name, notes, progress_callback)
            else:
                result = self.training_service.extend_signal(signal_name, notes, progress_callback)
            
            await websocket.send(json.dumps({
                "type": "training_complete",
                "signal_name": result.signal_name,
                "version": result.version,
                "sample_count": result.sample_count,
                "epochs_trained": result.epochs_trained,
                "early_stopped": result.early_stopped,
                "metrics": result.metrics,
                "auto_promoted": result.auto_promoted,
                "promotion_reason": result.promotion_reason,
            }))
        
        except Exception as e:
            await websocket.send(json.dumps({
                "type": "training_failed",
                "signal_name": signal_name,
                "error": str(e),
            }))
```
