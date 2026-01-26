"""
Training Service - Train detection heads with frozen backbone.

Key features:
    - Frozen backbone (no gradients)
    - Early stopping with patience=5
    - Progress callbacks for UI updates
    - Auto-promotion check after training
"""

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.ops import box_iou

from ..hydra.config import (
    DEFAULT_EPOCHS,
    EARLY_STOP_PATIENCE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_BATCH_SIZE,
    MIN_SAMPLES_FOR_TRAINING,
)
from ..hydra.version_manager import VersionManager
from .dataset import create_data_loaders
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
        
        self.version_manager = VersionManager(str(models_dir))
        self.split_manager = SplitManager(str(training_data_dir))
        
        # Training state
        self._is_training = False
        self._cancel_requested = False
        self._current_signal = None
    
    @property
    def is_training(self) -> bool:
        return self._is_training
    
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
        self._current_signal = signal_name
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
                raise ValueError(f"Need at least {MIN_SAMPLES_FOR_TRAINING} samples, got {sample_count}")
            
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
                model, train_loader, val_loader, callback, start_time
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
            self._current_signal = None
    
    def _build_model(self) -> FasterRCNN:
        """Build FasterRCNN with frozen backbone."""
        
        # Load backbone weights
        backbone_path = self.models_dir / "backbone" / "active.pth"
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone not found: {backbone_path}")
        
        backbone_state = torch.load(backbone_path, map_location=self.device, weights_only=False)
        
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
            head_state = torch.load(head_path, map_location=self.device, weights_only=False)
            model.load_state_dict(head_state, strict=False)
    
    def _train_loop(
        self,
        model: FasterRCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callback: Callable,
        start_time: float
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
                print(f"Training cancelled at epoch {epoch}")
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
            
            train_loss /= max(len(train_loader), 1)
            
            # Evaluate
            metrics = self._evaluate(model, val_loader)
            metrics["train_loss"] = train_loss
            
            # Check for improvement
            is_best = metrics["f1_score"] > best_f1
            if is_best:
                best_f1 = metrics["f1_score"]
                # Save head weights only (exclude backbone)
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()
                             if not k.startswith("backbone.")}
                best_metrics = metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Callback
            if callback:
                elapsed = time.time() - start_time
                progress = TrainingProgress(
                    epoch=epoch + 1,
                    total_epochs=DEFAULT_EPOCHS,
                    train_loss=train_loss,
                    val_loss=metrics["val_loss"],
                    f1_score=metrics["f1_score"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    is_best=is_best,
                    elapsed_sec=elapsed
                )
                callback(progress)
            
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, f1={metrics['f1_score']:.3f}" +
                  (" *" if is_best else ""))
            
            # Early stopping
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                return best_state, best_metrics, epoch + 1, True
        
        return best_state, best_metrics, DEFAULT_EPOCHS, False
    
    def _evaluate(self, model: FasterRCNN, val_loader: DataLoader) -> dict:
        """Evaluate model on validation set."""
        model.eval()
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_loss = 0.0
        
        with torch.inference_mode():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                
                outputs = model(images)
                
                for out, tgt in zip(outputs, targets):
                    pred_boxes = out["boxes"].cpu()
                    pred_scores = out["scores"].cpu()
                    true_boxes = tgt["boxes"]
                    
                    # Count TP, FP, FN
                    tp, fp, fn = self._count_matches(pred_boxes, pred_scores, true_boxes)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        return {
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "val_loss": total_loss / max(len(val_loader), 1),
        }
    
    def _count_matches(
        self, 
        pred_boxes: torch.Tensor, 
        pred_scores: torch.Tensor, 
        true_boxes: torch.Tensor,
        iou_thresh: float = 0.5, 
        score_thresh: float = 0.5
    ) -> Tuple[int, int, int]:
        """Count TP, FP, FN for F1 calculation."""
        # Filter by score
        mask = pred_scores >= score_thresh
        pred_boxes = pred_boxes[mask]
        
        if len(pred_boxes) == 0:
            return 0, 0, len(true_boxes)
        
        if len(true_boxes) == 0:
            return 0, len(pred_boxes), 0
        
        # Compute IoU matrix
        iou_matrix = box_iou(pred_boxes, true_boxes)
        
        # Greedy matching
        matched_gt = set()
        tp = 0
        for i in range(len(pred_boxes)):
            if iou_matrix.shape[1] == 0:
                break
            max_iou, max_j = iou_matrix[i].max(0)
            if max_iou >= iou_thresh and max_j.item() not in matched_gt:
                tp += 1
                matched_gt.add(max_j.item())
        
        fp = len(pred_boxes) - tp
        fn = len(true_boxes) - len(matched_gt)
        
        return tp, fp, fn
    
    def get_training_status(self) -> dict:
        """Get current training status."""
        return {
            "is_training": self._is_training,
            "current_signal": self._current_signal,
        }
