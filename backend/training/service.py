"""
Training Service - Train detection heads with frozen backbone.

Key features:
    - Frozen backbone (no gradients)
    - Research-based training presets (FAST, BALANCED, QUALITY)
    - Early stopping with configurable patience
    - Progress callbacks for UI updates
    - Auto-promotion check after training

Training presets based on research:
    - TFA (ICML 2020): Fine-tuning with frozen backbone
    - DeFRCN (ICCV 2021): SGD settings for detection  
    - Foundation Models (CVPR 2024): Adam with frozen backbone
    - Intel batch size research: Small batch → flat minima → better generalization
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
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou

from hydra.config import (
    TrainingPreset,
    TrainingConfig,
    get_training_config,
    get_preset_by_name,
    DEFAULT_PRESET,
    # Legacy imports for backwards compatibility
    DEFAULT_EPOCHS,
    EARLY_STOP_PATIENCE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_BATCH_SIZE,
    MIN_SAMPLES_FOR_TRAINING,
)
from hydra.version_manager import VersionManager
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
        preset: str = "balanced",
        notes: str = None,
        callback: Callable[[TrainingProgress], None] = None
    ) -> TrainingResult:
        """Train head for brand new signal.
        
        Args:
            signal_name: Name of the signal class
            preset: Training preset - "fast", "balanced", or "quality"
            notes: Optional training notes
            callback: Progress callback function
        """
        # Get config for preset
        config = get_training_config(get_preset_by_name(preset))
        
        # Create initial split
        self.split_manager.create_initial_split(signal_name)
        
        return self._train(signal_name, notes, callback, is_new=True, config=config)
    
    def extend_signal(
        self,
        signal_name: str,
        preset: str = "balanced",
        notes: str = None,
        callback: Callable[[TrainingProgress], None] = None
    ) -> TrainingResult:
        """Continue training with additional samples.
        
        Args:
            signal_name: Name of the signal class
            preset: Training preset - "fast", "balanced", or "quality"
            notes: Optional training notes
            callback: Progress callback function
        """
        # Get config for preset
        config = get_training_config(get_preset_by_name(preset))
        
        # Extend split with new samples
        self.split_manager.extend_split(signal_name)
        
        return self._train(signal_name, notes, callback, is_new=False, config=config)
    
    def cancel_training(self):
        """Request training cancellation."""
        self._cancel_requested = True
    
    def _train(
        self,
        signal_name: str,
        notes: str,
        callback: Callable,
        is_new: bool,
        config: TrainingConfig = None
    ) -> TrainingResult:
        """Internal training implementation.
        
        Args:
            signal_name: Name of the signal class
            notes: Optional training notes
            callback: Progress callback function
            is_new: Whether this is a new signal (vs extending)
            config: Training configuration (defaults to BALANCED preset)
        """
        # Use default config if not provided
        if config is None:
            config = get_training_config(DEFAULT_PRESET)
        
        self._is_training = True
        self._cancel_requested = False
        self._current_signal = signal_name
        self._current_config = config
        start_time = time.time()
        
        print(f"[Training] Starting with preset: epochs={config.epochs}, lr={config.learning_rate}, "
              f"batch={config.batch_size}, patience={config.early_stop_patience}")
        
        try:
            # Build model with frozen backbone
            model = self._build_model()
            
            # Load existing head weights if extending
            if not is_new:
                self._load_head_weights(model, signal_name)
            
            # Create data loaders with config batch size
            train_loader, val_loader = create_data_loaders(
                signal_name,
                str(self.training_data_dir),
                batch_size=config.batch_size
            )
            
            sample_count = len(train_loader.dataset) + len(val_loader.dataset)
            
            if sample_count < config.min_samples:
                raise ValueError(f"Need at least {config.min_samples} samples, got {sample_count}")
            
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
            
            # Training loop with config
            best_state, best_metrics, epochs_trained, early_stopped = self._train_loop(
                model, train_loader, val_loader, callback, start_time, config
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
        """Build FasterRCNN with frozen backbone and custom anchors for signal detection.
        
        Signal boxes are typically ~12px wide × ~84px tall (aspect ratio ~0.15).
        Default Faster R-CNN anchors (32-512px, aspects 0.5-2.0) are too big.
        Custom anchors: smaller sizes (8-128px) + narrow aspects (0.1-0.3).
        """
        
        # Load backbone weights
        backbone_path = self.models_dir / "backbone" / "active.pth"
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone not found: {backbone_path}")
        
        print(f"\n[DEBUG] Loading backbone from: {backbone_path}")
        backbone_state = torch.load(backbone_path, map_location=self.device, weights_only=False)
        print(f"[DEBUG] Backbone state keys: {len(backbone_state)} keys")
        print(f"[DEBUG] First 5 keys: {list(backbone_state.keys())[:5]}")
        
        # Create backbone
        backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=0)
        
        # Custom anchor generator for narrow signal boxes (~12x84 pixels, aspect ~0.15)
        # Sizes cover the range: min box ~4x65, max box ~58x171
        # Aspect ratios: 0.1, 0.15, 0.2, 0.3 for narrow vertical signals
        # IMPORTANT: FPN has 5 feature levels, so we need 5 tuples (one per level)
        anchor_generator = AnchorGenerator(
            sizes=((8, 16, 32, 64, 128),) * 5,  # Same sizes for all 5 FPN levels
            aspect_ratios=((0.1, 0.15, 0.2, 0.3),) * 5  # Same aspects for all 5 FPN levels
        )
        
        print(f"[DEBUG] Custom anchor generator:")
        print(f"  sizes: {anchor_generator.sizes}")
        print(f"  aspect_ratios: {anchor_generator.aspect_ratios}")
        
        # Create model with custom anchors
        model = FasterRCNN(
            backbone, 
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            # Lower RPN thresholds for small objects
            rpn_fg_iou_thresh=0.5,  # Default 0.7 is too strict for small boxes
            rpn_bg_iou_thresh=0.3,  # Default 0.3
        )
        
        # Load backbone weights
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        print(f"[DEBUG] Loaded weights - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"[DEBUG] First 5 missing: {missing[:5]}")
        
        # UNFREEZE BACKBONE - train everything (FULL END-TO-END)
        trainable_count = 0
        backbone_trainable = 0
        for name, param in model.named_parameters():
            param.requires_grad = True
            trainable_count += 1
            if name.startswith("backbone."):
                backbone_trainable += 1
        
        print(f"[DEBUG] ******* FULL END-TO-END TRAINING *******")
        print(f"[DEBUG] Total trainable params: {trainable_count}")
        print(f"[DEBUG] Backbone trainable params: {backbone_trainable} (UNFROZEN!)")
        print(f"[DEBUG] If backbone_trainable > 0, backbone IS training!")
        
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
        start_time: float,
        config: TrainingConfig = None
    ) -> Tuple[dict, dict, int, bool]:
        """Training loop with early stopping.
        
        Uses config for epochs, learning rate, and early stopping patience.
        """
        # Get values from config or use defaults
        if config is None:
            config = get_training_config(DEFAULT_PRESET)
        
        epochs = config.epochs
        learning_rate = config.learning_rate
        early_stop_patience = config.early_stop_patience
        
        # Only optimize non-frozen params
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        print(f"[Training] LR: {learning_rate}", flush=True)
        
        best_f1 = 0.0
        best_state = None
        best_metrics = None
        patience_counter = 0
        
        for epoch in range(epochs):
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
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
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
                    total_epochs=epochs,
                    train_loss=train_loss,
                    val_loss=metrics["val_loss"],
                    f1_score=metrics["f1_score"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    is_best=is_best,
                    elapsed_sec=elapsed
                )
                callback(progress)
            
            print(f"Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, f1={metrics['f1_score']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}" +
                  (" *" if is_best else ""), flush=True)
            
            # Early stopping DISABLED - run full epochs
            # if patience_counter >= early_stop_patience:
            #     print(f"Early stopping at epoch {epoch+1} (patience={early_stop_patience})")
            #     return best_state, best_metrics, epoch + 1, True
        
        return best_state, best_metrics, epochs, False
    
    def _evaluate(self, model: FasterRCNN, val_loader: DataLoader, debug_first: bool = True) -> dict:
        """Evaluate model on validation set."""
        model.eval()
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_loss = 0.0
        first_sample = True
        
        with torch.inference_mode():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                
                outputs = model(images)
                
                for out, tgt in zip(outputs, targets):
                    pred_boxes = out["boxes"].cpu()
                    pred_scores = out["scores"].cpu()
                    pred_labels = out["labels"].cpu()
                    true_boxes = tgt["boxes"]
                    
                    # DEBUG: Print first sample predictions
                    if first_sample and debug_first:
                        # Check score threshold
                        print(f"\n[DEBUG EVAL] roi_heads.score_thresh: {model.roi_heads.score_thresh}")
                        print(f"[DEBUG EVAL] First val sample:")
                        print(f"  True boxes: {true_boxes.shape[0]} boxes")
                        if len(true_boxes) > 0:
                            print(f"  True box[0]: {true_boxes[0].tolist()}")
                        print(f"  Pred boxes (default thresh): {pred_boxes.shape[0]} predictions")
                        print(f"  Pred scores: {pred_scores[:5].tolist() if len(pred_scores) > 0 else 'none'}")
                        
                        # Check with VERY LOW threshold to see raw predictions
                        original_thresh = model.roi_heads.score_thresh
                        model.roi_heads.score_thresh = 0.001
                        raw_outputs = model(images)
                        model.roi_heads.score_thresh = original_thresh
                        
                        raw_boxes = raw_outputs[0]["boxes"].cpu()
                        raw_scores = raw_outputs[0]["scores"].cpu()
                        print(f"  Raw preds (thresh=0.001): {len(raw_boxes)} boxes")
                        if len(raw_scores) > 0:
                            print(f"  Top 10 raw scores: {raw_scores[:10].tolist()}")
                            print(f"  Raw box[0]: {raw_boxes[0].tolist() if len(raw_boxes) > 0 else 'none'}")
                        
                        first_sample = False
                    
                    # Count TP, FP, FN
                    tp, fp, fn = self._count_matches(pred_boxes, pred_scores, true_boxes)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        print(f"[DEBUG EVAL] Total: TP={total_tp}, FP={total_fp}, FN={total_fn}")
        
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
