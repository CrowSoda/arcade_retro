"""
Crop classifier API handlers.

WebSocket commands:
- crop_detect: Run blob detection
- crop_train: Train Siamese model
- crop_infer: Run inference
- crop_label: Store labels
- crop_status: Get status
- crop_load_model: Load trained model
"""

from .handlers import (
    CropClassifierHandler,
    get_handler,
    handle_crop_command,
)

__all__ = [
    "CropClassifierHandler",
    "get_handler",
    "handle_crop_command",
]
