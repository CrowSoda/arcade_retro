"""
Data classes for server state management.

Extracted from server.py using strangler fig pattern.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ChannelState:
    """State for a single SDR channel."""

    channel_number: int
    center_freq_mhz: float = 825.0
    bandwidth_mhz: float = 20.0
    gain_db: float = 20.0
    state: str = "IDLE"
    active_capture_id: str | None = None


@dataclass
class CaptureSession:
    """Active capture session state."""

    capture_id: str
    signal_name: str
    rx_channel: int
    duration_seconds: int
    start_time: float
    file_path: str
    bytes_captured: int = 0
    is_active: bool = True


@dataclass
class InferenceSession:
    """Active inference session state."""

    session_id: str
    model_id: str
    source_type: str
    source_path: str | None
    rx_channel: int | None
    config: dict[str, Any]
    start_time: float
    chunk_count: int = 0
    detection_count: int = 0
    is_active: bool = True


@dataclass
class ModelState:
    """Loaded model state."""

    model_id: str
    model_name: str
    model_path: str
    model_hash: str
    backend: str
    num_classes: int
    class_names: list[str]
    engine: Any = None  # InferenceEngine - Any to avoid circular import
    loaded_at: float = 0.0
