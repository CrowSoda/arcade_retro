"""
API endpoint exposing backend capabilities to frontend.

The frontend queries this to know:
- What FFT sizes are available
- What colormaps are supported
- SDR frequency range
- Power modes
etc.

This prevents hardcoding values on the frontend that might differ from backend.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

from .settings import get_settings


@dataclass
class FFTCapabilities:
    """Available FFT configurations."""

    inference_nfft: int
    inference_locked: bool = True  # Cannot change without retraining
    waterfall_nfft: int = 65536
    waterfall_nfft_options: list[int] = field(
        default_factory=lambda: [4096, 8192, 16384, 32768, 65536]
    )


@dataclass
class DisplayCapabilities:
    """Display configuration options."""

    fps_options: list[int] = field(default_factory=lambda: [10, 15, 20, 30, 60])
    time_span_options: list[float] = field(default_factory=lambda: [1.0, 2.5, 5.0, 10.0])
    colormap_options: list[str] = field(
        default_factory=lambda: ["viridis", "plasma", "inferno", "magma", "turbo"]
    )
    default_colormap: str = "viridis"
    default_fps: int = 30
    default_time_span: float = 2.5


@dataclass
class SDRCapabilities:
    """SDR hardware capabilities."""

    min_freq_mhz: float = 30.0
    max_freq_mhz: float = 6000.0
    bandwidth_options_mhz: list[float] = field(
        default_factory=lambda: [5.0, 10.0, 20.0, 25.0, 40.0, 50.0]
    )
    sample_rate_mhz: float = 20.0
    max_gain_db: float = 34.0
    default_center_freq_mhz: float = 825.0
    default_bandwidth_mhz: float = 20.0
    default_gain_db: float = 20.0


@dataclass
class InferenceCapabilities:
    """Inference capabilities."""

    score_threshold_range: tuple = (0.0, 1.0)
    default_score_threshold: float = 0.5
    max_detections: int = 100
    tensorrt_available: bool = True
    dla_available: bool = False
    int8_available: bool = True


@dataclass
class PowerCapabilities:
    """Power mode options (Jetson)."""

    modes: list[str] = field(default_factory=lambda: ["15W", "30W", "50W"])
    default_mode: str = "30W"


@dataclass
class BackendCapabilities:
    """Complete backend capabilities for frontend configuration."""

    fft: FFTCapabilities
    display: DisplayCapabilities
    sdr: SDRCapabilities
    inference: InferenceCapabilities
    power: PowerCapabilities
    version: str = "1.0.0"
    api_version: str = "v1"


def get_capabilities() -> dict[str, Any]:
    """Get backend capabilities as JSON-serializable dict."""
    settings = get_settings()

    caps = BackendCapabilities(
        fft=FFTCapabilities(
            inference_nfft=settings.fft.inference_nfft,
            waterfall_nfft=settings.fft.waterfall_nfft,
        ),
        display=DisplayCapabilities(
            colormap_options=settings.display.colormap_options,
            default_colormap=settings.display.default_colormap,
            default_fps=settings.display.target_fps,
            default_time_span=settings.display.time_span_seconds,
        ),
        sdr=SDRCapabilities(
            min_freq_mhz=settings.sdr.min_freq_mhz,
            max_freq_mhz=settings.sdr.max_freq_mhz,
            sample_rate_mhz=settings.sdr.sample_rate_mhz,
            max_gain_db=settings.sdr.max_gain_db,
            default_center_freq_mhz=settings.sdr.default_center_freq_mhz,
            default_bandwidth_mhz=settings.sdr.default_bandwidth_mhz,
            default_gain_db=settings.sdr.default_gain_db,
        ),
        inference=InferenceCapabilities(
            default_score_threshold=settings.inference.score_threshold,
            max_detections=settings.inference.max_detections,
            tensorrt_available=settings.inference.use_tensorrt,
            dla_available=settings.inference.use_dla,
            int8_available=settings.inference.int8_quantization,
        ),
        power=PowerCapabilities(
            default_mode=settings.jetson.power_mode.value,
        ),
    )

    return asdict(caps)


def get_capabilities_json() -> str:
    """Get capabilities as JSON string."""
    import json

    return json.dumps(get_capabilities(), indent=2)
