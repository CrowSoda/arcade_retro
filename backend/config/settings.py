"""
Centralized configuration using Pydantic-settings v2.

Phase 4/Week 5 of the Hardening Roadmap.

Benefits:
- Single source of truth for all configuration
- Type-safe validation
- Environment variable overrides (G20_SERVER__GRPC_PORT=50052)
- Sensible defaults with documentation

Usage:
    from config.settings import get_settings
    settings = get_settings()
    print(settings.server.grpc_port)
"""

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PowerMode(str, Enum):
    """Jetson power modes."""

    LOW = "15W"
    BALANCED = "30W"
    HIGH = "50W"


class ServerSettings(BaseSettings):
    """Server configuration."""

    model_config = SettingsConfigDict(env_prefix="G20_SERVER_", env_nested_delimiter="__")

    grpc_port: int = Field(default=50051, description="gRPC server port")
    ws_port: int = Field(default=0, description="WebSocket port (0=auto pick free port)")
    max_workers: int = Field(default=10, description="gRPC thread pool size")
    localhost_only: bool = Field(default=True, description="Bind to localhost only")


class PathSettings(BaseSettings):
    """Path configuration."""

    model_config = SettingsConfigDict(env_prefix="G20_PATH_")

    base_dir: Path = Field(default=Path(__file__).parent.parent.parent)
    models_dir: Path | None = Field(default=None)
    data_dir: Path | None = Field(default=None)
    training_dir: Path | None = Field(default=None)
    config_dir: Path | None = Field(default=None)
    logs_dir: Path | None = Field(default=None)

    def model_post_init(self, __context):
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.training_dir is None:
            self.training_dir = self.base_dir / "training_data" / "signals"
        if self.config_dir is None:
            self.config_dir = self.base_dir / "config"
        if self.logs_dir is None:
            self.logs_dir = self.base_dir / "logs"


class FFTSettings(BaseSettings):
    """FFT configuration.

    CRITICAL: inference FFT is LOCKED - must match training parameters!
    """

    model_config = SettingsConfigDict(env_prefix="G20_FFT_")

    # Inference FFT - DO NOT CHANGE without retraining models!
    inference_nfft: int = Field(default=4096, description="LOCKED - matches training")
    inference_hop: int = Field(default=2048, description="LOCKED - 50% overlap")
    inference_dynamic_range_db: float = Field(default=80.0, description="LOCKED")
    inference_accumulation_frames: int = Field(default=6, description="Frames to accumulate")

    # Waterfall FFT - can be adjusted for display
    waterfall_nfft: int = Field(default=65536, description="High resolution for display")
    waterfall_dynamic_range_db: float = Field(default=60.0, description="Display range")


class InferenceSettings(BaseSettings):
    """Inference configuration."""

    model_config = SettingsConfigDict(env_prefix="G20_INFERENCE_")

    score_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Detection confidence threshold"
    )
    nms_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Non-max suppression threshold"
    )
    max_detections: int = Field(default=100, ge=1, description="Maximum detections per frame")
    use_tensorrt: bool = Field(default=True, description="Use TensorRT if available")
    use_dla: bool = Field(default=False, description="Use DLA for power efficiency (Jetson)")
    int8_quantization: bool = Field(default=True, description="Use INT8 quantization")

    # Class names - loaded from registry, but defaults here
    class_names: list[str] = Field(default=["background", "signal"])


class SDRSettings(BaseSettings):
    """SDR hardware configuration."""

    model_config = SettingsConfigDict(env_prefix="G20_SDR_")

    min_freq_mhz: float = Field(default=30.0, description="Minimum tunable frequency")
    max_freq_mhz: float = Field(default=6000.0, description="Maximum tunable frequency")
    default_center_freq_mhz: float = Field(default=825.0, description="Default center frequency")
    default_bandwidth_mhz: float = Field(default=20.0, description="Default bandwidth")
    sample_rate_mhz: float = Field(default=20.0, description="Sample rate")
    max_gain_db: float = Field(default=34.0, description="Maximum gain")
    default_gain_db: float = Field(default=20.0, description="Default gain")


class DisplaySettings(BaseSettings):
    """Display/waterfall configuration."""

    model_config = SettingsConfigDict(env_prefix="G20_DISPLAY_")

    target_fps: int = Field(default=30, description="Target frames per second")
    time_span_seconds: float = Field(default=2.5, description="Default time span")
    default_colormap: str = Field(default="viridis", description="Default colormap")
    colormap_options: list[str] = Field(
        default=["viridis", "plasma", "inferno", "magma", "turbo"],
        description="Available colormaps",
    )


class JetsonSettings(BaseSettings):
    """Jetson-specific configuration."""

    model_config = SettingsConfigDict(env_prefix="G20_JETSON_")

    power_mode: PowerMode = Field(default=PowerMode.BALANCED, description="Power mode")
    lock_clocks: bool = Field(default=True, description="Run jetson_clocks on startup")
    gpu_memory_limit_gb: float = Field(default=12.0, description="Max GPU memory")
    enable_mps: bool = Field(default=True, description="Enable Multi-Process Service")


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="G20_LOG_")

    level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")
    format: str = Field(default="json", description="Log format (json, text)")
    file_enabled: bool = Field(default=True, description="Enable file logging")
    console_enabled: bool = Field(default=True, description="Enable console logging")
    max_file_size_mb: int = Field(default=10, description="Max log file size before rotation")
    backup_count: int = Field(default=5, description="Number of backup log files")


class Settings(BaseSettings):
    """Root settings combining all sub-settings."""

    model_config = SettingsConfigDict(env_prefix="G20_", env_nested_delimiter="__")

    server: ServerSettings = Field(default_factory=ServerSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    fft: FFTSettings = Field(default_factory=FFTSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    sdr: SDRSettings = Field(default_factory=SDRSettings)
    display: DisplaySettings = Field(default_factory=DisplaySettings)
    jetson: JetsonSettings = Field(default_factory=JetsonSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Environment
    debug: bool = Field(default=False, description="Enable debug mode")


# Global singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings (useful for testing)."""
    global _settings
    _settings = Settings()
    return _settings
