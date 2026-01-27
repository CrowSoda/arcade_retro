"""
G20 Logger Configuration - Unified logging for Python backend with FILE TRACKING

**LOG FILES:**
- g20_demo/logs/g20_YYYY-MM-DD_HHMMSS.log - DTG-stamped log files
- Max 500MB total storage, oldest files auto-deleted
- Console: WARNING+ only (silent during normal ops)
- File: INFO+ (everything tracked)

Usage:
    from logger_config import get_logger

    logger = get_logger('FFT')
    logger.info('Processing frame', extra={'frame_idx': 42})
    logger.perf(f'Frame took {elapsed:.1f}ms')  # Only prints if perf enabled
"""

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Maximum total log storage in bytes (500 MB default)
MAX_LOG_STORAGE_BYTES = 500 * 1024 * 1024

# Individual log file max size before rotation (50 MB)
MAX_LOG_FILE_SIZE = 50 * 1024 * 1024

# Calculated max backups based on storage limit
MAX_BACKUP_COUNT = (MAX_LOG_STORAGE_BYTES // MAX_LOG_FILE_SIZE) - 1

_PERF_ENABLED = False
_PERF_INTERVAL = 30  # Only log every N perf calls
_perf_counters = {}


def _cleanup_old_logs(logs_dir: Path, max_bytes: int = MAX_LOG_STORAGE_BYTES):
    """Delete oldest log files when total size exceeds max_bytes."""
    try:
        log_files = sorted(
            logs_dir.glob("g20_*.log*"),
            key=lambda f: f.stat().st_mtime,
        )

        total_size = sum(f.stat().st_size for f in log_files)

        while total_size > max_bytes and len(log_files) > 1:
            oldest = log_files.pop(0)
            file_size = oldest.stat().st_size
            oldest.unlink()
            total_size -= file_size
    except Exception:
        pass  # Best effort cleanup


def _get_log_filename() -> str:
    """Generate DTG-stamped log filename: g20_YYYY-MM-DD_HHMMSS.log"""
    dtg = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"g20_{dtg}.log"


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production monitoring."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if provided
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def configure_logging(
    level: str = "INFO",
    perf_enabled: bool = False,
    perf_interval: int = 30,
    show_timestamps: bool = True,
    json_format: bool = False,
    modules: set | None = None,
    enable_file_logging: bool = True,
    max_storage_mb: int = 500,
):
    """
    Configure global logging settings with FILE TRACKING.

    Args:
        level: Minimum log level ('DEBUG', 'INFO', 'WARN', 'ERROR')
        perf_enabled: Whether to print performance logs
        perf_interval: Only print every N perf logs
        show_timestamps: Include timestamps in output (default: True)
        json_format: Use structured JSON logging (recommended for production)
        modules: If set, only show logs from these modules
        enable_file_logging: Save logs to files with rotation (default: True)
        max_storage_mb: Maximum log storage in MB (default: 500)
    """
    global _PERF_ENABLED, _PERF_INTERVAL, MAX_LOG_STORAGE_BYTES

    _PERF_ENABLED = perf_enabled
    _PERF_INTERVAL = perf_interval
    MAX_LOG_STORAGE_BYTES = max_storage_mb * 1024 * 1024

    # Create logs directory
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Configure formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s"
        if not show_timestamps:
            fmt = "[%(levelname)s] %(message)s"
        formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler (stdout) - ONLY WARNINGS AND ERRORS
    # Silent during normal operations, everything goes to file
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)  # SILENT for INFO - only warn/error

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    # FILE LOGGING - Single log file with DTG in name
    if enable_file_logging:
        # Clean up old logs before creating new one
        _cleanup_old_logs(logs_dir, MAX_LOG_STORAGE_BYTES)

        # Create log file with DTG timestamp
        log_filename = _get_log_filename()
        main_log_file = logs_dir / log_filename

        # Rotating file handler - rotates when file gets too big
        file_handler = RotatingFileHandler(
            main_log_file,
            maxBytes=MAX_LOG_FILE_SIZE,  # 50 MB per file
            backupCount=MAX_BACKUP_COUNT,  # Calculated from storage limit
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Silence noisy libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("grpc").setLevel(logging.WARNING)


def configure_production():
    """Apply production settings (minimal output)."""
    configure_logging(level="WARNING", perf_enabled=False)


def configure_development():
    """Apply development settings (verbose)."""
    configure_logging(level="DEBUG", perf_enabled=True, perf_interval=30)


# =============================================================================
# LOGGER CLASS
# =============================================================================


class G20Logger:
    """Logger wrapper with perf logging support."""

    def __init__(self, name: str):
        self.name = name
        self._logger = logging.getLogger(name)

    def debug(self, msg: str, extra: dict | None = None):
        """Debug level message."""
        self._log(logging.DEBUG, f"[{self.name}] {msg}", extra)

    def info(self, msg: str, extra: dict | None = None):
        """Info level message."""
        self._log(logging.INFO, f"[{self.name}] {msg}", extra)

    def warning(self, msg: str, extra: dict | None = None):
        """Warning level message."""
        self._log(logging.WARNING, f"[WARN] [{self.name}] {msg}", extra)

    def error(self, msg: str, extra: dict | None = None):
        """Error level message."""
        self._log(logging.ERROR, f"[ERR] [{self.name}] {msg}", extra)

    def _log(self, level: int, msg: str, extra: dict | None = None):
        """Internal logging with extra fields support."""
        if extra:
            # Create a LogRecord with extra fields
            record = self._logger.makeRecord(
                self._logger.name,
                level,
                "(unknown file)",
                0,
                msg,
                (),
                None,
            )
            record.extra_fields = extra
            self._logger.handle(record)
        else:
            self._logger.log(level, msg)

    def perf(self, msg: str) -> bool:
        """
        Performance log - throttled by perf_interval.

        Returns True if the message was actually logged.
        """
        if not _PERF_ENABLED:
            return False

        # Increment counter
        _perf_counters[self.name] = _perf_counters.get(self.name, 0) + 1

        # Check interval
        if _perf_counters[self.name] % _PERF_INTERVAL != 0:
            return False

        self._logger.info(f"[PERF] [{self.name}] {msg}")
        return True


def get_logger(name: str) -> G20Logger:
    """Get a logger instance for the given module name."""
    return G20Logger(name)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def reset_perf_counters():
    """Reset all performance counters."""
    global _perf_counters
    _perf_counters.clear()


def is_perf_enabled() -> bool:
    """Check if performance logging is enabled."""
    return _PERF_ENABLED


def get_log_storage_used() -> int:
    """Get current log storage used in bytes."""
    logs_dir = Path(__file__).parent.parent / "logs"
    if not logs_dir.exists():
        return 0
    return sum(f.stat().st_size for f in logs_dir.glob("g20_*.log*"))


# Auto-configure on import with FILE LOGGING enabled

if os.environ.get("G20_PROD", "").lower() in ("1", "true", "yes"):
    configure_production()
else:
    # Default: INFO level, perf disabled, FILE LOGGING ENABLED
    configure_logging(
        level="INFO",
        perf_enabled=False,
        show_timestamps=True,
        enable_file_logging=True,
        max_storage_mb=500,  # 500 MB max log storage
    )
