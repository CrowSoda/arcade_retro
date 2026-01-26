"""
G20 Logger Configuration - Unified logging for Python backend

Usage:
    from logger_config import get_logger, configure_logging
    
    logger = get_logger('FFT')
    logger.info('Processing frame')
    logger.perf(f'Frame took {elapsed:.1f}ms')  # Only prints if perf enabled
    
Configuration:
    configure_logging(level='INFO', perf_enabled=False, perf_interval=30)

Log Levels:
    DEBUG - Verbose debugging
    INFO  - Normal operation
    PERF  - Performance metrics (special handling)
    WARN  - Non-fatal issues
    ERROR - Errors needing attention
"""

import sys
import logging
from datetime import datetime
from typing import Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

_PERF_ENABLED = False
_PERF_INTERVAL = 30  # Only log every N perf calls
_perf_counters = {}


def configure_logging(
    level: str = 'INFO',
    perf_enabled: bool = False,
    perf_interval: int = 30,
    show_timestamps: bool = False,
    modules: Optional[set] = None,
):
    """
    Configure global logging settings.
    
    Args:
        level: Minimum log level ('DEBUG', 'INFO', 'WARN', 'ERROR')
        perf_enabled: Whether to print performance logs
        perf_interval: Only print every N perf logs
        show_timestamps: Include timestamps in output
        modules: If set, only show logs from these modules
    """
    global _PERF_ENABLED, _PERF_INTERVAL
    
    _PERF_ENABLED = perf_enabled
    _PERF_INTERVAL = perf_interval
    
    # Configure root logger
    fmt = '%(message)s'
    if show_timestamps:
        fmt = '%(asctime)s.%(msecs)03d ' + fmt
    
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt='%H:%M:%S',
        stream=sys.stdout,
        force=True,  # Override existing config
    )
    
    # Silence noisy libraries
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('grpc').setLevel(logging.WARNING)


def configure_production():
    """Apply production settings (minimal output)."""
    configure_logging(level='WARNING', perf_enabled=False)


def configure_development():
    """Apply development settings (verbose)."""
    configure_logging(level='DEBUG', perf_enabled=True, perf_interval=30)


# =============================================================================
# LOGGER CLASS
# =============================================================================

class G20Logger:
    """Logger wrapper with perf logging support."""
    
    def __init__(self, name: str):
        self.name = name
        self._logger = logging.getLogger(name)
    
    def debug(self, msg: str):
        """Debug level message."""
        self._logger.debug(f'[{self.name}] {msg}')
    
    def info(self, msg: str):
        """Info level message."""
        self._logger.info(f'[{self.name}] {msg}')
    
    def warning(self, msg: str):
        """Warning level message."""
        self._logger.warning(f'[WARN] [{self.name}] {msg}')
    
    def error(self, msg: str):
        """Error level message."""
        self._logger.error(f'[ERR] [{self.name}] {msg}')
    
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
        
        self._logger.info(f'[PERF] [{self.name}] {msg}')
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


# Auto-configure on import (development mode by default in debug)
import os
if os.environ.get('G20_PROD', '').lower() in ('1', 'true', 'yes'):
    configure_production()
else:
    # Default: INFO level, perf disabled to reduce noise
    configure_logging(level='INFO', perf_enabled=False)
