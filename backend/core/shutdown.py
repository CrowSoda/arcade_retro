"""
Graceful shutdown coordination for multi-threaded server.

Extracted from server.py using strangler fig pattern.
Provides cross-thread and async shutdown signaling.
"""

import asyncio
import atexit
import signal
import sys
import threading
from typing import Any

from logger_config import get_logger

logger = get_logger("shutdown")

# Threading event for cross-thread shutdown signaling
_shutdown_event = threading.Event()

# Asyncio event for async code (created per event loop)
_async_shutdown_event: asyncio.Event | None = None

# Track all resources that need cleanup
_cleanup_resources: list[Any] = []


def register_cleanup(resource: Any):
    """Register a resource for cleanup on shutdown."""
    _cleanup_resources.append(resource)


def is_shutting_down() -> bool:
    """Check if shutdown has been requested."""
    return _shutdown_event.is_set()


def request_shutdown():
    """Signal all components to begin shutdown."""
    logger.info("Shutdown requested")
    _shutdown_event.set()
    if _async_shutdown_event:
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(_async_shutdown_event.set)
        except RuntimeError:
            pass  # No running loop


def wait_for_shutdown(timeout: float | None = None) -> bool:
    """Wait for shutdown signal (blocking). Returns True if signaled."""
    return _shutdown_event.wait(timeout=timeout)


async def wait_for_shutdown_async():
    """Async wait for shutdown signal."""
    global _async_shutdown_event
    if _async_shutdown_event is None:
        _async_shutdown_event = asyncio.Event()
        if _shutdown_event.is_set():
            _async_shutdown_event.set()
    await _async_shutdown_event.wait()


def get_async_event() -> asyncio.Event:
    """Get or create the async shutdown event for current loop."""
    global _async_shutdown_event
    if _async_shutdown_event is None:
        _async_shutdown_event = asyncio.Event()
        if _shutdown_event.is_set():
            _async_shutdown_event.set()
    return _async_shutdown_event


def cleanup_all():
    """Clean up all registered resources."""
    logger.info("Cleaning up resources", extra={"resource_count": len(_cleanup_resources)})

    for resource in reversed(_cleanup_resources):
        try:
            if hasattr(resource, "close"):
                resource.close()
            elif hasattr(resource, "stop"):
                resource.stop()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    _cleanup_resources.clear()
    logger.info("Cleanup complete")


def _signal_handler(signum, frame):
    """Handle shutdown signals (SIGINT, SIGTERM, SIGBREAK)."""
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    logger.warning(f"Received {sig_name}, initiating graceful shutdown", extra={"signal": sig_name})
    request_shutdown()


def setup_signal_handlers():
    """Set up platform-appropriate signal handlers."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if sys.platform == "win32":
        # Windows: SIGBREAK is sent on Ctrl+Break and by taskkill
        signal.signal(signal.SIGBREAK, _signal_handler)
    logger.info("Signal handlers registered")


# Register cleanup on process exit
atexit.register(cleanup_all)
