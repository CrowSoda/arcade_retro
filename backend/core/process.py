"""
Parent process monitoring for orphan prevention.

Extracted from server.py using strangler fig pattern.
Monitors parent process and triggers shutdown if parent dies.
"""

import os
import sys
import threading
from collections.abc import Callable

from logger_config import get_logger

logger = get_logger("watchdog")

# Parent process ID (for orphan detection)
_parent_pid: int | None = None


def _is_parent_alive() -> bool:
    """Check if the parent process is still running."""
    if _parent_pid is None:
        return True  # No parent tracking, assume alive

    if sys.platform == "win32":
        # Windows: Use ctypes to check if process exists
        import ctypes

        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, _parent_pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    else:
        # Unix: Check if process exists
        try:
            os.kill(_parent_pid, 0)  # Signal 0 = check if process exists
            return True
        except (OSError, ProcessLookupError):
            return False


def _watchdog_thread(on_orphan: Callable[[], None], check_interval: float = 2.0):
    """Background thread that monitors parent process."""
    from . import shutdown

    logger.info(f"Watchdog started monitoring parent PID: {_parent_pid}")

    while not shutdown.is_shutting_down():
        if not _is_parent_alive():
            logger.warning(
                f"Parent process {_parent_pid} died, initiating shutdown",
                extra={"parent_pid": _parent_pid},
            )
            on_orphan()
            break

        # Check every N seconds
        shutdown.wait_for_shutdown(timeout=check_interval)

    logger.info("Watchdog stopped")


def start_parent_watchdog(
    on_orphan: Callable[[], None] | None = None,
    check_interval: float = 2.0,
):
    """
    Start the parent process watchdog thread.

    Args:
        on_orphan: Callback when parent dies. Defaults to request_shutdown().
        check_interval: How often to check parent (seconds).
    """
    global _parent_pid
    _parent_pid = os.getppid()

    # Don't monitor if parent is init (PID 1) or system process
    if _parent_pid <= 1:
        logger.info(f"Watchdog: Parent PID is {_parent_pid}, not monitoring")
        return

    # Default callback
    if on_orphan is None:
        from . import shutdown

        on_orphan = shutdown.request_shutdown

    watchdog = threading.Thread(
        target=_watchdog_thread,
        args=(on_orphan, check_interval),
        daemon=True,
        name="ParentWatchdog",
    )
    watchdog.start()


def get_parent_pid() -> int | None:
    """Get the monitored parent PID."""
    return _parent_pid
