"""
Runtime Info - Manages server.json for Flutter ↔ Backend port discovery.

This replaces stdout parsing with a proper file-based approach.

Usage:
    from runtime_info import write_server_info, clear_server_info

    # When server starts
    write_server_info(ws_port=60617, grpc_port=50051)

    # When server shuts down
    clear_server_info()

    # Flutter reads: g20_demo/runtime/server.json
"""

import json
import os
from datetime import datetime
from pathlib import Path

# Runtime directory relative to backend/
RUNTIME_DIR = Path(__file__).parent.parent / "runtime"
SERVER_INFO_FILE = RUNTIME_DIR / "server.json"


def write_server_info(ws_port: int, grpc_port: int = 50051) -> Path:
    """
    Write server connection info to runtime/server.json.

    Called when the WebSocket server is ready.
    Flutter reads this file to discover the port.

    Returns the path to the file.
    """
    RUNTIME_DIR.mkdir(exist_ok=True)

    info = {
        "ws_port": ws_port,
        "grpc_port": grpc_port,
        "pid": os.getpid(),
        "started_at": datetime.utcnow().isoformat() + "Z",
        "ready": True,
    }

    SERVER_INFO_FILE.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return SERVER_INFO_FILE


def clear_server_info():
    """
    Remove server.json on shutdown.

    This signals to Flutter that the server is no longer running.
    """
    try:
        if SERVER_INFO_FILE.exists():
            SERVER_INFO_FILE.unlink()
    except Exception:
        pass  # Best effort - file may already be gone


def get_server_info() -> dict | None:
    """
    Read server info (for testing/debugging).

    Returns None if server is not running.
    """
    try:
        if SERVER_INFO_FILE.exists():
            return json.loads(SERVER_INFO_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def is_server_running() -> bool:
    """
    Check if server is running by reading server.json and checking PID.
    """
    info = get_server_info()
    if not info:
        return False

    # Check if process is still alive
    pid = info.get("pid")
    if pid:
        try:
            # On Windows, this raises error if process doesn't exist
            os.kill(pid, 0)
            return True
        except OSError:
            # Process doesn't exist, clean up stale file
            clear_server_info()
            return False

    return False
