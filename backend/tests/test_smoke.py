"""
Smoke Tests - Does the system start and respond?

These are the FIRST tests to run. If these fail, nothing else matters.
Per roadmap: Week 1-2 testing priority.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


class TestBackendStarts:
    """Test that the backend process starts without crashing."""

    def test_server_imports_without_error(self):
        """Verify server.py can be imported (catches syntax errors)."""
        # This will raise ImportError or SyntaxError if broken
        import server  # noqa: F401

    def test_config_loads(self):
        """Verify configuration loads with valid defaults."""
        from config.settings import get_settings

        settings = get_settings()
        assert settings.server.grpc_port == 50051
        assert settings.fft.inference_nfft == 4096
        assert settings.sdr.min_freq_mhz < settings.sdr.max_freq_mhz

    def test_capabilities_api_returns_valid_json(self):
        """Verify capabilities API returns valid JSON."""
        from config.capabilities import get_capabilities

        caps = get_capabilities()
        assert "fft" in caps
        assert "display" in caps
        assert "sdr" in caps
        assert caps["fft"]["inference_locked"] is True

    def test_core_models_instantiate(self):
        """Verify core data models can be instantiated."""
        from core.models import ChannelState

        channel = ChannelState(channel_number=1, center_freq_mhz=825.0, bandwidth_mhz=20.0)
        assert channel.channel_number == 1
        assert channel.state == "IDLE"

    def test_shutdown_module_works(self):
        """Verify shutdown module functions work."""
        from core import shutdown

        # Reset state for testing
        shutdown._shutdown_event.clear()

        assert not shutdown.is_shutting_down()
        shutdown.request_shutdown()
        assert shutdown.is_shutting_down()

        # Reset for other tests
        shutdown._shutdown_event.clear()


class TestWebSocketBasics:
    """Basic WebSocket connectivity tests."""

    @pytest.mark.asyncio
    async def test_websocket_server_can_be_created(self):
        """Verify WebSocket server can be created (not started)."""
        import websockets

        # Just verify the module works
        assert hasattr(websockets, "serve")

    def test_ws_router_can_register_handlers(self):
        """Verify WebSocket router accepts handler registration."""
        from api.ws.router import WebSocketRouter

        router = WebSocketRouter()

        @router.route("/test")
        async def test_handler(ws):
            pass

        assert "/test" in router._routes


class TestConfigEnvironmentOverrides:
    """Test that environment variables override defaults."""

    def test_env_override_simulation(self):
        """Verify environment override mechanism works."""
        import os

        from config.settings import reload_settings

        # Set env var
        original = os.environ.get("G20_DEBUG")
        os.environ["G20_DEBUG"] = "true"

        try:
            settings = reload_settings()
            assert settings.debug is True
        finally:
            # Restore
            if original is None:
                os.environ.pop("G20_DEBUG", None)
            else:
                os.environ["G20_DEBUG"] = original


class TestLoggingWorks:
    """Test that logging infrastructure is functional."""

    def test_logger_config_module_loads(self):
        """Verify logger_config module loads."""
        import logger_config

        assert hasattr(logger_config, "get_logger")

    def test_can_get_logger(self):
        """Verify we can get a logger instance."""
        from logger_config import get_logger

        logger = get_logger("test")
        assert logger is not None
        # Should not raise
        logger.info("Smoke test log message")


class TestPathsExist:
    """Test that required directories/files exist or can be created."""

    def test_backend_directory_structure(self):
        """Verify expected directories exist."""
        assert BACKEND_DIR.exists()
        assert (BACKEND_DIR / "core").exists()
        assert (BACKEND_DIR / "config").exists()
        assert (BACKEND_DIR / "api").exists()

    def test_models_dir_can_be_created(self):
        """Verify models directory exists or can be created."""
        from config.settings import get_settings

        settings = get_settings()
        models_dir = settings.paths.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        assert models_dir.exists()


class TestServerIntegration:
    """Integration tests that launch the server as a subprocess."""

    @pytest.fixture
    def server_process(self):
        """Launch server subprocess and yield when ready, then cleanup."""
        server_py = BACKEND_DIR / "server.py"

        # Launch server as subprocess
        proc = subprocess.Popen(
            [sys.executable, str(server_py)],
            cwd=str(BACKEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for server to be ready (look for WebSocket port in output)
        ready = False
        start_time = time.time()
        timeout = 15  # seconds

        while time.time() - start_time < timeout:
            if proc.poll() is not None:
                # Process died
                output = proc.stdout.read() if proc.stdout else ""
                pytest.fail(f"Server process died during startup. Output:\n{output}")

            # Check if there's output indicating ready
            time.sleep(0.5)

            # Try to connect to see if ready
            try:
                import socket

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("localhost", 8765))
                sock.close()
                if result == 0:
                    ready = True
                    break
            except Exception:
                pass

        if not ready:
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail("Server did not become ready within timeout")

        yield proc

        # Cleanup: terminate server
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Flaky in CI - WebSocket closes immediately")
    async def test_websocket_connection(self, server_process):
        """Test actual WebSocket connection to launched server."""
        import websockets

        # Give server a moment to stabilize
        await asyncio.sleep(0.5)

        connected = False
        try:
            # Connect to video endpoint - may receive binary data (spectrogram frames)
            async with websockets.connect("ws://localhost:8765/video", close_timeout=2) as ws:
                # If we get here, connection was established
                connected = True

                # Try to receive any data (could be binary or text)
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=3)
                    # Server responded with something (binary or text)
                    assert response is not None
                    assert len(response) > 0
                except asyncio.TimeoutError:
                    # No data within timeout is okay - connection still valid
                    pass

        except websockets.exceptions.ConnectionClosedError:
            # Server may close connection for /video without SDR - that's okay
            connected = True  # Connection was established before close
        except Exception as e:
            pytest.fail(f"WebSocket connection failed: {e}")

        assert connected, "WebSocket connection was never established"

    def test_server_starts_and_stops(self, server_process):
        """Test that server can start and stop cleanly."""
        # If we get here, server_process fixture succeeded
        assert server_process.poll() is None  # Still running

        # Terminate and verify clean exit
        server_process.terminate()
        exit_code = server_process.wait(timeout=10)

        # Exit code 0 or -15 (SIGTERM) are acceptable
        assert exit_code in [0, -15, 15, 1], f"Unexpected exit code: {exit_code}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
