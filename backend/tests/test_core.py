"""
Core Module Tests - Models, shutdown, process management.

Tests for backend/core/ modules.
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


class TestChannelState:
    """Test ChannelState model."""

    def test_create_channel(self):
        """Create a basic channel."""
        from core.models import ChannelState

        channel = ChannelState(channel_number=1, center_freq_mhz=825.0, bandwidth_mhz=20.0)
        assert channel.channel_number == 1
        assert channel.center_freq_mhz == 825.0
        assert channel.bandwidth_mhz == 20.0
        assert channel.state == "IDLE"
        assert channel.gain_db == 20.0  # default

    def test_channel_state_transitions(self):
        """Channel can transition between states."""
        from core.models import ChannelState

        channel = ChannelState(channel_number=1, center_freq_mhz=825.0, bandwidth_mhz=20.0)
        channel.state = "CAPTURING"
        assert channel.state == "CAPTURING"
        channel.state = "PROCESSING"
        assert channel.state == "PROCESSING"


class TestCaptureSession:
    """Test CaptureSession model."""

    def test_create_capture_session(self):
        """Create a capture session."""
        from core.models import CaptureSession

        session = CaptureSession(
            capture_id="test-123",
            signal_name="test_signal",
            rx_channel=1,
            duration_seconds=10,
            start_time=time.time(),
            file_path="/tmp/capture.bin",
        )
        assert session.capture_id == "test-123"
        assert session.duration_seconds == 10
        assert session.bytes_captured == 0
        assert session.is_active is True


class TestInferenceSession:
    """Test InferenceSession model."""

    def test_create_inference_session(self):
        """Create an inference session."""
        from core.models import InferenceSession

        session = InferenceSession(
            session_id="inf-456",
            model_id="hydra_v1",
            source_type="live",
            source_path=None,
            rx_channel=1,
            config={},
            start_time=time.time(),
        )
        assert session.session_id == "inf-456"
        assert session.model_id == "hydra_v1"
        assert session.detection_count == 0
        assert session.is_active is True


class TestModelState:
    """Test ModelState model."""

    def test_create_model_state(self):
        """Create model state."""
        from core.models import ModelState

        state = ModelState(
            model_id="hydra_v1",
            model_name="Hydra V1",
            model_path="/models/hydra_v1.pt",
            model_hash="abc123",
            backend="pytorch",
            num_classes=10,
            class_names=["class_a", "class_b"],
        )
        assert state.model_name == "Hydra V1"
        assert state.backend == "pytorch"
        assert state.num_classes == 10


class TestShutdownModule:
    """Test shutdown coordination."""

    def test_shutdown_event_starts_clear(self):
        """Shutdown event should start not set."""
        from core import shutdown

        # Reset for testing
        shutdown._shutdown_event.clear()
        assert not shutdown.is_shutting_down()

    def test_request_shutdown_sets_event(self):
        """request_shutdown() should set the event."""
        from core import shutdown

        shutdown._shutdown_event.clear()
        shutdown.request_shutdown()
        assert shutdown.is_shutting_down()
        # Clean up
        shutdown._shutdown_event.clear()

    def test_wait_for_shutdown_with_timeout(self):
        """wait_for_shutdown with timeout should return False if not signaled."""
        from core import shutdown

        shutdown._shutdown_event.clear()
        result = shutdown.wait_for_shutdown(timeout=0.01)
        assert result is False

    def test_wait_for_shutdown_returns_true_when_set(self):
        """wait_for_shutdown should return True if event is set."""
        from core import shutdown

        shutdown._shutdown_event.clear()
        shutdown.request_shutdown()
        result = shutdown.wait_for_shutdown(timeout=0.01)
        assert result is True
        shutdown._shutdown_event.clear()

    def test_register_cleanup(self):
        """Can register resources for cleanup."""
        from core import shutdown

        mock_resource = MagicMock()
        mock_resource.close = MagicMock()

        # Clear existing
        shutdown._cleanup_resources.clear()
        shutdown.register_cleanup(mock_resource)
        assert mock_resource in shutdown._cleanup_resources

        # Clean up
        shutdown._cleanup_resources.clear()


class TestShutdownAsync:
    """Test async shutdown features."""

    @pytest.mark.asyncio
    async def test_get_async_event(self):
        """Can get async shutdown event."""
        from core import shutdown

        # Reset
        shutdown._shutdown_event.clear()
        shutdown._async_shutdown_event = None

        event = shutdown.get_async_event()
        assert isinstance(event, asyncio.Event)
        assert not event.is_set()

    @pytest.mark.asyncio
    async def test_async_event_mirrors_thread_event(self):
        """Async event should reflect thread event state."""
        from core import shutdown

        # Reset
        shutdown._shutdown_event.clear()
        shutdown._async_shutdown_event = None

        # Set thread event first, then get async event
        shutdown._shutdown_event.set()
        event = shutdown.get_async_event()
        # The async event should be set since thread event was set
        assert event.is_set()

        # Clean up
        shutdown._shutdown_event.clear()
        shutdown._async_shutdown_event = None


class TestProcessModule:
    """Test process management (if available)."""

    def test_process_module_imports(self):
        """Process module should import."""
        from core import process

        assert process is not None


class TestWebSocketRouter:
    """Test WebSocket router functionality."""

    def test_router_creates_empty(self):
        """Router should start with no routes."""
        from api.ws.router import WebSocketRouter

        router = WebSocketRouter()
        assert len(router._routes) == 0

    def test_route_decorator_registers(self):
        """@route decorator should register handlers."""
        from api.ws.router import WebSocketRouter

        router = WebSocketRouter()

        @router.route("/test")
        async def test_handler(ws):
            pass

        assert "/test" in router._routes
        assert router._routes["/test"] == test_handler

    def test_multiple_routes(self):
        """Can register multiple routes."""
        from api.ws.router import WebSocketRouter

        router = WebSocketRouter()

        @router.route("/a")
        async def handler_a(ws):
            pass

        @router.route("/b")
        async def handler_b(ws):
            pass

        @router.route("/c")
        async def handler_c(ws):
            pass

        assert len(router._routes) == 3
        assert "/a" in router._routes
        assert "/b" in router._routes
        assert "/c" in router._routes


class TestLoggerConfig:
    """Test logging configuration."""

    def test_get_logger_returns_logger(self):
        """get_logger should return a logger instance."""
        from logger_config import get_logger

        logger = get_logger("test_module")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")

    def test_loggers_have_same_name(self):
        """Loggers with same name use same underlying logger."""
        from logger_config import get_logger

        logger1 = get_logger("same_name_test")
        logger2 = get_logger("same_name_test")
        # G20Logger wraps underlying loggers; check they log to same name
        assert logger1.name == logger2.name

    def test_different_names_different_loggers(self):
        """Different names should return different loggers."""
        from logger_config import get_logger

        logger1 = get_logger("module_a")
        logger2 = get_logger("module_b")
        assert logger1 is not logger2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
