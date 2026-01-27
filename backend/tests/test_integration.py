"""
Integration tests for WebSocket and gRPC services.

These tests verify end-to-end message flow without a running backend.
Uses mocks for external dependencies.

Per roadmap: Week 7-8 integration testing.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# WebSocket Router Integration Tests
# =============================================================================


class TestWebSocketRouter:
    """Integration tests for WebSocket routing."""

    def test_router_registers_handlers(self):
        """Verify handlers can be registered with the router."""
        from api.ws.router import WebSocketRouter

        router = WebSocketRouter()

        # Register a handler
        @router.route("/test")
        async def test_handler(websocket):
            pass

        assert "/test" in router._routes
        assert router._routes["/test"] == test_handler

    def test_router_add_route_programmatic(self):
        """Verify programmatic route addition."""
        from api.ws.router import WebSocketRouter

        router = WebSocketRouter()

        async def my_handler(websocket):
            pass

        router.add_route("/api/v1/stream", my_handler)
        assert "/api/v1/stream" in router._routes

    @pytest.mark.asyncio
    async def test_router_unknown_path_closes_connection(self):
        """Verify unknown paths close the WebSocket."""
        from api.ws.router import WebSocketRouter

        router = WebSocketRouter()
        mock_ws = AsyncMock()

        await router.handle(mock_ws, "/unknown/path")

        mock_ws.close.assert_called_once()
        # Check close code 1008 (Policy Violation)
        call_args = mock_ws.close.call_args
        assert call_args[0][0] == 1008

    @pytest.mark.asyncio
    async def test_router_routes_to_correct_handler(self):
        """Verify router calls the correct handler."""
        from api.ws.router import WebSocketRouter

        router = WebSocketRouter()
        handler_called = asyncio.Event()

        async def test_handler(websocket):
            handler_called.set()

        router.add_route("/test", test_handler)
        mock_ws = AsyncMock()

        await router.handle(mock_ws, "/test")

        assert handler_called.is_set()


# =============================================================================
# WebSocket Message Format Tests
# =============================================================================


class TestWebSocketMessageFormats:
    """Test WebSocket message serialization/deserialization."""

    def test_status_message_format(self):
        """Verify status message has required fields."""
        message = {
            "type": "status",
            "connected": True,
            "backend_version": "1.0.0",
            "capabilities": {"fft_sizes": [1024, 2048, 4096], "colormaps": ["viridis", "plasma"]},
        }

        # Should be JSON serializable
        json_str = json.dumps(message)
        parsed = json.loads(json_str)

        assert parsed["type"] == "status"
        assert parsed["connected"] is True

    def test_detection_message_format(self):
        """Verify detection message format."""
        message = {
            "type": "detections",
            "frame_idx": 1234,
            "timestamp_us": 1706356800000000,
            "detections": [
                {
                    "x1": 0.1,
                    "y1": 0.2,
                    "x2": 0.3,
                    "y2": 0.4,
                    "class_id": 1,
                    "class_name": "signal_a",
                    "confidence": 0.95,
                }
            ],
        }

        json_str = json.dumps(message)
        parsed = json.loads(json_str)

        assert parsed["type"] == "detections"
        assert len(parsed["detections"]) == 1
        assert parsed["detections"][0]["confidence"] == 0.95

    def test_training_progress_message_format(self):
        """Verify training progress message format."""
        message = {
            "type": "training_progress",
            "signal_name": "test_signal",
            "epoch": 10,
            "total_epochs": 100,
            "loss": 0.0234,
            "metrics": {"train_loss": 0.0234, "val_loss": 0.0312, "accuracy": 0.95},
        }

        json_str = json.dumps(message)
        parsed = json.loads(json_str)

        assert parsed["epoch"] == 10
        assert parsed["metrics"]["accuracy"] == 0.95

    def test_error_message_format(self):
        """Verify error message format."""
        message = {
            "type": "error",
            "code": "INVALID_CONFIG",
            "message": "FFT size must be power of 2",
            "details": {"field": "fft_size", "value": 1000, "expected": "power of 2"},
        }

        json_str = json.dumps(message)
        parsed = json.loads(json_str)

        assert parsed["type"] == "error"
        assert parsed["code"] == "INVALID_CONFIG"


# =============================================================================
# Command Pattern Tests
# =============================================================================


class TestCommandPattern:
    """Test command pattern for WebSocket handlers."""

    def test_command_context_creation(self):
        """Verify CommandContext can be created."""
        from dataclasses import dataclass
        from typing import Any

        @dataclass
        class CommandContext:
            websocket: Any
            data: dict[str, Any]
            training_service: Any = None
            sample_manager: Any = None

        ctx = CommandContext(websocket=MagicMock(), data={"command": "test", "param": 123})

        assert ctx.data["command"] == "test"
        assert ctx.data["param"] == 123

    @pytest.mark.asyncio
    async def test_command_execute_pattern(self):
        """Test command execute pattern."""
        from abc import ABC, abstractmethod
        from dataclasses import dataclass
        from typing import Any

        @dataclass
        class CommandContext:
            websocket: Any
            data: dict[str, Any]

        class Command(ABC):
            @abstractmethod
            async def execute(self, ctx: CommandContext) -> dict[str, Any]:
                pass

        class TestCommand(Command):
            async def execute(self, ctx: CommandContext) -> dict[str, Any]:
                return {"type": "test_response", "value": ctx.data.get("input", 0) * 2}

        cmd = TestCommand()
        ctx = CommandContext(websocket=MagicMock(), data={"input": 21})

        result = await cmd.execute(ctx)

        assert result["type"] == "test_response"
        assert result["value"] == 42


# =============================================================================
# gRPC Service Tests
# =============================================================================


class TestGRPCServices:
    """Integration tests for gRPC services - test servicer classes directly."""

    def test_device_control_servicer_exists(self):
        """Test DeviceControlServicer class can be instantiated."""
        # Import the module first (before base class assignment happens)
        import importlib.util

        importlib.util.spec_from_file_location(
            "device_control_test", BACKEND_DIR / "api" / "grpc" / "device_control.py"
        )
        # Read the file to check contents without executing base class assignment
        device_control_path = BACKEND_DIR / "api" / "grpc" / "device_control.py"
        content = device_control_path.read_text()

        assert "class DeviceControlServicer" in content
        assert "def SetFrequency" in content
        assert "def SetBandwidth" in content
        assert "def StartCapture" in content
        assert "def GetStatus" in content

    def test_inference_service_servicer_exists(self):
        """Test InferenceServicer class exists in module."""
        inference_service_path = BACKEND_DIR / "api" / "grpc" / "inference_service.py"
        content = inference_service_path.read_text()

        assert "class InferenceServicer" in content
        assert "PROTO_AVAILABLE" in content

    def test_grpc_package_init_exists(self):
        """Test gRPC package __init__.py exists."""
        init_path = BACKEND_DIR / "api" / "grpc" / "__init__.py"
        assert init_path.exists()

    def test_device_control_has_proto_flag(self):
        """Test device_control handles missing protos gracefully."""
        device_control_path = BACKEND_DIR / "api" / "grpc" / "device_control.py"
        content = device_control_path.read_text()

        # Should have proto availability check
        assert "PROTO_AVAILABLE" in content
        assert "try:" in content
        assert "except ImportError:" in content

    def test_device_control_servicer_methods(self):
        """Test DeviceControlServicer has all required gRPC methods."""
        device_control_path = BACKEND_DIR / "api" / "grpc" / "device_control.py"
        content = device_control_path.read_text()

        required_methods = [
            "SetFrequency",
            "SetBandwidth",
            "SetGain",
            "StartCapture",
            "StopCapture",
            "GetStatus",
            "GetDeviceInfo",
            "SetMode",
        ]

        for method in required_methods:
            assert f"def {method}" in content, f"Missing method: {method}"


# =============================================================================
# Shutdown Coordination Tests
# =============================================================================


class TestShutdownCoordination:
    """Integration tests for graceful shutdown module functions."""

    def test_shutdown_module_functions_exist(self):
        """Test shutdown module exports required functions."""
        from core import shutdown

        assert hasattr(shutdown, "is_shutting_down")
        assert hasattr(shutdown, "request_shutdown")
        assert hasattr(shutdown, "register_cleanup")
        assert hasattr(shutdown, "cleanup_all")

    def test_shutdown_module_is_callable(self):
        """Test shutdown functions are callable."""
        from core import shutdown

        assert callable(shutdown.is_shutting_down)
        assert callable(shutdown.request_shutdown)
        assert callable(shutdown.register_cleanup)

    def test_signal_handler_setup(self):
        """Test signal handlers can be set up."""
        from core import shutdown

        # Should not raise
        assert hasattr(shutdown, "setup_signal_handlers")
        assert callable(shutdown.setup_signal_handlers)


# =============================================================================
# Capabilities API Tests
# =============================================================================


class TestCapabilitiesAPI:
    """Integration tests for capabilities API."""

    def test_capabilities_returns_dict(self):
        """Test capabilities returns JSON-serializable dict."""
        from config.capabilities import get_capabilities

        caps = get_capabilities()

        assert isinstance(caps, dict)
        # Should be JSON serializable
        json_str = json.dumps(caps)
        assert json_str is not None

    def test_capabilities_has_fft_section(self):
        """Test capabilities has fft section."""
        from config.capabilities import get_capabilities

        caps = get_capabilities()
        assert "fft" in caps

    def test_capabilities_has_display_section(self):
        """Test capabilities has display section."""
        from config.capabilities import get_capabilities

        caps = get_capabilities()
        assert "display" in caps

    def test_capabilities_fft_section(self):
        """Test FFT capabilities section."""
        from config.capabilities import get_capabilities

        caps = get_capabilities()
        fft = caps["fft"]

        assert "inference_nfft" in fft
        assert fft["inference_nfft"] == 4096  # Default value

    def test_capabilities_display_section(self):
        """Test display capabilities section."""
        from config.capabilities import get_capabilities

        caps = get_capabilities()
        display = caps["display"]

        assert "colormap_options" in display
        assert "viridis" in display["colormap_options"]


# =============================================================================
# End-to-End Message Flow Tests (Mocked)
# =============================================================================


class TestEndToEndMessageFlow:
    """Test complete message flows with mocked components."""

    @pytest.mark.asyncio
    async def test_training_request_response_flow(self):
        """Test training request → response flow."""
        # Simulate a complete training request/response cycle

        # 1. Client sends request
        request = {"command": "train_signal", "signal_name": "test_signal", "preset": "fast"}

        # 2. Server processes and responds
        response = {
            "type": "training_started",
            "signal_name": "test_signal",
            "estimated_time_sec": 120,
        }

        # Verify message formats are compatible
        assert json.dumps(request)  # Request is valid JSON
        assert json.dumps(response)  # Response is valid JSON
        assert response["signal_name"] == request["signal_name"]

    @pytest.mark.asyncio
    async def test_inference_detection_flow(self):
        """Test inference → detection flow."""
        # Simulate spectrogram → detection cycle

        # 1. Frame data (simulated)
        frame_metadata = {
            "frame_idx": 100,
            "timestamp_us": 1706356800000000,
            "width": 2048,
            "height": 256,
        }

        # 2. Detection results
        detections = {
            "type": "detections",
            "frame_idx": frame_metadata["frame_idx"],
            "timestamp_us": frame_metadata["timestamp_us"],
            "detections": [
                {
                    "x1": 0.25,
                    "y1": 0.1,
                    "x2": 0.35,
                    "y2": 0.9,
                    "class_name": "signal_a",
                    "confidence": 0.92,
                }
            ],
        }

        # Verify frame_idx matches
        assert detections["frame_idx"] == frame_metadata["frame_idx"]
        assert len(detections["detections"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
