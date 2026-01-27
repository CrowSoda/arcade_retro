"""
API Handlers Tests - WebSocket and gRPC handler tests.

Tests for backend/api/ws/handlers/ and backend/api/grpc/ modules.
Uses mocking to test handler logic without actual connections.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# Inference Handler Tests
# =============================================================================


class TestFormatDetections:
    """Test the _format_detections helper function."""

    def test_format_empty_detections(self):
        """Empty detections list returns empty list."""
        from api.ws.handlers.inference import _format_detections

        result = _format_detections([])
        assert result == []

    def test_format_single_detection(self):
        """Format single detection result."""
        from api.ws.handlers.inference import _format_detections

        detections = [
            {
                "boxes": [[100, 200, 300, 400]],
                "scores": [0.95],
                "labels": [1],
            }
        ]

        result = _format_detections(detections)

        assert len(result) == 1
        assert result[0]["detection_id"] == 0
        assert result[0]["class_id"] == 1
        assert result[0]["confidence"] == 0.95
        # Check normalized coords (divided by 1024)
        assert result[0]["x1"] == pytest.approx(100 / 1024)
        assert result[0]["y1"] == pytest.approx(200 / 1024)
        assert result[0]["x2"] == pytest.approx(300 / 1024)
        assert result[0]["y2"] == pytest.approx(400 / 1024)

    def test_format_multiple_detections(self):
        """Format multiple detections in single frame."""
        from api.ws.handlers.inference import _format_detections

        detections = [
            {
                "boxes": [[0, 0, 100, 100], [200, 200, 300, 300]],
                "scores": [0.9, 0.8],
                "labels": [1, 1],
            }
        ]

        result = _format_detections(detections)

        assert len(result) == 2
        assert result[0]["confidence"] == 0.9
        assert result[1]["confidence"] == 0.8

    def test_format_background_class(self):
        """Label 0 maps to background class name."""
        from api.ws.handlers.inference import _format_detections

        detections = [
            {
                "boxes": [[0, 0, 100, 100]],
                "scores": [0.7],
                "labels": [0],
            }
        ]

        result = _format_detections(detections)

        assert result[0]["class_name"] == "background"

    def test_format_signal_class(self):
        """Label 1 maps to signal class name."""
        from api.ws.handlers.inference import _format_detections

        detections = [
            {
                "boxes": [[0, 0, 100, 100]],
                "scores": [0.7],
                "labels": [1],
            }
        ]

        result = _format_detections(detections)

        assert result[0]["class_name"] == "creamy_chicken"


class TestInferenceHandlerMessages:
    """Test ws_inference_handler message handling."""

    @pytest.mark.asyncio
    async def test_handler_invalid_json(self):
        """Handler sends error for invalid JSON."""
        from api.ws.handlers.inference import ws_inference_handler

        mock_ws = AsyncMock()
        mock_ws.path = "/ws/inference"
        mock_ws.remote_address = ("127.0.0.1", 12345)

        # Simulate receiving invalid JSON then disconnecting
        mock_ws.__aiter__.return_value = ["not valid json"]

        await ws_inference_handler(mock_ws)

        # Should have sent error response
        calls = mock_ws.send.call_args_list
        assert len(calls) >= 1
        response = json.loads(calls[0][0][0])
        assert response["type"] == "error"
        assert "Invalid JSON" in response["message"]

    @pytest.mark.asyncio
    async def test_handler_stop_without_session(self):
        """Stop command without active session sends response."""
        from api.ws.handlers.inference import ws_inference_handler

        mock_ws = AsyncMock()
        mock_ws.path = "/ws/inference"
        mock_ws.remote_address = ("127.0.0.1", 12345)

        # Send stop command
        mock_ws.__aiter__.return_value = [json.dumps({"command": "stop"})]

        await ws_inference_handler(mock_ws)

        # Should have sent session_stopped
        calls = mock_ws.send.call_args_list
        assert len(calls) >= 1
        response = json.loads(calls[0][0][0])
        assert response["type"] == "session_stopped"

    @pytest.mark.asyncio
    async def test_handler_run_without_session(self):
        """Run command without start returns error."""
        from api.ws.handlers.inference import ws_inference_handler

        mock_ws = AsyncMock()
        mock_ws.path = "/ws/inference"
        mock_ws.remote_address = ("127.0.0.1", 12345)

        mock_ws.__aiter__.return_value = [json.dumps({"command": "run"})]

        await ws_inference_handler(mock_ws)

        calls = mock_ws.send.call_args_list
        assert len(calls) >= 1
        response = json.loads(calls[0][0][0])
        assert response["type"] == "error"
        assert "No active session" in response["message"]

    @pytest.mark.asyncio
    async def test_handler_infer_without_session(self):
        """Infer command without start returns error."""
        from api.ws.handlers.inference import ws_inference_handler

        mock_ws = AsyncMock()
        mock_ws.path = "/ws/inference"
        mock_ws.remote_address = ("127.0.0.1", 12345)

        mock_ws.__aiter__.return_value = [json.dumps({"command": "infer"})]

        await ws_inference_handler(mock_ws)

        calls = mock_ws.send.call_args_list
        response = json.loads(calls[0][0][0])
        assert response["type"] == "error"


# =============================================================================
# Pipeline Handler Tests
# =============================================================================


class TestPipelineHandler:
    """Test api/ws/handlers/pipeline.py module."""

    def test_module_imports(self):
        """pipeline handler module should import."""
        from api.ws.handlers import pipeline

        assert pipeline is not None

    def test_has_unified_handler(self):
        """Module has unified_pipeline_handler function."""
        from api.ws.handlers import pipeline

        assert hasattr(pipeline, "unified_pipeline_handler")

    def test_has_video_handler(self):
        """Module has video_pipeline_handler function."""
        from api.ws.handlers import pipeline

        assert hasattr(pipeline, "video_pipeline_handler")

    def test_has_helper_functions(self):
        """Module has helper functions."""
        from api.ws.handlers import pipeline

        assert hasattr(pipeline, "_find_iq_file")
        assert hasattr(pipeline, "_find_model")


# =============================================================================
# Training Handler Tests
# =============================================================================


class TestTrainingHandler:
    """Test api/ws/handlers/training.py module."""

    def test_module_imports(self):
        """training handler module should import."""
        from api.ws.handlers import training

        assert training is not None

    def test_has_handler_function(self):
        """Module has handler function."""
        from api.ws.handlers import training

        assert hasattr(training, "ws_training_handler")


# =============================================================================
# gRPC Tests - Skipped if stubs not available
# =============================================================================
# Note: gRPC service tests require generated protobuf stubs.
# These are covered in test_smoke.py which has proper skip handling.


# =============================================================================
# Router Tests
# =============================================================================


class TestWSRouter:
    """Test api/ws/router.py module."""

    def test_module_imports(self):
        """router module should import."""
        from api.ws import router

        assert router is not None

    def test_has_route_handler(self):
        """Router has route handling function."""
        from api.ws import router

        # Check for route function
        assert hasattr(router, "route_websocket") or hasattr(router, "ws_router")


# =============================================================================
# Integration: Handler JSON Protocol
# =============================================================================


class TestHandlerJSONProtocol:
    """Test JSON protocol structure for handlers."""

    def test_detection_frame_structure(self):
        """Detection frame has required fields."""
        from api.ws.handlers.inference import _format_detections

        # Create a detection frame like the handler would
        detections = [
            {
                "boxes": [[100, 100, 200, 200]],
                "scores": [0.85],
                "labels": [1],
            }
        ]

        det_list = _format_detections(detections)

        # Build frame like handler does
        frame = {
            "type": "detection_frame",
            "frame_id": 0,
            "timestamp_ms": 1234567890,
            "detections": det_list,
            "center_freq_mhz": 825.0,
            "bandwidth_mhz": 20.0,
        }

        # Verify JSON serializable
        json_str = json.dumps(frame)
        parsed = json.loads(json_str)

        assert parsed["type"] == "detection_frame"
        assert "frame_id" in parsed
        assert "timestamp_ms" in parsed
        assert "detections" in parsed
        assert len(parsed["detections"]) == 1

    def test_error_response_structure(self):
        """Error response has required fields."""
        error = {"type": "error", "message": "Test error message"}

        json_str = json.dumps(error)
        parsed = json.loads(json_str)

        assert parsed["type"] == "error"
        assert "message" in parsed

    def test_session_started_structure(self):
        """Session started response has required fields."""
        response = {
            "type": "session_started",
            "session_id": "abc12345",
            "model_path": "/models/test.pth",
            "backend": "pytorch",
        }

        json_str = json.dumps(response)
        parsed = json.loads(json_str)

        assert parsed["type"] == "session_started"
        assert "session_id" in parsed
        assert "model_path" in parsed

    def test_session_stopped_structure(self):
        """Session stopped response has required fields."""
        response = {"type": "session_stopped", "session_id": "abc12345"}

        json_str = json.dumps(response)
        parsed = json.loads(json_str)

        assert parsed["type"] == "session_stopped"
        assert "session_id" in parsed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
