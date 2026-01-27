"""
G20 Demo - gRPC + WebSocket Server
Implements DeviceControl and InferenceService

Two servers:
1. gRPC on port 50051 (for full proto support)
2. WebSocket on port 50052 (for Flutter - simpler JSON)

Usage:
    cd g20_demo/backend
    python -m grpc_tools.protoc -I../protos --python_out=./generated --grpc_python_out=./generated ../protos/*.proto
    python server.py --port 50051
"""

import asyncio
import atexit
import hashlib
import logging
import os
import signal
import sys
import threading
import time
import uuid
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np

# Import extracted core modules (strangler fig pattern - Phase 3 complete)
from core import shutdown as core_shutdown
from core.models import CaptureSession, ChannelState, InferenceSession, ModelState
from core.process import start_parent_watchdog

# Import enhanced logger
from logger_config import get_logger
from runtime_info import clear_server_info, write_server_info

# Import extracted API modules (strangler fig pattern)
# NOTE: gRPC servicers still defined locally due to proto inheritance complexity
# The extracted versions are available at api.grpc.device_control and api.grpc.inference_service

# Module logger
logger_server = get_logger("server")

# Aliases for backward compatibility during migration
_shutdown_event = core_shutdown._shutdown_event
_cleanup_resources = core_shutdown._cleanup_resources
_register_cleanup = core_shutdown.register_cleanup

# Async shutdown event - local ref for WebSocket handlers
_async_shutdown_event: asyncio.Event | None = None


def _cleanup_all():
    """Clean up all registered resources."""
    logger_server.info("Cleaning up resources", extra={"resource_count": len(_cleanup_resources)})

    # Clear server.json so Flutter knows we're down
    clear_server_info()

    for resource in _cleanup_resources:
        try:
            if hasattr(resource, "close"):
                resource.close()
            elif hasattr(resource, "stop"):
                resource.stop()
        except Exception as e:
            logger_server.error(f"Cleanup error: {e}")
    _cleanup_resources.clear()
    logger_server.info("Cleanup complete")


def _signal_handler(signum, frame):
    """Handle shutdown signals (SIGINT, SIGTERM, SIGBREAK)."""
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    logger_server.warning(
        f"Received {sig_name}, initiating graceful shutdown", extra={"signal": sig_name}
    )
    _shutdown_event.set()
    if _async_shutdown_event:
        # Schedule setting the async event from any thread
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(_async_shutdown_event.set)
        except RuntimeError:
            pass  # No running loop


def setup_signal_handlers():
    """Set up platform-appropriate signal handlers."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if sys.platform == "win32":
        # Windows: SIGBREAK is sent on Ctrl+Break and by taskkill
        signal.signal(signal.SIGBREAK, _signal_handler)
    logger_server.info("Shutdown: Signal handlers registered")


# Register cleanup on exit
atexit.register(_cleanup_all)

# Add generated proto stubs to path
GENERATED_DIR = Path(__file__).parent / "generated"
sys.path.insert(0, str(GENERATED_DIR))

# Import generated stubs (after running protoc)
PROTO_AVAILABLE = False
try:
    import control_pb2
    import control_pb2_grpc
    import inference_pb2
    import inference_pb2_grpc

    PROTO_AVAILABLE = True
except ImportError:
    logging.warning("Proto stubs not found. Run generate_stubs.sh first.")

# Local imports
from inference import InferenceEngine, SpectrogramPipeline
from unified_pipeline import video_ws_handler

# Hydra imports (Phase 5)
HYDRA_AVAILABLE = False
try:
    from dsp.simple_extract import extract_subband as simple_extract_subband
    from dsp.subband_extractor import (
        ExtractionParams,
        SubbandExtractor,
        _read_rfcap_header,
        extract_subband_from_file,
    )
    from hydra.detector import HydraDetector
    from hydra.version_manager import VersionManager
    from training.sample_manager import SampleManager
    from training.service import TrainingProgress, TrainingService
    from training.splits import SplitManager

    HYDRA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hydra modules not available: {e}")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("g20.server")

# Directories
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# NOTE: Data classes (ChannelState, CaptureSession, InferenceSession, ModelState)
# are now imported from core.models (strangler fig pattern)

# ============= Device Control Service =============


class DeviceControlServicer(control_pb2_grpc.DeviceControlServicer):
    """Simulates NV100 SDR hardware control."""

    def __init__(self):
        self.device_state = "CONNECTED"
        self.temperature_c = 42.5
        self.operating_mode = "AUTO_SCAN"
        self.manual_timeout = None

        self.channels = {
            1: ChannelState(channel_number=1, center_freq_mhz=825.0, bandwidth_mhz=20.0),
            2: ChannelState(channel_number=2, center_freq_mhz=825.0, bandwidth_mhz=5.0),
        }

        self.captures: dict[str, CaptureSession] = {}
        logger.info("DeviceControlServicer initialized")

    def SetFrequency(self, request, context):
        channel = request.rx_channel or 1
        freq = request.center_freq_mhz

        if freq < 30.0 or freq > 6000.0:
            return control_pb2.FrequencyResponse(
                success=False,
                error_message=f"Frequency {freq} MHz outside range (30-6000)",
                actual_freq_mhz=self.channels[channel].center_freq_mhz,
            )

        self.channels[channel].center_freq_mhz = freq
        logger.info(f"Channel {channel} → {freq} MHz")

        return control_pb2.FrequencyResponse(success=True, actual_freq_mhz=freq)

    def SetBandwidth(self, request, context):
        channel = request.rx_channel or 1
        bw = request.bandwidth_mhz

        if bw < 0.1 or bw > 50.0:
            return control_pb2.BandwidthResponse(
                success=False,
                error_message=f"Bandwidth {bw} MHz outside range (0.1-50)",
                actual_bw_mhz=self.channels[channel].bandwidth_mhz,
            )

        self.channels[channel].bandwidth_mhz = bw
        logger.info(f"Channel {channel} BW → {bw} MHz")

        return control_pb2.BandwidthResponse(success=True, actual_bw_mhz=bw)

    def SetGain(self, request, context):
        channel = request.rx_channel or 1
        gain = round(request.gain_db * 2) / 2  # 0.5 dB steps

        if gain < 0 or gain > 34:
            return control_pb2.GainResponse(
                success=False,
                error_message=f"Gain {gain} dB outside range (0-34)",
                actual_gain_db=self.channels[channel].gain_db,
            )

        self.channels[channel].gain_db = gain
        return control_pb2.GainResponse(success=True, actual_gain_db=gain)

    def StartCapture(self, request, context):
        channel = request.rx_channel or 2

        if self.channels[channel].state == "CAPTURING":
            return control_pb2.CaptureResponse(
                success=False, error_message=f"Channel {channel} busy"
            )

        capture_id = str(uuid.uuid4())[:8]
        signal_name = request.signal_name or f"capture_{capture_id}"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = str(DATA_DIR / "captures" / f"{signal_name}_{timestamp}.rfcap")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        session = CaptureSession(
            capture_id=capture_id,
            signal_name=signal_name,
            rx_channel=channel,
            duration_seconds=request.duration_seconds or 60,
            start_time=time.time(),
            file_path=file_path,
        )

        self.captures[capture_id] = session
        self.channels[channel].state = "CAPTURING"
        self.channels[channel].active_capture_id = capture_id

        logger.info(f"Capture started: {capture_id}")

        return control_pb2.CaptureResponse(success=True, capture_id=capture_id, file_path=file_path)

    def StopCapture(self, request, context):
        capture_id = request.capture_id

        if capture_id not in self.captures:
            return control_pb2.StopResponse(success=False, error_message="Not found")

        session = self.captures[capture_id]
        session.is_active = False
        self.channels[session.rx_channel].state = "IDLE"
        self.channels[session.rx_channel].active_capture_id = None

        logger.info(f"Capture stopped: {capture_id}")

        return control_pb2.StopResponse(
            success=True, file_path=session.file_path, bytes_captured=session.bytes_captured
        )

    def GetStatus(self, request, context):
        channels = []
        for ch_num, ch in self.channels.items():
            state_map = {"IDLE": 0, "SCANNING": 1, "CAPTURING": 2, "ERROR": 3}
            channels.append(
                control_pb2.ChannelStatus(
                    channel_number=ch_num,
                    center_freq_mhz=ch.center_freq_mhz,
                    bandwidth_mhz=ch.bandwidth_mhz,
                    gain_db=ch.gain_db,
                    state=state_map.get(ch.state, 0),
                    active_capture_id=ch.active_capture_id or "",
                )
            )

        return control_pb2.StatusResponse(
            state=control_pb2.DEVICE_CONNECTED,
            temperature_c=self.temperature_c,
            channels=channels,
            cpu_usage_percent=25.0,
            gpu_usage_percent=45.0,
            memory_used_bytes=512 * 1024 * 1024,
            disk_free_bytes=100 * 1024 * 1024 * 1024,
        )

    def GetDeviceInfo(self, request, context):
        return control_pb2.DeviceInfoResponse(
            device_name="Sidekiq NV100 (Simulated)",
            serial_number="SIM-00001",
            firmware_version="1.0.0",
            api_version="0.1.0",
            min_freq_mhz=30.0,
            max_freq_mhz=6000.0,
            max_bandwidth_mhz=50.0,
            max_gain_db=34.0,
            num_rx_channels=2,
            num_tx_channels=2,
        )

    def SetMode(self, request, context):
        modes = {0: "AUTO_SCAN", 1: "MANUAL", 2: "SENSE_ONLY"}
        self.operating_mode = modes.get(request.mode, "AUTO_SCAN")
        self.manual_timeout = (
            request.timeout_seconds if hasattr(request, "timeout_seconds") else None
        )

        logger.info(f"Mode → {self.operating_mode}")

        return control_pb2.ModeResponse(success=True, current_mode=request.mode)


# ============= Inference Service =============


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """ML inference pipeline with TensorRT support."""

    def __init__(self, device_control: DeviceControlServicer):
        self.device_control = device_control
        self.models: dict[str, ModelState] = {}
        self.active_model_id: str | None = None
        self.sessions: dict[str, InferenceSession] = {}
        self.spectrogram_pipeline = None

        self._scan_models()
        logger.info("InferenceServicer initialized")

    def _scan_models(self):
        """Scan models directory."""
        if not MODELS_DIR.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            return

        for f in MODELS_DIR.iterdir():
            if f.suffix in (".pth", ".trt", ".onnx"):
                model_id = hashlib.md5(f.name.encode()).hexdigest()[:8]
                backend = {".pth": "pytorch", ".trt": "tensorrt", ".onnx": "onnx"}.get(
                    f.suffix, "pytorch"
                )

                self.models[model_id] = ModelState(
                    model_id=model_id,
                    model_name=f.stem,
                    model_path=str(f),
                    model_hash=self._hash_file(str(f)),
                    backend=backend,
                    num_classes=2,
                    class_names=["background", "creamy_chicken"],
                )

        logger.info(f"Found {len(self.models)} models")

    @staticmethod
    def _hash_file(path: str) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()[:16]

    def _model_to_proto(self, m: ModelState) -> inference_pb2.ModelInfo:
        backend_map = {"pytorch": 0, "tensorrt": 1, "onnx": 2}
        return inference_pb2.ModelInfo(
            model_id=m.model_id,
            model_name=m.model_name,
            model_path=m.model_path,
            model_hash=m.model_hash,
            backend=backend_map.get(m.backend, 0),
            num_classes=m.num_classes,
            class_names=m.class_names,
            loaded_at_ms=int(m.loaded_at * 1000) if m.loaded_at else 0,
        )

    def LoadModel(self, request, context):
        model_path = request.model_path

        if not os.path.exists(model_path):
            return inference_pb2.LoadModelResponse(success=False, error_message="Model not found")

        try:
            model_id = hashlib.md5(model_path.encode()).hexdigest()[:8]
            engine = InferenceEngine(model_path, num_classes=2)

            model_state = ModelState(
                model_id=model_id,
                model_name=request.model_name or Path(model_path).stem,
                model_path=model_path,
                model_hash=self._hash_file(model_path),
                backend=engine.backend,
                num_classes=2,
                class_names=["background", "creamy_chicken"],
                engine=engine,
                loaded_at=time.time(),
            )

            self.models[model_id] = model_state
            if request.hot_swap or self.active_model_id is None:
                self.active_model_id = model_id

            logger.info(f"Loaded: {model_state.model_name} ({engine.backend})")

            return inference_pb2.LoadModelResponse(
                success=True, model_id=model_id, model_info=self._model_to_proto(model_state)
            )
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return inference_pb2.LoadModelResponse(success=False, error_message=str(e))

    def UnloadModel(self, request, context):
        model_id = request.model_id
        if model_id in self.models:
            del self.models[model_id]
            if self.active_model_id == model_id:
                self.active_model_id = None
            return inference_pb2.UnloadModelResponse(success=True)
        return inference_pb2.UnloadModelResponse(success=False, error_message="Not found")

    def GetModels(self, request, context):
        return inference_pb2.GetModelsResponse(
            models=[self._model_to_proto(m) for m in self.models.values()]
        )

    def GetActiveModel(self, request, context):
        if self.active_model_id and self.active_model_id in self.models:
            return inference_pb2.GetActiveModelResponse(
                has_active_model=True,
                active_model=self._model_to_proto(self.models[self.active_model_id]),
            )
        return inference_pb2.GetActiveModelResponse(has_active_model=False)

    def StartInference(self, request, context):
        model_id = request.model_id or self.active_model_id

        if not model_id or model_id not in self.models:
            return inference_pb2.StartInferenceResponse(success=False, error_message="No model")

        model = self.models[model_id]
        if model.engine is None:
            model.engine = InferenceEngine(model.model_path, model.num_classes)

        session_id = str(uuid.uuid4())[:8]

        source = request.source
        source_type = "file"
        source_path = None
        rx_channel = None

        if source.HasField("live"):
            source_type = "live"
            rx_channel = source.live.rx_channel or 1
        elif source.HasField("file"):
            source_type = "file"
            source_path = source.file.file_path

        config = {
            "score_threshold": request.config.score_threshold or 0.5,
            "batch_size": request.config.batch_size or 4,
            "precision": request.config.precision or "fp16",
            "chunk_ms": request.config.chunk_ms or 100,
            "nfft": request.config.nfft or 1024,
            "noverlap": request.config.noverlap or 768,
        }

        session = InferenceSession(
            session_id=session_id,
            model_id=model_id,
            source_type=source_type,
            source_path=source_path,
            rx_channel=rx_channel,
            config=config,
            start_time=time.time(),
        )

        self.sessions[session_id] = session
        self.active_model_id = model_id

        logger.info(f"Inference started: {session_id} on {source_type}")

        return inference_pb2.StartInferenceResponse(success=True, session_id=session_id)

    def StopInference(self, request, context):
        session_id = request.session_id

        if session_id not in self.sessions:
            return inference_pb2.StopInferenceResponse(success=False, error_message="Not found")

        session = self.sessions[session_id]
        session.is_active = False
        elapsed = int((time.time() - session.start_time) * 1000)

        stats = inference_pb2.InferenceStats(
            total_chunks=session.chunk_count,
            total_detections=session.detection_count,
            avg_inference_ms=10.0,
            throughput_chunks_per_sec=session.chunk_count / max(1, elapsed / 1000),
            duration_ms=elapsed,
        )

        logger.info(f"Inference stopped: {session_id}")

        return inference_pb2.StopInferenceResponse(success=True, stats=stats)

    def StreamDetections(self, request, context):
        """Stream detection frames from inference."""
        session_id = request.session_id

        if session_id not in self.sessions:
            context.abort(grpc.StatusCode.NOT_FOUND, "Session not found")
            return

        session = self.sessions[session_id]
        model = self.models.get(session.model_id)

        if not model or not model.engine:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model not loaded")
            return

        # Initialize spectrogram pipeline
        if self.spectrogram_pipeline is None:
            self.spectrogram_pipeline = SpectrogramPipeline(
                nfft=session.config["nfft"],
                noverlap=session.config["noverlap"],
                out_size=1024,
            )

        # Get RF context
        channel = self.device_control.channels.get(session.rx_channel or 1)
        center_freq = channel.center_freq_mhz if channel else 825.0
        bandwidth = channel.bandwidth_mhz if channel else 20.0

        # Open file if needed
        file_handle = None
        sample_rate = 20e6
        chunk_samples = int(sample_rate * session.config["chunk_ms"] / 1000)

        if session.source_type == "file" and session.source_path:
            if not os.path.exists(session.source_path):
                context.abort(grpc.StatusCode.NOT_FOUND, "File not found")
                return
            file_handle = open(session.source_path, "rb")

        try:
            frame_id = 0

            while session.is_active and context.is_active():
                # Read IQ
                if file_handle:
                    raw = file_handle.read(chunk_samples * 8)
                    if not raw:
                        file_handle.seek(0)
                        raw = file_handle.read(chunk_samples * 8)
                    iq_data = np.frombuffer(raw, dtype=np.complex64)
                else:
                    iq_data = (
                        np.random.randn(chunk_samples) + 1j * np.random.randn(chunk_samples)
                    ).astype(np.complex64)

                # Generate spectrogram & infer
                spec = self.spectrogram_pipeline.process(iq_data)
                detections = model.engine.infer(spec, session.config["score_threshold"])

                # Build detection messages
                det_protos = []
                for det_result in detections:
                    boxes = det_result.get("boxes", [])
                    scores = det_result.get("scores", [])
                    labels = det_result.get("labels", [])

                    for i in range(len(boxes)):
                        box = boxes[i]
                        det_protos.append(
                            inference_pb2.Detection(
                                detection_id=i,
                                class_name=model.class_names[int(labels[i])]
                                if int(labels[i]) < len(model.class_names)
                                else "unknown",
                                class_id=int(labels[i]),
                                confidence=float(scores[i]),
                                x1=float(box[0]) / 1024,
                                y1=float(box[1]) / 1024,
                                x2=float(box[2]) / 1024,
                                y2=float(box[3]) / 1024,
                                freq_center_mhz=center_freq,
                                freq_bandwidth_mhz=bandwidth * (box[3] - box[1]) / 1024,
                            )
                        )

                # Build frame
                frame = inference_pb2.DetectionFrame(
                    frame_id=frame_id,
                    timestamp_ms=int(time.time() * 1000),
                    elapsed_seconds=time.time() - session.start_time,
                    detections=det_protos,
                    center_freq_mhz=center_freq,
                    bandwidth_mhz=bandwidth,
                )

                yield frame

                frame_id += 1
                session.chunk_count += 1
                session.detection_count += len(det_protos)

                time.sleep(0.033)  # ~30 FPS

        finally:
            if file_handle:
                file_handle.close()

    def StartTraining(self, request, context):
        return inference_pb2.StartTrainingResponse(
            success=False, error_message="Not implemented yet"
        )

    def StopTraining(self, request, context):
        return inference_pb2.StopTrainingResponse(
            success=False, error_message="Not implemented yet"
        )

    def GetTrainingStatus(self, request, context):
        return inference_pb2.TrainingStatus(training_id="", state=0)

    def StreamTrainingProgress(self, request, context):
        return


# ============= Server =============


def serve(port: int = 50051, max_workers: int = 10):
    """Start gRPC server."""
    if not PROTO_AVAILABLE:
        logger.error("Proto stubs not found!")
        logger.error("Run: ./generate_stubs.sh (or generate_stubs.bat on Windows)")
        return

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )

    device_control = DeviceControlServicer()
    inference_service = InferenceServicer(device_control)

    control_pb2_grpc.add_DeviceControlServicer_to_server(device_control, server)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(inference_service, server)

    try:
        server.add_insecure_port(f"0.0.0.0:{port}")
        server.start()
        logger.info(f"gRPC server started on port {port}")
    except RuntimeError as e:
        logger.warning(f"gRPC failed to bind port {port}: {e}")
        return
    logger.info("Services: DeviceControl, InferenceService")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(grace=5)


# ============= WebSocket Server (for Flutter) =============


async def ws_inference_handler(websocket):
    """Handle WebSocket inference requests from Flutter."""
    import json
    import traceback

    # websockets 16.0 API: use websocket.path directly
    client_addr = getattr(websocket, "remote_address", "unknown")
    ws_path = websocket.path if hasattr(websocket, "path") else "/"
    logger_server.info(f"WS: Client connected from {client_addr}")
    logger_server.info(f"WS: Path: {ws_path}")
    logger.info("WebSocket client connected")

    session_id = None
    engine = None
    pipeline = None
    is_running = False
    score_thresh = 0.5
    inference_task = None

    async def run_inference_loop(ws, eng, pipe, score_th, params):
        """Background task for continuous inference streaming."""
        nonlocal is_running
        file_path = params.get("file_path", "")
        center_freq = params.get("center_freq_mhz", 825.0)
        bandwidth = params.get("bandwidth_mhz", 20.0)
        sample_rate = params.get("sample_rate", 20e6)
        chunk_ms = params.get("chunk_ms", 200)

        # Find a file if not specified
        if not file_path:
            for f in DATA_DIR.iterdir():
                if f.suffix in (".iq", ".cf32", ".bin"):
                    file_path = str(f)
                    break

        frame_id = 0
        chunk_samples = int(sample_rate * chunk_ms / 1000)

        if file_path and os.path.exists(file_path):
            logger.info(f"[WS] Running inference on: {file_path}")
            with open(file_path, "rb") as f:
                while is_running:
                    try:
                        raw = f.read(chunk_samples * 8)
                        if not raw or len(raw) < chunk_samples * 8:
                            f.seek(0)
                            continue

                        iq_data = np.frombuffer(raw, dtype=np.complex64)
                        spec = pipe.process(iq_data)
                        detections = eng.infer(spec, score_th)

                        det_list = []
                        for det_result in detections:
                            for i, (box, score, label) in enumerate(
                                zip(
                                    det_result.get("boxes", []),
                                    det_result.get("scores", []),
                                    det_result.get("labels", []),
                                    strict=False,
                                )
                            ):
                                det_list.append(
                                    {
                                        "detection_id": i,
                                        "class_id": int(label),
                                        "class_name": "creamy_chicken"
                                        if label == 1
                                        else "background",
                                        "confidence": float(score),
                                        "x1": float(box[0]) / 1024,
                                        "y1": float(box[1]) / 1024,
                                        "x2": float(box[2]) / 1024,
                                        "y2": float(box[3]) / 1024,
                                    }
                                )

                        await ws.send(
                            json.dumps(
                                {
                                    "type": "detection_frame",
                                    "frame_id": frame_id,
                                    "timestamp_ms": int(time.time() * 1000),
                                    "elapsed_seconds": frame_id * chunk_ms / 1000,
                                    "detections": det_list,
                                    "center_freq_mhz": center_freq,
                                    "bandwidth_mhz": bandwidth,
                                }
                            )
                        )

                        frame_id += 1
                        if frame_id % 10 == 0:
                            logger.info(f"[WS] Frame {frame_id}: {len(det_list)} detections")
                        await asyncio.sleep(chunk_ms / 1000)
                    except Exception as e:
                        logger.error(f"[WS] Inference error: {e}")
                        break
        else:
            # Demo mode with random data
            logger.info("[WS] Demo mode - generating random data")
            while is_running:
                try:
                    iq_data = (
                        np.random.randn(chunk_samples) + 1j * np.random.randn(chunk_samples)
                    ).astype(np.complex64) * 0.1
                    spec = pipe.process(iq_data)
                    detections = eng.infer(spec, score_th)

                    det_list = []
                    for det_result in detections:
                        for i, (box, score, label) in enumerate(
                            zip(
                                det_result.get("boxes", []),
                                det_result.get("scores", []),
                                det_result.get("labels", []),
                                strict=False,
                            )
                        ):
                            det_list.append(
                                {
                                    "detection_id": i,
                                    "class_id": int(label),
                                    "class_name": "creamy_chicken" if label == 1 else "background",
                                    "confidence": float(score),
                                    "x1": float(box[0]) / 1024,
                                    "y1": float(box[1]) / 1024,
                                    "x2": float(box[2]) / 1024,
                                    "y2": float(box[3]) / 1024,
                                }
                            )

                    await ws.send(
                        json.dumps(
                            {
                                "type": "detection_frame",
                                "frame_id": frame_id,
                                "timestamp_ms": int(time.time() * 1000),
                                "elapsed_seconds": frame_id * chunk_ms / 1000,
                                "detections": det_list,
                                "center_freq_mhz": center_freq,
                                "bandwidth_mhz": bandwidth,
                            }
                        )
                    )

                    frame_id += 1
                    if frame_id % 10 == 0:
                        logger.info(
                            f"[WS] Frame {frame_id}: {len(det_list)} detections (demo mode)"
                        )
                    await asyncio.sleep(chunk_ms / 1000)
                except Exception as e:
                    logger.error(f"[WS] Inference error: {e}")
                    break

        logger.info(f"[WS] Inference loop ended, {frame_id} frames")

    try:
        logger_server.info("WS: Waiting for messages...")
        async for message in websocket:
            print(
                f"[WS] Received: {message[:200] if len(str(message)) > 200 else message}",
                flush=True,
            )
            try:
                data = json.loads(message)
                cmd = data.get("command")
                logger_server.info(f"WS: Command: {cmd}")

                if cmd == "start":
                    # Start inference session
                    model_path = data.get("model_path", "")
                    score_thresh = data.get("score_threshold", 0.5)
                    nfft = data.get("nfft", 4096)
                    noverlap = data.get("noverlap", 2048)

                    # Find model
                    if not model_path:
                        # Find first .pth in models dir
                        for f in MODELS_DIR.iterdir():
                            if f.suffix == ".pth":
                                model_path = str(f)
                                break

                    if not model_path or not os.path.exists(model_path):
                        await websocket.send(
                            json.dumps(
                                {"type": "error", "message": f"Model not found: {model_path}"}
                            )
                        )
                        continue

                    # Load engine
                    logger.info(f"Loading model: {model_path}")
                    engine = InferenceEngine(model_path, num_classes=2)
                    pipeline = SpectrogramPipeline(nfft=nfft, noverlap=noverlap)

                    session_id = str(uuid.uuid4())[:8]
                    is_running = True

                    await websocket.send(
                        json.dumps(
                            {
                                "type": "session_started",
                                "session_id": session_id,
                                "model_path": model_path,
                                "backend": engine.backend,
                            }
                        )
                    )

                    logger.info(f"WebSocket inference started: {session_id}")

                elif cmd == "run":
                    # Continuous inference - run in BACKGROUND TASK
                    if not engine or not pipeline:
                        await websocket.send(
                            json.dumps(
                                {"type": "error", "message": "No active session - call start first"}
                            )
                        )
                        continue

                    # Cancel existing task if any
                    if inference_task and not inference_task.done():
                        inference_task.cancel()

                    is_running = True
                    inference_task = asyncio.create_task(
                        run_inference_loop(websocket, engine, pipeline, score_thresh, data)
                    )
                    logger.info("[WS] Started inference background task")

                elif cmd == "stop":
                    is_running = False
                    await websocket.send(
                        json.dumps({"type": "session_stopped", "session_id": session_id})
                    )
                    logger.info(f"WebSocket inference stopped: {session_id}")

                elif cmd == "infer":
                    # Run inference on provided IQ data (base64 encoded)
                    if not engine or not pipeline:
                        await websocket.send(
                            json.dumps({"type": "error", "message": "No active session"})
                        )
                        continue

                    import base64

                    iq_b64 = data.get("iq_data")
                    if iq_b64:
                        iq_bytes = base64.b64decode(iq_b64)
                        iq_data = np.frombuffer(iq_bytes, dtype=np.complex64)

                        # Generate spectrogram and run inference
                        spec = pipeline.process(iq_data)
                        detections = engine.infer(spec, score_thresh)

                        # Format response
                        det_list = []
                        for det_result in detections:
                            boxes = det_result.get("boxes", [])
                            scores = det_result.get("scores", [])
                            labels = det_result.get("labels", [])

                            for i in range(len(boxes)):
                                box = boxes[i]
                                det_list.append(
                                    {
                                        "detection_id": i,
                                        "class_id": int(labels[i]),
                                        "class_name": "creamy_chicken"
                                        if labels[i] == 1
                                        else "background",
                                        "confidence": float(scores[i]),
                                        "x1": float(box[0]) / 1024,
                                        "y1": float(box[1]) / 1024,
                                        "x2": float(box[2]) / 1024,
                                        "y2": float(box[3]) / 1024,
                                    }
                                )

                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "detection_frame",
                                    "frame_id": data.get("frame_id", 0),
                                    "timestamp_ms": int(time.time() * 1000),
                                    "detections": det_list,
                                }
                            )
                        )

            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
            except Exception as e:
                logger_server.info(f"WS: ERROR in command handler: {e}")
                traceback.print_exc()
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))

    except Exception as e:
        logger_server.info(f"WS: Handler crashed: {e}")
        traceback.print_exc()
    finally:
        logger_server.info("WS: Client disconnected")


# ============= Unified Pipeline WebSocket Handler =============


async def unified_pipeline_handler(websocket):
    """WebSocket handler for unified IQ pipeline - State of the Art.

    Single data source → Waterfall + Inference → Flutter
    """
    import json
    import traceback

    logger_server.info("Unified: Handler started")

    try:
        from unified_pipeline import UnifiedServer

        logger_server.info("Unified: Import successful")
    except ImportError as e:
        logger_server.info(f"Unified: IMPORT FAILED: {e}")
        traceback.print_exc()
        await websocket.send(json.dumps({"type": "error", "message": f"Import error: {e}"}))
        return

    # Find IQ file and model
    iq_file = None
    logger_server.info("Unified: DATA_DIR: {DATA_DIR}")
    try:
        for f in DATA_DIR.iterdir():
            logger_server.info("Unified: Found file: {f.name}")
            if f.suffix in (".sigmf-data", ".iq", ".cf32", ".bin"):
                iq_file = str(f)
                break
    except Exception as e:
        logger_server.info(f"Unified: Error scanning data dir: {e}")

    model_path = None
    logger_server.info("Unified: MODELS_DIR: {MODELS_DIR}")
    try:
        for f in MODELS_DIR.iterdir():
            logger_server.info("Unified: Found model: {f.name}")
            if f.suffix == ".pth":
                model_path = str(f)
                break
    except Exception as e:
        logger_server.info(f"Unified: Error scanning models dir: {e}")

    if not iq_file:
        msg = f"No IQ file found in {DATA_DIR}"
        logger_server.info(f"Unified: ERROR: {msg}")
        await websocket.send(json.dumps({"type": "error", "message": msg}))
        return

    if not model_path:
        msg = f"No model found in {MODELS_DIR}"
        logger_server.info(f"Unified: ERROR: {msg}")
        await websocket.send(json.dumps({"type": "error", "message": msg}))
        return

    logger_server.info(f"Unified: IQ: {iq_file}")
    logger_server.info(f"Unified: Model: {model_path}")

    try:
        logger_server.info("Unified: Creating UnifiedServer...")
        server = UnifiedServer(iq_file, model_path)
        logger_server.info("Unified: UnifiedServer created!")
    except Exception as e:
        logger_server.info(f"Unified: FAILED to create server: {e}")
        traceback.print_exc()
        await websocket.send(json.dumps({"type": "error", "message": f"Server init error: {e}"}))
        return

    # Start pipeline in background task
    pipeline_task = asyncio.create_task(server.run_pipeline(websocket))

    try:
        async for message in websocket:
            data = json.loads(message)
            cmd = data.get("command")

            if cmd == "stop":
                server.stop()
                break
            elif cmd == "status":
                await websocket.send(
                    json.dumps(
                        {
                            "type": "status",
                            "is_running": server.is_running,
                        }
                    )
                )
    except Exception as e:
        logger.error(f"[Unified] Handler error: {e}")
    finally:
        server.stop()
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass


# ============= Video Streaming Pipeline Handler (NEW) =============


async def video_pipeline_handler(websocket):
    """WebSocket handler for VIDEO STREAMING pipeline.

    Uses H.264/NVENC encoding (or JPEG fallback) for efficient streaming.
    Full frame waterfall buffer accumulated on backend.
    """
    import json
    import traceback

    logger_server.info("Video: Handler started")

    try:
        from unified_pipeline import VideoStreamServer

        logger_server.info("Video: Import successful")
    except ImportError as e:
        logger_server.info(f"Video: IMPORT FAILED: {e}")
        traceback.print_exc()
        await websocket.send(json.dumps({"type": "error", "message": f"Import error: {e}"}))
        return

    # Find IQ file and model
    iq_file = None
    logger_server.info("Video: DATA_DIR: {DATA_DIR}")
    try:
        for f in DATA_DIR.iterdir():
            logger_server.info("Video: Found file: {f.name}")
            if f.suffix in (".sigmf-data", ".iq", ".cf32", ".bin"):
                iq_file = str(f)
                break
    except Exception as e:
        logger_server.info(f"Video: Error scanning data dir: {e}")

    model_path = None
    logger_server.info("Video: MODELS_DIR: {MODELS_DIR}")
    try:
        for f in MODELS_DIR.iterdir():
            logger_server.info("Video: Found model: {f.name}")
            if f.suffix == ".pth":
                model_path = str(f)
                break
    except Exception as e:
        logger_server.info(f"Video: Error scanning models dir: {e}")

    if not iq_file:
        msg = f"No IQ file found in {DATA_DIR}"
        logger_server.info(f"Video: ERROR: {msg}")
        await websocket.send(json.dumps({"type": "error", "message": msg}))
        return

    if not model_path:
        msg = f"No model found in {MODELS_DIR}"
        logger_server.info(f"Video: ERROR: {msg}")
        await websocket.send(json.dumps({"type": "error", "message": msg}))
        return

    logger_server.info(f"Video: IQ: {iq_file}")
    logger_server.info(f"Video: Model: {model_path}")

    try:
        logger_server.info("Video: Creating VideoStreamServer...")
        server = VideoStreamServer(iq_file, model_path)
        logger_server.info("Video: VideoStreamServer created!")
    except Exception as e:
        logger_server.info(f"Video: FAILED to create server: {e}")
        traceback.print_exc()
        await websocket.send(json.dumps({"type": "error", "message": f"Server init error: {e}"}))
        return

    # Start pipeline in background task
    pipeline_task = asyncio.create_task(server.run_pipeline(websocket))

    try:
        async for message in websocket:
            data = json.loads(message)
            cmd = data.get("command")

            if cmd == "stop":
                server.stop()
                break
            elif cmd == "status":
                await websocket.send(
                    json.dumps(
                        {
                            "type": "status",
                            "is_running": server.is_running,
                            "pts": server.current_pts,
                        }
                    )
                )
    except Exception as e:
        logger.error(f"[Video] Handler error: {e}")
    finally:
        server.stop()
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass


# ============= Training WebSocket Handler (Hydra Phase 5) =============


async def ws_training_handler(websocket):
    """WebSocket handler for Hydra training and version management.

    Commands:
        - get_registry: Get all signals and versions
        - get_version_history: Get version history for a signal
        - train_signal: Train new or extend existing signal
        - cancel_training: Cancel running training
        - promote_version: Promote a version
        - rollback_signal: Rollback to previous version
        - save_sample: Save training sample (IQ + boxes)
        - get_samples: List samples for a signal
    """
    import json
    import traceback

    logger_server.info("Training: Handler started")
    logger_server.info("Training: HYDRA_AVAILABLE = {HYDRA_AVAILABLE}")

    # Wrap EVERYTHING in try/except to ensure we send error before closing
    try:
        await _ws_training_handler_impl(websocket)
    except Exception as e:
        logger_server.info(f"Training: CRITICAL ERROR in handler: {e}")
        traceback.print_exc()
        try:
            await websocket.send(
                json.dumps(
                    {
                        "type": "init_error",
                        "message": f"CRITICAL HANDLER ERROR:\n{e}\n\n{traceback.format_exc()}",
                    }
                )
            )
            # Wait a moment for message to send before closing
            await asyncio.sleep(0.5)
        except Exception as e:
            logger_server.info(f"Training: Failed to send error to client: {e}")
        # DON'T re-raise - just let handler close gracefully


async def _ws_training_handler_impl(websocket):
    """Actual implementation of training handler."""
    import json
    import traceback

    logger_server.info("Training: _ws_training_handler_impl ENTRY")

    # Don't immediately close - try to initialize services
    training_service = None
    version_manager = None
    sample_manager = None
    split_manager = None

    # Store any initialization errors so we can send them to the client
    init_errors = []

    # Always try to import SampleManager first (it has minimal dependencies)
    try:
        from training.sample_manager import SampleManager

        sample_manager = SampleManager(str(BASE_DIR / "training_data" / "signals"))
        logger_server.info("Training: SampleManager initialized: {sample_manager.base_dir}")
    except Exception as e:
        error_msg = f"SampleManager failed: {e}\n{traceback.format_exc()}"
        init_errors.append(error_msg)
        logger_server.info(f"Training: {error_msg}")

    if HYDRA_AVAILABLE:
        # Full Hydra available - initialize all services (TrainingService already imported globally)
        logger_server.info("Training: HYDRA_AVAILABLE=True, initializing services...")
        logger_server.info("Training:   MODELS_DIR: {MODELS_DIR}")
        logger.info(f"[Training]   training_data_dir: {BASE_DIR / 'training_data' / 'signals'}")
        try:
            logger_server.info("Training:   Creating TrainingService...")
            training_service = TrainingService(
                models_dir=str(MODELS_DIR),
                training_data_dir=str(BASE_DIR / "training_data" / "signals"),
            )
            logger_server.info("Training:   OK TrainingService created")

            logger_server.info("Training:   Creating VersionManager...")
            version_manager = VersionManager(str(MODELS_DIR))
            logger_server.info("Training:   OK VersionManager created")

            logger_server.info("Training:   Creating SplitManager...")
            split_manager = SplitManager(str(BASE_DIR / "training_data" / "signals"))
            logger_server.info("Training:   OK SplitManager created")

            logger_server.info("Training: OK Full training services initialized")
        except Exception as e:
            error_msg = f"Training services failed: {e}\n{traceback.format_exc()}"
            init_errors.append(error_msg)
            logger_server.info(f"Training: FAILED: {error_msg}")
    else:
        # HYDRA_AVAILABLE=False - try importing training-related modules separately
        logger_server.info("Training: HYDRA_AVAILABLE=False, trying minimal imports...")
        try:
            from hydra.version_manager import VersionManager as VM

            version_manager = VM(str(MODELS_DIR))
            logger_server.info("Training: VersionManager OK")
        except Exception as e:
            error_msg = f"VersionManager failed: {e}\n{traceback.format_exc()}"
            init_errors.append(error_msg)
            logger_server.info(f"Training: {error_msg}")

        try:
            from training.splits import SplitManager as SM

            split_manager = SM(str(BASE_DIR / "training_data" / "signals"))
            logger_server.info("Training: SplitManager OK")
        except Exception as e:
            error_msg = f"SplitManager failed: {e}\n{traceback.format_exc()}"
            init_errors.append(error_msg)
            logger_server.info(f"Training: {error_msg}")

        try:
            from training.service import TrainingService as TS

            training_service = TS(
                models_dir=str(MODELS_DIR),
                training_data_dir=str(BASE_DIR / "training_data" / "signals"),
            )
            logger_server.info("Training: TrainingService OK")
        except Exception as e:
            error_msg = f"TrainingService failed: {e}\n{traceback.format_exc()}"
            init_errors.append(error_msg)
            logger_server.info(f"Training: {error_msg}")

    # If there were any init errors, send them to the client immediately
    if init_errors:
        full_error = "INITIALIZATION ERRORS:\n" + "\n---\n".join(init_errors)
        logger_server.info("Training: Sending init errors to client")
        await websocket.send(
            json.dumps(
                {
                    "type": "init_error",
                    "message": full_error,
                    "sample_manager_ok": sample_manager is not None,
                    "training_service_ok": training_service is not None,
                }
            )
        )

    training_task = None

    def make_progress_callback(ws):
        """Create callback for training progress updates."""

        async def send_progress(progress: TrainingProgress):
            try:
                await ws.send(
                    json.dumps(
                        {
                            "type": "training_progress",
                            "signal_name": training_service._current_signal,
                            "epoch": progress.epoch,
                            "total_epochs": progress.total_epochs,
                            "train_loss": progress.train_loss,
                            "val_loss": progress.val_loss,
                            "f1_score": progress.f1_score,
                            "precision": progress.precision,
                            "recall": progress.recall,
                            "is_best": progress.is_best,
                            "elapsed_sec": progress.elapsed_sec,
                        }
                    )
                )
            except Exception as e:
                logger_server.info(f"Training: Progress send error: {e}")

        def callback(progress: TrainingProgress):
            # Schedule async send from sync callback
            try:
                asyncio.get_event_loop().create_task(send_progress(progress))
            except RuntimeError:
                pass

        return callback

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                cmd = data.get("command")
                if cmd not in ("save_sample",):  # Silent commands
                    logger_server.info(f"Training: Command: {cmd}")

                # =====================
                # Registry & Versions
                # =====================

                if cmd == "get_registry":
                    registry = version_manager.get_registry()
                    await websocket.send(json.dumps({"type": "registry", **registry}))

                elif cmd == "get_version_history":
                    signal_name = data.get("signal_name")
                    if not signal_name:
                        await websocket.send(
                            json.dumps({"type": "error", "message": "signal_name required"})
                        )
                        continue

                    versions = version_manager.get_version_history(signal_name)
                    info = version_manager.get_signal_info(signal_name) or {}

                    await websocket.send(
                        json.dumps(
                            {
                                "type": "version_history",
                                "signal_name": signal_name,
                                "active_version": info.get("active_head_version"),
                                "versions": versions,
                            }
                        )
                    )

                elif cmd == "promote_version":
                    signal_name = data.get("signal_name")
                    version = data.get("version")

                    if not signal_name or not version:
                        await websocket.send(
                            json.dumps(
                                {"type": "error", "message": "signal_name and version required"}
                            )
                        )
                        continue

                    try:
                        version_manager.promote_version(
                            signal_name, version, "Manual promotion from UI"
                        )
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "version_promoted",
                                    "signal_name": signal_name,
                                    "version": version,
                                }
                            )
                        )
                    except Exception as e:
                        await websocket.send(json.dumps({"type": "error", "message": str(e)}))

                elif cmd == "rollback_signal":
                    signal_name = data.get("signal_name")

                    if not signal_name:
                        await websocket.send(
                            json.dumps({"type": "error", "message": "signal_name required"})
                        )
                        continue

                    try:
                        new_version = version_manager.rollback(signal_name)
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "version_rollback",
                                    "signal_name": signal_name,
                                    "active_version": new_version,
                                }
                            )
                        )
                    except Exception as e:
                        await websocket.send(json.dumps({"type": "error", "message": str(e)}))

                # =====================
                # Training
                # =====================

                elif cmd == "train_signal":
                    signal_name = data.get("signal_name")
                    notes = data.get("notes")
                    is_new = data.get("is_new", False)

                    if not signal_name:
                        await websocket.send(
                            json.dumps({"type": "error", "message": "signal_name required"})
                        )
                        continue

                    if training_service is None:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": "training_service not initialized - training modules failed to import",
                                }
                            )
                        )
                        continue

                    if training_service.is_training:
                        await websocket.send(
                            json.dumps({"type": "error", "message": "Training already in progress"})
                        )
                        continue

                    # Run training in background
                    async def run_training():
                        try:
                            callback = make_progress_callback(websocket)

                            if is_new:
                                result = training_service.train_new_signal(
                                    signal_name, notes=notes, callback=callback
                                )
                            else:
                                result = training_service.extend_signal(
                                    signal_name, notes=notes, callback=callback
                                )

                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "training_complete",
                                        "signal_name": result.signal_name,
                                        "version": result.version,
                                        "sample_count": result.sample_count,
                                        "epochs_trained": result.epochs_trained,
                                        "early_stopped": result.early_stopped,
                                        "metrics": result.metrics,
                                        "training_time_sec": result.training_time_sec,
                                        "previous_version": result.previous_version,
                                        "previous_metrics": result.previous_metrics,
                                        "auto_promoted": result.auto_promoted,
                                        "promotion_reason": result.promotion_reason,
                                    }
                                )
                            )
                        except Exception as e:
                            logger_server.info(f"Training: Error: {e}")
                            traceback.print_exc()
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "training_failed",
                                        "signal_name": signal_name,
                                        "error": str(e),
                                    }
                                )
                            )

                    training_task = asyncio.create_task(run_training())

                elif cmd == "cancel_training":
                    if training_service is not None:
                        training_service.cancel_training()
                    await websocket.send(json.dumps({"type": "training_cancelled"}))

                elif cmd == "get_training_status":
                    if training_service is None:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "training_status",
                                    "is_training": False,
                                    "current_signal": None,
                                    "error": "training_service not initialized",
                                }
                            )
                        )
                        continue
                    status = training_service.get_training_status()
                    await websocket.send(json.dumps({"type": "training_status", **status}))

                # =====================
                # Training Samples
                # =====================

                elif cmd == "save_sample":
                    signal_name = data.get("signal_name")
                    iq_data_b64 = data.get("iq_data")
                    boxes = data.get("boxes", [])
                    metadata = data.get("metadata", {})

                    if not signal_name or not iq_data_b64:
                        await websocket.send(
                            json.dumps(
                                {"type": "error", "message": "signal_name and iq_data required"}
                            )
                        )
                        continue

                    if sample_manager is None:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": "sample_manager not initialized - training modules failed to import",
                                }
                            )
                        )
                        continue

                    try:
                        sample_id, is_new = sample_manager.save_sample(
                            signal_name, iq_data_b64, boxes, metadata
                        )
                        # Silent - no spam

                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "sample_saved",
                                    "signal_name": signal_name,
                                    "sample_id": sample_id,
                                    "is_new": is_new,  # False if duplicate was skipped
                                    "total_samples": sample_manager.get_sample_count(signal_name),
                                }
                            )
                        )
                    except Exception as e:
                        logger_server.info(f"Training: ERROR saving sample: {e}")
                        traceback.print_exc()
                        await websocket.send(
                            json.dumps(
                                {"type": "error", "message": f"save_sample failed: {str(e)}"}
                            )
                        )

                elif cmd == "get_samples":
                    signal_name = data.get("signal_name")

                    if not signal_name:
                        await websocket.send(
                            json.dumps({"type": "error", "message": "signal_name required"})
                        )
                        continue

                    samples = sample_manager.list_samples(signal_name)
                    split_summary = split_manager.get_split_summary(signal_name)

                    await websocket.send(
                        json.dumps(
                            {
                                "type": "samples_list",
                                "signal_name": signal_name,
                                "samples": samples,
                                "split": split_summary,
                            }
                        )
                    )

                elif cmd == "delete_sample":
                    signal_name = data.get("signal_name")
                    sample_id = data.get("sample_id")

                    if not signal_name or not sample_id:
                        await websocket.send(
                            json.dumps(
                                {"type": "error", "message": "signal_name and sample_id required"}
                            )
                        )
                        continue

                    deleted = sample_manager.delete_sample(signal_name, sample_id)
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "sample_deleted" if deleted else "error",
                                "signal_name": signal_name,
                                "sample_id": sample_id,
                                "message": None if deleted else "Sample not found",
                            }
                        )
                    )

                # =====================
                # Sub-Band Extraction
                # =====================

                elif cmd == "extract_subband":
                    # Extract sub-band from RFCAP file
                    # {
                    #   "command": "extract_subband",
                    #   "source_file": "captures/MAN_123456Z_825MHz.rfcap",
                    #   "output_file": "training_data/signals/unk/samples/0001.rfcap",
                    #   "center_offset_hz": 500000,
                    #   "bandwidth_hz": 2000000,
                    #   "start_sec": 5.0,
                    #   "duration_sec": 10.0,
                    #   "stopband_db": 60.0  (optional)
                    # }
                    source_file = data.get("source_file")
                    output_file = data.get("output_file")
                    center_offset_hz = data.get("center_offset_hz", 0)
                    bandwidth_hz = data.get("bandwidth_hz")
                    data.get("start_sec", 0)
                    data.get("duration_sec")
                    data.get("stopband_db", 60.0)

                    if not source_file or not bandwidth_hz:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": "source_file and bandwidth_hz required",
                                }
                            )
                        )
                        continue

                    # Resolve paths relative to BASE_DIR
                    source_path = (
                        str(BASE_DIR / source_file)
                        if not os.path.isabs(source_file)
                        else source_file
                    )
                    output_path = (
                        str(BASE_DIR / output_file)
                        if output_file and not os.path.isabs(output_file)
                        else output_file
                    )

                    # Generate output path if not specified
                    if not output_path:
                        output_dir = BASE_DIR / "training_data" / "extracted"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_path = str(output_dir / f"subband_{timestamp}.rfcap")
                    else:
                        # Ensure output directory exists
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    if not os.path.exists(source_path):
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": f"Source file not found: {source_path}",
                                }
                            )
                        )
                        continue

                    # Send progress updates
                    async def send_extraction_progress(progress):
                        try:
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "extraction_progress",
                                        "progress": progress,
                                        "source_file": source_file,
                                    }
                                )
                            )
                        except Exception as e:
                            logger_server.info(f"Training: Extraction progress send error: {e}")

                    def progress_callback(p):
                        try:
                            asyncio.get_event_loop().create_task(send_extraction_progress(p))
                        except RuntimeError:
                            pass

                    try:
                        # Read source header to get original center and sample rate
                        source_header = _read_rfcap_header(source_path)
                        original_center_hz = source_header["center_freq"]
                        original_sample_rate = source_header["sample_rate"]
                        new_center_hz = original_center_hz + center_offset_hz

                        logger.info(
                            f"[Extract] Using FAST simple_extract: {bandwidth_hz / 1e6:.2f}MHz BW, offset {center_offset_hz / 1e6:.2f}MHz",
                        )
                        logger.info(
                            f"[Extract] Original: center={original_center_hz / 1e6:.2f}MHz, rate={original_sample_rate / 1e6:.2f}Msps",
                        )
                        logger.info(
                            f"[Extract] Target: center={new_center_hz / 1e6:.2f}MHz, bw={bandwidth_hz / 1e6:.2f}MHz",
                        )

                        # Run FAST extraction (101 taps vs 4095 taps)
                        result = await asyncio.to_thread(
                            simple_extract_subband,
                            input_path=source_path,
                            output_path=output_path,
                            original_center_hz=original_center_hz,
                            original_sample_rate=original_sample_rate,
                            new_center_hz=new_center_hz,
                            new_bandwidth_hz=bandwidth_hz,
                            num_taps=101,  # Fast filter
                            progress_callback=progress_callback,  # Track progress!
                        )

                        logger_server.info("Extract: DONE! Output: {result['output_path']}")
                        logger.info(
                            f"[Extract] Output rate: {result['new_sample_rate'] / 1e6:.2f}Msps, samples: {result['output_samples']}",
                        )

                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "subband_extracted",
                                    "output_file": result["output_path"],
                                    "source_rate_hz": result["original_sample_rate"],
                                    "output_rate_hz": result["new_sample_rate"],
                                    "bandwidth_hz": result["new_bandwidth_hz"],
                                    "center_offset_hz": result["shift_hz"],
                                    "output_center_freq_hz": result["new_center_hz"],
                                    "input_samples": result["input_samples"],
                                    "output_samples": result["output_samples"],
                                    "decimation_ratio": result["decimation"],
                                    "filter_taps": result["filter_taps"],
                                    "processing_time_sec": 0.0,  # simple_extract doesn't track this
                                }
                            )
                        )
                    except Exception as e:
                        logger_server.info(f"Training: Extraction error: {e}")
                        traceback.print_exc()
                        await websocket.send(
                            json.dumps(
                                {"type": "error", "command": "extract_subband", "message": str(e)}
                            )
                        )

                else:
                    await websocket.send(
                        json.dumps({"type": "error", "message": f"Unknown command: {cmd}"})
                    )

            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
            except Exception as e:
                logger_server.info(f"Training: Command error: {e}")
                traceback.print_exc()
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))

    except Exception as e:
        logger_server.info(f"Training: Handler error: {e}")
        traceback.print_exc()
    finally:
        if training_task and not training_task.done():
            training_service.cancel_training()
            training_task.cancel()
        logger_server.info("Training: Handler closed")


async def ws_router(websocket):
    """Route WebSocket connections based on path."""
    import json
    import traceback

    try:
        # websockets 16.0+ API: try multiple ways to get path
        ws_path = "/"

        # Try request.path first (websockets 16.0+)
        if hasattr(websocket, "request") and hasattr(websocket.request, "path"):
            ws_path = websocket.request.path
        # Fallback to direct path attribute
        elif hasattr(websocket, "path"):
            ws_path = websocket.path
        # Try raw_path on request
        elif hasattr(websocket, "request") and hasattr(websocket.request, "raw_path"):
            ws_path = (
                websocket.request.raw_path.decode()
                if isinstance(websocket.request.raw_path, bytes)
                else websocket.request.raw_path
            )

        logger_server.info(f"Router: Path: {ws_path}")
    except Exception as e:
        logger_server.info(f"Router: Error getting path: {e}")
        ws_path = "/"

    try:
        if "/training" in ws_path:
            logger_server.info("Router: Routing to ws_training_handler (Hydra training/versioning)")
            await ws_training_handler(websocket)
        elif "/video" in ws_path:
            logger.info(
                "[Router] Routing to video_ws_handler (H.264/JPEG streaming with set_time_span support)",
            )
            # Find IQ file and model for video_ws_handler
            iq_file = None
            model_path = None
            for f in DATA_DIR.iterdir():
                if f.suffix in (".sigmf-data", ".iq", ".cf32", ".bin"):
                    iq_file = str(f)
                    break
            for f in MODELS_DIR.iterdir():
                if f.suffix == ".pth":
                    model_path = str(f)
                    break
            if iq_file and model_path:
                await video_ws_handler(websocket, iq_file, model_path)
            else:
                import json

                await websocket.send(
                    json.dumps(
                        {
                            "type": "error",
                            "message": f"Missing IQ file or model: iq={iq_file}, model={model_path}",
                        }
                    )
                )
        elif "/unified" in ws_path:
            logger_server.info("Router: Routing to unified_pipeline_handler (row-by-row)")
            await unified_pipeline_handler(websocket)
        elif "/inference" in ws_path:
            logger_server.info("Router: Routing to ws_inference_handler")
            await ws_inference_handler(websocket)
        else:
            logger.info(f"[Router] Default route (path={ws_path}) -> unified_pipeline_handler")
            await unified_pipeline_handler(websocket)
    except Exception as e:
        logger_server.info(f"Router: EXCEPTION in handler: {e}")
        traceback.print_exc()


async def run_websocket_server(port: int = 0):
    """Run WebSocket server for Flutter clients.

    If port=0, OS picks a free port and we print it for parent process to read.
    This is the professional KISS approach - no port conflicts ever.

    Uses _async_shutdown_event for graceful shutdown coordination.
    """
    global _async_shutdown_event

    try:
        import websockets

        logger_server.info("SERVER: websockets version: {websockets.__version__}")
    except ImportError:
        logger_server.info("SERVER: ERROR: websockets package not found!")
        return

    # Create async shutdown event for this event loop
    _async_shutdown_event = asyncio.Event()

    # If threading event is already set, set async event too
    if _shutdown_event.is_set():
        _async_shutdown_event.set()

    server = None
    try:
        # Let OS pick port if port=0, or use specified port
        # Use ws_router to dispatch based on path
        # max_size=100MB to handle large IQ data for training samples
        server = await websockets.serve(
            ws_router,
            "127.0.0.1",  # localhost only
            port,
            reuse_address=True,
            max_size=100 * 1024 * 1024,  # 100 MB max message size
        )

        # Register server for cleanup
        _register_cleanup(server)

        # Get the actual port (especially useful when port=0)
        actual_port = server.sockets[0].getsockname()[1]

        # Write server info to file for Flutter to read (replaces stdout parsing)
        info_file = write_server_info(ws_port=actual_port, grpc_port=50051)
        logger_server.info(f"SERVER: WebSocket server READY on ws://127.0.0.1:{actual_port}")
        logger_server.info(f"SERVER: Connection info written to {info_file}")

        # Wait for shutdown signal instead of running forever
        await _async_shutdown_event.wait()
        logger_server.info("SERVER: Shutdown signal received, closing WebSocket server...")

    except OSError as e:
        logger_server.info(f"SERVER: ERROR: WebSocket failed to bind: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if server:
            server.close()
            await server.wait_closed()
            logger_server.info("SERVER: WebSocket server closed")


def serve_both(grpc_port: int = 50051, ws_port: int = 50052, max_workers: int = 10):
    """Start both gRPC and WebSocket servers with coordinated shutdown."""

    logger_server.info(f"MAIN: serve_both called: gRPC={grpc_port}, WS={ws_port}")

    # Set up signal handlers first
    setup_signal_handlers()

    # Start parent process watchdog - auto-exit if parent (Flutter) dies
    start_parent_watchdog()

    grpc_server = None

    # Start gRPC in a thread
    def run_grpc():
        nonlocal grpc_server
        logger_server.info(f"MAIN: gRPC thread starting on port {grpc_port}...")
        if not PROTO_AVAILABLE:
            logger_server.info("MAIN: gRPC not available (proto stubs missing)")
            return

        grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )

        device_control = DeviceControlServicer()
        inference_service = InferenceServicer(device_control)

        control_pb2_grpc.add_DeviceControlServicer_to_server(device_control, grpc_server)
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(inference_service, grpc_server)

        try:
            grpc_server.add_insecure_port(f"0.0.0.0:{grpc_port}")
            grpc_server.start()
            logger.info(f"gRPC server started on port {grpc_port}")
        except RuntimeError as e:
            logger.warning(f"gRPC failed to bind port {grpc_port}: {e}")
            return

        # Wait for shutdown signal
        _shutdown_event.wait()
        logger_server.info("MAIN: gRPC thread received shutdown signal, stopping...")
        grpc_server.stop(grace=5)
        logger_server.info("MAIN: gRPC server stopped")

    grpc_thread = threading.Thread(target=run_grpc, daemon=True)
    grpc_thread.start()
    logger_server.info("MAIN: gRPC thread started, now starting WebSocket server...")

    # Run WebSocket in main thread (asyncio)
    try:
        asyncio.run(run_websocket_server(ws_port))
    except KeyboardInterrupt:
        logger_server.info("MAIN: KeyboardInterrupt received")
    finally:
        logger_server.info("MAIN: Triggering shutdown...")
        _shutdown_event.set()

        # Wait for gRPC thread to finish
        grpc_thread.join(timeout=10)
        if grpc_thread.is_alive():
            logger_server.info("MAIN: Warning: gRPC thread did not stop gracefully")

        # Final cleanup
        _cleanup_all()
        logger_server.info("MAIN: Shutdown complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="G20 gRPC + WebSocket Server")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port")
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket port (default: 8765, avoids Windows reserved ranges)",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--grpc-only", action="store_true", help="Only start gRPC server")
    parser.add_argument("--ws-only", action="store_true", help="Only start WebSocket server")
    args = parser.parse_args()

    # WebSocket port - explicit None check so port 0 works (OS picks free port)
    ws_port = args.ws_port if args.ws_port is not None else args.port + 1

    if args.grpc_only:
        serve(port=args.port, max_workers=args.workers)
    elif args.ws_only:
        asyncio.run(run_websocket_server(ws_port))
    else:
        serve_both(grpc_port=args.port, ws_port=ws_port, max_workers=args.workers)
