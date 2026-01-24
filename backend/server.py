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

import os
import sys
import time
import uuid
import asyncio
import logging
import hashlib
from pathlib import Path
from concurrent import futures
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

import grpc
import numpy as np

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
from inference import InferenceEngine, MultiModelEngine, SpectrogramPipeline
from unified_pipeline import video_ws_handler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("g20.server")

# Directories
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# ============= Data Classes =============

@dataclass
class ChannelState:
    channel_number: int
    center_freq_mhz: float = 825.0
    bandwidth_mhz: float = 20.0
    gain_db: float = 20.0
    state: str = "IDLE"
    active_capture_id: Optional[str] = None


@dataclass
class CaptureSession:
    capture_id: str
    signal_name: str
    rx_channel: int
    duration_seconds: int
    start_time: float
    file_path: str
    bytes_captured: int = 0
    is_active: bool = True


@dataclass
class InferenceSession:
    session_id: str
    model_id: str
    source_type: str
    source_path: Optional[str]
    rx_channel: Optional[int]
    config: Dict[str, Any]
    start_time: float
    chunk_count: int = 0
    detection_count: int = 0
    is_active: bool = True


@dataclass
class ModelState:
    model_id: str
    model_name: str
    model_path: str
    model_hash: str
    backend: str
    num_classes: int
    class_names: List[str]
    engine: Optional[InferenceEngine] = None
    loaded_at: float = 0.0


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
        
        self.captures: Dict[str, CaptureSession] = {}
        logger.info("DeviceControlServicer initialized")
    
    def SetFrequency(self, request, context):
        channel = request.rx_channel or 1
        freq = request.center_freq_mhz
        
        if freq < 30.0 or freq > 6000.0:
            return control_pb2.FrequencyResponse(
                success=False,
                error_message=f"Frequency {freq} MHz outside range (30-6000)",
                actual_freq_mhz=self.channels[channel].center_freq_mhz
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
                actual_bw_mhz=self.channels[channel].bandwidth_mhz
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
                actual_gain_db=self.channels[channel].gain_db
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
        
        return control_pb2.CaptureResponse(
            success=True, capture_id=capture_id, file_path=file_path
        )
    
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
            channels.append(control_pb2.ChannelStatus(
                channel_number=ch_num,
                center_freq_mhz=ch.center_freq_mhz,
                bandwidth_mhz=ch.bandwidth_mhz,
                gain_db=ch.gain_db,
                state=state_map.get(ch.state, 0),
                active_capture_id=ch.active_capture_id or ""
            ))
        
        return control_pb2.StatusResponse(
            state=control_pb2.DEVICE_CONNECTED,
            temperature_c=self.temperature_c,
            channels=channels,
            cpu_usage_percent=25.0,
            gpu_usage_percent=45.0,
            memory_used_bytes=512 * 1024 * 1024,
            disk_free_bytes=100 * 1024 * 1024 * 1024
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
            num_tx_channels=2
        )
    
    def SetMode(self, request, context):
        modes = {0: "AUTO_SCAN", 1: "MANUAL", 2: "SENSE_ONLY"}
        self.operating_mode = modes.get(request.mode, "AUTO_SCAN")
        self.manual_timeout = request.timeout_seconds if hasattr(request, 'timeout_seconds') else None
        
        logger.info(f"Mode → {self.operating_mode}")
        
        return control_pb2.ModeResponse(success=True, current_mode=request.mode)


# ============= Inference Service =============

class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """ML inference pipeline with TensorRT support."""
    
    def __init__(self, device_control: DeviceControlServicer):
        self.device_control = device_control
        self.models: Dict[str, ModelState] = {}
        self.active_model_id: Optional[str] = None
        self.sessions: Dict[str, InferenceSession] = {}
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
                backend = {".pth": "pytorch", ".trt": "tensorrt", ".onnx": "onnx"}.get(f.suffix, "pytorch")
                
                self.models[model_id] = ModelState(
                    model_id=model_id,
                    model_name=f.stem,
                    model_path=str(f),
                    model_hash=self._hash_file(str(f)),
                    backend=backend,
                    num_classes=2,
                    class_names=["background", "creamy_chicken"]
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
            loaded_at_ms=int(m.loaded_at * 1000) if m.loaded_at else 0
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
                loaded_at=time.time()
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
                active_model=self._model_to_proto(self.models[self.active_model_id])
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
            start_time=time.time()
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
            duration_ms=elapsed
        )
        
        logger.info(f"Inference stopped: {session_id}")
        
        return inference_pb2.StopInferenceResponse(success=True, stats=stats)
    
    def StreamDetections(self, request, context):
        """Stream detection frames from inference."""
        session_id = request.session_id
        include_spectrogram = request.include_spectrogram
        
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
            import torch
            
            while session.is_active and context.is_active():
                # Read IQ
                if file_handle:
                    raw = file_handle.read(chunk_samples * 8)
                    if not raw:
                        file_handle.seek(0)
                        raw = file_handle.read(chunk_samples * 8)
                    iq_data = np.frombuffer(raw, dtype=np.complex64)
                else:
                    iq_data = (np.random.randn(chunk_samples) + 1j * np.random.randn(chunk_samples)).astype(np.complex64)
                
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
                        det_protos.append(inference_pb2.Detection(
                            detection_id=i,
                            class_name=model.class_names[int(labels[i])] if int(labels[i]) < len(model.class_names) else "unknown",
                            class_id=int(labels[i]),
                            confidence=float(scores[i]),
                            x1=float(box[0]) / 1024,
                            y1=float(box[1]) / 1024,
                            x2=float(box[2]) / 1024,
                            y2=float(box[3]) / 1024,
                            freq_center_mhz=center_freq,
                            freq_bandwidth_mhz=bandwidth * (box[3] - box[1]) / 1024,
                        ))
                
                # Build frame
                frame = inference_pb2.DetectionFrame(
                    frame_id=frame_id,
                    timestamp_ms=int(time.time() * 1000),
                    elapsed_seconds=time.time() - session.start_time,
                    detections=det_protos,
                    center_freq_mhz=center_freq,
                    bandwidth_mhz=bandwidth
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
        return inference_pb2.StartTrainingResponse(success=False, error_message="Not implemented yet")
    
    def StopTraining(self, request, context):
        return inference_pb2.StopTrainingResponse(success=False, error_message="Not implemented yet")
    
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
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
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
    import sys
    import traceback
    
    # websockets 16.0 API: use websocket.path directly
    client_addr = getattr(websocket, 'remote_address', 'unknown')
    ws_path = websocket.path if hasattr(websocket, 'path') else '/'
    print(f"[WS] Client connected from {client_addr}", flush=True)
    print(f"[WS] Path: {ws_path}", flush=True)
    logger.info(f"WebSocket client connected")
    
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
                            for i, (box, score, label) in enumerate(zip(
                                det_result.get("boxes", []),
                                det_result.get("scores", []),
                                det_result.get("labels", [])
                            )):
                                det_list.append({
                                    "detection_id": i,
                                    "class_id": int(label),
                                    "class_name": "creamy_chicken" if label == 1 else "background",
                                    "confidence": float(score),
                                    "x1": float(box[0]) / 1024, "y1": float(box[1]) / 1024,
                                    "x2": float(box[2]) / 1024, "y2": float(box[3]) / 1024,
                                })
                        
                        await ws.send(json.dumps({
                            "type": "detection_frame",
                            "frame_id": frame_id,
                            "timestamp_ms": int(time.time() * 1000),
                            "elapsed_seconds": frame_id * chunk_ms / 1000,
                            "detections": det_list,
                            "center_freq_mhz": center_freq,
                            "bandwidth_mhz": bandwidth,
                        }))
                        
                        frame_id += 1
                        if frame_id % 10 == 0:
                            logger.info(f"[WS] Frame {frame_id}: {len(det_list)} detections")
                        await asyncio.sleep(chunk_ms / 1000)
                    except Exception as e:
                        logger.error(f"[WS] Inference error: {e}")
                        break
        else:
            # Demo mode with random data
            logger.info(f"[WS] Demo mode - generating random data")
            while is_running:
                try:
                    iq_data = (np.random.randn(chunk_samples) + 1j * np.random.randn(chunk_samples)).astype(np.complex64) * 0.1
                    spec = pipe.process(iq_data)
                    detections = eng.infer(spec, score_th)
                    
                    det_list = []
                    for det_result in detections:
                        for i, (box, score, label) in enumerate(zip(
                            det_result.get("boxes", []),
                            det_result.get("scores", []),
                            det_result.get("labels", [])
                        )):
                            det_list.append({
                                "detection_id": i,
                                "class_id": int(label),
                                "class_name": "creamy_chicken" if label == 1 else "background",
                                "confidence": float(score),
                                "x1": float(box[0]) / 1024, "y1": float(box[1]) / 1024,
                                "x2": float(box[2]) / 1024, "y2": float(box[3]) / 1024,
                            })
                    
                    await ws.send(json.dumps({
                        "type": "detection_frame",
                        "frame_id": frame_id,
                        "timestamp_ms": int(time.time() * 1000),
                        "elapsed_seconds": frame_id * chunk_ms / 1000,
                        "detections": det_list,
                        "center_freq_mhz": center_freq,
                        "bandwidth_mhz": bandwidth,
                    }))
                    
                    frame_id += 1
                    if frame_id % 10 == 0:
                        logger.info(f"[WS] Frame {frame_id}: {len(det_list)} detections (demo mode)")
                    await asyncio.sleep(chunk_ms / 1000)
                except Exception as e:
                    logger.error(f"[WS] Inference error: {e}")
                    break
        
        logger.info(f"[WS] Inference loop ended, {frame_id} frames")
    
    try:
        print(f"[WS] Waiting for messages...", flush=True)
        async for message in websocket:
            print(f"[WS] Received: {message[:200] if len(str(message)) > 200 else message}", flush=True)
            try:
                data = json.loads(message)
                cmd = data.get("command")
                print(f"[WS] Command: {cmd}", flush=True)
                
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
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"Model not found: {model_path}"
                        }))
                        continue
                    
                    # Load engine
                    logger.info(f"Loading model: {model_path}")
                    engine = InferenceEngine(model_path, num_classes=2)
                    pipeline = SpectrogramPipeline(nfft=nfft, noverlap=noverlap)
                    
                    session_id = str(uuid.uuid4())[:8]
                    is_running = True
                    
                    await websocket.send(json.dumps({
                        "type": "session_started",
                        "session_id": session_id,
                        "model_path": model_path,
                        "backend": engine.backend
                    }))
                    
                    logger.info(f"WebSocket inference started: {session_id}")
                
                elif cmd == "run":
                    # Continuous inference - run in BACKGROUND TASK
                    if not engine or not pipeline:
                        await websocket.send(json.dumps({"type": "error", "message": "No active session - call start first"}))
                        continue
                    
                    # Cancel existing task if any
                    if inference_task and not inference_task.done():
                        inference_task.cancel()
                    
                    is_running = True
                    inference_task = asyncio.create_task(
                        run_inference_loop(websocket, engine, pipeline, score_thresh, data)
                    )
                    logger.info(f"[WS] Started inference background task")
                
                elif cmd == "stop":
                    is_running = False
                    await websocket.send(json.dumps({
                        "type": "session_stopped",
                        "session_id": session_id
                    }))
                    logger.info(f"WebSocket inference stopped: {session_id}")
                
                elif cmd == "infer":
                    # Run inference on provided IQ data (base64 encoded)
                    if not engine or not pipeline:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "No active session"
                        }))
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
                                det_list.append({
                                    "detection_id": i,
                                    "class_id": int(labels[i]),
                                    "class_name": "creamy_chicken" if labels[i] == 1 else "background",
                                    "confidence": float(scores[i]),
                                    "x1": float(box[0]) / 1024,
                                    "y1": float(box[1]) / 1024,
                                    "x2": float(box[2]) / 1024,
                                    "y2": float(box[3]) / 1024,
                                })
                        
                        await websocket.send(json.dumps({
                            "type": "detection_frame",
                            "frame_id": data.get("frame_id", 0),
                            "timestamp_ms": int(time.time() * 1000),
                            "detections": det_list
                        }))
                
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
            except Exception as e:
                print(f"[WS] ERROR in command handler: {e}", flush=True)
                traceback.print_exc()
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))
    
    except Exception as e:
        print(f"[WS] Handler crashed: {e}", flush=True)
        traceback.print_exc()
    finally:
        print(f"[WS] Client disconnected", flush=True)


# ============= Unified Pipeline WebSocket Handler =============

async def unified_pipeline_handler(websocket):
    """WebSocket handler for unified IQ pipeline - State of the Art.
    
    Single data source → Waterfall + Inference → Flutter
    """
    import json
    import traceback
    
    print("[Unified] Handler started", flush=True)
    
    try:
        from unified_pipeline import UnifiedServer
        print("[Unified] Import successful", flush=True)
    except ImportError as e:
        print(f"[Unified] IMPORT FAILED: {e}", flush=True)
        traceback.print_exc()
        await websocket.send(json.dumps({'type': 'error', 'message': f'Import error: {e}'}))
        return
    
    # Find IQ file and model
    iq_file = None
    print(f"[Unified] DATA_DIR: {DATA_DIR}", flush=True)
    try:
        for f in DATA_DIR.iterdir():
            print(f"[Unified] Found file: {f.name}", flush=True)
            if f.suffix in ('.sigmf-data', '.iq', '.cf32', '.bin'):
                iq_file = str(f)
                break
    except Exception as e:
        print(f"[Unified] Error scanning data dir: {e}", flush=True)
    
    model_path = None
    print(f"[Unified] MODELS_DIR: {MODELS_DIR}", flush=True)
    try:
        for f in MODELS_DIR.iterdir():
            print(f"[Unified] Found model: {f.name}", flush=True)
            if f.suffix == '.pth':
                model_path = str(f)
                break
    except Exception as e:
        print(f"[Unified] Error scanning models dir: {e}", flush=True)
    
    if not iq_file:
        msg = f'No IQ file found in {DATA_DIR}'
        print(f"[Unified] ERROR: {msg}", flush=True)
        await websocket.send(json.dumps({'type': 'error', 'message': msg}))
        return
    
    if not model_path:
        msg = f'No model found in {MODELS_DIR}'
        print(f"[Unified] ERROR: {msg}", flush=True)
        await websocket.send(json.dumps({'type': 'error', 'message': msg}))
        return
    
    print(f"[Unified] IQ: {iq_file}", flush=True)
    print(f"[Unified] Model: {model_path}", flush=True)
    
    try:
        print("[Unified] Creating UnifiedServer...", flush=True)
        server = UnifiedServer(iq_file, model_path)
        print("[Unified] UnifiedServer created!", flush=True)
    except Exception as e:
        print(f"[Unified] FAILED to create server: {e}", flush=True)
        traceback.print_exc()
        await websocket.send(json.dumps({'type': 'error', 'message': f'Server init error: {e}'}))
        return
    
    # Start pipeline in background task
    pipeline_task = asyncio.create_task(server.run_pipeline(websocket))
    
    try:
        async for message in websocket:
            data = json.loads(message)
            cmd = data.get('command')
            
            if cmd == 'stop':
                server.stop()
                break
            elif cmd == 'status':
                await websocket.send(json.dumps({
                    'type': 'status',
                    'is_running': server.is_running,
                }))
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
    
    print("[Video] Handler started", flush=True)
    
    try:
        from unified_pipeline import VideoStreamServer
        print("[Video] Import successful", flush=True)
    except ImportError as e:
        print(f"[Video] IMPORT FAILED: {e}", flush=True)
        traceback.print_exc()
        await websocket.send(json.dumps({'type': 'error', 'message': f'Import error: {e}'}))
        return
    
    # Find IQ file and model
    iq_file = None
    print(f"[Video] DATA_DIR: {DATA_DIR}", flush=True)
    try:
        for f in DATA_DIR.iterdir():
            print(f"[Video] Found file: {f.name}", flush=True)
            if f.suffix in ('.sigmf-data', '.iq', '.cf32', '.bin'):
                iq_file = str(f)
                break
    except Exception as e:
        print(f"[Video] Error scanning data dir: {e}", flush=True)
    
    model_path = None
    print(f"[Video] MODELS_DIR: {MODELS_DIR}", flush=True)
    try:
        for f in MODELS_DIR.iterdir():
            print(f"[Video] Found model: {f.name}", flush=True)
            if f.suffix == '.pth':
                model_path = str(f)
                break
    except Exception as e:
        print(f"[Video] Error scanning models dir: {e}", flush=True)
    
    if not iq_file:
        msg = f'No IQ file found in {DATA_DIR}'
        print(f"[Video] ERROR: {msg}", flush=True)
        await websocket.send(json.dumps({'type': 'error', 'message': msg}))
        return
    
    if not model_path:
        msg = f'No model found in {MODELS_DIR}'
        print(f"[Video] ERROR: {msg}", flush=True)
        await websocket.send(json.dumps({'type': 'error', 'message': msg}))
        return
    
    print(f"[Video] IQ: {iq_file}", flush=True)
    print(f"[Video] Model: {model_path}", flush=True)
    
    try:
        print("[Video] Creating VideoStreamServer...", flush=True)
        server = VideoStreamServer(iq_file, model_path)
        print("[Video] VideoStreamServer created!", flush=True)
    except Exception as e:
        print(f"[Video] FAILED to create server: {e}", flush=True)
        traceback.print_exc()
        await websocket.send(json.dumps({'type': 'error', 'message': f'Server init error: {e}'}))
        return
    
    # Start pipeline in background task
    pipeline_task = asyncio.create_task(server.run_pipeline(websocket))
    
    try:
        async for message in websocket:
            data = json.loads(message)
            cmd = data.get('command')
            
            if cmd == 'stop':
                server.stop()
                break
            elif cmd == 'status':
                await websocket.send(json.dumps({
                    'type': 'status',
                    'is_running': server.is_running,
                    'pts': server.current_pts,
                }))
    except Exception as e:
        logger.error(f"[Video] Handler error: {e}")
    finally:
        server.stop()
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass


async def ws_router(websocket):
    """Route WebSocket connections based on path."""
    import json
    import traceback
    
    try:
        # websockets 16.0+ API: try multiple ways to get path
        ws_path = '/'
        
        # Try request.path first (websockets 16.0+)
        if hasattr(websocket, 'request') and hasattr(websocket.request, 'path'):
            ws_path = websocket.request.path
        # Fallback to direct path attribute
        elif hasattr(websocket, 'path'):
            ws_path = websocket.path
        # Try raw_path on request
        elif hasattr(websocket, 'request') and hasattr(websocket.request, 'raw_path'):
            ws_path = websocket.request.raw_path.decode() if isinstance(websocket.request.raw_path, bytes) else websocket.request.raw_path
        
        print(f"[Router] Path: {ws_path}", flush=True)
    except Exception as e:
        print(f"[Router] Error getting path: {e}", flush=True)
        ws_path = '/'
    
    try:
        if '/video' in ws_path:
            print("[Router] Routing to video_ws_handler (H.264/JPEG streaming with set_time_span support)", flush=True)
            # Find IQ file and model for video_ws_handler
            iq_file = None
            model_path = None
            for f in DATA_DIR.iterdir():
                if f.suffix in ('.sigmf-data', '.iq', '.cf32', '.bin'):
                    iq_file = str(f)
                    break
            for f in MODELS_DIR.iterdir():
                if f.suffix == '.pth':
                    model_path = str(f)
                    break
            if iq_file and model_path:
                await video_ws_handler(websocket, iq_file, model_path)
            else:
                import json
                await websocket.send(json.dumps({'type': 'error', 'message': f'Missing IQ file or model: iq={iq_file}, model={model_path}'}))
        elif '/unified' in ws_path:
            print("[Router] Routing to unified_pipeline_handler (row-by-row)", flush=True)
            await unified_pipeline_handler(websocket)
        elif '/inference' in ws_path:
            print("[Router] Routing to ws_inference_handler", flush=True)
            await ws_inference_handler(websocket)
        else:
            print(f"[Router] Default route (path={ws_path}) -> unified_pipeline_handler", flush=True)
            await unified_pipeline_handler(websocket)
    except Exception as e:
        print(f"[Router] EXCEPTION in handler: {e}", flush=True)
        traceback.print_exc()


async def run_websocket_server(port: int = 0):
    """Run WebSocket server for Flutter clients.
    
    If port=0, OS picks a free port and we print it for parent process to read.
    This is the professional KISS approach - no port conflicts ever.
    """
    try:
        import websockets
        print(f"[SERVER] websockets version: {websockets.__version__}", flush=True)
    except ImportError:
        print("[SERVER] ERROR: websockets package not found!", flush=True)
        return
    
    try:
        # Let OS pick port if port=0, or use specified port
        # Use ws_router to dispatch based on path
        server = await websockets.serve(
            ws_router,
            "127.0.0.1",  # localhost only
            port,
            reuse_address=True
        )
        
        # Get the actual port (especially useful when port=0)
        actual_port = server.sockets[0].getsockname()[1]
        
        # CRITICAL: Print port for parent process (Flutter) to read
        print(f"WS_PORT:{actual_port}", flush=True)
        print(f"[SERVER] WebSocket server READY on ws://127.0.0.1:{actual_port}", flush=True)
        
        await asyncio.Future()  # Run forever
    except OSError as e:
        print(f"[SERVER] ERROR: WebSocket failed to bind: {e}", flush=True)
        import traceback
        traceback.print_exc()


def serve_both(grpc_port: int = 50051, ws_port: int = 50052, max_workers: int = 10):
    """Start both gRPC and WebSocket servers."""
    import threading
    
    print(f"[MAIN] serve_both called: gRPC={grpc_port}, WS={ws_port}", flush=True)
    
    # Start gRPC in a thread
    def run_grpc():
        print(f"[MAIN] gRPC thread starting on port {grpc_port}...", flush=True)
        if PROTO_AVAILABLE:
            serve(port=grpc_port, max_workers=max_workers)
        else:
            print("[MAIN] gRPC not available (proto stubs missing)", flush=True)
    
    grpc_thread = threading.Thread(target=run_grpc, daemon=True)
    grpc_thread.start()
    print("[MAIN] gRPC thread started, now starting WebSocket server...", flush=True)
    
    # Run WebSocket in main thread (asyncio)
    try:
        asyncio.run(run_websocket_server(ws_port))
    except KeyboardInterrupt:
        print("[MAIN] Shutting down...", flush=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="G20 gRPC + WebSocket Server")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port (default: 8765, avoids Windows reserved ranges)")
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
