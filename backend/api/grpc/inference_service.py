"""
InferenceServicer - ML inference pipeline with TensorRT support.

Extracted from server.py using strangler fig pattern.
"""

import hashlib
import logging
import os

# Import generated proto stubs
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import grpc
import numpy as np
from core.models import InferenceSession, ModelState

GENERATED_DIR = Path(__file__).parent.parent.parent / "generated"
sys.path.insert(0, str(GENERATED_DIR))

try:
    import inference_pb2
    import inference_pb2_grpc

    PROTO_AVAILABLE = True
except ImportError:
    PROTO_AVAILABLE = False
    inference_pb2 = None
    inference_pb2_grpc = None

# Import inference engine
try:
    from inference import InferenceEngine, SpectrogramPipeline

    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    InferenceEngine = None
    SpectrogramPipeline = None

logger = logging.getLogger("g20.server")

# Directories
BASE_DIR = Path(__file__).parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


class InferenceServicer:
    """ML inference pipeline with TensorRT support."""

    def __init__(self, device_control: Any):
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

    def _model_to_proto(self, m: ModelState):
        if not PROTO_AVAILABLE:
            return None
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
            engine = InferenceEngine(model_path, num_classes=2) if INFERENCE_AVAILABLE else None

            model_state = ModelState(
                model_id=model_id,
                model_name=request.model_name or Path(model_path).stem,
                model_path=model_path,
                model_hash=self._hash_file(model_path),
                backend=engine.backend if engine else "pytorch",
                num_classes=2,
                class_names=["background", "creamy_chicken"],
                engine=engine,
                loaded_at=time.time(),
            )

            self.models[model_id] = model_state
            if request.hot_swap or self.active_model_id is None:
                self.active_model_id = model_id

            logger.info(f"Loaded: {model_state.model_name}")

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
        if model.engine is None and INFERENCE_AVAILABLE:
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
        if self.spectrogram_pipeline is None and INFERENCE_AVAILABLE:
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
                spec = (
                    self.spectrogram_pipeline.process(iq_data)
                    if self.spectrogram_pipeline
                    else None
                )
                detections = (
                    model.engine.infer(spec, session.config["score_threshold"])
                    if spec is not None
                    else []
                )

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


# Make servicer available if protos are built
if PROTO_AVAILABLE:
    InferenceServicer.__bases__ = (inference_pb2_grpc.InferenceServiceServicer,)
