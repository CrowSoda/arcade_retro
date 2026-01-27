"""
WebSocket inference handler for Flutter clients.

Extracted from server.py using strangler fig pattern.
"""

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

logger = logging.getLogger("g20.server")

# Directory paths
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


async def ws_inference_handler(websocket):
    """Handle WebSocket inference requests from Flutter."""

    client_addr = getattr(websocket, "remote_address", "unknown")
    ws_path = websocket.path if hasattr(websocket, "path") else "/"
    logger.info(f"WS: Client connected from {client_addr}")
    logger.info(f"WS: Path: {ws_path}")

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

                        det_list = _format_detections(detections)

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

                    det_list = _format_detections(detections)

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
                        logger.info(f"[WS] Frame {frame_id}: {len(det_list)} detections (demo)")
                    await asyncio.sleep(chunk_ms / 1000)
                except Exception as e:
                    logger.error(f"[WS] Inference error: {e}")
                    break

        logger.info(f"[WS] Inference loop ended, {frame_id} frames")

    try:
        logger.info("WS: Waiting for messages...")
        async for message in websocket:
            print(
                f"[WS] Received: {message[:200] if len(str(message)) > 200 else message}",
                flush=True,
            )
            try:
                data = json.loads(message)
                cmd = data.get("command")
                logger.info(f"WS: Command: {cmd}")

                if cmd == "start":
                    # Start inference session
                    from inference import InferenceEngine, SpectrogramPipeline

                    model_path = data.get("model_path", "")
                    score_thresh = data.get("score_threshold", 0.5)
                    nfft = data.get("nfft", 4096)
                    noverlap = data.get("noverlap", 2048)

                    # Find model
                    if not model_path:
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

                        spec = pipeline.process(iq_data)
                        detections = engine.infer(spec, score_thresh)
                        det_list = _format_detections(detections)

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
                logger.info(f"WS: ERROR in command handler: {e}")
                traceback.print_exc()
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))

    except Exception as e:
        logger.info(f"WS: Handler crashed: {e}")
        traceback.print_exc()
    finally:
        logger.info("WS: Client disconnected")


def _format_detections(detections) -> list:
    """Format detection results for JSON response."""
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
    return det_list
