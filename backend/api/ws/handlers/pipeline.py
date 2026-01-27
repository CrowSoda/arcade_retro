"""
WebSocket pipeline handlers for unified and video streaming.

Extracted from server.py using strangler fig pattern.
"""

import asyncio
import json
import logging
import traceback
from pathlib import Path

logger = logging.getLogger("g20.server")

# Directory paths
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


async def unified_pipeline_handler(websocket):
    """WebSocket handler for unified IQ pipeline.

    Single data source → Waterfall + Inference → Flutter
    """

    logger.info("Unified: Handler started")

    try:
        from unified_pipeline import UnifiedServer

        logger.info("Unified: Import successful")
    except ImportError as e:
        logger.info(f"Unified: IMPORT FAILED: {e}")
        traceback.print_exc()
        await websocket.send(json.dumps({"type": "error", "message": f"Import error: {e}"}))
        return

    # Find IQ file and model
    iq_file = _find_iq_file()
    model_path = _find_model()

    if not iq_file:
        msg = f"No IQ file found in {DATA_DIR}"
        logger.info(f"Unified: ERROR: {msg}")
        await websocket.send(json.dumps({"type": "error", "message": msg}))
        return

    if not model_path:
        msg = f"No model found in {MODELS_DIR}"
        logger.info(f"Unified: ERROR: {msg}")
        await websocket.send(json.dumps({"type": "error", "message": msg}))
        return

    logger.info(f"Unified: IQ: {iq_file}")
    logger.info(f"Unified: Model: {model_path}")

    try:
        logger.info("Unified: Creating UnifiedServer...")
        server = UnifiedServer(iq_file, model_path)
        logger.info("Unified: UnifiedServer created!")
    except Exception as e:
        logger.info(f"Unified: FAILED to create server: {e}")
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


async def video_pipeline_handler(websocket):
    """WebSocket handler for VIDEO STREAMING pipeline.

    Uses H.264/NVENC encoding (or JPEG fallback) for efficient streaming.
    Full frame waterfall buffer accumulated on backend.
    """

    logger.info("Video: Handler started")

    try:
        from unified_pipeline import VideoStreamServer

        logger.info("Video: Import successful")
    except ImportError as e:
        logger.info(f"Video: IMPORT FAILED: {e}")
        traceback.print_exc()
        await websocket.send(json.dumps({"type": "error", "message": f"Import error: {e}"}))
        return

    # Find IQ file and model
    iq_file = _find_iq_file()
    model_path = _find_model()

    if not iq_file:
        msg = f"No IQ file found in {DATA_DIR}"
        logger.info(f"Video: ERROR: {msg}")
        await websocket.send(json.dumps({"type": "error", "message": msg}))
        return

    if not model_path:
        msg = f"No model found in {MODELS_DIR}"
        logger.info(f"Video: ERROR: {msg}")
        await websocket.send(json.dumps({"type": "error", "message": msg}))
        return

    logger.info(f"Video: IQ: {iq_file}")
    logger.info(f"Video: Model: {model_path}")

    try:
        logger.info("Video: Creating VideoStreamServer...")
        server = VideoStreamServer(iq_file, model_path)
        logger.info("Video: VideoStreamServer created!")
    except Exception as e:
        logger.info(f"Video: FAILED to create server: {e}")
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


def _find_iq_file() -> str | None:
    """Find an IQ file in the data directory."""
    try:
        for f in DATA_DIR.iterdir():
            if f.suffix in (".sigmf-data", ".iq", ".cf32", ".bin"):
                return str(f)
    except Exception as e:
        logger.info(f"Error scanning data dir: {e}")
    return None


def _find_model() -> str | None:
    """Find a model file in the models directory."""
    try:
        for f in MODELS_DIR.iterdir():
            if f.suffix == ".pth":
                return str(f)
    except Exception as e:
        logger.info(f"Error scanning models dir: {e}")
    return None
