"""
Training WebSocket handler for Hydra training and version management.

Extracted from server.py using strangler fig pattern.
This is the largest handler (~400 lines).

Commands:
    - get_registry: Get all signals and versions
    - get_version_history: Get version history for a signal
    - train_signal: Train new or extend existing signal
    - cancel_training: Cancel running training
    - promote_version: Promote a version
    - rollback_signal: Rollback to previous version
    - save_sample: Save training sample (IQ + boxes)
    - get_samples: List samples for a signal
    - extract_subband: Extract sub-band from RFCAP file
"""

import asyncio
import json
import logging
import os
import time
import traceback
from pathlib import Path

logger = logging.getLogger("g20.server")

# Directory paths
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


async def ws_training_handler(websocket):
    """WebSocket handler for Hydra training and version management."""
    logger.info("Training: Handler started")

    # Wrap EVERYTHING in try/except to ensure we send error before closing
    try:
        await _ws_training_handler_impl(websocket)
    except Exception as e:
        logger.info(f"Training: CRITICAL ERROR in handler: {e}")
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
            await asyncio.sleep(0.5)
        except Exception as send_err:
            logger.info(f"Training: Failed to send error to client: {send_err}")


async def _ws_training_handler_impl(websocket):
    """Actual implementation of training handler."""
    logger.info("Training: _ws_training_handler_impl ENTRY")

    # Initialize services
    training_service = None
    version_manager = None
    sample_manager = None
    split_manager = None
    init_errors = []

    # Try importing training modules
    try:
        from training.sample_manager import SampleManager

        sample_manager = SampleManager(str(BASE_DIR / "training_data" / "signals"))
        logger.info(f"Training: SampleManager initialized: {sample_manager.base_dir}")
    except Exception as e:
        error_msg = f"SampleManager failed: {e}\n{traceback.format_exc()}"
        init_errors.append(error_msg)
        logger.info(f"Training: {error_msg}")

    # Try importing Hydra modules
    try:
        from hydra.version_manager import VersionManager
        from training.service import TrainingService
        from training.splits import SplitManager

        version_manager = VersionManager(str(MODELS_DIR))
        split_manager = SplitManager(str(BASE_DIR / "training_data" / "signals"))
        training_service = TrainingService(
            models_dir=str(MODELS_DIR),
            training_data_dir=str(BASE_DIR / "training_data" / "signals"),
        )
        logger.info("Training: Full training services initialized")
    except Exception as e:
        error_msg = f"Training services failed: {e}\n{traceback.format_exc()}"
        init_errors.append(error_msg)
        logger.info(f"Training: {error_msg}")

    # Send init errors if any
    if init_errors:
        full_error = "INITIALIZATION ERRORS:\n" + "\n---\n".join(init_errors)
        logger.info("Training: Sending init errors to client")
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

        async def send_progress(progress):
            try:
                await ws.send(
                    json.dumps(
                        {
                            "type": "training_progress",
                            "signal_name": training_service._current_signal
                            if training_service
                            else None,
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
                logger.info(f"Training: Progress send error: {e}")

        def callback(progress):
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
                if cmd not in ("save_sample",):
                    logger.info(f"Training: Command: {cmd}")

                # =====================
                # Registry & Versions
                # =====================

                if cmd == "get_registry":
                    if version_manager:
                        registry = version_manager.get_registry()
                        await websocket.send(json.dumps({"type": "registry", **registry}))
                    else:
                        await websocket.send(
                            json.dumps(
                                {"type": "error", "message": "version_manager not initialized"}
                            )
                        )

                elif cmd == "get_version_history":
                    signal_name = data.get("signal_name")
                    if not signal_name:
                        await websocket.send(
                            json.dumps({"type": "error", "message": "signal_name required"})
                        )
                        continue
                    if version_manager:
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
                    else:
                        await websocket.send(
                            json.dumps(
                                {"type": "error", "message": "version_manager not initialized"}
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
                        if version_manager:
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
                        else:
                            await websocket.send(
                                json.dumps(
                                    {"type": "error", "message": "version_manager not initialized"}
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
                        if version_manager:
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
                        else:
                            await websocket.send(
                                json.dumps(
                                    {"type": "error", "message": "version_manager not initialized"}
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
                                {"type": "error", "message": "training_service not initialized"}
                            )
                        )
                        continue

                    if training_service.is_training:
                        await websocket.send(
                            json.dumps({"type": "error", "message": "Training already in progress"})
                        )
                        continue

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
                            logger.info(f"Training: Error: {e}")
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
                                    "error": "not initialized",
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
                                {"type": "error", "message": "sample_manager not initialized"}
                            )
                        )
                        continue

                    try:
                        sample_id, is_new = sample_manager.save_sample(
                            signal_name, iq_data_b64, boxes, metadata
                        )
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "sample_saved",
                                    "signal_name": signal_name,
                                    "sample_id": sample_id,
                                    "is_new": is_new,
                                    "total_samples": sample_manager.get_sample_count(signal_name),
                                }
                            )
                        )
                    except Exception as e:
                        logger.info(f"Training: ERROR saving sample: {e}")
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

                    if sample_manager and split_manager:
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
                    else:
                        await websocket.send(
                            json.dumps({"type": "error", "message": "managers not initialized"})
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
                    if sample_manager:
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
                    else:
                        await websocket.send(
                            json.dumps(
                                {"type": "error", "message": "sample_manager not initialized"}
                            )
                        )

                # =====================
                # Sub-Band Extraction
                # =====================

                elif cmd == "extract_subband":
                    await _handle_extract_subband(websocket, data)

                else:
                    await websocket.send(
                        json.dumps({"type": "error", "message": f"Unknown command: {cmd}"})
                    )

            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
            except Exception as e:
                logger.info(f"Training: Command error: {e}")
                traceback.print_exc()
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))

    except Exception as e:
        logger.info(f"Training: Handler error: {e}")
        traceback.print_exc()
    finally:
        if training_task and not training_task.done():
            if training_service:
                training_service.cancel_training()
            training_task.cancel()
        logger.info("Training: Handler closed")


async def _handle_extract_subband(websocket, data: dict):
    """Handle extract_subband command separately to keep main handler cleaner."""
    source_file = data.get("source_file")
    output_file = data.get("output_file")
    center_offset_hz = data.get("center_offset_hz", 0)
    bandwidth_hz = data.get("bandwidth_hz")

    if not source_file or not bandwidth_hz:
        await websocket.send(
            json.dumps({"type": "error", "message": "source_file and bandwidth_hz required"})
        )
        return

    # Resolve paths
    source_path = str(BASE_DIR / source_file) if not os.path.isabs(source_file) else source_file
    output_path = (
        str(BASE_DIR / output_file)
        if output_file and not os.path.isabs(output_file)
        else output_file
    )

    if not output_path:
        output_dir = BASE_DIR / "training_data" / "extracted"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"subband_{timestamp}.rfcap")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(source_path):
        await websocket.send(
            json.dumps({"type": "error", "message": f"Source file not found: {source_path}"})
        )
        return

    try:
        from dsp.simple_extract import extract_subband as simple_extract_subband
        from dsp.subband_extractor import _read_rfcap_header

        source_header = _read_rfcap_header(source_path)
        original_center_hz = source_header["center_freq"]
        original_sample_rate = source_header["sample_rate"]
        new_center_hz = original_center_hz + center_offset_hz

        async def send_progress(progress):
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
            except Exception:
                pass

        def progress_callback(p):
            try:
                asyncio.get_event_loop().create_task(send_progress(p))
            except RuntimeError:
                pass

        result = await asyncio.to_thread(
            simple_extract_subband,
            input_path=source_path,
            output_path=output_path,
            original_center_hz=original_center_hz,
            original_sample_rate=original_sample_rate,
            new_center_hz=new_center_hz,
            new_bandwidth_hz=bandwidth_hz,
            num_taps=101,
            progress_callback=progress_callback,
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
                }
            )
        )
    except Exception as e:
        logger.info(f"Training: Extraction error: {e}")
        traceback.print_exc()
        await websocket.send(
            json.dumps({"type": "error", "command": "extract_subband", "message": str(e)})
        )
