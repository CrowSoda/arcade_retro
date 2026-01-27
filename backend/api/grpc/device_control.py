"""
DeviceControlServicer - Simulates NV100 SDR hardware control.

Extracted from server.py using strangler fig pattern.
"""

import logging
import os

# Import generated proto stubs
import sys
import time
import uuid
from pathlib import Path

from core.models import CaptureSession, ChannelState

GENERATED_DIR = Path(__file__).parent.parent.parent / "generated"
sys.path.insert(0, str(GENERATED_DIR))

try:
    import control_pb2
    import control_pb2_grpc

    PROTO_AVAILABLE = True
except ImportError:
    PROTO_AVAILABLE = False
    control_pb2 = None
    control_pb2_grpc = None

logger = logging.getLogger("g20.server")

# Directories
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"


class DeviceControlServicer:
    """Simulates NV100 SDR hardware control via gRPC."""

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


# Make servicer available if protos are built
if PROTO_AVAILABLE:
    # Inherit from the generated base class
    DeviceControlServicer.__bases__ = (control_pb2_grpc.DeviceControlServicer,)
