#!/usr/bin/env python3
"""
TENSORCADE Model Wrapper for G20 Demo System

This wrapper loads your trained Faster R-CNN model and replicates TENSORCADE's
exact preprocessing pipeline. Use this to:
1. Verify the model loads correctly
2. Generate baseline detections for test comparison
3. Run inference in the G20 pipeline

TENSORCADE Preprocessing (from workers.py):
1. torch.stft with hann window
2. fftshift
3. 10*log10(|X|² + 1e-12) for PSD
4. Normalize: (psd - vmin) / (vmax - vmin), vmin = vmax - 80dB
5. Resize to 1024x1024 with bilinear interpolation
6. Flip vertically (torch.flip dim=2)
7. Expand to 3 channels (grayscale -> RGB)
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


@dataclass
class TensorcadeConfig:
    """Configuration matching TENSORCADE app_settings."""

    nfft: int = 4096
    noverlap: int = 2048  # 50% overlap
    dynamic_range: float = 80.0
    out_size: int = 1024
    backbone: str = "resnet18"
    num_classes: int = 2
    score_threshold: float = 0.5


@dataclass
class Detection:
    """Detection output format compatible with G20 tracker."""

    frame_id: int
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized 0-1
    box_pixels: tuple[float, float, float, float]  # Original pixel coords


class TensorcadeWrapper:
    """
    Wraps your trained TENSORCADE Faster R-CNN model.
    Replicates exact preprocessing from workers.py.
    """

    def __init__(
        self,
        model_path: str,
        config: TensorcadeConfig | None = None,
        device: str | None = None,
    ):
        self.config = config or TensorcadeConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Build model architecture
        self.model = self._build_model()

        # Load trained weights
        self._load_weights(model_path)

        # Pre-allocate window (stays on GPU)
        self._window = torch.hann_window(self.config.nfft, device=self.device)
        self._hop = self.config.nfft - self.config.noverlap

        print(f"[TensorcadeWrapper] Loaded model from {model_path}")
        print(f"[TensorcadeWrapper] Device: {self.device}")
        print(
            f"[TensorcadeWrapper] Config: nfft={self.config.nfft}, "
            f"dynamic_range={self.config.dynamic_range}dB, "
            f"out_size={self.config.out_size}"
        )

    def _build_model(self) -> torch.nn.Module:
        """Build Faster R-CNN matching TENSORCADE architecture."""
        if self.config.backbone == "resnet50":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config.num_classes)
        else:
            # ResNet18 backbone (TENSORCADE default)
            backbone = resnet_fpn_backbone(
                "resnet18",
                weights=None,
                trainable_layers=5,
            )
            model = torchvision.models.detection.FasterRCNN(
                backbone, num_classes=self.config.num_classes
            )

        return model.to(self.device)

    def _load_weights(self, model_path: str):
        """Load weights handling different checkpoint formats."""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different save formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Assume it's a raw state dict
                state_dict = checkpoint
        else:
            # Full model was saved
            self.model = checkpoint.to(self.device)
            self.model.eval()
            return

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess_iq(self, iq_samples: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess IQ samples exactly as TENSORCADE does.

        Args:
            iq_samples: Complex64 IQ samples

        Returns:
            (model_input, spectrogram_for_display)
            - model_input: Tensor ready for inference [1, 3, 1024, 1024]
            - spectrogram_for_display: numpy array [1024, 1024] float32 0-1
        """
        # Move to GPU
        chunk_gpu = torch.from_numpy(iq_samples).to(self.device)

        # 1. STFT (matches TENSORCADE workers.py lines 1388-1396)
        Zxx = torch.stft(
            chunk_gpu,
            n_fft=self.config.nfft,
            hop_length=self._hop,
            win_length=self.config.nfft,
            window=self._window,
            center=False,
            return_complex=True,
        )

        # 2. FFT shift + PSD in dB (lines 1405-1406)
        Zxx_shifted = torch.fft.fftshift(Zxx, dim=0)
        sxx_db = 10 * torch.log10(Zxx_shifted.abs().square() + 1e-12)

        # 3. Normalize (lines 1414-1416)
        vmax = sxx_db.max()
        vmin = vmax - self.config.dynamic_range
        sxx_norm = ((sxx_db - vmin) / (vmax - vmin + 1e-12)).clamp_(0, 1)

        # 4. Resize to out_size x out_size (lines 1425-1431)
        sxx_norm = sxx_norm.unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]
        resized = F.interpolate(
            sxx_norm,
            size=(self.config.out_size, self.config.out_size),
            mode="bilinear",
            align_corners=False,
        )

        # 5. Flip vertically (line 1435)
        resized_flipped = torch.flip(resized, dims=[2])

        # 6. Expand to 3 channels (line 1438)
        img_3ch = resized_flipped.expand(-1, 3, -1, -1)

        # Get display version
        display = resized_flipped[0, 0].cpu().numpy()

        return img_3ch, display

    @torch.no_grad()
    def detect(
        self,
        iq_samples: np.ndarray,
        frame_id: int = 0,
    ) -> tuple[list[Detection], np.ndarray]:
        """
        Run detection on IQ samples.

        Args:
            iq_samples: Complex64 IQ samples
            frame_id: Frame identifier for tracking

        Returns:
            (detections, spectrogram_display)
        """
        # Preprocess
        model_input, display = self.preprocess_iq(iq_samples)

        # Inference
        outputs = self.model(model_input)[0]

        # Extract results
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        # Filter by score and convert to Detection objects
        detections = []
        for box, score, label in zip(boxes, scores, labels, strict=False):
            if score < self.config.score_threshold:
                continue

            x1, y1, x2, y2 = box

            detections.append(
                Detection(
                    frame_id=frame_id,
                    class_id=int(label),
                    class_name="signal" if label == 1 else "background",
                    confidence=float(score),
                    # Normalized coordinates (0-1)
                    box=(
                        x1 / self.config.out_size,
                        y1 / self.config.out_size,
                        x2 / self.config.out_size,
                        y2 / self.config.out_size,
                    ),
                    # Original pixel coordinates
                    box_pixels=(float(x1), float(y1), float(x2), float(y2)),
                )
            )

        return detections, display

    def detect_from_spectrogram(
        self,
        spectrogram: np.ndarray,
        frame_id: int = 0,
    ) -> list[Detection]:
        """
        Run detection on a pre-computed spectrogram.

        Args:
            spectrogram: 2D numpy array [H, W] normalized 0-1
            frame_id: Frame identifier

        Returns:
            List of detections
        """
        # Resize if needed
        if spectrogram.shape != (self.config.out_size, self.config.out_size):
            import cv2

            spectrogram = cv2.resize(
                spectrogram,
                (self.config.out_size, self.config.out_size),
                interpolation=cv2.INTER_AREA,
            )

        # Convert to tensor [1, 3, H, W]
        tensor = torch.from_numpy(spectrogram).float().to(self.device)
        tensor = tensor.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)

        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)[0]

        # Convert to detections
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        detections = []
        for box, score, label in zip(boxes, scores, labels, strict=False):
            if score < self.config.score_threshold:
                continue

            x1, y1, x2, y2 = box
            detections.append(
                Detection(
                    frame_id=frame_id,
                    class_id=int(label),
                    class_name="signal" if label == 1 else "background",
                    confidence=float(score),
                    box=(
                        x1 / self.config.out_size,
                        y1 / self.config.out_size,
                        x2 / self.config.out_size,
                        y2 / self.config.out_size,
                    ),
                    box_pixels=(float(x1), float(y1), float(x2), float(y2)),
                )
            )

        return detections


def load_iq_file(path: str, chunk_samples: int, chunk_index: int = 0) -> np.ndarray:
    """Load a chunk of IQ data from a .sigmf-data file."""
    with open(path, "rb") as f:
        f.seek(chunk_index * chunk_samples * 8)  # complex64 = 8 bytes
        raw = f.read(chunk_samples * 8)
    return np.frombuffer(raw, dtype=np.complex64)


# Quick test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tensorcade_wrapper.py <model.pth> [iq_file.sigmf-data] [sample_rate]")
        print("\nThis will:")
        print("  1. Load the model and verify it works")
        print("  2. Optionally run inference on IQ data")
        sys.exit(0)

    model_path = sys.argv[1]

    print(f"\n{'='*60}")
    print("  TENSORCADE Model Compatibility Test")
    print(f"{'='*60}\n")

    # Load model
    try:
        wrapper = TensorcadeWrapper(model_path)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)

    # Test with random data
    print("Testing with random IQ data...")
    sample_rate = 20e6
    chunk_ms = 200
    chunk_samples = int(sample_rate * chunk_ms / 1000)

    fake_iq = (np.random.randn(chunk_samples) + 1j * np.random.randn(chunk_samples)).astype(
        np.complex64
    )

    try:
        detections, display = wrapper.detect(fake_iq, frame_id=0)
        print("✓ Inference works")
        print(f"  - Spectrogram shape: {display.shape}")
        print(f"  - Detections: {len(detections)}")
        if detections:
            for d in detections[:3]:
                print(f"    - {d.class_name}: conf={d.confidence:.2f}, box={d.box_pixels}")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Test with real IQ data if provided
    if len(sys.argv) >= 3:
        iq_path = sys.argv[2]
        sample_rate = float(sys.argv[3]) if len(sys.argv) >= 4 else 20e6

        print(f"\nTesting with real IQ data: {iq_path}")
        print(f"Sample rate: {sample_rate/1e6:.1f} MHz")

        chunk_samples = int(sample_rate * 200 / 1000)  # 200ms chunk

        try:
            iq_data = load_iq_file(iq_path, chunk_samples, chunk_index=0)
            detections, display = wrapper.detect(iq_data, frame_id=0)
            print("✓ Real data inference works")
            print(f"  - Detections: {len(detections)}")
            for d in detections:
                print(
                    f"    - {d.class_name}: conf={d.confidence:.3f}, "
                    f"box=({d.box_pixels[0]:.0f},{d.box_pixels[1]:.0f})-"
                    f"({d.box_pixels[2]:.0f},{d.box_pixels[3]:.0f})"
                )
        except Exception as e:
            print(f"✗ Real data test failed: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print("  Test Complete")
    print(f"{'='*60}\n")
