# Inference Pipeline

## End-to-End Flow

```
Spectrogram (1024×1024)
         │
         ▼
┌─────────────────────┐
│  Blob Detection     │  ~5-10ms (CPU)
│  (Otsu + Region)    │  No GPU needed
└─────────┬───────────┘
          │ List[BBox] (~50-500 candidates)
          ▼
┌─────────────────────┐
│  Crop Extraction    │  ~1-2ms (CPU)
│  + Preprocessing    │  Letterbox to 64×64
└─────────┬───────────┘
          │ List[Tensor] (N, 1, 64, 64)
          ▼
┌─────────────────────┐
│  Batch CNN Infer.   │  ~10-20ms (GPU, batched)
│  (FP16 optional)    │  Single forward pass
└─────────┬───────────┘
          │ List[confidence]
          ▼
┌─────────────────────┐
│  Threshold + NMS    │  ~1-2ms (CPU)
│  Filter results     │
└─────────┬───────────┘
          │
          ▼
    Final Detections

Target: <50ms end-to-end
```

---

## Full Implementation

```python
import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms


@dataclass
class Detection:
    """Single detection result."""
    bbox: dict  # x_min, y_min, x_max, y_max
    confidence: float
    signal_name: str


class TwoStageDetector:
    """
    Two-stage signal detector: blob detection + crop classification.

    Position-invariant by design - classifier never sees position.
    """

    def __init__(
        self,
        classifier: torch.nn.Module,
        signal_name: str,
        device: str = 'cuda',
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        padding_pct: float = 0.15,
        crop_size: tuple = (64, 64),
    ):
        self.classifier = classifier.to(device)
        self.classifier.eval()
        self.signal_name = signal_name
        self.device = device
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.padding_pct = padding_pct
        self.crop_size = crop_size

        # Import blob detector (use existing implementation)
        from blob_detector import BlobDetector
        self.blob_detector = BlobDetector()

    @torch.inference_mode()
    def detect(self, spectrogram: np.ndarray) -> list[Detection]:
        """
        Run full detection pipeline.

        Args:
            spectrogram: (H, W) grayscale spectrogram, float32 0-1

        Returns:
            List of Detection objects
        """
        t0 = time.perf_counter()

        # --- Stage 1: Blob detection ---
        raw_bboxes = self.blob_detector.detect(spectrogram)
        t_blob = time.perf_counter()

        if not raw_bboxes:
            return []

        # --- Crop extraction with padding ---
        padded_bboxes = [
            self._add_padding(bbox, spectrogram.shape)
            for bbox in raw_bboxes
        ]

        crops = []
        for bbox in padded_bboxes:
            crop = self._extract_and_preprocess(spectrogram, bbox)
            crops.append(crop)

        batch = torch.stack(crops).to(self.device)
        t_preprocess = time.perf_counter()

        # --- Stage 2: Classification ---
        if hasattr(self.classifier, 'predict_proba'):
            confidences = self.classifier.predict_proba(batch).squeeze()
        else:
            logits = self.classifier(batch)
            confidences = torch.sigmoid(logits).squeeze()

        confidences = confidences.cpu().numpy()
        t_classify = time.perf_counter()

        # --- Post-processing: threshold + NMS ---
        detections = []
        for bbox, conf in zip(padded_bboxes, confidences):
            if conf >= self.score_threshold:
                detections.append(Detection(
                    bbox=bbox,
                    confidence=float(conf),
                    signal_name=self.signal_name,
                ))

        # Apply NMS if multiple detections
        if len(detections) > 1:
            detections = self._apply_nms(detections)

        t_end = time.perf_counter()

        # Timing stats
        self._last_timing = {
            'blob_ms': (t_blob - t0) * 1000,
            'preprocess_ms': (t_preprocess - t_blob) * 1000,
            'classify_ms': (t_classify - t_preprocess) * 1000,
            'postprocess_ms': (t_end - t_classify) * 1000,
            'total_ms': (t_end - t0) * 1000,
            'num_candidates': len(raw_bboxes),
            'num_detections': len(detections),
        }

        return detections

    def _add_padding(self, bbox: dict, image_shape: tuple) -> dict:
        """Add padding around bounding box."""
        h, w = image_shape[:2]

        box_w = bbox['x_max'] - bbox['x_min']
        box_h = bbox['y_max'] - bbox['y_min']

        pad_x = int(box_w * self.padding_pct)
        pad_y = int(box_h * self.padding_pct)

        return {
            'x_min': max(0, bbox['x_min'] - pad_x),
            'y_min': max(0, bbox['y_min'] - pad_y),
            'x_max': min(w, bbox['x_max'] + pad_x),
            'y_max': min(h, bbox['y_max'] + pad_y),
        }

    def _extract_and_preprocess(
        self,
        image: np.ndarray,
        bbox: dict,
    ) -> torch.Tensor:
        """Extract crop and preprocess for classification."""
        import cv2

        # Extract
        x1, y1 = bbox['x_min'], bbox['y_min']
        x2, y2 = bbox['x_max'], bbox['y_max']
        crop = image[y1:y2, x1:x2].copy()

        if crop.size == 0:
            return torch.zeros(1, *self.crop_size, dtype=torch.float32)

        # Letterbox resize
        target_h, target_w = self.crop_size
        crop_h, crop_w = crop.shape[:2]

        scale = min(target_h / crop_h, target_w / crop_w)
        new_h, new_w = int(crop_h * scale), int(crop_w * scale)

        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        result = np.zeros((target_h, target_w), dtype=np.float32)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        result[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # Normalize
        mean, std = result.mean(), result.std()
        if std > 1e-8:
            result = (result - mean) / std

        return torch.from_numpy(result).unsqueeze(0)

    def _apply_nms(self, detections: list[Detection]) -> list[Detection]:
        """Apply Non-Maximum Suppression."""
        boxes = torch.tensor([
            [d.bbox['x_min'], d.bbox['y_min'], d.bbox['x_max'], d.bbox['y_max']]
            for d in detections
        ], dtype=torch.float32)

        scores = torch.tensor([d.confidence for d in detections])

        keep_indices = nms(boxes, scores, self.nms_threshold)

        return [detections[i] for i in keep_indices]

    def get_timing_stats(self) -> dict:
        """Get timing from last inference."""
        return getattr(self, '_last_timing', {})
```

---

## Dynamic Batch Processing

For async APIs, collect crops and batch them for efficiency:

```python
class AsyncBatchProcessor:
    """
    Collects inference requests and processes in batches.

    Maximizes GPU utilization by batching across requests.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        max_batch_size: int = 32,
        max_wait_ms: float = 15.0,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self._queue = asyncio.Queue()
        self._worker_task = None

    async def start(self):
        """Start the batch processing worker."""
        self._worker_task = asyncio.create_task(self._batch_worker())

    async def stop(self):
        """Stop the worker."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def predict(self, crop: torch.Tensor) -> float:
        """
        Submit single crop for prediction.

        Returns confidence score when batch is processed.
        """
        future = asyncio.Future()
        await self._queue.put((crop, future))
        return await future

    async def predict_batch(self, crops: torch.Tensor) -> list[float]:
        """Submit batch and wait for results."""
        futures = []
        for crop in crops:
            future = asyncio.Future()
            await self._queue.put((crop, future))
            futures.append(future)

        return [await f for f in futures]

    async def _batch_worker(self):
        """Worker that collects and processes batches."""
        while True:
            batch = []
            futures = []

            # Wait for first item
            try:
                crop, future = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self.max_wait_ms / 1000,
                )
                batch.append(crop)
                futures.append(future)
            except asyncio.TimeoutError:
                continue

            # Collect more items (non-blocking)
            deadline = time.time() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch_size and time.time() < deadline:
                try:
                    crop, future = self._queue.get_nowait()
                    batch.append(crop)
                    futures.append(future)
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)

            # Process batch
            if batch:
                results = await asyncio.to_thread(
                    self._infer_batch, torch.stack(batch)
                )

                for future, result in zip(futures, results):
                    future.set_result(float(result))

    @torch.inference_mode()
    def _infer_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Run inference on batch."""
        batch = batch.to(self.device)

        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(batch)
        else:
            logits = self.model(batch)
            probs = torch.sigmoid(logits)

        return probs.cpu().numpy().squeeze()
```

---

## FastAPI Backend

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np

app = FastAPI(title="RF Signal Detector")


class DetectionResponse(BaseModel):
    bbox: dict
    confidence: float
    signal_name: str


class DetectResponse(BaseModel):
    detections: list[DetectionResponse]
    timing_ms: float


@app.on_event("startup")
async def startup():
    # Load model
    from models.classifier import SignalClassifier

    model = SignalClassifier()
    model.load_state_dict(torch.load("models/classifier.pth"))

    app.state.detector = TwoStageDetector(
        classifier=model,
        signal_name="target_signal",
    )


@app.post("/detect", response_model=DetectResponse)
async def detect_signals(
    image: UploadFile = File(...),
    threshold: float = 0.5,
):
    """
    Detect signals in spectrogram.

    Args:
        image: Grayscale spectrogram image (PNG/JPG)
        threshold: Confidence threshold (0-1)

    Returns:
        List of detections with bounding boxes and confidence
    """
    # Read image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise HTTPException(400, "Invalid image")

    # Normalize to 0-1
    img = img.astype(np.float32) / 255.0

    # Update threshold
    app.state.detector.score_threshold = threshold

    # Detect
    detections = app.state.detector.detect(img)
    timing = app.state.detector.get_timing_stats()

    return DetectResponse(
        detections=[
            DetectionResponse(
                bbox=d.bbox,
                confidence=d.confidence,
                signal_name=d.signal_name,
            )
            for d in detections
        ],
        timing_ms=timing.get('total_ms', 0),
    )
```

---

## Optimization: TensorRT / ONNX

For production deployment, export to ONNX and optimize with TensorRT:

```python
def export_to_onnx(model: torch.nn.Module, output_path: str):
    """Export model to ONNX format."""
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 1, 64, 64)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'},
        },
        opset_version=17,
    )

    print(f"Exported to {output_path}")


# TensorRT optimization (requires tensorrt package)
def optimize_with_tensorrt(onnx_path: str, trt_path: str):
    """Optimize ONNX model with TensorRT."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    engine = builder.build_serialized_network(network, config)

    with open(trt_path, 'wb') as f:
        f.write(engine)

    print(f"TensorRT engine saved to {trt_path}")
```

---

## Performance Targets

| Component | Target | Actual (estimate) |
|-----------|--------|-------------------|
| Blob detection | <15ms | 5-10ms |
| Crop preprocessing | <5ms | 1-2ms |
| CNN classification | <25ms | 10-20ms |
| Post-processing | <5ms | 1-2ms |
| **Total** | **<50ms** | **~30ms** |

With TensorRT FP16 optimization, classification can drop to <5ms for a total of <20ms.
