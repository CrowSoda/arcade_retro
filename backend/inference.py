"""
G20 Demo - TensorRT Inference Engine
Auto-fallback: TensorRT → ONNX → PyTorch

Optimized for:
- Development: RTX 4090 (PyTorch FP16)
- Production: Jetson Orin NX (TensorRT FP16)
"""

import os
import time
import logging
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("g20.inference")

# Check TensorRT availability
TRT_AVAILABLE = False
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    logger.info(f"TensorRT {trt.__version__} available")
except ImportError:
    logger.warning("TensorRT not available, will use PyTorch")

# Check ONNX availability
ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info(f"ONNX Runtime {ort.__version__} available")
except ImportError:
    pass


class InferenceEngine:
    """
    TensorRT-accelerated inference with automatic fallback.
    
    Priority: TensorRT → ONNX → PyTorch
    """
    
    def __init__(
        self,
        model_path: str,
        num_classes: int = 2,  # background + 1 signal class (e.g. creamy_chicken)
        device: str = "cuda",
        precision: str = "fp16"
    ):
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.precision = precision
        
        self._backend = None
        self._engine = None
        self._context = None
        self._ort_session = None
        self._pytorch_model = None
        
        self._load_engine()
        
        # Timing stats
        self._inference_times = []
    
    def _load_engine(self):
        """Load the appropriate inference backend."""
        ext = Path(self.model_path).suffix.lower()
        
        if ext == ".trt" and TRT_AVAILABLE:
            self._load_tensorrt()
        elif ext == ".onnx" and ONNX_AVAILABLE:
            self._load_onnx()
        elif ext == ".pth":
            self._load_pytorch()
        else:
            # Try finding converted versions
            base = self.model_path.rsplit(".", 1)[0]
            if TRT_AVAILABLE and os.path.exists(f"{base}.trt"):
                self.model_path = f"{base}.trt"
                self._load_tensorrt()
            elif ONNX_AVAILABLE and os.path.exists(f"{base}.onnx"):
                self.model_path = f"{base}.onnx"
                self._load_onnx()
            else:
                self._load_pytorch()
    
    def _load_tensorrt(self):
        """Load TensorRT engine."""
        logger.info(f"Loading TensorRT engine: {self.model_path}")
        
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            self._engine = runtime.deserialize_cuda_engine(f.read())
        
        self._context = self._engine.create_execution_context()
        self._backend = "tensorrt"
        self._setup_trt_buffers()
        logger.info("TensorRT engine loaded")
    
    def _setup_trt_buffers(self):
        """Pre-allocate GPU buffers for TensorRT."""
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        self._bindings = []
        self._inputs = []
        self._outputs = []
        
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = list(self._engine.get_tensor_shape(name))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            
            # Handle dynamic shapes
            shape = [s if s > 0 else 8 for s in shape]
            size = int(np.prod(shape))
            
            mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self._bindings.append(int(mem))
            
            info = {"name": name, "shape": shape, "dtype": dtype, "mem": mem}
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._inputs.append(info)
            else:
                self._outputs.append(info)
    
    def _load_onnx(self):
        """Load ONNX Runtime session."""
        logger.info(f"Loading ONNX model: {self.model_path}")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self._ort_session = ort.InferenceSession(self.model_path, opts, providers=providers)
        self._backend = "onnx"
        logger.info(f"ONNX loaded: {self._ort_session.get_providers()}")
    
    def _load_pytorch(self):
        """Load PyTorch model (fallback)."""
        logger.info(f"Loading PyTorch model: {self.model_path}")
        
        import torchvision
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        
        # ResNet18 backbone with 5 trainable layers (matches tensorcade training)
        backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=5)
        self._pytorch_model = torchvision.models.detection.FasterRCNN(
            backbone, num_classes=self.num_classes
        )
        
        state = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self._pytorch_model.load_state_dict(state)
        self._pytorch_model.to(self.device)
        self._pytorch_model.eval()
        
        if self.precision == "fp16" and self.device.type == "cuda":
            self._pytorch_model.half()
        
        self._backend = "pytorch"
        logger.info("PyTorch model loaded")
    
    @property
    def backend(self) -> str:
        return self._backend
    
    def infer(
        self,
        images: torch.Tensor,
        score_threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Run inference on batch of images.
        
        Args:
            images: Tensor [B, 3, H, W] in range [0, 1]
            score_threshold: Minimum confidence
        
        Returns:
            List of dicts with 'boxes', 'scores', 'labels'
        """
        start = time.perf_counter()
        
        if self._backend == "tensorrt":
            results = self._infer_trt(images, score_threshold)
        elif self._backend == "onnx":
            results = self._infer_onnx(images, score_threshold)
        else:
            results = self._infer_pytorch(images, score_threshold)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._inference_times.append(elapsed_ms)
        
        return results
    
    def _infer_trt(self, images: torch.Tensor, score_threshold: float) -> List[Dict]:
        """TensorRT inference."""
        import pycuda.driver as cuda
        
        input_np = images.cpu().numpy().astype(np.float32)
        cuda.memcpy_htod(self._inputs[0]["mem"], input_np)
        
        self._context.execute_v2(self._bindings)
        
        outputs = []
        for out in self._outputs:
            arr = np.empty(out["shape"], dtype=out["dtype"])
            cuda.memcpy_dtoh(arr, out["mem"])
            outputs.append(arr)
        
        return self._parse_detections(outputs, images.shape[0], score_threshold)
    
    def _infer_onnx(self, images: torch.Tensor, score_threshold: float) -> List[Dict]:
        """ONNX Runtime inference."""
        input_name = self._ort_session.get_inputs()[0].name
        input_np = images.cpu().numpy().astype(np.float32)
        
        outputs = self._ort_session.run(None, {input_name: input_np})
        return self._parse_detections(outputs, images.shape[0], score_threshold)
    
    def _infer_pytorch(self, images: torch.Tensor, score_threshold: float) -> List[Dict]:
        """PyTorch inference."""
        images = images.to(self.device)
        
        if self.precision == "fp16" and self.device.type == "cuda":
            images = images.half()
        
        with torch.inference_mode():
            outputs = self._pytorch_model(images)
        
        results = []
        for out in outputs:
            mask = out["scores"] >= score_threshold
            results.append({
                "boxes": out["boxes"][mask].cpu().numpy(),
                "scores": out["scores"][mask].cpu().numpy(),
                "labels": out["labels"][mask].cpu().numpy()
            })
        
        return results
    
    def _parse_detections(self, outputs: List[np.ndarray], batch_size: int, threshold: float) -> List[Dict]:
        """Parse raw outputs to detection format."""
        results = []
        
        if len(outputs) >= 3:
            boxes, labels, scores = outputs[0], outputs[1], outputs[2]
            mask = scores >= threshold
            results.append({
                "boxes": boxes[mask],
                "scores": scores[mask],
                "labels": labels[mask]
            })
        
        return results
    
    @property
    def avg_inference_ms(self) -> float:
        """Average inference time in milliseconds."""
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times[-100:]) / min(len(self._inference_times), 100)


class MultiModelEngine:
    """
    Parallel inference across N models using CUDA streams.
    
    Usage:
        engine = MultiModelEngine([
            ("model_a.pth", 6),
            ("model_b.pth", 4),
        ])
        results = engine.infer_parallel(spectrogram_batch)
    """
    
    def __init__(
        self,
        model_configs: List[Tuple[str, int]],
        device: str = "cuda",
        precision: str = "fp16"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.precision = precision
        self.num_models = len(model_configs)
        
        # Load all engines
        self.engines: List[InferenceEngine] = []
        for path, num_classes in model_configs:
            engine = InferenceEngine(path, num_classes, device, precision)
            self.engines.append(engine)
            logger.info(f"Loaded: {path} (backend={engine.backend})")
        
        # CUDA streams for parallel execution
        if self.device.type == "cuda":
            self.streams = [torch.cuda.Stream() for _ in range(self.num_models)]
        else:
            self.streams = [None] * self.num_models
        
        logger.info(f"MultiModelEngine: {self.num_models} models loaded")
    
    def infer_parallel(
        self,
        images: torch.Tensor,
        score_threshold: float = 0.9
    ) -> List[List[Dict[str, Any]]]:
        """
        Run all models in parallel.
        
        Returns:
            List of results, one per model
        """
        results = [None] * self.num_models
        
        if self.device.type == "cuda":
            for i, (engine, stream) in enumerate(zip(self.engines, self.streams)):
                with torch.cuda.stream(stream):
                    results[i] = engine.infer(images, score_threshold)
            torch.cuda.synchronize()
        else:
            for i, engine in enumerate(self.engines):
                results[i] = engine.infer(images, score_threshold)
        
        return results
    
    def infer_merged(
        self,
        images: torch.Tensor,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Run all models and merge with NMS."""
        from torchvision.ops import nms
        
        all_results = self.infer_parallel(images, score_threshold)
        
        merged = []
        for batch_idx in range(images.shape[0]):
            boxes_list, scores_list, labels_list, model_ids = [], [], [], []
            
            for model_idx, model_results in enumerate(all_results):
                if batch_idx < len(model_results):
                    det = model_results[batch_idx]
                    n = len(det.get("scores", []))
                    if n > 0:
                        boxes_list.extend(det["boxes"])
                        scores_list.extend(det["scores"])
                        labels_list.extend(det["labels"])
                        model_ids.extend([model_idx] * n)
            
            if boxes_list:
                boxes_t = torch.tensor(boxes_list, dtype=torch.float32)
                scores_t = torch.tensor(scores_list, dtype=torch.float32)
                keep = nms(boxes_t, scores_t, nms_threshold).numpy()
                
                merged.append({
                    "boxes": np.array(boxes_list)[keep],
                    "scores": np.array(scores_list)[keep],
                    "labels": np.array(labels_list)[keep],
                    "model_ids": np.array(model_ids)[keep],
                })
            else:
                merged.append({"boxes": np.array([]), "scores": np.array([]), "labels": np.array([]), "model_ids": np.array([])})
        
        return merged
    
    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark parallel vs sequential."""
        dummy = torch.randn(1, 3, 512, 512, device=self.device)
        if self.precision == "fp16" and self.device.type == "cuda":
            dummy = dummy.half()
        
        # Warmup
        for _ in range(10):
            self.infer_parallel(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Parallel
        start = time.time()
        for _ in range(num_iterations):
            self.infer_parallel(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        parallel_ms = (time.time() - start) / num_iterations * 1000
        
        # Sequential
        start = time.time()
        for _ in range(num_iterations):
            for engine in self.engines:
                engine.infer(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        sequential_ms = (time.time() - start) / num_iterations * 1000
        
        return {
            "parallel_ms": parallel_ms,
            "sequential_ms": sequential_ms,
            "speedup": sequential_ms / parallel_ms,
            "num_models": self.num_models,
        }


class SpectrogramPipeline:
    """
    GPU-accelerated B&W spectrogram for ResNet detection.
    Matches tensorcade: NFFT=4096, noverlap=2048, 80dB, 1024x1024 output.
    """
    
    def __init__(
        self,
        nfft: int = 4096,
        noverlap: int = 2048,
        out_size: int = 1024,
        dynamic_range_db: float = 80.0,
        device: str = "cuda"
    ):
        self.nfft = nfft
        self.hop_length = nfft - noverlap  # hop = nfft - noverlap
        self.noverlap = noverlap
        self.out_size = out_size
        self.dynamic_range_db = dynamic_range_db
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.window = torch.hann_window(nfft, device=self.device)
        
        logger.info(f"SpectrogramPipeline: nfft={nfft}, noverlap={noverlap}, "
                   f"hop={self.hop_length}, dynrange={dynamic_range_db}dB, "
                   f"output={out_size}x{out_size}, BLACK AND WHITE")
    
    def process(self, iq_data: np.ndarray) -> torch.Tensor:
        """
        Convert IQ to BLACK AND WHITE spectrogram tensor.
        
        Pipeline: IQ → STFT → dB → normalize → grayscale image → expand to 3ch
        
        Args:
            iq_data: Complex64 numpy array
        
        Returns:
            Tensor [1, 3, out_size, out_size] - Grayscale expanded to 3 channels
        """
        # Convert to torch complex
        if iq_data.dtype == np.complex128:
            iq_data = iq_data.astype(np.complex64)
        chunk = torch.from_numpy(iq_data).to(self.device)
        
        # Compute STFT
        Zxx = torch.stft(
            chunk, 
            n_fft=self.nfft, 
            hop_length=self.hop_length,
            win_length=self.nfft, 
            window=self.window,
            center=False, 
            return_complex=True
        )
        
        # FFT shift to put DC in center
        Zxx = torch.fft.fftshift(Zxx, dim=0)
        
        # Convert to power in dB
        power = Zxx.abs().square()
        sxx_db = 10 * torch.log10(power + 1e-12)
        
        # Normalize with dynamic range (0=black/weak, 1=white/strong)
        vmax = sxx_db.max()
        vmin = vmax - self.dynamic_range_db
        sxx_norm = ((sxx_db - vmin) / (vmax - vmin + 1e-12)).clamp_(0, 1)
        
        # Flip frequency axis (high freq at top)
        sxx_norm = torch.flip(sxx_norm, dims=[0])
        
        # Resize to model input size [1, 1, out_size, out_size]
        sxx_norm = sxx_norm.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            sxx_norm, 
            size=(self.out_size, self.out_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Expand grayscale to 3 channels (same B&W image × 3)
        # This is how it was trained - NOT a colormap!
        return resized.expand(-1, 3, -1, -1)
    
    def process_batch(self, iq_chunks: List[np.ndarray]) -> torch.Tensor:
        """Process multiple IQ chunks into a batch."""
        batch = [self.process(chunk) for chunk in iq_chunks]
        return torch.cat(batch, dim=0)


def export_to_tensorrt(
    pytorch_path: str,
    output_path: str,
    num_classes: int = 6,
    input_size: Tuple[int, int] = (512, 512),
    fp16: bool = True
) -> str:
    """
    Export PyTorch model to TensorRT.
    
    Returns path to .trt file.
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT not available")
    
    onnx_path = output_path.replace(".trt", ".onnx")
    
    # Step 1: PyTorch → ONNX
    logger.info(f"Exporting to ONNX: {onnx_path}")
    
    import torchvision
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    
    backbone = resnet_fpn_backbone("resnet18", weights=None, trainable_layers=3)
    model = torchvision.models.detection.FasterRCNN(backbone, num_classes=num_classes)
    model.load_state_dict(torch.load(pytorch_path, map_location="cpu", weights_only=False))
    model.eval()
    
    dummy = torch.randn(1, 3, input_size[0], input_size[1])
    torch.onnx.export(
        model, dummy, onnx_path, opset_version=17,
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={
            "images": {0: "batch"},
            "boxes": {0: "dets"},
            "labels": {0: "dets"},
            "scores": {0: "dets"}
        }
    )
    
    # Step 2: ONNX → TensorRT
    logger.info(f"Converting to TensorRT: {output_path}")
    
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"Parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 enabled")
    
    logger.info("Building TensorRT engine (may take several minutes)...")
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        raise RuntimeError("TRT build failed")
    
    with open(output_path, "wb") as f:
        f.write(engine)
    
    logger.info(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path (.pth)")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--export-trt", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-multi", type=int, help="Benchmark N models")
    args = parser.parse_args()
    
    if args.export_trt:
        out = args.model.replace(".pth", ".trt")
        export_to_tensorrt(args.model, out, args.num_classes)
        print(f"Exported: {out}")
    
    if args.benchmark:
        engine = InferenceEngine(args.model, args.num_classes)
        print(f"Backend: {engine.backend}")
        
        dummy = torch.randn(1, 3, 512, 512)
        if engine.device.type == "cuda":
            dummy = dummy.cuda()
        
        for _ in range(10):
            engine.infer(dummy)
        
        times = []
        for _ in range(100):
            start = time.perf_counter()
            engine.infer(dummy)
            times.append((time.perf_counter() - start) * 1000)
        
        print(f"Avg: {sum(times)/len(times):.2f} ms")
    
    if args.benchmark_multi:
        configs = [(args.model, args.num_classes)] * args.benchmark_multi
        multi = MultiModelEngine(configs)
        results = multi.benchmark()
        
        print(f"\n{args.benchmark_multi} Models:")
        print(f"  Sequential: {results['sequential_ms']:.2f} ms")
        print(f"  Parallel:   {results['parallel_ms']:.2f} ms")
        print(f"  Speedup:    {results['speedup']:.2f}x")
