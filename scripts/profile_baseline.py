#!/usr/bin/env python3
"""
G20 Pre-Jetson Profiling
Run on dev machine. Re-run on Jetson later for comparison.

Only measures metrics that transfer across hardware:
- Serialization overhead
- Python/async overhead
- Memory leak detection
- Code path timing

Does NOT measure (wait for Jetson):
- GPU transfer times (architecture-dependent)
- Absolute inference times (GPU-dependent)
- Memory budget (16GB shared vs discrete)
"""

import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Measurement:
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    iterations: int


def measure(func, iterations=100, warmup=10) -> Measurement:
    name = func.__name__
    for _ in range(warmup):
        func()
    times = []
    for _ in range(iterations):
        gc.disable()
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
        gc.enable()
    t = np.array(times)
    return Measurement(
        name=name,
        mean_ms=float(np.mean(t)),
        std_ms=float(np.std(t)),
        min_ms=float(np.min(t)),
        max_ms=float(np.max(t)),
        p95_ms=float(np.percentile(t, 95)),
        p99_ms=float(np.percentile(t, 99)),
        iterations=iterations,
    )


def profile_serialization() -> list[Measurement]:
    """Serialization overhead - transfers to Jetson."""
    results = []

    # Test sizes matching your IQ frames
    sizes = [
        (4096 * 6, "24k_complex64"),  # Typical frame
        (65536 * 6, "384k_complex64"),  # Large batch
    ]

    for num_samples, label in sizes:
        data = np.random.randn(num_samples).astype(np.complex64)
        float_data = data.real.astype(np.float32)

        # Raw bytes (baseline)
        def raw_bytes():
            encoded = data.tobytes()
            return np.frombuffer(encoded, dtype=np.complex64)

        raw_bytes.__name__ = f"raw_bytes_{label}"
        results.append(measure(raw_bytes))

        # JSON (worst case, small subset)
        small = float_data[:1000].tolist()

        def json_encode():
            return json.dumps(small)

        json_encode.__name__ = "json_1k_floats"
        if label == "24k_complex64":
            results.append(measure(json_encode, iterations=50))

        # Struct pack (header simulation)
        import struct

        header_fmt = "=4sQIII"  # magic, timestamp, frame_idx, num_samples, flags

        def struct_pack():
            return struct.pack(header_fmt, b"IQ01", 123456789, 42, num_samples, 0)

        struct_pack.__name__ = "struct_header_pack"
        if label == "24k_complex64":
            results.append(measure(struct_pack))

        # Combined: header + payload
        def frame_serialize():
            header = struct.pack(header_fmt, b"IQ01", 123456789, 42, num_samples, 0)
            return header + data.tobytes()

        frame_serialize.__name__ = f"frame_serialize_{label}"
        results.append(measure(frame_serialize))

    return results


def profile_numpy_ops() -> list[Measurement]:
    """NumPy operations - transfers to Jetson."""
    results = []

    nfft = 4096
    num_frames = 6
    # Complex data for complex FFT
    complex_data = np.random.randn(nfft * num_frames).astype(np.complex64)
    complex_frames = complex_data.reshape(num_frames, nfft)
    # Real data for rfft (rfft requires real input)
    real_data = np.random.randn(nfft * num_frames).astype(np.float32)
    real_frames = real_data.reshape(num_frames, nfft)
    window = np.hanning(nfft).astype(np.float32)

    def numpy_fft_single():
        return np.fft.fft(complex_data[:nfft])

    results.append(measure(numpy_fft_single))

    def numpy_fft_batch():
        return np.fft.fft(complex_frames, axis=1)

    results.append(measure(numpy_fft_batch))

    def numpy_magnitude():
        fft_result = np.fft.rfft(real_frames, axis=1)
        return np.abs(fft_result) ** 2

    results.append(measure(numpy_magnitude))

    def numpy_full_spectrogram():
        windowed = real_frames * window
        fft_result = np.fft.rfft(windowed, axis=1)
        magnitude = np.abs(fft_result) ** 2
        db = 10 * np.log10(magnitude + 1e-10)
        normalized = (db - db.min()) / (db.max() - db.min() + 1e-10)
        return normalized

    results.append(measure(numpy_full_spectrogram))

    return results


def profile_python_overhead() -> list[Measurement]:
    """Pure Python overhead - transfers to Jetson."""
    results = []

    # Function call overhead
    def empty_func():
        pass

    def call_overhead():
        for _ in range(1000):
            empty_func()

    results.append(measure(call_overhead))

    # Dict operations (config lookup simulation)
    config = {f"key_{i}": f"value_{i}" for i in range(100)}

    def dict_lookup():
        for i in range(1000):
            _ = config.get(f"key_{i % 100}")

    results.append(measure(dict_lookup))

    # List append (buffer accumulation)
    def list_append():
        lst = []
        for i in range(10000):
            lst.append(i)
        return lst

    results.append(measure(list_append))

    # Dataclass creation
    @dataclass
    class Detection:
        x1: float
        y1: float
        x2: float
        y2: float
        class_id: int
        confidence: float

    def dataclass_create():
        dets = []
        for i in range(100):
            dets.append(Detection(0.1, 0.2, 0.3, 0.4, 1, 0.95))
        return dets

    results.append(measure(dataclass_create))

    return results


def profile_async_overhead() -> list[Measurement]:
    """Async overhead - transfers to Jetson."""
    import asyncio

    results = []

    async def async_noop():
        pass

    async def measure_async_call():
        for _ in range(1000):
            await async_noop()

    def async_call_overhead():
        asyncio.run(measure_async_call())

    results.append(measure(async_call_overhead, iterations=20))

    # Queue operations
    async def measure_queue():
        q = asyncio.Queue(maxsize=100)
        for i in range(1000):
            await q.put(i)
            await q.get()

    def queue_overhead():
        asyncio.run(measure_queue())

    results.append(measure(queue_overhead, iterations=20))

    return results


def profile_file_io() -> list[Measurement]:
    """File I/O overhead - transfers to Jetson."""
    import tempfile

    results = []

    # JSON config read (simulates your config loading)
    config_data = {"signals": [{"name": f"sig_{i}", "freq": i * 1e6} for i in range(100)]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    def json_config_read():
        with open(config_path) as f:
            return json.load(f)

    results.append(measure(json_config_read, iterations=50))

    # Binary file read (simulates IQ file load)
    iq_data = np.random.randn(65536).astype(np.complex64)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        iq_data.tofile(f)
        iq_path = f.name

    def binary_file_read():
        return np.fromfile(iq_path, dtype=np.complex64)

    results.append(measure(binary_file_read, iterations=50))

    # Cleanup
    Path(config_path).unlink()
    Path(iq_path).unlink()

    return results


def check_memory_growth() -> dict[str, Any]:
    """Check for obvious memory leaks."""
    import tracemalloc

    tracemalloc.start()

    # Simulate workload
    for _ in range(100):
        data = np.random.randn(4096 * 6).astype(np.complex64)
        fft = np.fft.fft(data)
        del data, fft

    snapshot1 = tracemalloc.take_snapshot()

    for _ in range(100):
        data = np.random.randn(4096 * 6).astype(np.complex64)
        fft = np.fft.fft(data)
        del data, fft

    snapshot2 = tracemalloc.take_snapshot()

    tracemalloc.stop()

    stats = snapshot2.compare_to(snapshot1, "lineno")
    top_growth = []
    for stat in stats[:5]:
        if stat.size_diff > 1024:  # Only report >1KB growth
            top_growth.append(
                {"location": str(stat.traceback), "size_diff_kb": stat.size_diff / 1024}
            )

    return {"iterations": 100, "top_growth": top_growth, "leak_detected": len(top_growth) > 0}


def main():
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "platform": sys.platform,
        },
        "serialization": [],
        "numpy_ops": [],
        "python_overhead": [],
        "async_overhead": [],
        "file_io": [],
        "memory": {},
    }

    print("Profiling serialization...")
    results["serialization"] = [asdict(m) for m in profile_serialization()]

    print("Profiling numpy ops...")
    results["numpy_ops"] = [asdict(m) for m in profile_numpy_ops()]

    print("Profiling python overhead...")
    results["python_overhead"] = [asdict(m) for m in profile_python_overhead()]

    print("Profiling async overhead...")
    results["async_overhead"] = [asdict(m) for m in profile_async_overhead()]

    print("Profiling file I/O...")
    results["file_io"] = [asdict(m) for m in profile_file_io()]

    print("Checking memory growth...")
    results["memory"] = check_memory_growth()

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for category in ["serialization", "numpy_ops", "python_overhead", "async_overhead", "file_io"]:
        print(f"\n{category.upper()}:")
        for m in results[category]:
            print(
                f"  {m['name']}: {m['mean_ms']:.3f} ± {m['std_ms']:.3f} ms (p99: {m['p99_ms']:.3f})"
            )

    print("\nMEMORY:")
    print(f"  leak_detected: {results['memory']['leak_detected']}")
    if results["memory"]["top_growth"]:
        for g in results["memory"]["top_growth"]:
            print(f"  growth: {g['size_diff_kb']:.1f} KB at {g['location'][:60]}")

    # Save
    output_path = Path("profile_baseline.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
