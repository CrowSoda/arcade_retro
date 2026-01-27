# G20 Production Hardening Roadmap - CORRECTIONS & UPDATES

**IMPORTANT: Read this file FIRST before the weekly plans**

This document contains critical corrections to the original roadmap based on real-world constraints and practical engineering feedback.

---

## 🔴 CRITICAL CORRECTIONS

### Correction 1: Zero-Copy Memory Approach
**Original (WRONG):** Use `.cuda()` on pinned tensors for zero-copy
**Problem:** `.cuda()` actually COPIES data, defeating the purpose

**CORRECTED APPROACH:**
```python
# Option A: Use CuPy for true cudaHostAlloc
import cupy as cp

# Allocate mapped memory via CuPy
mapped_mem = cp.cuda.alloc_pinned_memory(buffer_size)
# Access from CPU
cpu_array = np.frombuffer(mapped_mem, dtype=np.complex64)
# GPU kernel reads same physical memory - NO COPY

# Option B: BENCHMARK FIRST
# Jetson unified memory may already be fast enough!
# Profile before implementing complex zero-copy:

import time
import numpy as np
import torch

def benchmark_transfer():
    data = np.random.randn(4096 * 6).astype(np.complex64)

    # Test 1: Standard copy
    start = time.perf_counter()
    for _ in range(1000):
        gpu_data = torch.from_numpy(data).cuda()
    torch.cuda.synchronize()
    standard_time = (time.perf_counter() - start) / 1000 * 1000  # ms

    print(f"Standard transfer: {standard_time:.3f} ms")
    print(f"If < 0.5ms, zero-copy complexity may not be worth it")

# RUN THIS BENCHMARK BEFORE IMPLEMENTING ZERO-COPY
benchmark_transfer()
```

---

### Correction 2: Shared Memory Ring Buffer
**Original (WRONG):** POSIX shared memory with manual lock-free SPSC
**Problems:**
- No memory barriers = data races
- Linux-only (Jetson is Linux, but limits Windows dev)
- Lock-free is overkill for this use case

**CORRECTED APPROACH:**
```python
# Use Python's built-in multiprocessing.shared_memory (cross-platform)
from multiprocessing import shared_memory
import numpy as np
import threading

class SimplifiedSharedBuffer:
    """Cross-platform shared memory with mutex protection.
    Lock-free is overkill for ~30fps IQ streaming."""

    def __init__(self, name: str, size: int, create: bool = False):
        if create:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        else:
            self.shm = shared_memory.SharedMemory(name=name)

        self.lock = threading.Lock()  # Simple mutex, not lock-free

    def write(self, data: np.ndarray):
        with self.lock:
            self.shm.buf[:len(data.tobytes())] = data.tobytes()

    def read(self, dtype, shape) -> np.ndarray:
        with self.lock:
            return np.frombuffer(self.shm.buf, dtype=dtype).reshape(shape).copy()

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()
```

---

### Correction 3: FlatBuffers is Overkill
**Original (WRONG):** FlatBuffers for IQ frame transport
**Problem:** FlatBuffers adds complexity for arrays where raw bytes work fine

**CORRECTED APPROACH:**
```python
# For IQ data: Just use raw bytes
# For metadata: JSON is fine (decodes in ~1ms, not hot path)

# INSTEAD OF:
# FlatBuffers schema, codegen, encoder, decoder...

# USE THIS:
import json
import struct

class SimpleFrameProtocol:
    """Simple binary protocol. FlatBuffers is overkill."""

    HEADER_FORMAT = '<QIIff'  # timestamp, frame_idx, num_rows, noise_floor, center_freq
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    @staticmethod
    def encode_frame(timestamp_us, frame_idx, rows_rgba, noise_floor, center_freq):
        header = struct.pack(
            SimpleFrameProtocol.HEADER_FORMAT,
            timestamp_us, frame_idx, len(rows_rgba), noise_floor, center_freq
        )
        return header + rows_rgba.tobytes()

    @staticmethod
    def decode_frame(data: bytes):
        header = struct.unpack(SimpleFrameProtocol.HEADER_FORMAT,
                              data[:SimpleFrameProtocol.HEADER_SIZE])
        pixel_data = data[SimpleFrameProtocol.HEADER_SIZE:]
        return {
            'timestamp_us': header[0],
            'frame_idx': header[1],
            'num_rows': header[2],
            'noise_floor': header[3],
            'center_freq': header[4],
            'pixels': np.frombuffer(pixel_data, dtype=np.uint8)
        }

# For detections (small structured data): JSON is fine
detections_json = json.dumps({
    'boxes': [[0.1, 0.2, 0.3, 0.4, 0.95, 'signal']],
    'timestamp': 1234567890
})
```

---

### Correction 4: Dependency Injection Simplification
**Original (WRONG):** dependency-injector library + global singletons
**Problem:** Adds framework complexity and global state confusion

**CORRECTED APPROACH:**
```python
# INSTEAD OF:
# from dependency_injector import containers, providers
# class Container(containers.DeclarativeContainer): ...
# _coordinator: Optional[ShutdownCoordinator] = None  # Global singleton

# USE SIMPLE CONSTRUCTOR INJECTION:
class InferenceHandler:
    """Dependencies passed via constructor. No framework needed."""

    def __init__(
        self,
        shutdown_coordinator: ShutdownCoordinator,
        model_loader: ModelLoader,
        sample_manager: SampleManager
    ):
        self.shutdown = shutdown_coordinator
        self.model_loader = model_loader
        self.sample_manager = sample_manager

# Wire up in main():
def main():
    # Create dependencies explicitly
    shutdown = ShutdownCoordinator()
    model_loader = ModelLoader(models_dir=Path("models"))
    sample_manager = SampleManager(training_dir=Path("training_data"))

    # Inject via constructor
    inference_handler = InferenceHandler(
        shutdown_coordinator=shutdown,
        model_loader=model_loader,
        sample_manager=sample_manager
    )

    # No global state, no framework, easy to test
```

---

### Correction 5: CUDA Graphs Likely Unnecessary
**Original (WRONG):** CUDA Graph capture of entire pipeline
**Problem:** Kernel launch overhead is ~50μs, not 1ms+. Graphs add complexity for marginal gain.

**CORRECTED APPROACH:**
```python
# SKIP CUDA Graphs UNLESS profiling shows kernel launch >1ms

# PROFILE FIRST:
import torch
import time

def profile_kernel_launches(num_iterations=1000):
    """Measure actual kernel launch overhead before optimizing."""

    # Typical pipeline operations
    data = torch.randn(6, 4096, dtype=torch.complex64, device='cuda')
    window = torch.hann_window(4096, device='cuda')

    # Warmup
    for _ in range(10):
        windowed = data * window
        fft = torch.fft.rfft(windowed)
        mag = torch.abs(fft)
    torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    for _ in range(num_iterations):
        windowed = data * window
        fft = torch.fft.rfft(windowed)
        mag = torch.abs(fft)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iterations * 1000  # ms

    print(f"Average kernel launch overhead: {elapsed:.3f} ms")
    print(f"If < 1ms, CUDA Graphs add complexity for minimal gain")
    return elapsed

# RUN THIS BEFORE IMPLEMENTING CUDA GRAPHS
overhead = profile_kernel_launches()
if overhead < 1.0:
    print("✓ Skip CUDA Graphs - overhead is acceptable")
else:
    print("✗ Consider CUDA Graphs - overhead is significant")
```

---

### Correction 6: Testing Strategy Order
**Original (WRONG):** Start with Hypothesis property-based tests
**Problem:** Property-based testing is advanced. Start simple, build up.

**CORRECTED TESTING ORDER:**
```
1. SMOKE TESTS (Week 1-2)
   - Does the backend start?
   - Does the WebSocket connect?
   - Does inference return something?

2. INTEGRATION TESTS (Week 3-4)
   - WebSocket message flow end-to-end
   - Training pipeline produces output
   - Detection matches expected format

3. UNIT TESTS (Week 5-6)
   - Individual function behavior
   - Edge cases
   - Error handling

4. PROPERTY-BASED TESTS (Week 7-8) - LAST
   - Hypothesis for DSP invariants
   - Only after other tests exist
```

**Example progression:**
```python
# Week 1: Smoke test
def test_backend_starts():
    process = subprocess.Popen(['python', 'backend/server.py'])
    time.sleep(3)
    assert process.poll() is None  # Still running
    process.terminate()

# Week 3: Integration test
async def test_websocket_echo():
    async with websockets.connect('ws://localhost:8765') as ws:
        await ws.send('{"command": "status"}')
        response = await asyncio.wait_for(ws.recv(), timeout=5)
        assert 'type' in json.loads(response)

# Week 5: Unit test
def test_coordinate_conversion():
    result = pixel_to_frequency(512, width=1024, center=825e6, bw=20e6)
    assert abs(result - 825e6) < 1e3  # Within 1 kHz

# Week 7: Property test (LAST)
@given(st.floats(min_value=0, max_value=1024))
def test_coordinate_roundtrip(pixel):
    freq = pixel_to_frequency(pixel, ...)
    back = frequency_to_pixel(freq, ...)
    assert abs(back - pixel) < 1e-5
```

---

### Correction 7: FFI Memory Safety
**Original (WRONG):** FFI with manual calloc/free that leaks on exception
**Problem:** If exception occurs before `finally`, memory leaks

**CORRECTED APPROACH:**
```dart
// Option A: Use Arena pattern for automatic cleanup
import 'package:ffi/ffi.dart';

void applyColormapSafe(Float32List dbData, Uint8List rgbaOutput, ...) {
  using((arena) {
    // Arena automatically frees on scope exit (even on exception)
    final dbPtr = arena<Float>(dbData.length);
    final rgbaPtr = arena<Uint8>(rgbaOutput.length);

    dbPtr.asTypedList(dbData.length).setAll(0, dbData);

    _applyColormapNative(dbPtr, rgbaPtr, dbData.length, ...);

    rgbaOutput.setAll(0, rgbaPtr.asTypedList(rgbaOutput.length));
    // No manual free needed - arena handles it
  });
}

// Option B: Skip FFI entirely for colormap
// Dart is fast enough for 2048-pixel colormap at 30fps
void applyColormapDart(Float32List dbData, Uint8List rgbaOutput, ...) {
  for (int i = 0; i < dbData.length; i++) {
    final normalized = ((dbData[i] - dbMin) / (dbMax - dbMin)).clamp(0.0, 1.0);
    final idx = (normalized * 255).toInt();
    rgbaOutput[i * 4 + 0] = VIRIDIS_LUT[idx * 3 + 0];
    rgbaOutput[i * 4 + 1] = VIRIDIS_LUT[idx * 3 + 1];
    rgbaOutput[i * 4 + 2] = VIRIDIS_LUT[idx * 3 + 2];
    rgbaOutput[i * 4 + 3] = 255;
  }
}
// Profile before deciding FFI is needed!
```

---

### Correction 8: server.py Migration Strategy
**Original (WRONG):** Extract all modules at once
**Problem:** No migration path, high risk of breaking everything

**CORRECTED APPROACH: Strangler Fig Pattern**
```python
# MIGRATION STRATEGY: Copy → Import → Delete → Test at each step

# WEEK 1: Copy shutdown code to new file, import in server.py
# server.py:
from core.shutdown import ShutdownCoordinator  # NEW
# ... keep old code as fallback, but use new import

# WEEK 2: After tests pass, delete old shutdown code from server.py

# WEEK 3: Copy next module (gRPC services)
# ... repeat pattern

# Each step has:
# 1. Copy function/class to new module
# 2. Add imports to server.py
# 3. Run ALL existing tests
# 4. Delete old code from server.py
# 5. Run tests again

# SAFETY: If tests fail after delete, git checkout server.py and retry
```

---

### Correction 9: Rollback Criteria Per Week
**Original (WRONG):** No explicit success metrics or rollback procedures

**CORRECTED: Each week needs:**
```markdown
## Week N Completion Criteria

### Success Metrics
- [ ] Metric 1: Specific, measurable
- [ ] Metric 2: Specific, measurable
- [ ] All existing tests still pass

### Rollback Trigger
If ANY of these occur, rollback and reassess:
- Inference latency increases >50%
- Memory usage increases >2GB
- Tests fail and can't be fixed in 1 day
- Flutter UI becomes unresponsive

### Rollback Procedure
1. `git checkout main -- <changed files>`
2. Run smoke tests
3. Document what went wrong
4. Adjust plan before retrying
```

---

### Correction 10: Realistic Timeline
**Original (WRONG):** 10 weeks
**Reality:** 15-22 weeks with profiling-first approach

**CORRECTED TIMELINE:**

| Phase | Weeks | Focus |
|-------|-------|-------|
| **Phase 0: Baseline** | 1-2 | Profile EVERYTHING before optimizing |
| **Phase 1: Quick Wins** | 3-4 | Dead code, logging, async I/O, Ruff |
| **Phase 2: Code Quality** | 5-8 | Split server.py (strangler fig), tests |
| **Phase 3: Performance** | 9-12 | Zero-copy (if needed), TensorRT (if needed) |
| **Phase 4: IPC** | 13-15 | Shared memory (if needed), protocol simplification |
| **Phase 5: Flutter** | 16-18 | FFI (if needed), shaders (if needed) |
| **Phase 6: CI/CD** | 19-22 | ARM64 runners, A/B partitions, monitoring |

**KEY CHANGE: Profile BEFORE optimizing**
```python
# WEEK 1-2: Create baseline benchmarks
def create_baseline():
    """Measure current performance before ANY changes."""
    metrics = {
        'inference_latency_ms': measure_inference_latency(),
        'memory_usage_gb': measure_memory_usage(),
        'ipc_latency_us': measure_ipc_latency(),
        'startup_time_s': measure_startup_time(),
        'gpu_utilization_pct': measure_gpu_utilization(),
    }

    # Save to file for comparison
    with open('baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics

# Then decide what to optimize based on actual numbers!
```

---

## Updated Week Order

**NEW ORDER (Profile-First):**

1. **Weeks 1-2:** Baseline profiling + quick wins (dead code, logging)
2. **Weeks 3-4:** Smoke tests + basic integration tests
3. **Weeks 5-8:** Split server.py using strangler fig pattern
4. **Weeks 9-10:** Unit tests, coverage ratchet
5. **Weeks 11-12:** Config centralization (Pydantic)
6. **Weeks 13-14:** Performance optimization (ONLY if profiling shows need)
7. **Weeks 15-16:** IPC optimization (ONLY if profiling shows need)
8. **Weeks 17-18:** Flutter optimization (ONLY if profiling shows need)
9. **Weeks 19-20:** CI/CD setup
10. **Weeks 21-22:** Production hardening, monitoring, A/B updates

---

## Summary of Changes

| Original | Corrected |
|----------|-----------|
| `.cuda()` for zero-copy | CuPy or benchmark first |
| Lock-free SPSC ring buffer | `multiprocessing.shared_memory` + mutex |
| FlatBuffers everywhere | Raw bytes for arrays, JSON for metadata |
| dependency-injector library | Simple constructor injection |
| CUDA Graphs mandatory | Profile first, likely unnecessary |
| Start with Hypothesis | Start with smoke tests |
| FFI with manual memory | Arena pattern or skip FFI |
| Big-bang server.py split | Strangler fig pattern |
| No rollback criteria | Explicit metrics + rollback per week |
| 10 weeks | 15-22 weeks realistic |

---

**Read the weekly files (Harden_roadmap_1.md through _5.md) with these corrections in mind.**
# G20 Production Hardening Roadmap

**Target Platform:** Jetson AGX Orin (16GB)
**Project:** G20 RF Signal Detection Platform
**Created:** January 27, 2026
**Status:** Active Development → Production Hardening

---

## Executive Summary

This roadmap transforms the G20 RF detection platform from a working prototype into a production-ready Jetson Orin deployment. It synthesizes:
1. **Code Review Findings** - 140 files, ~28,300 lines, critical technical debt identified
2. **Production Engineering Guide** - Zero-copy pipelines, TensorRT optimization, proper IPC
3. **10-Week Implementation Plan** - Phased approach from quick wins to CI/CD hardening

### Current State Assessment

| Metric | Current | Target |
|--------|---------|--------|
| Test Coverage | <1% | 70%+ |
| God Modules | 3 (1450+ lines each) | 0 (max 300 lines) |
| Duplicate Code | ~770 lines | 0 |
| Print Statements | 200+ | 0 (proper logging) |
| Config Sources | 8+ hardcoded locations | 1 central config |
| Sync File I/O | 3+ locations blocking UI | 0 |

### Architecture Target

```
┌─────────────────────────────────────────────────────────────────┐
│                     JETSON ORIN (16GB)                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  SDR (PCIe)  │───▶│ Zero-Copy    │───▶│   cuFFT      │      │
│  │  Epiq NV100  │    │ DMA Buffer   │    │   (GPU)      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                    │              │
│         │            cudaHostAlloc         Triple Buffer        │
│         │            (Mapped)              CUDA Streams         │
│         ▼                   ▼                    ▼              │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Shared Memory Ring Buffer               │      │
│  │              (POSIX + Lock-free SPSC)                │      │
│  │              IQ Data: <1μs latency                   │      │
│  └──────────────────────────────────────────────────────┘      │
│         │                                       │               │
│         ▼                                       ▼               │
│  ┌──────────────┐                      ┌──────────────┐        │
│  │  TensorRT    │                      │  Waterfall   │        │
│  │  INT8 + DLA  │                      │  Row-Strip   │        │
│  │  CUDA Graph  │                      │  Streaming   │        │
│  └──────────────┘                      └──────────────┘        │
│         │                                       │               │
│         └───────────────┬───────────────────────┘               │
│                         ▼                                       │
│              ┌──────────────────┐                               │
│              │   FlatBuffers    │                               │
│              │   (18ns decode)  │                               │
│              └──────────────────┘                               │
│                         │                                       │
│                         ▼                                       │
│              ┌──────────────────┐                               │
│              │  Flutter UI      │                               │
│              │  (Impeller GPU)  │                               │
│              │  Fragment Shader │                               │
│              └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## DO IT TODAY (Immediate Actions)

### 🔴 Critical - Complete Before Any Other Work

```bash
# 1. Install Ruff and run on codebase
pip install ruff
cd g20_demo/backend
ruff check . --fix

# 2. Delete dead code files
rm backend/waterfall_buffer.py
rm junk.txt junk.txt222 diff.txt spec_dioff.txt

# 3. Replace duplicate colormap implementations
# In unified_pipeline.py, DELETE the 5 duplicate colormap functions (150 lines)
# ADD: from .colormaps import COLORMAP_LUTS, get_colormap

# 4. Fix Flutter sync I/O (causes UI freeze)
# In signal_database.dart and config_screen.dart:
# Change: file.readAsStringSync() → await file.readAsString()
```

### Validation Checklist
- [ ] `ruff check backend/` passes with no errors
- [ ] `waterfall_buffer.py` deleted
- [ ] Junk files deleted from root
- [ ] Only ONE colormap implementation exists (in `colormaps.py`)
- [ ] No `readAsStringSync()` calls in Flutter code
---

## WEEK 1: Quick Wins & Code Quality Foundation

### Goals
- Eliminate all duplicate code
- Replace print() with structured logging
- Add Ruff + pre-commit hooks
- Fix synchronous I/O in Flutter

### Day 1-2: Dead Code Elimination

#### Files to Delete
```
backend/waterfall_buffer.py          # Marked "unused, kept for future" - DELETE
junk.txt                             # Root junk file
junk.txt222                          # Root junk file
diff.txt                             # Debug artifact
spec_dioff.txt                       # Debug artifact
backend/debug_samples.png            # Debug artifact
```

#### Legacy Code to Remove

**`lib/features/live_detection/providers/inference_provider.dart`**
Remove the "backward compatibility" legacy providers:
```dart
// DELETE THESE (lines ~80-150):
// ============ Legacy providers for backward compatibility ============
final inferenceManagerProvider = ...
class LiveInferenceNotifier extends StateNotifier<LiveInferenceState> { ... }
final liveInferenceProvider = ...
final autoStartInferenceProvider = ...
```

**`lib/features/training/training_screen.dart`**
Delete unused method:
```dart
// DELETE: Widget _buildLabelsTable() { ... }  // Never called, only _buildLabelsTableCompact used
```

### Day 2-3: Colormap Consolidation (HIGH IMPACT)

**Current State:** 180 lines duplicated across 3 files
- `backend/colormaps.py` ✅ KEEP (correct implementation)
- `backend/unified_pipeline.py` ❌ DELETE 150 lines of duplicates
- `backend/waterfall_buffer.py` ❌ DELETE entire file (already dead code)

**Action for `unified_pipeline.py`:**
```python
# DELETE these functions (lines ~50-200):
# def _generate_viridis_lut(): ...
# def _generate_plasma_lut(): ...
# def _generate_inferno_lut(): ...
# def _generate_magma_lut(): ...
# def _generate_turbo_lut(): ...
# VIRIDIS_LUT = _generate_viridis_lut()
# PLASMA_LUT = _generate_plasma_lut()
# ... etc

# REPLACE WITH:
from .colormaps import COLORMAP_LUTS, get_colormap, apply_colormap_db
```

### Day 3-4: Logging Infrastructure

**Replace 200+ print() statements with structured logging**

**`backend/logger_config.py` already exists but is ignored!**

```python
# In every backend file, REPLACE:
print(f"[WS] Client connected from {client_addr}", flush=True)

# WITH:
from .logger_config import get_logger
logger = get_logger(__name__)
logger.info("WS client connected", extra={"client": client_addr})
```

**Files requiring print() → logging conversion:**
| File | Print Statements | Priority |
|------|------------------|----------|
| `server.py` | 50+ | HIGH |
| `unified_pipeline.py` | 50+ | HIGH |
| `training/service.py` | 20+ | MEDIUM |
| `training/sample_manager.py` | 15+ | MEDIUM |
| `hydra/detector.py` | 10+ | MEDIUM |

**Add structured logging config:**
```python
# backend/logger_config.py - ENHANCE existing file
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        return json.dumps(log_entry)
```

### Day 4-5: Pre-commit Hooks Setup

**Create `.pre-commit-config.yaml`:**
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json

  - repo: local
    hooks:
      - id: dart-analyze
        name: dart analyze
        entry: dart analyze lib/
        language: system
        files: \.dart$
```

**Create `pyproject.toml` for Ruff:**
```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "S",    # flake8-bandit (security)
    "UP",   # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.lint.per-file-ignores]
"**/generated/*.py" = ["ALL"]  # Ignore generated proto files
```

### Day 5: Flutter Async I/O Fixes

**`lib/core/database/signal_database.dart`:**
```dart
// BEFORE (blocks UI thread):
SignalDatabaseNotifier() : super(_loadFromDiskSync());

static List<SignalEntry> _loadFromDiskSync() {
  final jsonStr = file.readAsStringSync();  // BLOCKS!
  ...
}

// AFTER (async loading):
SignalDatabaseNotifier() : super([]) {
  _loadFromDisk();  // Fire async load
}

Future<void> _loadFromDisk() async {
  try {
    final file = File(_filePath);
    if (await file.exists()) {
      final jsonStr = await file.readAsString();  // ASYNC
      state = _parseEntries(jsonStr);
    }
  } catch (e) {
    debugPrint('[SignalDB] Load error: $e');
  }
}
```

**`lib/features/config/config_screen.dart`:**
```dart
// BEFORE:
static List<Mission> _loadFromDiskSync() {
  final jsonStr = file.readAsStringSync();  // BLOCKS!
  ...
}

// AFTER:
class MissionsNotifier extends StateNotifier<List<Mission>> {
  MissionsNotifier() : super([]) {
    _loadFromDisk();
  }

  Future<void> _loadFromDisk() async {
    final file = File(_filePath);
    if (await file.exists()) {
      final jsonStr = await file.readAsString();
      state = _parseMissions(jsonStr);
    }
  }
}
```

### Week 1 Deliverables Checklist
- [ ] All dead code files deleted
- [ ] Legacy providers removed from inference_provider.dart
- [ ] Colormap duplication eliminated (150 lines saved)
- [ ] All print() replaced with logger calls
- [ ] Pre-commit hooks configured and working
- [ ] Ruff passing on all Python files
- [ ] Flutter async I/O converted (3 files)
- [ ] No `readAsStringSync()` calls remain

---

## WEEK 2: Zero-Copy Pipeline Implementation

### Goals
- Implement cudaHostAlloc mapped memory for SDR-to-GPU transfers
- Create triple-buffered pipeline with CUDA streams
- Verify no cudaMemcpy in hot path
- Benchmark latency improvements

### Understanding Jetson Orin's Unified Memory

**Key Insight:** On Jetson, CPU and GPU share the same physical DRAM. Traditional `cudaMemcpy` is wasteful.

```
┌─────────────────────────────────────────────────────┐
│                 JETSON ORIN MEMORY                  │
│                                                     │
│   CPU ─────┐                    ┌───── GPU         │
│            │                    │                   │
│            ▼                    ▼                   │
│   ┌─────────────────────────────────────────┐      │
│   │        UNIFIED DRAM (16GB)              │      │
│   │                                         │      │
│   │   ┌─────────────────────────────────┐   │      │
│   │   │  cudaHostAlloc (Mapped)         │   │      │
│   │   │  - CPU writes via pointer       │   │      │
│   │   │  - GPU reads same memory        │   │      │
│   │   │  - NO COPY REQUIRED             │   │      │
│   │   └─────────────────────────────────┘   │      │
│   └─────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────┘
```

### Day 1-2: Zero-Copy Buffer Allocation

**Create `backend/memory/zero_copy.py`:**
```python
"""
Zero-copy buffer management for Jetson Orin.
Exploits unified memory architecture for SDR-to-GPU transfers.
"""
import torch
import numpy as np
from typing import Tuple, Optional

class ZeroCopyBuffer:
    """
    Allocates pinned mapped memory accessible by both CPU and GPU.
    On Jetson Orin, this enables true zero-copy SDR-to-GPU transfers.
    """

    def __init__(self, size_bytes: int, dtype: np.dtype = np.complex64):
        self.size_bytes = size_bytes
        self.dtype = dtype
        self.num_samples = size_bytes // np.dtype(dtype).itemsize

        # Allocate pinned memory with cudaHostAllocMapped flag
        # This creates memory that both CPU and GPU can access directly
        self._host_tensor = torch.zeros(
            self.num_samples,
            dtype=torch.complex64,
            device='cpu'
        ).pin_memory()

        # Get GPU-accessible pointer (same physical memory)
        self._device_tensor = self._host_tensor.cuda()

    @property
    def cpu_array(self) -> np.ndarray:
        """NumPy view for CPU/SDR writes."""
        return self._host_tensor.numpy()

    @property
    def gpu_tensor(self) -> torch.Tensor:
        """GPU tensor for CUDA operations (same memory, no copy)."""
        return self._device_tensor

    def sync(self):
        """Ensure CPU writes are visible to GPU."""
        torch.cuda.synchronize()


class TripleBuffer:
    """
    Triple-buffered pipeline for continuous streaming.
    While buffer A receives SDR data, B undergoes FFT, C runs inference.
    """

    def __init__(self, buffer_size: int):
        self.buffers = [ZeroCopyBuffer(buffer_size) for _ in range(3)]
        self.streams = [torch.cuda.Stream() for _ in range(3)]

        # Buffer roles rotate each frame
        self.write_idx = 0   # SDR writes here
        self.fft_idx = 1     # FFT processes here
        self.infer_idx = 2   # Inference reads here

    def rotate(self):
        """Rotate buffer assignments for next frame."""
        self.write_idx = (self.write_idx + 1) % 3
        self.fft_idx = (self.fft_idx + 1) % 3
        self.infer_idx = (self.infer_idx + 1) % 3

    @property
    def write_buffer(self) -> ZeroCopyBuffer:
        return self.buffers[self.write_idx]

    @property
    def fft_buffer(self) -> ZeroCopyBuffer:
        return self.buffers[self.fft_idx]

    @property
    def infer_buffer(self) -> ZeroCopyBuffer:
        return self.buffers[self.infer_idx]
```

### Day 2-3: Integrate with GPUSpectrogramProcessor

**Modify `backend/gpu_fft.py`:**
```python
from memory.zero_copy import TripleBuffer

class GPUSpectrogramProcessor:
    def __init__(self, nfft: int = 4096, ...):
        # ... existing init ...

        # Add triple-buffered memory management
        buffer_size = nfft * 6 * 8  # 6 frames of complex64
        self.triple_buffer = TripleBuffer(buffer_size)

        # Pre-allocate FFT window on GPU (stays there)
        self.window = torch.hann_window(nfft, device='cuda')

    def process_zero_copy(self, buffer_idx: int) -> torch.Tensor:
        """
        Process FFT using zero-copy buffer.
        No cudaMemcpy - GPU reads directly from mapped memory.
        """
        buffer = self.triple_buffer.buffers[buffer_idx]
        stream = self.triple_buffer.streams[buffer_idx]

        with torch.cuda.stream(stream):
            # GPU reads directly from mapped memory
            iq_gpu = buffer.gpu_tensor

            # Apply window (in-place on GPU)
            windowed = iq_gpu * self.window

            # FFT (entirely on GPU)
            fft_result = torch.fft.rfft(windowed)

            # Magnitude (on GPU)
            magnitude = torch.abs(fft_result)

            return magnitude
```

### Day 3-4: Verify No cudaMemcpy in Hot Path

**Add validation script `scripts/verify_zero_copy.py`:**
```python
"""
Verify zero-copy pipeline has no hidden copies.
Uses CUDA events to profile memory operations.
"""
import torch
import numpy as np
from backend.memory.zero_copy import ZeroCopyBuffer, TripleBuffer

def verify_zero_copy():
    print("=== Zero-Copy Verification ===\n")

    # Create buffer
    buffer = ZeroCopyBuffer(4096 * 8)  # 4096 complex64 samples

    # Write from "SDR" (CPU side)
    buffer.cpu_array[:] = np.random.randn(4096) + 1j * np.random.randn(4096)

    # Sync to ensure writes visible
    buffer.sync()

    # Profile GPU access
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    # GPU reads from same memory - should be ~0ms if truly zero-copy
    gpu_data = buffer.gpu_tensor
    result = torch.abs(torch.fft.rfft(gpu_data))

    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end)
    print(f"GPU access time: {elapsed:.3f} ms")
    print(f"Expected: <0.1 ms for zero-copy")
    print(f"If >1 ms, there's a hidden copy!")

    # Verify data integrity
    cpu_fft = np.abs(np.fft.rfft(buffer.cpu_array))
    gpu_fft = result.cpu().numpy()

    max_diff = np.max(np.abs(cpu_fft - gpu_fft))
    print(f"\nMax FFT difference (CPU vs GPU): {max_diff:.2e}")
    print(f"Expected: <1e-5 (numerical precision)")

    return elapsed < 0.5 and max_diff < 1e-4

if __name__ == "__main__":
    success = verify_zero_copy()
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: Zero-copy verification")
```

### Day 4-5: Benchmark Latency Improvements

**Expected latencies on 30W mode:**
| Operation | Before (with copies) | After (zero-copy) |
|-----------|---------------------|-------------------|
| IQ to GPU | ~500μs | <10μs |
| cuFFT 4096-point | ~100μs | ~100μs |
| Total hot path | ~700μs | ~150μs |

**Create `scripts/benchmark_pipeline.py`:**
```python
import time
import torch
import numpy as np
from backend.memory.zero_copy import TripleBuffer
from backend.gpu_fft import GPUSpectrogramProcessor

def benchmark_pipeline(iterations: int = 1000):
    processor = GPUSpectrogramProcessor(nfft=4096)

    # Warmup
    for _ in range(10):
        processor.process_zero_copy(0)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        processor.triple_buffer.rotate()
        processor.process_zero_copy(processor.triple_buffer.fft_idx)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_latency = (end - start) / iterations * 1000  # ms
    print(f"Average pipeline latency: {avg_latency:.3f} ms")
    print(f"Throughput: {1000/avg_latency:.1f} frames/sec")

    return avg_latency

if __name__ == "__main__":
    benchmark_pipeline()
```

### Week 2 Deliverables Checklist
- [ ] `backend/memory/zero_copy.py` implemented
- [ ] TripleBuffer with CUDA streams working
- [ ] GPUSpectrogramProcessor uses zero-copy buffers
- [ ] `verify_zero_copy.py` passes
- [ ] No cudaMemcpy in hot path (verified with nsys profile)
- [ ] Benchmark shows <200μs pipeline latency
- [ ] Memory usage stable (no leaks over 1 hour)
---

## WEEK 3: Split server.py God Module

### Goals
- Break 1450-line server.py into 8-10 focused modules
- Implement command pattern for WebSocket handlers
- Wire dependencies with dependency-injector
- Each new module under 300 lines

### The Problem: 11 Responsibilities in One File

```
server.py (1450 lines) currently handles:
├── Global shutdown coordination (~100 lines)
├── Parent process watchdog (~50 lines)
├── Data classes (ChannelState, CaptureSession, etc.)
├── DeviceControlServicer gRPC (~150 lines)
├── InferenceServicer gRPC (~200 lines)
├── WebSocket inference handler (~100 lines)
├── WebSocket unified pipeline handler (~100 lines)
├── WebSocket video pipeline handler (~100 lines)
├── WebSocket training handler (~400 lines!)
├── WebSocket router (~50 lines)
└── Server startup logic (~200 lines)
```

### Target Structure

```
backend/
├── api/
│   ├── __init__.py
│   ├── grpc/
│   │   ├── __init__.py
│   │   ├── device_control.py      # DeviceControlServicer
│   │   └── inference_service.py   # InferenceServicer
│   └── ws/
│       ├── __init__.py
│       ├── router.py              # WebSocket routing
│       └── handlers/
│           ├── __init__.py
│           ├── inference.py       # ws_inference_handler
│           ├── pipeline.py        # unified/video pipeline
│           └── training.py        # Training commands
├── core/
│   ├── __init__.py
│   ├── shutdown.py               # Shutdown coordination
│   ├── process.py                # Parent watchdog
│   └── models.py                 # Data classes
├── server.py                     # Just startup (~100 lines)
└── container.py                  # Dependency injection
```

### Day 1: Extract Core Modules

**Create `backend/core/shutdown.py`:**
```python
"""Graceful shutdown coordination for multi-threaded server."""
import asyncio
import threading
import signal
import sys
from typing import List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ShutdownCoordinator:
    """Coordinates graceful shutdown across threads and async tasks."""

    def __init__(self):
        self._shutdown_event = threading.Event()
        self._async_shutdown_event: Optional[asyncio.Event] = None
        self._cleanup_resources: List[Any] = []
        self._lock = threading.Lock()

    def register_cleanup(self, resource: Any):
        """Register a resource for cleanup on shutdown."""
        with self._lock:
            self._cleanup_resources.append(resource)

    def is_shutting_down(self) -> bool:
        return self._shutdown_event.is_set()

    def request_shutdown(self):
        """Signal all components to begin shutdown."""
        logger.info("Shutdown requested")
        self._shutdown_event.set()
        if self._async_shutdown_event:
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(self._async_shutdown_event.set)
            except RuntimeError:
                pass  # No running loop

    def wait(self, timeout: float = None) -> bool:
        """Wait for shutdown signal."""
        return self._shutdown_event.wait(timeout=timeout)

    async def wait_async(self):
        """Async wait for shutdown signal."""
        if self._async_shutdown_event is None:
            self._async_shutdown_event = asyncio.Event()
            if self._shutdown_event.is_set():
                self._async_shutdown_event.set()
        await self._async_shutdown_event.wait()

    def cleanup(self):
        """Run cleanup on all registered resources."""
        logger.info(f"Cleaning up {len(self._cleanup_resources)} resources")
        for resource in reversed(self._cleanup_resources):
            try:
                if hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, 'stop'):
                    resource.stop()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def install_signal_handlers(self):
        """Install signal handlers for graceful shutdown."""
        def handler(signum, frame):
            logger.info(f"Received signal {signum}")
            self.request_shutdown()

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
        if sys.platform == 'win32':
            signal.signal(signal.SIGBREAK, handler)


# Global instance (singleton pattern)
_coordinator: Optional[ShutdownCoordinator] = None

def get_shutdown_coordinator() -> ShutdownCoordinator:
    global _coordinator
    if _coordinator is None:
        _coordinator = ShutdownCoordinator()
    return _coordinator
```

**Create `backend/core/process.py`:**
```python
"""Parent process monitoring for orphan prevention."""
import os
import sys
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ParentWatchdog:
    """Monitor parent process and exit if it dies (prevents orphans)."""

    def __init__(self, parent_pid: Optional[int] = None, check_interval: float = 1.0):
        self.parent_pid = parent_pid
        self.check_interval = check_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self, on_orphan: callable):
        """Start monitoring parent process."""
        if self.parent_pid is None:
            logger.warning("No parent PID provided, watchdog disabled")
            return

        def monitor():
            while not self._stop_event.wait(self.check_interval):
                if not self._is_parent_alive():
                    logger.warning(f"Parent {self.parent_pid} died, triggering shutdown")
                    on_orphan()
                    break

        self._thread = threading.Thread(target=monitor, daemon=True)
        self._thread.start()
        logger.info(f"Watchdog started, monitoring parent PID {self.parent_pid}")

    def stop(self):
        """Stop the watchdog."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _is_parent_alive(self) -> bool:
        """Check if parent process is still running."""
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, self.parent_pid
            )
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            try:
                os.kill(self.parent_pid, 0)
                return True
            except OSError:
                return False
```

**Create `backend/core/models.py`:**
```python
"""Data classes for server state management."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass
class ChannelState:
    """State for a single SDR channel."""
    center_freq_hz: float = 0.0
    bandwidth_hz: float = 0.0
    is_tuned: bool = False
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class CaptureSession:
    """Active capture session state."""
    session_id: str
    output_path: str
    start_time: datetime = field(default_factory=datetime.now)
    samples_captured: int = 0
    is_active: bool = True

@dataclass
class InferenceSession:
    """Active inference session state."""
    session_id: str
    model_id: str
    start_time: datetime = field(default_factory=datetime.now)
    frames_processed: int = 0
    detections_total: int = 0
    is_active: bool = True

@dataclass
class ModelState:
    """Loaded model state."""
    model_id: str
    model_path: str
    class_names: List[str]
    is_loaded: bool = False
    engine: Optional[Any] = None
```

### Day 2: Extract gRPC Services

**Create `backend/api/grpc/device_control.py`:**
```python
"""gRPC Device Control Service implementation."""
import grpc
import logging
from typing import Dict

from ...generated import control_pb2, control_pb2_grpc
from ...core.models import ChannelState, CaptureSession
from ...core.shutdown import get_shutdown_coordinator

logger = logging.getLogger(__name__)

class DeviceControlServicer(control_pb2_grpc.DeviceControlServicer):
    """Handles SDR device control via gRPC."""

    def __init__(self):
        self.channels: Dict[int, ChannelState] = {}
        self.captures: Dict[str, CaptureSession] = {}
        self._shutdown = get_shutdown_coordinator()

    def TuneChannel(self, request, context):
        logger.info(f"TuneChannel: ch={request.channel_id}, freq={request.center_freq_hz}")

        if self._shutdown.is_shutting_down():
            context.abort(grpc.StatusCode.UNAVAILABLE, "Server shutting down")

        self.channels[request.channel_id] = ChannelState(
            center_freq_hz=request.center_freq_hz,
            bandwidth_hz=request.bandwidth_hz,
            is_tuned=True
        )

        return control_pb2.TuneResponse(
            success=True,
            actual_freq_hz=request.center_freq_hz,
            actual_bandwidth_hz=request.bandwidth_hz
        )

    def GetChannelStatus(self, request, context):
        channel = self.channels.get(request.channel_id)
        if channel is None:
            return control_pb2.ChannelStatus(is_tuned=False)

        return control_pb2.ChannelStatus(
            is_tuned=channel.is_tuned,
            center_freq_hz=channel.center_freq_hz,
            bandwidth_hz=channel.bandwidth_hz
        )

    def StartCapture(self, request, context):
        session = CaptureSession(
            session_id=request.session_id,
            output_path=request.output_path
        )
        self.captures[request.session_id] = session
        logger.info(f"Capture started: {request.session_id}")

        return control_pb2.CaptureResponse(
            success=True,
            session_id=request.session_id
        )

    def StopCapture(self, request, context):
        session = self.captures.pop(request.session_id, None)
        if session:
            session.is_active = False
            logger.info(f"Capture stopped: {request.session_id}")
            return control_pb2.CaptureResponse(success=True)
        else:
            context.abort(grpc.StatusCode.NOT_FOUND, "Session not found")
```

**Create `backend/api/grpc/inference_service.py`:**
```python
"""gRPC Inference Service implementation."""
import grpc
import logging
from pathlib import Path
from typing import Dict, Optional

from ...generated import inference_pb2, inference_pb2_grpc
from ...core.models import ModelState, InferenceSession
from ...core.shutdown import get_shutdown_coordinator

logger = logging.getLogger(__name__)

class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    """Handles model inference via gRPC."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models: Dict[str, ModelState] = {}
        self.sessions: Dict[str, InferenceSession] = {}
        self.active_model_id: Optional[str] = None
        self._shutdown = get_shutdown_coordinator()
        self._scan_models()

    def _scan_models(self):
        """Scan models directory for available models."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return

        for model_path in self.models_dir.glob("*.pth"):
            model_id = model_path.stem
            self.models[model_id] = ModelState(
                model_id=model_id,
                model_path=str(model_path),
                class_names=["background", "signal"]  # TODO: Load from config
            )
        logger.info(f"Found {len(self.models)} models")

    def ListModels(self, request, context):
        models = [
            inference_pb2.ModelInfo(
                model_id=m.model_id,
                model_path=m.model_path,
                is_loaded=m.is_loaded
            )
            for m in self.models.values()
        ]
        return inference_pb2.ListModelsResponse(models=models)

    def LoadModel(self, request, context):
        model = self.models.get(request.model_id)
        if model is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Model not found: {request.model_id}")

        # TODO: Actually load the model
        model.is_loaded = True
        self.active_model_id = request.model_id
        logger.info(f"Model loaded: {request.model_id}")

        return inference_pb2.LoadModelResponse(success=True)

    def UnloadModel(self, request, context):
        model = self.models.get(request.model_id)
        if model and model.is_loaded:
            model.is_loaded = False
            model.engine = None
            if self.active_model_id == request.model_id:
                self.active_model_id = None
            logger.info(f"Model unloaded: {request.model_id}")

        return inference_pb2.UnloadModelResponse(success=True)
```

### Day 3: Extract WebSocket Handlers

**Create `backend/api/ws/router.py`:**
```python
"""WebSocket routing to appropriate handlers."""
import logging
from typing import Callable, Dict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class WebSocketRouter:
    """Routes WebSocket connections to appropriate handlers based on path."""

    def __init__(self):
        self._routes: Dict[str, Callable] = {}

    def route(self, path: str):
        """Decorator to register a handler for a path."""
        def decorator(handler: Callable):
            self._routes[path] = handler
            logger.debug(f"Registered WS handler: {path}")
            return handler
        return decorator

    def add_route(self, path: str, handler: Callable):
        """Programmatically add a route."""
        self._routes[path] = handler

    async def handle(self, websocket, path: str):
        """Route incoming connection to appropriate handler."""
        # Normalize path
        parsed = urlparse(path)
        clean_path = parsed.path.rstrip('/')

        handler = self._routes.get(clean_path)
        if handler is None:
            # Try prefix matching
            for route_path, route_handler in self._routes.items():
                if clean_path.startswith(route_path):
                    handler = route_handler
                    break

        if handler is None:
            logger.warning(f"No handler for path: {clean_path}")
            await websocket.close(1008, f"Unknown path: {clean_path}")
            return

        logger.info(f"Routing {clean_path} to {handler.__name__}")
        await handler(websocket)


# Global router instance
ws_router = WebSocketRouter()
```

### Day 4: Command Pattern for Training Handler

**Create `backend/api/ws/handlers/training.py`:**
```python
"""Training WebSocket handler with command pattern."""
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CommandContext:
    """Context passed to command handlers."""
    websocket: Any
    data: Dict[str, Any]
    training_service: Any  # Injected dependency
    sample_manager: Any    # Injected dependency

class Command(ABC):
    """Base class for training commands."""

    @abstractmethod
    async def execute(self, ctx: CommandContext) -> Dict[str, Any]:
        """Execute the command and return response."""
        pass

class GetRegistryCommand(Command):
    async def execute(self, ctx: CommandContext) -> Dict[str, Any]:
        registry = ctx.training_service.get_registry()
        return {"type": "registry", "data": registry}

class TrainSignalCommand(Command):
    async def execute(self, ctx: CommandContext) -> Dict[str, Any]:
        signal_name = ctx.data.get("signal_name")
        preset = ctx.data.get("preset", "balanced")

        if not signal_name:
            return {"type": "error", "message": "signal_name required"}

        # Start training (returns immediately, progress via callbacks)
        ctx.training_service.train_signal(
            signal_name=signal_name,
            preset=preset,
            progress_callback=lambda p: _send_progress(ctx.websocket, p)
        )
        return {"type": "training_started", "signal_name": signal_name}

class SaveSampleCommand(Command):
    async def execute(self, ctx: CommandContext) -> Dict[str, Any]:
        signal_name = ctx.data.get("signal_name")
        iq_data = ctx.data.get("iq_data")
        boxes = ctx.data.get("boxes", [])
        metadata = ctx.data.get("metadata", {})

        if not signal_name or not iq_data:
            return {"type": "error", "message": "signal_name and iq_data required"}

        sample_id, is_new = ctx.sample_manager.save_sample(
            signal_name=signal_name,
            iq_data_b64=iq_data,
            boxes=boxes,
            metadata=metadata
        )
        return {
            "type": "sample_saved",
            "sample_id": sample_id,
            "is_new": is_new
        }

# Command registry
COMMANDS: Dict[str, Command] = {
    "get_registry": GetRegistryCommand(),
    "train_signal": TrainSignalCommand(),
    "save_sample": SaveSampleCommand(),
    # Add more commands here...
}

async def training_handler(websocket, training_service, sample_manager):
    """Main training WebSocket handler using command pattern."""
    logger.info("Training client connected")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                cmd = data.get("command")

                handler = COMMANDS.get(cmd)
                if handler is None:
                    response = {"type": "error", "message": f"Unknown command: {cmd}"}
                else:
                    ctx = CommandContext(
                        websocket=websocket,
                        data=data,
                        training_service=training_service,
                        sample_manager=sample_manager
                    )
                    response = await handler.execute(ctx)

                await websocket.send(json.dumps(response))

            except json.JSONDecodeError as e:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Invalid JSON: {e}"
                }))

    except Exception as e:
        logger.error(f"Training handler error: {e}")
    finally:
        logger.info("Training client disconnected")

async def _send_progress(websocket, progress):
    """Send training progress update."""
    await websocket.send(json.dumps({
        "type": "progress",
        "epoch": progress.epoch,
        "loss": progress.loss,
        "metrics": progress.metrics
    }))
```

### Day 5: Dependency Injection Container

**Create `backend/container.py`:**
```python
"""Dependency injection container using dependency-injector."""
from dependency_injector import containers, providers
from pathlib import Path

from .core.shutdown import ShutdownCoordinator
from .core.process import ParentWatchdog
from .api.grpc.device_control import DeviceControlServicer
from .api.grpc.inference_service import InferenceServicer
from .training.service import TrainingService
from .training.sample_manager import SampleManager
from .hydra.detector import HydraDetector

class Container(containers.DeclarativeContainer):
    """DI container for all server dependencies."""

    # Configuration
    config = providers.Configuration()

    # Core services
    shutdown_coordinator = providers.Singleton(ShutdownCoordinator)

    parent_watchdog = providers.Singleton(
        ParentWatchdog,
        parent_pid=config.parent_pid,
        check_interval=1.0
    )

    # Paths
    base_dir = providers.Object(Path(__file__).parent.parent)
    models_dir = providers.Factory(lambda base: base / "models", base_dir)
    data_dir = providers.Factory(lambda base: base / "data", base_dir)
    training_dir = providers.Factory(lambda base: base / "training_data" / "signals", base_dir)

    # gRPC services
    device_control_servicer = providers.Singleton(DeviceControlServicer)

    inference_servicer = providers.Singleton(
        InferenceServicer,
        models_dir=models_dir
    )

    # ML components
    hydra_detector = providers.Singleton(
        HydraDetector,
        models_dir=models_dir
    )

    sample_manager = providers.Singleton(
        SampleManager,
        training_dir=training_dir
    )

    training_service = providers.Singleton(
        TrainingService,
        sample_manager=sample_manager,
        hydra_detector=hydra_detector
    )
```

### Week 3 Deliverables Checklist
- [ ] `backend/core/shutdown.py` extracted and working
- [ ] `backend/core/process.py` extracted and working
- [ ] `backend/core/models.py` with all data classes
- [ ] `backend/api/grpc/device_control.py` extracted
- [ ] `backend/api/grpc/inference_service.py` extracted
- [ ] `backend/api/ws/router.py` with clean routing
- [ ] `backend/api/ws/handlers/training.py` with command pattern
- [ ] `backend/container.py` with dependency injection
- [ ] `backend/server.py` reduced to ~100 lines (just startup)
- [ ] All existing functionality still works
- [ ] No module exceeds 300 lines
---

## WEEK 4: IPC Overhaul - Shared Memory + FlatBuffers

### Goals
- Replace gRPC for local IQ data transfer with POSIX shared memory
- Implement lock-free ring buffer for IQ streaming
- Adopt FlatBuffers for hot-path serialization (18ns vs 1179ns Protobuf)
- Keep gRPC for remote/network communication only

### Why This Matters

**Current IPC Latency:**
| Method | Latency | Use Case |
|--------|---------|----------|
| gRPC (Protobuf) | ~1.2ms decode | Currently used everywhere |
| Unix Socket | 2-3μs | Control messages |
| Shared Memory | <1μs | IQ data streaming |
| FlatBuffers | 18ns decode | Hot-path data |

### Day 1-2: Shared Memory Ring Buffer

**Create `backend/ipc/shared_ring.py`:**
```python
"""Lock-free SPSC ring buffer over POSIX shared memory."""
import mmap
import os
import struct
from typing import Optional
import numpy as np

class SharedRingBuffer:
    """
    Single-Producer Single-Consumer lock-free ring buffer.
    Uses POSIX shared memory for cross-process zero-copy IQ transfer.

    Memory layout:
    [0:8]   - write_index (uint64, producer updates)
    [8:16]  - read_index (uint64, consumer updates)
    [16:24] - capacity (uint64, immutable)
    [24:32] - item_size (uint64, immutable)
    [32:]   - ring buffer data
    """

    HEADER_SIZE = 32

    def __init__(self, name: str, capacity: int, item_size: int, create: bool = False):
        self.name = name
        self.capacity = capacity
        self.item_size = item_size
        self.total_size = self.HEADER_SIZE + (capacity * item_size)

        # Create or open shared memory
        if create:
            # Remove existing if present
            try:
                os.unlink(f"/dev/shm/{name}")
            except FileNotFoundError:
                pass

            self.fd = os.open(
                f"/dev/shm/{name}",
                os.O_CREAT | os.O_RDWR,
                0o666
            )
            os.ftruncate(self.fd, self.total_size)
        else:
            self.fd = os.open(f"/dev/shm/{name}", os.O_RDWR)

        # Memory-map the file
        self.mmap = mmap.mmap(self.fd, self.total_size)

        if create:
            # Initialize header
            struct.pack_into('QQQ Q', self.mmap, 0,
                           0,           # write_index
                           0,           # read_index
                           capacity,    # capacity
                           item_size)   # item_size

    def _read_header(self):
        return struct.unpack_from('QQQQ', self.mmap, 0)

    @property
    def write_index(self) -> int:
        return struct.unpack_from('Q', self.mmap, 0)[0]

    @write_index.setter
    def write_index(self, value: int):
        struct.pack_into('Q', self.mmap, 0, value)

    @property
    def read_index(self) -> int:
        return struct.unpack_from('Q', self.mmap, 8)[0]

    @read_index.setter
    def read_index(self, value: int):
        struct.pack_into('Q', self.mmap, 8, value)

    def push(self, data: bytes) -> bool:
        """Push item to ring buffer. Returns False if full."""
        assert len(data) == self.item_size

        write_idx = self.write_index
        read_idx = self.read_index

        # Check if full
        next_write = (write_idx + 1) % self.capacity
        if next_write == read_idx:
            return False  # Buffer full

        # Write data
        offset = self.HEADER_SIZE + (write_idx * self.item_size)
        self.mmap[offset:offset + self.item_size] = data

        # Update write index (memory barrier implicit in mmap)
        self.write_index = next_write
        return True

    def pop(self) -> Optional[bytes]:
        """Pop item from ring buffer. Returns None if empty."""
        write_idx = self.write_index
        read_idx = self.read_index

        # Check if empty
        if read_idx == write_idx:
            return None

        # Read data
        offset = self.HEADER_SIZE + (read_idx * self.item_size)
        data = bytes(self.mmap[offset:offset + self.item_size])

        # Update read index
        self.read_index = (read_idx + 1) % self.capacity
        return data

    def push_numpy(self, array: np.ndarray) -> bool:
        """Push numpy array directly."""
        return self.push(array.tobytes())

    def pop_numpy(self, dtype=np.complex64) -> Optional[np.ndarray]:
        """Pop as numpy array."""
        data = self.pop()
        if data is None:
            return None
        return np.frombuffer(data, dtype=dtype)

    def close(self):
        """Clean up resources."""
        self.mmap.close()
        os.close(self.fd)

    def unlink(self):
        """Remove shared memory segment."""
        try:
            os.unlink(f"/dev/shm/{self.name}")
        except FileNotFoundError:
            pass
```

### Day 2-3: FlatBuffers Schema for Hot Paths

**Create `protos/iq_frame.fbs`:**
```flatbuffers
// FlatBuffers schema for IQ frame transport
// 18ns decode vs 1179ns for Protobuf

namespace g20.ipc;

// Detection box in normalized coordinates
struct DetectionBox {
  x1: float;
  y1: float;
  x2: float;
  y2: float;
  class_id: uint8;
  confidence: float;
}

// Single spectrogram row (RGBA pixels)
table SpectrogramRow {
  row_index: uint32;
  absolute_row: uint32;
  pixels: [uint8];  // RGBA data
  psd_db: [float];  // Power spectral density
}

// Frame with detections
table InferenceFrame {
  timestamp_us: uint64;
  frame_index: uint32;
  rows: [SpectrogramRow];
  detections: [DetectionBox];
  noise_floor_db: float;
  center_freq_hz: double;
  bandwidth_hz: double;
}

// Metadata message
table StreamMetadata {
  width: uint32;
  height: uint32;
  fft_size: uint32;
  sample_rate_hz: double;
  center_freq_hz: double;
  colormap_id: uint8;
}

root_type InferenceFrame;
```

**Generate Python bindings:**
```bash
flatc --python -o backend/generated/ protos/iq_frame.fbs
```

**Create `backend/ipc/flatbuffers_codec.py`:**
```python
"""FlatBuffers encoding/decoding for hot-path IPC."""
import flatbuffers
from ..generated.g20.ipc import InferenceFrame, SpectrogramRow, DetectionBox, StreamMetadata
import numpy as np

class FrameEncoder:
    """Encode frames to FlatBuffers format."""

    def __init__(self, initial_size: int = 1024):
        self.builder = flatbuffers.Builder(initial_size)

    def encode_frame(
        self,
        timestamp_us: int,
        frame_index: int,
        rows: list,
        detections: list,
        noise_floor_db: float,
        center_freq_hz: float,
        bandwidth_hz: float
    ) -> bytes:
        """Encode a complete inference frame."""
        self.builder.Reset()

        # Encode rows
        row_offsets = []
        for row in rows:
            pixels_offset = self.builder.CreateNumpyVector(row['pixels'])
            psd_offset = self.builder.CreateNumpyVector(row['psd_db'].astype(np.float32))

            SpectrogramRow.Start(self.builder)
            SpectrogramRow.AddRowIndex(self.builder, row['row_index'])
            SpectrogramRow.AddAbsoluteRow(self.builder, row['absolute_row'])
            SpectrogramRow.AddPixels(self.builder, pixels_offset)
            SpectrogramRow.AddPsdDb(self.builder, psd_offset)
            row_offsets.append(SpectrogramRow.End(self.builder))

        rows_vector = self.builder.CreateVector(row_offsets)

        # Encode detections (inline structs, no offset needed)
        InferenceFrame.StartDetectionsVector(self.builder, len(detections))
        for det in reversed(detections):  # FlatBuffers builds in reverse
            DetectionBox.Create(
                self.builder,
                det['x1'], det['y1'], det['x2'], det['y2'],
                det['class_id'], det['confidence']
            )
        detections_vector = self.builder.EndVector()

        # Build frame
        InferenceFrame.Start(self.builder)
        InferenceFrame.AddTimestampUs(self.builder, timestamp_us)
        InferenceFrame.AddFrameIndex(self.builder, frame_index)
        InferenceFrame.AddRows(self.builder, rows_vector)
        InferenceFrame.AddDetections(self.builder, detections_vector)
        InferenceFrame.AddNoiseFloorDb(self.builder, noise_floor_db)
        InferenceFrame.AddCenterFreqHz(self.builder, center_freq_hz)
        InferenceFrame.AddBandwidthHz(self.builder, bandwidth_hz)
        frame = InferenceFrame.End(self.builder)

        self.builder.Finish(frame)
        return bytes(self.builder.Output())


class FrameDecoder:
    """Zero-copy FlatBuffers frame decoder."""

    def decode_frame(self, data: bytes) -> dict:
        """
        Decode frame with zero-copy access to arrays.
        ~18ns for basic access vs ~1179ns for Protobuf.
        """
        frame = InferenceFrame.InferenceFrame.GetRootAs(data, 0)

        return {
            'timestamp_us': frame.TimestampUs(),
            'frame_index': frame.FrameIndex(),
            'noise_floor_db': frame.NoiseFloorDb(),
            'center_freq_hz': frame.CenterFreqHz(),
            'bandwidth_hz': frame.BandwidthHz(),
            'rows': self._decode_rows(frame),
            'detections': self._decode_detections(frame)
        }

    def _decode_rows(self, frame) -> list:
        rows = []
        for i in range(frame.RowsLength()):
            row = frame.Rows(i)
            rows.append({
                'row_index': row.RowIndex(),
                'absolute_row': row.AbsoluteRow(),
                'pixels': row.PixelsAsNumpy(),  # Zero-copy numpy view
                'psd_db': row.PsdDbAsNumpy()
            })
        return rows

    def _decode_detections(self, frame) -> list:
        detections = []
        for i in range(frame.DetectionsLength()):
            det = frame.Detections(i)
            detections.append({
                'x1': det.X1(),
                'y1': det.Y1(),
                'x2': det.X2(),
                'y2': det.Y2(),
                'class_id': det.ClassId(),
                'confidence': det.Confidence()
            })
        return detections
```

### Day 4-5: Unix Socket for Control Messages

**Create `backend/ipc/control_socket.py`:**
```python
"""Unix domain socket for control messages (2-3μs latency)."""
import asyncio
import json
import os
import logging
from typing import Callable, Dict, Any

logger = logging.getLogger(__name__)

class ControlServer:
    """Unix socket server for control messages."""

    def __init__(self, socket_path: str = "/tmp/g20_control.sock"):
        self.socket_path = socket_path
        self.handlers: Dict[str, Callable] = {}
        self._server = None

    def register_handler(self, command: str, handler: Callable):
        """Register a command handler."""
        self.handlers[command] = handler

    async def start(self):
        """Start the control server."""
        # Remove existing socket
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=self.socket_path
        )
        logger.info(f"Control server listening on {self.socket_path}")

    async def stop(self):
        """Stop the control server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_client(self, reader, writer):
        """Handle incoming control connection."""
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break

                request = json.loads(line.decode())
                command = request.get('command')

                handler = self.handlers.get(command)
                if handler:
                    result = await handler(request)
                else:
                    result = {'error': f'Unknown command: {command}'}

                response = json.dumps(result) + '\n'
                writer.write(response.encode())
                await writer.drain()

        except Exception as e:
            logger.error(f"Control handler error: {e}")
        finally:
            writer.close()


class ControlClient:
    """Unix socket client for sending control messages."""

    def __init__(self, socket_path: str = "/tmp/g20_control.sock"):
        self.socket_path = socket_path
        self._reader = None
        self._writer = None

    async def connect(self):
        """Connect to control server."""
        self._reader, self._writer = await asyncio.open_unix_connection(
            self.socket_path
        )

    async def send_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """Send command and wait for response."""
        request = {'command': command, **kwargs}
        self._writer.write((json.dumps(request) + '\n').encode())
        await self._writer.drain()

        response = await self._reader.readline()
        return json.loads(response.decode())

    async def close(self):
        """Close connection."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
```

### Week 4 Deliverables Checklist
- [ ] `backend/ipc/shared_ring.py` with SPSC ring buffer
- [ ] `protos/iq_frame.fbs` FlatBuffers schema
- [ ] `backend/ipc/flatbuffers_codec.py` encoder/decoder
- [ ] `backend/ipc/control_socket.py` Unix socket for control
- [ ] Benchmark: shared memory <1μs IQ transfer
- [ ] Benchmark: FlatBuffers decode <50ns
- [ ] gRPC retained for remote/network only

---

## WEEK 5: Configuration Centralization

### Goals
- Single source of truth for all configuration
- Pydantic-settings v2 for type-safe validation
- Backend exposes capabilities API to frontend
- Environment-based overrides for deployment

### Day 1-2: Centralized Config with Pydantic

**Create `backend/config/settings.py`:**
```python
"""Centralized configuration using Pydantic-settings v2."""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import List, Optional
from enum import Enum

class PowerMode(str, Enum):
    LOW = "15W"
    BALANCED = "30W"
    HIGH = "50W"

class ServerSettings(BaseSettings):
    """Server configuration."""
    model_config = SettingsConfigDict(
        env_prefix="G20_SERVER_",
        env_nested_delimiter="__"
    )

    grpc_port: int = Field(default=50051, description="gRPC server port")
    ws_port: int = Field(default=0, description="WebSocket port (0=auto)")
    max_workers: int = Field(default=10, description="gRPC thread pool size")

class PathSettings(BaseSettings):
    """Path configuration."""
    model_config = SettingsConfigDict(env_prefix="G20_PATH_")

    base_dir: Path = Field(default=Path(__file__).parent.parent.parent)
    models_dir: Path = Field(default=None)
    data_dir: Path = Field(default=None)
    training_dir: Path = Field(default=None)
    config_dir: Path = Field(default=None)

    def model_post_init(self, __context):
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.training_dir is None:
            self.training_dir = self.base_dir / "training_data" / "signals"
        if self.config_dir is None:
            self.config_dir = self.base_dir / "config"

class FFTSettings(BaseSettings):
    """FFT configuration - CRITICAL: inference FFT is locked!"""
    model_config = SettingsConfigDict(env_prefix="G20_FFT_")

    # Inference FFT - DO NOT CHANGE without retraining models!
    inference_nfft: int = Field(default=4096, description="LOCKED - matches training")
    inference_hop: int = Field(default=2048, description="LOCKED - 50% overlap")
    inference_dynamic_range_db: float = Field(default=80.0, description="LOCKED")
    inference_accumulation_frames: int = Field(default=6, description="Frames to accumulate")

    # Waterfall FFT - can be adjusted
    waterfall_nfft: int = Field(default=65536, description="High resolution for display")
    waterfall_dynamic_range_db: float = Field(default=60.0, description="Display range")

class InferenceSettings(BaseSettings):
    """Inference configuration."""
    model_config = SettingsConfigDict(env_prefix="G20_INFERENCE_")

    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1)
    use_tensorrt: bool = Field(default=True)
    use_dla: bool = Field(default=False, description="Use DLA for power efficiency")
    int8_quantization: bool = Field(default=True)

    # Class names - loaded from registry, but defaults here
    class_names: List[str] = Field(default=["background", "signal"])

class JetsonSettings(BaseSettings):
    """Jetson-specific configuration."""
    model_config = SettingsConfigDict(env_prefix="G20_JETSON_")

    power_mode: PowerMode = Field(default=PowerMode.BALANCED)
    lock_clocks: bool = Field(default=True, description="Run jetson_clocks")
    gpu_memory_limit_gb: float = Field(default=12.0, description="Max GPU memory")
    enable_mps: bool = Field(default=True, description="Multi-Process Service")

class Settings(BaseSettings):
    """Root settings combining all sub-settings."""
    model_config = SettingsConfigDict(
        env_prefix="G20_",
        env_nested_delimiter="__"
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    fft: FFTSettings = Field(default_factory=FFTSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    jetson: JetsonSettings = Field(default_factory=JetsonSettings)

    # Environment
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

# Global singleton
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

### Day 3: Capabilities API

**Create `backend/api/capabilities.py`:**
```python
"""API endpoint exposing backend capabilities to frontend."""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from ..config.settings import get_settings

@dataclass
class FFTCapabilities:
    """Available FFT configurations."""
    inference_nfft: int
    inference_locked: bool = True  # Cannot change without retraining
    waterfall_nfft_options: List[int] = None

    def __post_init__(self):
        if self.waterfall_nfft_options is None:
            self.waterfall_nfft_options = [4096, 8192, 16384, 32768, 65536]

@dataclass
class DisplayCapabilities:
    """Display configuration options."""
    fps_options: List[int] = None
    time_span_options: List[float] = None
    colormap_options: List[str] = None

    def __post_init__(self):
        if self.fps_options is None:
            self.fps_options = [10, 15, 20, 30, 60]
        if self.time_span_options is None:
            self.time_span_options = [1.0, 2.5, 5.0, 10.0]
        if self.colormap_options is None:
            self.colormap_options = ["viridis", "plasma", "inferno", "magma", "turbo"]

@dataclass
class HardwareCapabilities:
    """SDR hardware capabilities."""
    min_freq_mhz: float = 30.0
    max_freq_mhz: float = 6000.0
    bandwidth_options_mhz: List[float] = None
    sample_rate_mhz: float = 20.0

    def __post_init__(self):
        if self.bandwidth_options_mhz is None:
            self.bandwidth_options_mhz = [5.0, 10.0, 20.0, 25.0, 40.0, 50.0]

@dataclass
class BackendCapabilities:
    """Complete backend capabilities for frontend configuration."""
    fft: FFTCapabilities
    display: DisplayCapabilities
    hardware: HardwareCapabilities
    power_modes: List[str]
    version: str

def get_capabilities() -> Dict[str, Any]:
    """Get backend capabilities as JSON-serializable dict."""
    settings = get_settings()

    caps = BackendCapabilities(
        fft=FFTCapabilities(
            inference_nfft=settings.fft.inference_nfft
        ),
        display=DisplayCapabilities(),
        hardware=HardwareCapabilities(),
        power_modes=["15W", "30W", "50W"],
        version="1.0.0"
    )

    return asdict(caps)
```

### Day 4-5: Migrate Hardcoded Values

**Files requiring updates to use centralized config:**

| File | Hardcoded Values | Action |
|------|-----------------|--------|
| `server.py` | Ports, paths, timeouts | Use `get_settings()` |
| `unified_pipeline.py` | FFT sizes, class names | Use `get_settings()` |
| `inference.py` | Class names, thresholds | Use `get_settings()` |
| `training/service.py` | Epochs, patience | Use `get_settings()` |
| `hydra/config.py` | Paths, presets | Use `get_settings()` |

**Example migration for `inference.py`:**
```python
# BEFORE:
class SignalInference:
    def __init__(self):
        self.class_names = ["background", "creamy_chicken"]  # HARDCODED

# AFTER:
from config.settings import get_settings

class SignalInference:
    def __init__(self):
        settings = get_settings()
        self.class_names = settings.inference.class_names
```

### Week 5 Deliverables Checklist
- [ ] `backend/config/settings.py` with Pydantic-settings v2
- [ ] All settings have type validation
- [ ] Environment variable overrides working (`G20_SERVER__GRPC_PORT=50052`)
- [ ] `backend/api/capabilities.py` exposes config to frontend
- [ ] All hardcoded values migrated to settings
- [ ] `config/inference.yaml` now actually loaded (was ignored before!)
- [ ] Frontend queries capabilities API instead of hardcoding
- [ ] Documentation for all config options
---

## WEEK 6: TensorRT Optimization

### Goals
- INT8 quantization with histogram calibration
- CUDA Graph capture for entire FFT→inference pipeline
- DLA testing for power efficiency
- Benchmark latency and memory usage

### Memory Budget Planning (16GB Orin)

```
Total VRAM: 16 GB
├── OS/System:           ~2.0 GB
├── cuFFT Workspace:     ~0.5 GB
├── TensorRT Engine:     ~1.0 GB (INT8)
├── Activation Memory:   ~2.0 GB
├── Safety Margin:       ~1.0 GB
└── Available for Data:  ~9.5 GB
    ├── Triple Buffer:   ~0.5 GB (IQ data)
    ├── Spectrogram:     ~0.5 GB
    └── Detection:       ~8.5 GB available
```

### Day 1-2: INT8 Calibration

**Create `scripts/calibrate_tensorrt.py`:**
```python
"""
INT8 calibration for TensorRT with histogram calibration.
Histogram calibration works better than entropy for RF signal data.
"""
import tensorrt as trt
import numpy as np
from pathlib import Path
import torch

class HistogramCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Histogram-based INT8 calibrator.
    Better than entropy for RF spectrograms with wide dynamic range.
    """

    def __init__(self, calibration_data: list, cache_file: str = "calibration.cache"):
        super().__init__()
        self.calibration_data = calibration_data
        self.cache_file = cache_file
        self.current_idx = 0
        self.batch_size = 1

        # Allocate device memory for calibration batch
        sample = calibration_data[0]
        self.device_input = torch.zeros_like(sample, device='cuda')

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_idx >= len(self.calibration_data):
            return None

        # Copy calibration data to device
        self.device_input.copy_(self.calibration_data[self.current_idx])
        self.current_idx += 1

        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self):
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def build_int8_engine(
    onnx_path: str,
    calibration_data: list,
    output_path: str,
    workspace_mb: int = 2048,
    dla_core: int = -1  # -1 = GPU only, 0 or 1 = use DLA
) -> bool:
    """
    Build INT8 TensorRT engine with histogram calibration.

    Args:
        onnx_path: Path to ONNX model
        calibration_data: List of calibration tensors
        output_path: Path to save .engine file
        workspace_mb: TensorRT workspace size in MB
        dla_core: DLA core to use (-1 for GPU only)
    """
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            return False

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

    # Enable INT8
    config.set_flag(trt.BuilderFlag.INT8)

    # Keep first/last conv in FP16 for accuracy (1-2% mAP improvement)
    config.set_flag(trt.BuilderFlag.FP16)

    # Set calibrator
    calibrator = HistogramCalibrator(calibration_data)
    config.int8_calibrator = calibrator

    # DLA configuration
    if dla_core >= 0:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        print(f"Using DLA core {dla_core} with GPU fallback")

    # Build engine
    print("Building INT8 engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build engine")
        return False

    # Save engine
    with open(output_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"Engine saved to {output_path}")
    return True


def load_calibration_spectrograms(data_dir: Path, num_samples: int = 100) -> list:
    """Load real spectrograms for calibration."""
    calibration_data = []

    # Load from training data
    for npz_file in sorted(data_dir.glob("**/*.npz"))[:num_samples]:
        data = np.load(npz_file)
        if 'spectrogram' in data:
            spec = data['spectrogram']
            # Normalize to [0, 1]
            spec = spec.astype(np.float32) / 255.0
            # Expand to 3 channels
            spec = np.stack([spec, spec, spec], axis=0)
            calibration_data.append(torch.from_numpy(spec).unsqueeze(0))

    print(f"Loaded {len(calibration_data)} calibration samples")
    return calibration_data
```

### Day 3-4: CUDA Graph Capture

**Create `backend/inference/cuda_graph.py`:**
```python
"""
CUDA Graph capture for entire FFT→inference pipeline.
Reduces kernel launch overhead to ~2.5μs + 1ns per node.
"""
import torch
import torch.cuda
from typing import Optional, Callable

class CUDAGraphPipeline:
    """
    Captures entire processing pipeline as CUDA Graph for minimal launch overhead.

    Pipeline stages:
    1. FFT (cuFFT)
    2. Magnitude computation
    3. dB conversion
    4. Normalization
    5. TensorRT inference
    """

    def __init__(self, fft_size: int, num_frames: int = 6):
        self.fft_size = fft_size
        self.num_frames = num_frames
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.graph_output: Optional[torch.Tensor] = None

        # Allocate static buffers (required for graph capture)
        self.input_buffer = torch.zeros(
            num_frames * fft_size,
            dtype=torch.complex64,
            device='cuda'
        )
        self.fft_output = torch.zeros(
            num_frames, fft_size // 2 + 1,
            dtype=torch.complex64,
            device='cuda'
        )
        self.magnitude = torch.zeros(
            num_frames, fft_size // 2 + 1,
            dtype=torch.float32,
            device='cuda'
        )
        self.spectrogram = torch.zeros(
            1, 3, 1024, 1024,  # Batch, Channels, H, W
            dtype=torch.float32,
            device='cuda'
        )

        # Hann window (pre-computed)
        self.window = torch.hann_window(fft_size, device='cuda')

    def capture(self, inference_fn: Callable[[torch.Tensor], torch.Tensor]):
        """
        Capture the entire pipeline as a CUDA Graph.
        Must be called after warmup.
        """
        # Warmup (required before capture)
        print("Warming up pipeline...")
        for _ in range(10):
            self._run_pipeline(inference_fn)
        torch.cuda.synchronize()

        # Capture graph
        print("Capturing CUDA Graph...")
        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self.graph_output = self._run_pipeline(inference_fn)

        print(f"Graph captured with output shape: {self.graph_output.shape}")

    def _run_pipeline(self, inference_fn: Callable) -> torch.Tensor:
        """Run the full pipeline (used during capture and replay)."""
        # Reshape input for batched FFT
        iq_frames = self.input_buffer.view(self.num_frames, self.fft_size)

        # Apply window
        windowed = iq_frames * self.window

        # FFT
        fft_result = torch.fft.rfft(windowed, dim=1)

        # Magnitude
        magnitude = torch.abs(fft_result)

        # dB conversion with floor
        db = 20 * torch.log10(magnitude + 1e-10)

        # Normalize to [0, 1] range (assuming 80 dB dynamic range)
        normalized = (db + 80) / 80
        normalized = torch.clamp(normalized, 0, 1)

        # Resize to inference input (1024x1024)
        # Stack to 3 channels
        spec = normalized.unsqueeze(0).expand(3, -1, -1)
        spec = torch.nn.functional.interpolate(
            spec.unsqueeze(0),
            size=(1024, 1024),
            mode='bilinear',
            align_corners=False
        )

        # Run inference
        detections = inference_fn(spec)

        return detections

    def replay(self, iq_data: torch.Tensor) -> torch.Tensor:
        """
        Replay the captured graph with new input data.
        ~2.5μs launch overhead vs ~50μs for individual kernels.
        """
        if self.graph is None:
            raise RuntimeError("Graph not captured. Call capture() first.")

        # Copy input (the only CPU→GPU transfer)
        self.input_buffer.copy_(iq_data)

        # Replay graph
        self.graph.replay()

        return self.graph_output

    def benchmark(self, iterations: int = 1000) -> dict:
        """Benchmark graph replay performance."""
        import time

        # Generate test input
        test_input = torch.randn(
            self.num_frames * self.fft_size,
            dtype=torch.complex64,
            device='cuda'
        )

        # Warmup
        for _ in range(10):
            self.replay(test_input)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            self.replay(test_input)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / iterations) * 1000

        return {
            'iterations': iterations,
            'total_time_s': elapsed,
            'avg_latency_ms': avg_ms,
            'throughput_fps': 1000 / avg_ms
        }
```

### Day 5: DLA Testing for Power Efficiency

**Create `scripts/benchmark_dla.py`:**
```python
"""
Benchmark DLA vs GPU inference for power efficiency testing.
DLA provides 3-5x better power efficiency for INT8 inference.
In 15W mode, DLA contributes 74% of total DL performance.
"""
import tensorrt as trt
import torch
import time
import subprocess

def get_power_stats() -> dict:
    """Get current power usage via tegrastats."""
    # This is simplified - real implementation would parse tegrastats output
    result = subprocess.run(
        ['tegrastats', '--interval', '100'],
        capture_output=True,
        timeout=1
    )
    # Parse power from output
    return {'gpu_power_mw': 0, 'dla_power_mw': 0, 'total_power_mw': 0}

def benchmark_inference(engine_path: str, iterations: int = 1000) -> dict:
    """Benchmark inference with a TensorRT engine."""
    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers
    input_shape = (1, 3, 1024, 1024)
    input_buffer = torch.zeros(input_shape, dtype=torch.float32, device='cuda')
    output_buffer = torch.zeros((1, 100, 6), dtype=torch.float32, device='cuda')  # Detections

    # Warmup
    for _ in range(10):
        context.execute_v2([input_buffer.data_ptr(), output_buffer.data_ptr()])
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        context.execute_v2([input_buffer.data_ptr(), output_buffer.data_ptr()])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        'engine': engine_path,
        'iterations': iterations,
        'total_time_s': elapsed,
        'avg_latency_ms': (elapsed / iterations) * 1000,
        'throughput_fps': iterations / elapsed
    }

def compare_gpu_vs_dla():
    """Compare GPU-only vs DLA inference."""
    print("=" * 60)
    print("GPU vs DLA Benchmark")
    print("=" * 60)

    # Build engines if not exists
    # ... (engine building code)

    # Benchmark GPU engine
    print("\nBenchmarking GPU engine...")
    gpu_results = benchmark_inference("model_gpu_int8.engine")
    print(f"  Latency: {gpu_results['avg_latency_ms']:.2f} ms")
    print(f"  Throughput: {gpu_results['throughput_fps']:.1f} FPS")

    # Benchmark DLA engine
    print("\nBenchmarking DLA engine...")
    dla_results = benchmark_inference("model_dla_int8.engine")
    print(f"  Latency: {dla_results['avg_latency_ms']:.2f} ms")
    print(f"  Throughput: {dla_results['throughput_fps']:.1f} FPS")

    # Compare
    print("\nComparison:")
    speedup = gpu_results['avg_latency_ms'] / dla_results['avg_latency_ms']
    print(f"  DLA vs GPU latency ratio: {speedup:.2f}x")
    print(f"  (Values > 1 mean DLA is slower but more power efficient)")

    print("\nRecommendations:")
    if dla_results['avg_latency_ms'] < 10:
        print("  ✅ DLA latency acceptable for real-time (<10ms)")
        print("  ✅ Use DLA for field deployment (3-5x power savings)")
    else:
        print("  ⚠️ DLA latency too high for real-time")
        print("  ⚠️ Use GPU for latency-critical applications")

if __name__ == "__main__":
    compare_gpu_vs_dla()
```

### Week 6 Deliverables Checklist
- [ ] `scripts/calibrate_tensorrt.py` with histogram calibration
- [ ] INT8 engine built and validated (accuracy within 1-2% of FP32)
- [ ] `backend/inference/cuda_graph.py` with graph capture
- [ ] Graph replay latency <5ms
- [ ] DLA engine built with GPU fallback
- [ ] Power efficiency benchmarks documented
- [ ] Memory usage within 12GB budget

---

## WEEK 7-8: Test Coverage Sprint

### Goals
- Qodo (CodiumAI) for AI-generated tests on critical functions
- Hypothesis for property-based DSP testing
- Target 70% coverage on backend
- Coverage ratchet in CI

### Testing Strategy

```
Test Pyramid for G20:

        /\
       /  \     E2E Tests (5%)
      /    \    - Full pipeline detection accuracy
     /------\
    /        \  Integration Tests (25%)
   /          \ - WebSocket message flow
  /            \- gRPC service calls
 /--------------\
/                \ Unit Tests (70%)
 - DSP functions (FFT, filters)
 - Coordinate conversion
 - Data models
 - Configuration validation
```

### Day 1-2: Hypothesis Property-Based Tests for DSP

**Create `tests/backend/test_dsp_properties.py`:**
```python
"""
Property-based tests for DSP functions using Hypothesis.
Verifies mathematical invariants that must always hold.
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

from backend.gpu_fft import GPUSpectrogramProcessor
from backend.dsp.filters import design_lowpass_filter
from backend.dsp.subband_extractor import SubbandExtractor

# Strategy for complex IQ data
complex_arrays = arrays(
    dtype=np.complex64,
    shape=st.integers(min_value=1024, max_value=65536),
    elements=st.complex_numbers(
        min_magnitude=0, max_magnitude=1,
        allow_nan=False, allow_infinity=False
    )
)

class TestFFTProperties:
    """Property-based tests for FFT processing."""

    @given(complex_arrays)
    @settings(max_examples=50, deadline=None)
    def test_parseval_theorem(self, iq_data):
        """
        Parseval's theorem: Energy in time domain equals energy in frequency domain.
        ∑|x[n]|² = (1/N) ∑|X[k]|²
        """
        # Compute time-domain energy
        time_energy = np.sum(np.abs(iq_data) ** 2)

        # Compute frequency-domain energy
        fft_result = np.fft.fft(iq_data)
        freq_energy = np.sum(np.abs(fft_result) ** 2) / len(iq_data)

        # Should be equal within numerical precision
        np.testing.assert_allclose(time_energy, freq_energy, rtol=1e-5)

    @given(complex_arrays)
    @settings(max_examples=50, deadline=None)
    def test_fft_linearity(self, iq_data):
        """FFT is linear: FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)"""
        a, b = 2.0, 3.0
        x = iq_data
        y = np.roll(iq_data, 100)  # Shifted version

        # Combined transform
        combined_fft = np.fft.fft(a * x + b * y)

        # Sum of individual transforms
        sum_fft = a * np.fft.fft(x) + b * np.fft.fft(y)

        np.testing.assert_allclose(combined_fft, sum_fft, rtol=1e-5)

    @given(complex_arrays)
    @settings(max_examples=50, deadline=None)
    def test_ifft_inverts_fft(self, iq_data):
        """IFFT(FFT(x)) = x (within numerical precision)."""
        reconstructed = np.fft.ifft(np.fft.fft(iq_data))
        np.testing.assert_allclose(reconstructed, iq_data, rtol=1e-5)


class TestFilterProperties:
    """Property-based tests for filter design."""

    @given(
        st.floats(min_value=0.01, max_value=0.49),  # Normalized cutoff
        st.integers(min_value=60, max_value=100)     # Stopband attenuation
    )
    @settings(max_examples=30, deadline=None)
    def test_filter_passband_gain(self, cutoff, stopband_db):
        """Passband gain should be approximately 0 dB."""
        taps = design_lowpass_filter(
            cutoff_normalized=cutoff,
            stopband_db=stopband_db,
            transition_width=0.1
        )

        # Compute frequency response
        from scipy.signal import freqz
        w, h = freqz(taps, worN=8000)

        # Find passband (< 0.8 * cutoff)
        passband_mask = w < (0.8 * cutoff * np.pi)
        passband_gain_db = 20 * np.log10(np.abs(h[passband_mask]) + 1e-10)

        # Passband ripple should be < 1 dB
        assert np.max(np.abs(passband_gain_db)) < 1.0

    @given(
        st.floats(min_value=0.01, max_value=0.49),
        st.integers(min_value=60, max_value=100)
    )
    @settings(max_examples=30, deadline=None)
    def test_filter_stopband_attenuation(self, cutoff, stopband_db):
        """Stopband should achieve specified attenuation."""
        taps = design_lowpass_filter(
            cutoff_normalized=cutoff,
            stopband_db=stopband_db,
            transition_width=0.1
        )

        from scipy.signal import freqz
        w, h = freqz(taps, worN=8000)

        # Find stopband (> cutoff + transition)
        stopband_mask = w > ((cutoff + 0.1) * np.pi)
        if np.any(stopband_mask):
            stopband_gain_db = 20 * np.log10(np.abs(h[stopband_mask]) + 1e-10)
            # Should achieve at least 90% of specified attenuation
            assert np.max(stopband_gain_db) < -0.9 * stopband_db


class TestSubbandExtraction:
    """Property-based tests for subband extraction."""

    @given(
        st.floats(min_value=1e6, max_value=10e6),   # Bandwidth Hz
        st.floats(min_value=-5e6, max_value=5e6),  # Center offset Hz
    )
    @settings(max_examples=20, deadline=None)
    def test_output_sample_count(self, bandwidth_hz, offset_hz):
        """Output sample count should match decimation ratio."""
        assume(abs(offset_hz) < 8e6)  # Within Nyquist

        source_rate = 20e6
        input_samples = 100000

        # Calculate expected decimation
        decimation = int(source_rate / bandwidth_hz)
        expected_output = input_samples // decimation

        # Create test input
        test_input = np.zeros(input_samples, dtype=np.complex64)

        # This would call the actual extractor
        # output = extractor.extract(test_input)
        # assert len(output) == expected_output
        pass  # Placeholder for actual implementation
```

### Day 3-4: Qodo AI-Generated Tests

**Run Qodo on critical functions:**
```bash
# Install Qodo CLI
pip install qodo

# Generate tests for critical modules
qodo test backend/training/sample_manager.py --output tests/backend/test_sample_manager.py
qodo test backend/hydra/detector.py --output tests/backend/test_detector.py
qodo test backend/inference.py --output tests/backend/test_inference.py
```

**Example Qodo-generated test structure:**
```python
# tests/backend/test_sample_manager.py
# Generated by Qodo with manual enhancements

import pytest
import numpy as np
from pathlib import Path
from backend.training.sample_manager import SampleManager

class TestSampleManager:
    @pytest.fixture
    def sample_manager(self, tmp_path):
        return SampleManager(training_dir=tmp_path)

    @pytest.fixture
    def valid_iq_data(self):
        """Valid base64-encoded IQ data."""
        import base64
        iq = np.random.randn(4096).astype(np.complex64)
        return base64.b64encode(iq.tobytes()).decode()

    def test_save_sample_creates_files(self, sample_manager, valid_iq_data, tmp_path):
        """Verify save_sample creates NPZ and JSON files."""
        sample_id, is_new = sample_manager.save_sample(
            signal_name="test_signal",
            iq_data_b64=valid_iq_data,
            boxes=[{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}],
            metadata={"fft_size": 4096}
        )

        assert is_new is True
        assert (tmp_path / "test_signal" / "samples" / f"{sample_id}.npz").exists()
        assert (tmp_path / "test_signal" / "samples" / f"{sample_id}.json").exists()

    def test_save_sample_deterministic_id(self, sample_manager, valid_iq_data):
        """Same input should produce same sample ID."""
        boxes = [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}]

        id1, _ = sample_manager.save_sample("sig", valid_iq_data, boxes, {})
        id2, _ = sample_manager.save_sample("sig", valid_iq_data, boxes, {})

        assert id1 == id2  # Deterministic

    def test_save_sample_rejects_invalid_boxes(self, sample_manager, valid_iq_data):
        """Invalid box coordinates should raise ValueError."""
        invalid_boxes = [{"x_min": 0.5, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}]  # x_min > x_max

        with pytest.raises(ValueError):
            sample_manager.save_sample("sig", valid_iq_data, invalid_boxes, {})
```

### Day 5: Coverage Ratchet in CI

**Create `.github/workflows/test.yml`:**
```yaml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-backend:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r backend/requirements.txt
          pip install pytest pytest-cov hypothesis

      - name: Run tests with coverage
        run: |
          pytest tests/backend/ \
            --cov=backend \
            --cov-report=xml \
            --cov-report=term-missing \
            --cov-fail-under=70

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  test-flutter:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: subosito/flutter-action@v2
        with:
          flutter-version: '3.29.0'

      - name: Run Flutter tests
        run: flutter test --coverage

      - name: Check coverage
        run: |
          # Parse lcov and check coverage
          pip install lcov_cobertura
          lcov_cobertura coverage/lcov.info -o coverage.xml

  lint:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Run Ruff
        uses: chartboost/ruff-action@v1

      - name: Run Dart analyze
        run: |
          flutter pub get
          dart analyze lib/
```

### Week 7-8 Deliverables Checklist
- [ ] `tests/backend/test_dsp_properties.py` with Hypothesis tests
- [ ] Qodo-generated tests for sample_manager, detector, inference
- [ ] All critical coordinate conversion functions tested
- [ ] pytest-cov configured with 70% threshold
- [ ] Coverage ratchet in CI (fails if coverage drops)
- [ ] GitHub Actions workflow for tests + linting
- [ ] Hardware-in-loop test markers (`pytest -m "hardware"`)
- [ ] Test fixtures for mock RF signals
---

## WEEK 9: Flutter Cleanup & Performance

### Goals
- FFI for performance-critical paths
- Fragment shader colormap on GPU
- Fix async file I/O everywhere
- Impeller verification (GPU-accelerated rendering)

### Day 1-2: FFI for Native Performance

**Flutter 3.29+ allows FFI calls on platform thread - no async boundary overhead!**

**Create `lib/native/ffi_bindings.dart`:**
```dart
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

/// Native FFI bindings for performance-critical operations.
/// FFI calls are now allowed on platform thread (Flutter 3.29+).
class NativeBindings {
  static late DynamicLibrary _lib;
  static bool _initialized = false;

  static void initialize() {
    if (_initialized) return;

    if (Platform.isWindows) {
      _lib = DynamicLibrary.open('g20_native.dll');
    } else if (Platform.isLinux) {
      _lib = DynamicLibrary.open('libg20_native.so');
    } else {
      throw UnsupportedError('Platform not supported');
    }

    _initialized = true;
  }

  /// Apply colormap to spectrogram data on native side.
  /// Much faster than Dart loops for large arrays.
  static void applyColormap(
    Float32List dbData,
    Uint8List rgbaOutput,
    int colormapId,
    double dbMin,
    double dbMax,
  ) {
    final applyColormapNative = _lib.lookupFunction<
      Void Function(Pointer<Float>, Pointer<Uint8>, Int32, Int32, Float, Float),
      void Function(Pointer<Float>, Pointer<Uint8>, int, int, double, double)
    >('apply_colormap');

    final dbPtr = calloc<Float>(dbData.length);
    final rgbaPtr = calloc<Uint8>(rgbaOutput.length);

    try {
      // Copy input
      dbPtr.asTypedList(dbData.length).setAll(0, dbData);

      // Call native
      applyColormapNative(
        dbPtr, rgbaPtr, dbData.length, colormapId, dbMin, dbMax
      );

      // Copy output
      rgbaOutput.setAll(0, rgbaPtr.asTypedList(rgbaOutput.length));
    } finally {
      calloc.free(dbPtr);
      calloc.free(rgbaPtr);
    }
  }

  /// Batch decode multiple FlatBuffer frames.
  static List<Map<String, dynamic>> decodeFlatBufferBatch(Uint8List data) {
    // Native FlatBuffer decoding is ~10x faster than Dart
    // Implementation would use flatbuffers C library
    throw UnimplementedError('Native FlatBuffer decoding');
  }
}
```

**Native C implementation `native/g20_native.c`:**
```c
#include <stdint.h>
#include <math.h>

// Viridis colormap LUT (256 entries)
static const uint8_t VIRIDIS_LUT[256][3] = {
    {68, 1, 84}, {68, 2, 86}, /* ... full LUT ... */ {253, 231, 37}
};

void apply_colormap(
    const float* db_data,
    uint8_t* rgba_output,
    int length,
    int colormap_id,
    float db_min,
    float db_max
) {
    const float range = db_max - db_min;

    for (int i = 0; i < length; i++) {
        // Normalize to [0, 255]
        float normalized = (db_data[i] - db_min) / range;
        if (normalized < 0) normalized = 0;
        if (normalized > 1) normalized = 1;

        int idx = (int)(normalized * 255);

        // Apply colormap
        rgba_output[i * 4 + 0] = VIRIDIS_LUT[idx][0];  // R
        rgba_output[i * 4 + 1] = VIRIDIS_LUT[idx][1];  // G
        rgba_output[i * 4 + 2] = VIRIDIS_LUT[idx][2];  // B
        rgba_output[i * 4 + 3] = 255;                   // A
    }
}
```

### Day 3: Fragment Shader Colormap

**Create `lib/shaders/colormap.frag`:**
```glsl
#version 320 es
precision highp float;

// Spectrogram texture (grayscale dB values normalized to 0-1)
uniform sampler2D u_spectrogram;

// Colormap texture (256x1 RGB)
uniform sampler2D u_colormap;

// Parameters
uniform float u_db_min;
uniform float u_db_max;

in vec2 v_texCoord;
out vec4 fragColor;

void main() {
    // Sample spectrogram (grayscale)
    float db_normalized = texture(u_spectrogram, v_texCoord).r;

    // Clamp and remap to colormap index
    float colormap_idx = clamp(db_normalized, 0.0, 1.0);

    // Sample colormap (1D texture lookup)
    vec3 color = texture(u_colormap, vec2(colormap_idx, 0.5)).rgb;

    fragColor = vec4(color, 1.0);
}
```

**Integrate with Flutter's CustomPainter:**
```dart
import 'dart:ui' as ui;
import 'package:flutter/material.dart';

class ShaderSpectrogramPainter extends CustomPainter {
  final ui.Image spectrogramTexture;
  final ui.Image colormapTexture;
  final ui.FragmentProgram? shaderProgram;

  ShaderSpectrogramPainter({
    required this.spectrogramTexture,
    required this.colormapTexture,
    this.shaderProgram,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (shaderProgram != null) {
      // Use GPU shader for colormap
      final shader = shaderProgram!.fragmentShader();
      shader.setImageSampler(0, spectrogramTexture);
      shader.setImageSampler(1, colormapTexture);
      shader.setFloat(0, 0.0);  // db_min
      shader.setFloat(1, 1.0);  // db_max

      canvas.drawRect(
        Rect.fromLTWH(0, 0, size.width, size.height),
        Paint()..shader = shader,
      );
    } else {
      // Fallback to texture
      canvas.drawImageRect(
        spectrogramTexture,
        Rect.fromLTWH(0, 0, spectrogramTexture.width.toDouble(),
                      spectrogramTexture.height.toDouble()),
        Rect.fromLTWH(0, 0, size.width, size.height),
        Paint(),
      );
    }
  }

  @override
  bool shouldRepaint(covariant ShaderSpectrogramPainter old) {
    return spectrogramTexture != old.spectrogramTexture;
  }
}
```

### Day 4-5: Impeller Verification & State Batching

**Verify Impeller is active:**
```dart
void main() {
  // Impeller is default in Flutter 3.16+ iOS, 3.22+ Android
  // For desktop, may need explicit flag

  debugPrint('Impeller enabled: ${ui.PlatformDispatcher.instance.impellerEnabled}');

  runApp(const G20App());
}
```

**Batch state updates to 60fps:**
```dart
class SpectrogramStateNotifier extends StateNotifier<SpectrogramState> {
  Timer? _batchTimer;
  final List<SpectrogramRow> _pendingRows = [];

  SpectrogramStateNotifier() : super(SpectrogramState.initial());

  void addRow(SpectrogramRow row) {
    _pendingRows.add(row);

    // Start batch timer if not running
    _batchTimer ??= Timer(const Duration(milliseconds: 16), _flushBatch);
  }

  void _flushBatch() {
    if (_pendingRows.isEmpty) return;

    // Update state with all pending rows at once
    state = state.copyWith(
      rows: [...state.rows, ..._pendingRows],
      lastUpdate: DateTime.now(),
    );

    _pendingRows.clear();
    _batchTimer = null;
  }

  @override
  void dispose() {
    _batchTimer?.cancel();
    super.dispose();
  }
}
```

### Week 9 Deliverables Checklist
- [ ] FFI bindings for colormap application
- [ ] Native C library built for Windows/Linux
- [ ] Fragment shader colormap working
- [ ] Impeller enabled and verified
- [ ] State updates batched to 60fps
- [ ] No per-sample rebuilds
- [ ] Memory profile shows no Flutter-side leaks

---

## WEEK 10: CI/CD & Production Hardening

### Goals
- ARM64 GitHub runners for native builds
- nvpmodel/jetson_clocks startup script
- A/B partition for safe updates
- Production monitoring with jtop/Prometheus

### Day 1-2: GitHub Actions with ARM64 Runners

**Create `.github/workflows/build.yml`:**
```yaml
name: Build & Deploy

on:
  push:
    tags: ['v*']
  workflow_dispatch:

jobs:
  build-backend:
    # GitHub native ARM64 runners (no QEMU emulation!)
    runs-on: ubuntu-24.04-arm

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r backend/requirements.txt
          pip install pyinstaller

      - name: Build backend executable
        run: |
          cd backend
          pyinstaller --onefile --name g20_backend server.py

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: backend-arm64
          path: backend/dist/g20_backend

  build-flutter:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: subosito/flutter-action@v2
        with:
          flutter-version: '3.29.0'

      - name: Build Linux release
        run: |
          flutter build linux --release

      - name: Package with native libraries
        run: |
          mkdir -p dist
          cp -r build/linux/x64/release/bundle/* dist/
          cp native/libg20_native.so dist/lib/
          tar -czvf g20-linux-x64.tar.gz dist/

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: flutter-linux
          path: g20-linux-x64.tar.gz

  build-tensorrt-engine:
    # Build on actual Jetson hardware (self-hosted runner)
    runs-on: [self-hosted, jetson-orin]

    steps:
      - uses: actions/checkout@v4

      - name: Build TensorRT engines
        run: |
          cd scripts
          python calibrate_tensorrt.py \
            --onnx models/detector.onnx \
            --calibration-data training_data/signals/ \
            --output models/detector_int8.engine \
            --dla-core 0

      - name: Upload engines
        uses: actions/upload-artifact@v4
        with:
          name: tensorrt-engines
          path: models/*.engine
```

### Day 3: Production Startup Script

**Create `scripts/jetson_startup.sh`:**
```bash
#!/bin/bash
# G20 Production Startup Script for Jetson Orin
# Run as root or with sudo

set -e

echo "=== G20 Production Startup ==="
echo "Date: $(date)"

# 1. Set power mode (30W balanced for field deployment)
echo "[1/6] Setting power mode to 30W..."
nvpmodel -m 2  # MODE_30W
sleep 2

# 2. Lock clocks for consistent performance
echo "[2/6] Locking clocks..."
jetson_clocks
sleep 1

# 3. Start MPS for multi-process GPU access
echo "[3/6] Starting MPS..."
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

# Set memory limits for MPS
echo "set_default_device_pinned_mem_limit 0 6G" | nvidia-cuda-mps-control

# 4. Configure memory limits
echo "[4/6] Configuring memory..."
export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=6G"

# 5. Start monitoring
echo "[5/6] Starting monitoring..."
# Start tegrastats logging
tegrastats --interval 1000 --logfile /var/log/g20/tegrastats.log &

# Start jtop Prometheus exporter (if installed)
if command -v jtop &> /dev/null; then
    jtop --prometheus --port 9100 &
fi

# 6. Launch G20 backend
echo "[6/6] Starting G20 backend..."
cd /opt/g20
./g20_backend \
    --config /etc/g20/config.yaml \
    --log-level INFO \
    --log-file /var/log/g20/backend.log &

BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Write PID file
echo $BACKEND_PID > /var/run/g20/backend.pid

# Wait for backend to be ready
echo "Waiting for backend..."
for i in {1..30}; do
    if curl -s http://localhost:50052/health > /dev/null 2>&1; then
        echo "Backend ready!"
        break
    fi
    sleep 1
done

echo "=== G20 Startup Complete ==="
echo "Monitoring: http://localhost:9100/metrics"
echo "WebSocket: ws://localhost:50052"
```

**Create systemd service `/etc/systemd/system/g20.service`:**
```ini
[Unit]
Description=G20 RF Signal Detection
After=network.target nvidia-persistenced.service
Requires=nvidia-persistenced.service

[Service]
Type=forking
ExecStart=/opt/g20/scripts/jetson_startup.sh
ExecStop=/opt/g20/scripts/jetson_shutdown.sh
PIDFile=/var/run/g20/backend.pid
Restart=on-failure
RestartSec=10
User=g20
Group=g20

# Resource limits
MemoryMax=14G
CPUQuota=800%

# Logging
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Day 4: A/B Partition for Safe Updates

**Configure A/B partitioning during flash:**
```bash
# During Jetson flash, enable A/B partitioning
sudo ./flash.sh -r --ROOTFS_AB=1 jetson-agx-orin-devkit mmcblk0p1
```

**Create update script `scripts/safe_update.sh`:**
```bash
#!/bin/bash
# Safe A/B partition update for G20

set -e

NEW_VERSION=$1
if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

echo "=== G20 Safe Update to $NEW_VERSION ==="

# 1. Determine current/inactive slots
CURRENT_SLOT=$(nvbootctrl get-current-slot)
if [ "$CURRENT_SLOT" = "0" ]; then
    INACTIVE_SLOT=1
else
    INACTIVE_SLOT=0
fi

echo "Current slot: $CURRENT_SLOT"
echo "Target slot: $INACTIVE_SLOT"

# 2. Download new version to inactive partition
echo "Downloading update..."
wget -O /tmp/g20-$NEW_VERSION.tar.gz \
    "https://releases.example.com/g20/g20-$NEW_VERSION-arm64.tar.gz"

# 3. Extract to inactive rootfs
INACTIVE_MOUNT="/mnt/slot${INACTIVE_SLOT}"
mkdir -p $INACTIVE_MOUNT
mount /dev/mmcblk0p$((INACTIVE_SLOT + 1)) $INACTIVE_MOUNT

echo "Installing to inactive slot..."
tar -xzf /tmp/g20-$NEW_VERSION.tar.gz -C $INACTIVE_MOUNT/opt/g20/

# 4. Update version file
echo "$NEW_VERSION" > $INACTIVE_MOUNT/opt/g20/VERSION

# 5. Unmount
umount $INACTIVE_MOUNT

# 6. Mark inactive slot as bootable
nvbootctrl set-slot-as-unbootable $INACTIVE_SLOT
nvbootctrl set-active-boot-slot $INACTIVE_SLOT

echo "Update staged. Rebooting to apply..."
echo "If boot fails, system will automatically rollback."

# 7. Reboot
reboot
```

### Day 5: Production Monitoring

**Prometheus metrics export via jtop:**
```yaml
# /etc/prometheus/prometheus.yml
scrape_configs:
  - job_name: 'jetson'
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 5s

  - job_name: 'g20'
    static_configs:
      - targets: ['localhost:50052']
    metrics_path: /metrics
    scrape_interval: 5s
```

**Alert rules `alerts/g20_alerts.yml`:**
```yaml
groups:
  - name: g20_alerts
    rules:
      - alert: GPUTemperatureHigh
        expr: jetson_gpu_temp > 85
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "GPU temperature above 85°C"

      - alert: MemoryUsageHigh
        expr: jetson_mem_used / jetson_mem_total > 0.75
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Memory usage above 75% ({{$value | humanizePercentage}})"

      - alert: InferenceLatencyHigh
        expr: g20_inference_latency_ms > 15
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Inference latency above 15ms"

      - alert: DetectionRateLow
        expr: rate(g20_detections_total[5m]) < 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Detection rate below 0.1/sec for 10 minutes"
```

### Week 10 Deliverables Checklist
- [ ] GitHub Actions with ARM64 runners
- [ ] Self-hosted Jetson runner for TensorRT builds
- [ ] `jetson_startup.sh` with nvpmodel + jetson_clocks
- [ ] systemd service file
- [ ] A/B partitioning configured
- [ ] `safe_update.sh` for OTA updates
- [ ] Prometheus metrics export
- [ ] Alert rules for GPU temp, memory, latency
- [ ] Grafana dashboard for monitoring

---

## Production Deployment Checklist

### Pre-Deployment Verification

```
[ ] All tests passing (70%+ coverage)
[ ] Ruff linting clean
[ ] No sync file I/O in Flutter
[ ] Memory usage under 12GB sustained
[ ] Inference latency <10ms
[ ] Zero-copy pipeline verified
[ ] TensorRT INT8 engine accuracy validated
```

### Deployment Steps

```bash
# 1. Flash Jetson with A/B partitioning
sudo ./flash.sh -r --ROOTFS_AB=1 jetson-agx-orin-devkit mmcblk0p1

# 2. Install G20 package
sudo dpkg -i g20-1.0.0-arm64.deb

# 3. Configure
sudo cp /opt/g20/config/production.yaml /etc/g20/config.yaml
sudo vim /etc/g20/config.yaml  # Adjust for deployment

# 4. Enable service
sudo systemctl enable g20
sudo systemctl start g20

# 5. Verify
sudo systemctl status g20
curl http://localhost:50052/health
```

### Expected Production Latencies

| Operation | Target | Measured |
|-----------|--------|----------|
| IQ to GPU (zero-copy) | <10μs | ___μs |
| cuFFT 4096-point | <100μs | ___μs |
| TensorRT INT8 inference | <5ms | ___ms |
| Shared memory IQ read | <1μs | ___μs |
| FlatBuffers decode | <50ns | ___ns |
| End-to-end pipeline | <10ms | ___ms |

### Monitoring Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| GPU Temperature | 85°C | 90°C |
| Memory Usage | 12GB (75%) | 14GB (87%) |
| Inference Latency | 10ms | 15ms |
| Detection Rate | <1/min | <0.1/min |
| Error Rate | >1% | >5% |

---

## Summary: 10-Week Transformation

| Week | Focus | Key Deliverables |
|------|-------|------------------|
| 1 | Quick Wins | Ruff, logging, dead code removal, async I/O |
| 2 | Zero-Copy Pipeline | cudaHostAlloc, triple buffer, CUDA streams |
| 3 | Split server.py | 8-10 modules, command pattern, DI container |
| 4 | IPC Overhaul | Shared memory, FlatBuffers, Unix sockets |
| 5 | Config Centralization | Pydantic-settings, capabilities API |
| 6 | TensorRT Optimization | INT8, CUDA Graphs, DLA testing |
| 7-8 | Test Coverage Sprint | Hypothesis, Qodo, 70% coverage, CI |
| 9 | Flutter Cleanup | FFI, shaders, Impeller, state batching |
| 10 | CI/CD + Hardening | ARM64 runners, A/B updates, monitoring |

### Final Metrics Target

```
Before Hardening:
├── Test Coverage: <1%
├── God Modules: 3
├── Duplicate Lines: 770
├── Print Statements: 200+
└── Pipeline Latency: ~50ms

After Hardening:
├── Test Coverage: 70%+
├── God Modules: 0
├── Duplicate Lines: 0
├── Print Statements: 0 (proper logging)
└── Pipeline Latency: <10ms
```

---

**Document Version:** 1.0
**Created:** January 27, 2026
**Platform:** Jetson AGX Orin (16GB)
**Flutter:** 3.29+
**Python:** 3.10+
**TensorRT:** 8.6+

**To merge these files:**
```bash
cat Harden_roadmap_1.md Harden_roadmap_2.md Harden_roadmap_3a.md \
    Harden_roadmap_3b.md Harden_roadmap_4.md Harden_roadmap_5.md \
    > Harden_roadmap.md
```
# G20 Roadmap Revision — Based on Profiling

**READ THIS AFTER `Harden_roadmap_0_CORRECTIONS.md`**

This document supersedes specific sections of the original roadmap based on actual profiling data.

---

## Profiling Results (Dev Machine Baseline)

```
SERIALIZATION:
  raw_bytes_24k_complex64: 0.006ms
  json_1k_floats: 0.297ms
  struct_header_pack: 0.000ms
  frame_serialize_24k_complex64: 0.006ms

ASYNC_OVERHEAD:
  async_call_overhead: 0.418ms / 1000 calls
  queue_overhead: 0.766ms / 1000 ops

MEMORY:
  leak_detected: False
```

---

## REMOVED FROM ROADMAP

### ❌ FlatBuffers Integration — REMOVED

**Reason:** Raw bytes serialization is 0.006ms. FlatBuffers adds schema overhead with no speed benefit for array data.

**Do not:**
- Add FlatBuffers dependency
- Create .fbs schema files
- Generate FlatBuffers bindings

**Instead:**
```python
# IQ frames — use raw bytes + struct header
header = struct.pack('=4sQII', b'IQ01', timestamp_us, frame_idx, num_samples)
message = header + iq_array.tobytes()

# Detections — use struct pack for fixed-size data
det_format = '=ffffif'  # x1, y1, x2, y2, class_id, confidence
det_bytes = b''.join(struct.pack(det_format, d.x1, d.y1, d.x2, d.y2, d.class_id, d.conf) for d in detections)
```

---

### ❌ Shared Memory IPC Overhaul — REMOVED

**Reason:** Current serialization is 0.006ms. Shared memory would save ~5μs. Irrelevant to total latency budget.

**Do not:**
- Implement POSIX shared memory ring buffers
- Add `multiprocessing.shared_memory` infrastructure
- Replace WebSocket/gRPC with shared memory for local IPC

**Keep:**
- Existing WebSocket for frontend communication
- Existing gRPC for service communication
- Raw bytes over existing transports

---

### ❌ Lock-free Data Structures — REMOVED

**Reason:** Queue overhead is 0.8μs per operation. Not a bottleneck.

**Do not:**
- Implement custom lock-free queues
- Add `faster-fifo` or similar dependencies

**Keep:**
- `asyncio.Queue` for async coordination
- Standard `queue.Queue` for thread communication

---

## RETAINED IN ROADMAP

### ✅ Week 1: Quick Wins
No changes. Proceed as planned:
- Ruff + pre-commit
- Delete dead code
- Replace print() with logging
- Fix sync file I/O in Flutter

### ✅ Week 3: Split server.py
No changes. Proceed as planned:
- Extract modules using strangler fig pattern
- Simple constructor injection (no DI framework)

### ✅ Week 5: Config Centralization
No changes. Proceed as planned:
- Pydantic-settings v2
- Single source of truth
- Capabilities API

### ✅ Weeks 7-8: Testing
No changes. Proceed as planned:
- Smoke tests first
- Integration tests
- Unit tests
- 70% coverage target

### ⏸️ Week 2: Zero-Copy Pipeline — DEFERRED
**Wait for Jetson.** Cannot profile GPU transfer on discrete GPU.

### ⏸️ Week 6: TensorRT Optimization — DEFERRED
**Wait for Jetson.** Engine is device-specific.

### ⏸️ Week 10: Deployment Hardening — DEFERRED
**Wait for Jetson.** Jetson-specific scripts.

---

## REVISED TIMELINE

| Week | Task | Status |
|------|------|--------|
| 1 | Quick wins | **DO NOW** |
| 2 | ~~Zero-copy~~ | WAIT FOR JETSON |
| 3-4 | Split server.py | **DO NOW** |
| 5 | Config centralization | **DO NOW** |
| 6 | ~~TensorRT~~ | WAIT FOR JETSON |
| 7-8 | Testing | **DO NOW** |
| 9 | Flutter cleanup | **DO NOW** |
| 10 | ~~Deployment~~ | WAIT FOR JETSON |

**Effective work before Jetson:** Weeks 1, 3-5, 7-9 (code quality)
**Work after Jetson arrives:** Weeks 2, 6, 10 (GPU optimization + deployment)

---

## SERIALIZATION PROTOCOL — FINAL SPEC

Use this for all IQ data transport:

```python
import struct
import numpy as np

# Frame header format (24 bytes)
HEADER_FORMAT = '=4sQIIf'  # magic, timestamp_us, frame_idx, num_samples, reserved
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

def serialize_iq_frame(timestamp_us: int, frame_idx: int, iq_data: np.ndarray) -> bytes:
    """Serialize IQ frame with header. ~0.006ms for 24k samples."""
    header = struct.pack(HEADER_FORMAT, b'IQ01', timestamp_us, frame_idx, len(iq_data), 0.0)
    return header + iq_data.astype(np.complex64).tobytes()

def deserialize_iq_frame(data: bytes) -> tuple:
    """Deserialize IQ frame. Zero-copy numpy view."""
    magic, timestamp_us, frame_idx, num_samples, _ = struct.unpack_from(HEADER_FORMAT, data)
    if magic != b'IQ01':
        raise ValueError(f"Bad magic: {magic}")
    iq_data = np.frombuffer(data, dtype=np.complex64, offset=HEADER_SIZE, count=num_samples)
    return timestamp_us, frame_idx, iq_data
```

For detections (variable count per frame):

```python
from dataclasses import dataclass

@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float

# Detection format (24 bytes each)
DET_FORMAT = '=ffffif'  # x1, y1, x2, y2, class_id, confidence
DET_SIZE = struct.calcsize(DET_FORMAT)

def serialize_detections(detections: list) -> bytes:
    """Serialize detection list. ~0.001ms for 10 detections."""
    header = struct.pack('=I', len(detections))
    body = b''.join(
        struct.pack(DET_FORMAT, d.x1, d.y1, d.x2, d.y2, d.class_id, d.confidence)
        for d in detections
    )
    return header + body

def deserialize_detections(data: bytes) -> list:
    """Deserialize detection list."""
    count = struct.unpack_from('=I', data)[0]
    detections = []
    offset = 4
    for _ in range(count):
        x1, y1, x2, y2, class_id, conf = struct.unpack_from(DET_FORMAT, data, offset)
        detections.append(Detection(x1, y1, x2, y2, class_id, conf))
        offset += DET_SIZE
    return detections
```

---

## WHAT TO DO NOW (Pre-Jetson)

### Immediate Actions

1. **Run Ruff on codebase**
   ```bash
   pip install ruff
   ruff check backend/ --fix
   ruff format backend/
   ```

2. **Delete dead files**
   - `backend/waterfall_buffer.py`
   - `junk.txt`, `junk.txt222`
   - Any unused imports

3. **Replace print() with logging**
   ```python
   # Before
   print(f"Processing frame {idx}")

   # After
   logger.info("Processing frame %d", idx)
   ```

4. **Fix sync file I/O in Flutter**
   ```dart
   // Before
   final content = file.readAsStringSync();

   // After
   final content = await file.readAsString();
   ```

5. **Add pre-commit hooks**
   ```yaml
   # .pre-commit-config.yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.4.4
       hooks:
         - id: ruff
         - id: ruff-format
   ```

---

## SUMMARY

**Removed (profiling proved unnecessary):**
- FlatBuffers
- Shared memory IPC
- Lock-free queues
- CUDA Graphs (likely)

**Retained:**
- Code quality work (weeks 1, 3-5, 7-9)
- Raw bytes + struct pack for serialization
- Simple constructor injection

**Deferred (need Jetson):**
- GPU optimization (weeks 2, 6, 10)
- TensorRT engine building
- nvpmodel/jetson_clocks scripts

---

## Re-profile on Jetson

When Jetson arrives, run:
```bash
python scripts/profile_baseline.py
```

Compare results to validate assumptions. If serialization is still <1ms on Jetson ARM, the simplified approach holds.
