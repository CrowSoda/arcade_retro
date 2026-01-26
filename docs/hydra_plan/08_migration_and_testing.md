# Migration Plan & Testing Requirements

## Migration Steps

### Phase 0: Preparation (Day 1)

**Checklist:**
- [ ] Backup current `models/creamy_chicken_fold3.pth`
- [ ] Document current model hash: `sha256sum models/creamy_chicken_fold3.pth`
- [ ] Verify model file size (~100MB)
- [ ] Create backup directory: `mkdir models/legacy`
- [ ] Copy model to backup: `cp models/creamy_chicken_fold3.pth models/legacy/`
- [ ] Verify CUDA works: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Install any missing dependencies

**Rollback:** If any step fails, original model remains unchanged.

---

### Phase 1: Directory Setup (Day 1)

```bash
# Create new directory structure
cd g20_demo

# Create directories
mkdir -p models/backbone
mkdir -p models/heads
mkdir -p training_data/signals

# Create backend modules
mkdir -p backend/hydra
mkdir -p backend/training

# Create __init__.py files
touch backend/hydra/__init__.py
touch backend/training/__init__.py
```

**Verify:**
```bash
ls -la models/
# Should show: backbone/, heads/, legacy/, creamy_chicken_fold3.pth
```

---

### Phase 2: Backbone Extraction (Day 2-3)

**Step 1: Copy extractor code**
```bash
# Create the extractor file
# (copy code from 03_backend_phase1.md)
```

**Step 2: Run extraction**
```bash
cd g20_demo
python -m backend.hydra.backbone_extractor \
    --input models/creamy_chicken_fold3.pth \
    --output-dir models/ \
    --signal-name creamy_chicken \
    --validate \
    --backup
```

**Step 3: Verify**
```bash
# Check file sizes
ls -lh models/backbone/v1.pth        # ~55MB
ls -lh models/heads/creamy_chicken/v1.pth  # ~10MB

# Check metadata
cat models/backbone/metadata.json
cat models/heads/creamy_chicken/metadata.json
cat models/registry.json

# Verify symlinks
ls -la models/backbone/active.pth
ls -la models/heads/creamy_chicken/active.pth
```

**Rollback:** Delete `models/backbone`, `models/heads`, `models/registry.json`

---

### Phase 3: Hydra Detector Integration (Day 4-5)

**Step 1: Create detector files**
- Copy `detector.py` from plan
- Copy `config.py` from plan
- Copy `version_manager.py` from plan

**Step 2: Create integration test**
```python
# test_hydra_detector.py
from backend.hydra.detector import HydraDetector
import torch

detector = HydraDetector("models/")
detector.load_backbone()
detector.load_heads(["creamy_chicken"])

# Test inference
spec = torch.randn(1, 3, 1024, 1024)
results = detector.detect(spec, score_threshold=0.3)
print(f"Detections: {results}")
print(f"Timing: {detector.get_timing_stats()}")
```

**Step 3: Modify unified_pipeline.py**

Add feature flag for gradual rollout:
```python
# At top of unified_pipeline.py
USE_HYDRA_DETECTOR = True  # Toggle for rollback

# In __init__
if USE_HYDRA_DETECTOR:
    from backend.hydra.detector import HydraDetector
    self.detector = HydraDetector("models/")
    self.detector.load_backbone()
else:
    self._load_model(model_path)  # Original code
```

**Verify:**
- Start backend: `python backend/server.py`
- Connect Flutter app
- Detection should work exactly as before
- Check logs for timing info

**Rollback:** Set `USE_HYDRA_DETECTOR = False`

---

### Phase 4: Training Service (Day 6-8)

**Step 1: Create training files**
- Copy `service.py` from plan
- Copy `dataset.py` (implement basic version)
- Copy `splits.py` (implement basic version)

**Step 2: Create training data directory for testing**
```bash
mkdir -p training_data/signals/creamy_chicken/samples
mkdir -p training_data/signals/creamy_chicken/splits
```

**Step 3: Test training with mock data**
```python
# Create a few test samples manually
# Then run:
from backend.training.service import TrainingService

ts = TrainingService()
result = ts.train_new_signal("creamy_chicken", notes="Test training")
print(result)
```

**Verify:**
- New version created in `models/heads/creamy_chicken/`
- Metadata updated
- Registry updated

---

### Phase 5: WebSocket API (Day 9-10)

**Step 1: Add new commands to server.py**

**Step 2: Test with websocket client**
```python
import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://localhost:XXXX") as ws:
        # Test registry
        await ws.send(json.dumps({"command": "get_registry"}))
        resp = await ws.recv()
        print(json.loads(resp))
        
        # Test version history
        await ws.send(json.dumps({
            "command": "get_version_history",
            "signal_name": "creamy_chicken"
        }))
        resp = await ws.recv()
        print(json.loads(resp))

asyncio.run(test())
```

---

### Phase 6: Flutter Integration (Day 11-15)

**Step 1: Add new providers**
- Create `training_provider.dart`
- Create `signal_versions_provider.dart`

**Step 2: Test providers independently**

**Step 3: Update training screen UI**

**Step 4: Full integration test**
- Load capture file
- Draw bounding boxes
- Click train
- Verify progress updates
- Verify completion dialog
- Verify version history updates

---

## Testing Requirements

### Unit Tests

**File: `backend/hydra/tests/test_backbone_extractor.py`**
```python
def test_categorize_weights():
    """Verify weights are correctly categorized."""
    
def test_extract_backbone_produces_valid_file():
    """Extracted backbone can be loaded."""
    
def test_extract_head_produces_valid_file():
    """Extracted head can be loaded."""
    
def test_validate_extraction_passes_for_correct_split():
    """Validation succeeds when split is correct."""
```

**File: `backend/hydra/tests/test_detector.py`**
```python
def test_load_backbone():
    """Backbone loads without error."""
    
def test_load_single_head():
    """Single head loads after backbone."""
    
def test_load_multiple_heads():
    """Multiple heads can be loaded."""
    
def test_unload_heads():
    """Heads can be unloaded to free memory."""
    
def test_detect_returns_correct_format():
    """Detection output has expected structure."""
    
def test_detect_matches_original_model():
    """Hydra output matches original single-model output."""
```

**File: `backend/hydra/tests/test_version_manager.py`**
```python
def test_create_version():
    """New version is created and saved."""
    
def test_should_auto_promote_first_version():
    """First version always auto-promotes."""
    
def test_should_auto_promote_with_improvement():
    """Version with >2% F1 improvement promotes."""
    
def test_should_not_promote_without_improvement():
    """Version with <2% improvement doesn't promote."""
    
def test_promote_version_updates_active():
    """Promoting updates active symlink."""
    
def test_rollback():
    """Rollback reverts to previous version."""
    
def test_cleanup_keeps_n_versions():
    """Cleanup keeps specified number of versions."""
```

### Integration Tests

**File: `backend/tests/integration/test_training_flow.py`**
```python
def test_new_signal_training_end_to_end():
    """Complete flow: samples → training → version → promote."""
    
def test_extend_signal_training():
    """Adding samples and retraining works."""
    
def test_training_with_early_stopping():
    """Early stopping triggers correctly."""
```

**File: `backend/tests/integration/test_mission_flow.py`**
```python
def test_load_mission_heads():
    """Mission signals load correct heads."""
    
def test_inference_with_multiple_heads():
    """Multiple heads detect their respective signals."""
```

### Performance Benchmarks

**File: `backend/benchmarks/benchmark_inference.py`**
```python
"""
Run with: python -m backend.benchmarks.benchmark_inference

Expected results:
    Original single model:     ~10ms per inference
    Hydra 1 head:              ~10ms (same)
    Hydra 6 heads:             ~35ms (3.5x, not 6x!)
    Hydra 12 heads:            ~45ms (4.5x, not 12x!)
"""

def benchmark_original_model():
    """Baseline: original single-model inference."""
    
def benchmark_hydra_1_head():
    """Hydra with single head should match original."""
    
def benchmark_hydra_6_heads():
    """Hydra with 6 heads."""
    
def benchmark_hydra_12_heads():
    """Hydra with 12 heads."""

def compare_outputs():
    """Verify Hydra outputs match original."""
```

**File: `backend/benchmarks/benchmark_memory.py`**
```python
"""
Run with: python -m backend.benchmarks.benchmark_memory

Expected results:
    Original single model:     ~100MB GPU
    Hydra backbone only:       ~55MB GPU
    Hydra backbone + 1 head:   ~65MB GPU
    Hydra backbone + 6 heads:  ~115MB GPU
    Hydra backbone + 12 heads: ~175MB GPU
"""

def measure_original_memory():
    """Baseline memory usage."""
    
def measure_hydra_backbone():
    """Backbone-only memory."""
    
def measure_hydra_incremental():
    """Memory as heads are added one by one."""
```

---

## Success Criteria

### Phase 1-2 Complete ✓
- [ ] Backbone extracted: `models/backbone/v1.pth` exists
- [ ] Head extracted: `models/heads/creamy_chicken/v1.pth` exists  
- [ ] Validation passes: outputs identical to original
- [ ] Unit tests pass

### Phase 3 Complete ✓
- [ ] Flutter connects to backend
- [ ] Detections appear on waterfall
- [ ] No visible change in behavior
- [ ] Inference time ≤ original

### Phase 4 Complete ✓
- [ ] Can train via CLI
- [ ] Produces valid head file
- [ ] Metrics calculated correctly
- [ ] Auto-promotion logic works

### Phase 5 Complete ✓
- [ ] WebSocket commands work
- [ ] Progress messages stream correctly
- [ ] Version history loads

### Phase 6 Complete ✓
- [ ] Full training flow in Flutter
- [ ] Progress overlay displays
- [ ] Completion dialog shows comparison
- [ ] Version history widget works
- [ ] Rollback works

### Final Acceptance ✓
- [ ] 6 signals detect in < 50ms
- [ ] Memory < 150MB for 6 signals
- [ ] All existing tests pass
- [ ] No regressions in detection quality

---

## Rollback Procedures

### Emergency Rollback (Backend)

If Hydra causes production issues:

```bash
# 1. Disable Hydra in unified_pipeline.py
# Set USE_HYDRA_DETECTOR = False

# 2. Restart backend
# Flutter will reconnect automatically

# 3. Verify detection works with original model
```

### Version Rollback (Per-Signal)

If a training produces bad results:

```bash
# Via WebSocket
{"command": "rollback_signal", "signal_name": "creamy_chicken"}

# Or via CLI
python -m backend.hydra.version_manager --signal creamy_chicken --rollback
```

### Full Rollback (Nuclear Option)

If everything is broken:

```bash
# 1. Stop backend

# 2. Remove Hydra files
rm -rf models/backbone models/heads models/registry.json

# 3. Restore original model
cp models/legacy/creamy_chicken_fold3.pth models/

# 4. Revert code changes to unified_pipeline.py

# 5. Restart backend
```

---

## Quick Reference Commands

```bash
# Extract backbone (one-time)
python -m backend.hydra.backbone_extractor -i models/creamy_chicken_fold3.pth --validate --backup

# Train new signal
python -m backend.training.service --signal wifi_24 --new

# Extend signal
python -m backend.training.service --signal creamy_chicken --extend

# Promote version
python -m backend.hydra.version_manager --signal creamy_chicken --promote 3

# Rollback
python -m backend.hydra.version_manager --signal creamy_chicken --rollback

# Run benchmarks
python -m backend.benchmarks.benchmark_inference
python -m backend.benchmarks.benchmark_memory

# Run tests
python -m pytest backend/hydra/tests/ -v
python -m pytest backend/tests/integration/ -v
```
