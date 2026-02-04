# Accuracy Expectations and Failure Modes

## Performance Trajectory

| Labels | Approach | Expected Accuracy | Notes |
|--------|----------|-------------------|-------|
| 25 | Siamese + pseudo-labeling | 75-85% | Sufficient for active learning loop |
| 50 | Siamese + augmentation | 78-88% | Improving with diversity |
| 100 | Shallow CNN + focal loss | 82-90% | Transition point |
| 250 | CNN + augmentation | 88-94% | Production-viable |
| 500 | CNN + hard negative mining | 92-97% | Full production |

---

## Accuracy Caveats

These estimates assume:

1. **Representative training distribution**
   - All signal modulation types are represented
   - SNR distribution matches deployment conditions
   - Labeled examples include edge cases

2. **Proper augmentation**
   - Noise levels match worst-case deployment SNR
   - Power scaling covers expected TX power range
   - Frequency shifts cover expected IF variations

3. **Balanced class distribution**
   - Or proper handling via focal loss / oversampling

---

## Primary Failure Modes

### 1. Distribution Shift (Training ≠ Deployment)

**Symptom:** High validation accuracy, poor field performance.

**Causes:**
- Training data from clean lab environment
- Deployment has interference, multipath, fading
- Different receiver noise characteristics

**Mitigations:**
- Include augmentation with noise levels matching worst-case deployment SNR
- Collect labels from multiple capture sessions / locations
- Validate on held-out data from different conditions

**Detection:**
```python
def detect_distribution_shift(train_crops, deploy_crops, model):
    """Compare feature distributions."""
    train_features = model.features(train_crops).mean(dim=0)
    deploy_features = model.features(deploy_crops).mean(dim=0)

    divergence = F.kl_div(
        F.log_softmax(train_features, dim=0),
        F.softmax(deploy_features, dim=0),
    )

    return divergence.item()  # High = distribution shift
```

---

### 2. Class Imbalance Causing Threshold Sensitivity

**Symptom:** Model predicts mostly "not signal"; small threshold changes cause large recall swings.

**Causes:**
- Blob detection generates many more negatives than positives
- BCE loss doesn't handle imbalance well
- Model learns to predict majority class

**Mitigations:**
- Use focal loss from the start (γ=2.0, α=0.25)
- Calibrate threshold using precision-recall curves, not ROC
- Consider oversampling positives during training

**Diagnosis:**
```python
def analyze_threshold_sensitivity(model, val_crops, val_labels):
    """Check how sensitive metrics are to threshold."""
    probs = model.predict_proba(val_crops).squeeze()

    results = []
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= thresh).float()
        tp = ((preds == 1) & (val_labels == 1)).sum()
        fp = ((preds == 1) & (val_labels == 0)).sum()
        fn = ((preds == 0) & (val_labels == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        results.append({
            'threshold': thresh,
            'precision': precision.item(),
            'recall': recall.item(),
        })

    return results  # Look for stable regions
```

---

### 3. Similar-Looking False Positives (Blob Artifacts)

**Symptom:** Consistent false positives on specific noise patterns.

**Causes:**
- Blob detector triggers on systematic noise patterns
- Classifier can't distinguish signal from specific artifacts
- Insufficient negative diversity in training

**Mitigations:**
- Hard negative mining: identify high-confidence false positives
- Add these to training set explicitly
- Increase blob detector specificity (higher thresholds)

**Hard Negative Mining:**
```python
def mine_hard_negatives(model, unlabeled_crops, k=50):
    """Find high-confidence false positives for labeling."""
    probs = model.predict_proba(unlabeled_crops).squeeze()

    # High confidence positives that are likely wrong
    # (assuming most unlabeled are negative)
    _, hard_indices = probs.topk(k)

    return hard_indices.tolist()  # Send to human for labeling
```

---

### 4. Novel Modulation Types Unseen in Training

**Symptom:** New signals classified with low confidence or incorrect label.

**Causes:**
- Model trained only on known signal types
- New signal has different spectral characteristics
- Out-of-distribution detection needed

**Mitigations:**
- Monitor prediction entropy in production
- Flag high-entropy samples for human review
- Implement continuous active learning

**Entropy Monitoring:**
```python
def compute_prediction_entropy(probs):
    """Binary entropy: high at 0.5, low at 0 or 1."""
    entropy = -probs * torch.log2(probs + 1e-8) - (1 - probs) * torch.log2(1 - probs + 1e-8)
    return entropy

def flag_uncertain_predictions(model, crops, entropy_threshold=0.8):
    """Flag predictions with high uncertainty for review."""
    probs = model.predict_proba(crops)
    entropy = compute_prediction_entropy(probs)

    uncertain_mask = entropy > entropy_threshold
    return torch.where(uncertain_mask)[0].tolist()
```

---

### 5. Siamese Network Mode Collapse

**Symptom:** All embeddings collapse to similar values.

**Causes:**
- Contrastive loss margin too small
- Imbalanced positive/negative pairs
- Learning rate too high

**Mitigations:**
- Use triplet loss with proper margin
- Ensure diverse negative sampling
- Monitor embedding variance during training

**Detection:**
```python
def check_embedding_collapse(model, crops):
    """Check if embeddings have collapsed."""
    embeddings = model.encode(crops)

    # Variance should be healthy (not near zero)
    variance = embeddings.var(dim=0).mean()

    # Pairwise distances should have spread
    distances = torch.cdist(embeddings, embeddings)
    distance_std = distances.std()

    return {
        'embedding_variance': variance.item(),
        'distance_std': distance_std.item(),
        'collapsed': variance < 0.01 or distance_std < 0.1,
    }
```

---

## Monitoring and Continuous Improvement

### Production Logging

```python
@app.post("/detect")
async def detect_signals(...):
    detections = detector.detect(spectrogram)

    # Log uncertain predictions for review
    uncertain = [d for d in detections if 0.4 < d.confidence < 0.7]
    if uncertain:
        await log_for_review(uncertain, image_id)

    # Log high-confidence predictions for validation
    confident = [d for d in detections if d.confidence > 0.95]
    await log_for_validation(confident, image_id)

    return detections
```

### Periodic Review Workflow

1. **Weekly:** Review uncertain predictions (0.4-0.7 confidence)
2. **Monthly:** Validate random sample of confident predictions
3. **Quarterly:** Retrain with accumulated new labels
4. **On drift detection:** Trigger active learning session

---

## Success Criteria

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Precision | 80% | 90% | 95% |
| Recall | 75% | 85% | 92% |
| F1 Score | 77% | 87% | 93% |
| Inference time | <100ms | <50ms | <25ms |
| Labels needed | <50 | <30 | <20 |
