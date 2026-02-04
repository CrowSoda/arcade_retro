# Data Augmentation for RF Spectrograms

## Overview

RF spectrograms require **domain-specific augmentations**. Generic image augmentations (rotation, horizontal flip) destroy signal semantics.

---

## Safe vs. Unsafe Augmentations

### ✅ Safe (Use Aggressively)

| Augmentation | Implementation | Rationale |
|--------------|----------------|-----------|
| Frequency shift | Vertical translation ±20% | Same signal at different IFs |
| Time shift | Horizontal translation ±20% | Reinforces position invariance |
| Power scaling | Multiply by 0.5-2.0 | Simulates varying TX power |
| SNR augmentation | Add Gaussian noise | Critical for real-world robustness |
| SpecAugment masking | Mask freq/time bands | Prevents overfitting to specific features |

### ❌ Unsafe (Avoid)

| Augmentation | Why It's Bad |
|--------------|--------------|
| Horizontal flip | Reverses time direction |
| Vertical flip | Inverts frequency axis |
| Large rotations | Destroys time-frequency axes |
| CutMix | Mixes unrelated signals |
| Color jitter | N/A for spectrograms |

---

## Full Implementation

```python
import numpy as np
import random
import torch


class SpectrogramAugment:
    """
    Spectrogram-specific augmentations for RF signal detection.

    Research on Panoradio dataset shows networks trained WITHOUT
    noise augmentation achieve only 30-50% accuracy on real-world
    data versus 90%+ WITH proper augmentation.
    """

    def __init__(
        self,
        p_freq_shift: float = 0.5,
        p_time_shift: float = 0.5,
        p_power_scale: float = 0.5,
        p_noise: float = 0.7,  # Higher - noise is critical
        p_freq_mask: float = 0.3,
        p_time_mask: float = 0.3,
    ):
        self.p_freq_shift = p_freq_shift
        self.p_time_shift = p_time_shift
        self.p_power_scale = p_power_scale
        self.p_noise = p_noise
        self.p_freq_mask = p_freq_mask
        self.p_time_mask = p_time_mask

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to a single crop.

        Args:
            crop: (H, W) numpy array, normalized

        Returns:
            Augmented crop
        """
        crop = crop.copy()

        # --- Frequency shift (vertical translation) ---
        if random.random() < self.p_freq_shift:
            shift = int(crop.shape[0] * random.uniform(-0.2, 0.2))
            crop = np.roll(crop, shift, axis=0)
            # Zero edges to avoid wrap-around artifacts
            if shift > 0:
                crop[:shift, :] = 0
            elif shift < 0:
                crop[shift:, :] = 0

        # --- Time shift (horizontal translation) ---
        if random.random() < self.p_time_shift:
            shift = int(crop.shape[1] * random.uniform(-0.2, 0.2))
            crop = np.roll(crop, shift, axis=1)
            if shift > 0:
                crop[:, :shift] = 0
            elif shift < 0:
                crop[:, shift:] = 0

        # --- Power scaling ---
        if random.random() < self.p_power_scale:
            scale = random.uniform(0.5, 2.0)
            crop = crop * scale

        # --- SNR augmentation (additive noise) ---
        if random.random() < self.p_noise:
            crop = self._add_noise(crop)

        # --- SpecAugment: frequency masking ---
        if random.random() < self.p_freq_mask:
            crop = self._freq_mask(crop)

        # --- SpecAugment: time masking ---
        if random.random() < self.p_time_mask:
            crop = self._time_mask(crop)

        return crop

    def _add_noise(self, crop: np.ndarray) -> np.ndarray:
        """Add Gaussian noise at random SNR."""
        # Random SNR between -10 dB and +20 dB
        snr_db = random.uniform(-10, 20)

        # Calculate noise power based on signal power
        signal_power = (crop ** 2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))

        noise = np.random.normal(0, np.sqrt(noise_power), crop.shape)
        return crop + noise.astype(np.float32)

    def _freq_mask(self, crop: np.ndarray, max_width_pct: float = 0.15) -> np.ndarray:
        """Mask a random frequency band."""
        h, w = crop.shape
        mask_width = int(h * random.uniform(0.05, max_width_pct))
        mask_start = random.randint(0, h - mask_width)

        crop[mask_start:mask_start + mask_width, :] = 0
        return crop

    def _time_mask(self, crop: np.ndarray, max_width_pct: float = 0.15) -> np.ndarray:
        """Mask a random time segment."""
        h, w = crop.shape
        mask_width = int(w * random.uniform(0.05, max_width_pct))
        mask_start = random.randint(0, w - mask_width)

        crop[:, mask_start:mask_start + mask_width] = 0
        return crop


class SpectrogramAugmentTorch:
    """
    PyTorch-native version for GPU augmentation.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to a batch.

        Args:
            batch: (B, 1, H, W) tensor

        Returns:
            Augmented batch
        """
        B, C, H, W = batch.shape

        for i in range(B):
            if random.random() < self.p:
                # Frequency shift
                shift = int(H * random.uniform(-0.2, 0.2))
                batch[i] = torch.roll(batch[i], shift, dims=1)

            if random.random() < self.p:
                # Power scaling
                scale = random.uniform(0.5, 2.0)
                batch[i] = batch[i] * scale

            if random.random() < self.p:
                # Noise injection
                snr_db = random.uniform(-10, 20)
                signal_power = batch[i].pow(2).mean()
                noise_std = torch.sqrt(signal_power / (10 ** (snr_db / 10)))
                noise = torch.randn_like(batch[i]) * noise_std
                batch[i] = batch[i] + noise

        return batch
```

---

## Usage in Training

```python
# In training loop

augment = SpectrogramAugment(
    p_freq_shift=0.5,
    p_time_shift=0.5,
    p_power_scale=0.5,
    p_noise=0.7,
    p_freq_mask=0.3,
    p_time_mask=0.3,
)

# During batch preparation
for epoch in range(epochs):
    for batch_idx, (crops, labels) in enumerate(train_loader):
        # Apply augmentation to training crops
        augmented = []
        for crop in crops:
            aug_crop = augment(crop.numpy().squeeze())
            augmented.append(torch.from_numpy(aug_crop).unsqueeze(0))

        crops = torch.stack(augmented)

        # Continue with training...
```

---

## Augmentation During Training vs. Inference

| Phase | Augmentation |
|-------|--------------|
| Training | Apply all augmentations |
| Validation | No augmentation |
| Inference | No augmentation |
| Test-Time Augmentation (optional) | Average predictions across augmented versions |

---

## Research Backing

### SNR Augmentation is Critical

From Panoradio dataset experiments:
- Without noise aug: 30-50% accuracy on real data
- With noise aug: 90%+ accuracy on real data

Real-world RF environments have varying SNR. Training only on clean signals creates brittleness.

### SpecAugment Prevents Overfitting

From "SpecAugment" (Park et al., 2019):
- Masking prevents model from relying on specific frequency/time features
- Forces learning of robust, distributed representations
- Originally for speech but applies directly to spectrograms

### Power Scaling Simulates Distance

RF power varies with distance (inverse square law). Power scaling augmentation teaches the model that signal appearance is independent of absolute power level.
