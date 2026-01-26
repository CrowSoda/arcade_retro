#!/usr/bin/env python3
"""
Verify that training sample boxes land on bright signal regions.

This is the KEY diagnostic tool for the coordinate fix (Jan 2026).
If boxes land on BRIGHT regions (mean > 150), the fix is working.
If boxes land on DARK regions (mean < 100), there's still a mismatch.

Usage:
    cd g20_demo/backend
    python verify_sample.py <signal_name> [sample_id]
    
Examples:
    python verify_sample.py Creamy_Pork           # Verify first 5 samples
    python verify_sample.py Creamy_Pork abc123    # Verify specific sample
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def verify_sample(signal_name: str, sample_id: str = None, training_data_dir: str = "training_data/signals"):
    """
    Verify that training sample boxes land on bright signal regions.
    
    Args:
        signal_name: Signal class name (e.g., "Creamy_Pork")
        sample_id: Specific sample ID, or None for first 5
        training_data_dir: Base training data directory
    """
    base_dir = Path(training_data_dir) / signal_name / "samples"
    
    if not base_dir.exists():
        print(f"ERROR: No samples found at {base_dir}")
        return False
    
    # Get sample IDs to verify
    if sample_id:
        samples = [sample_id]
    else:
        # Get first 5 samples
        npz_files = sorted(base_dir.glob("*.npz"))
        if not npz_files:
            print(f"ERROR: No .npz files found in {base_dir}")
            return False
        samples = [f.stem for f in npz_files[:5]]
    
    print(f"\n{'='*60}")
    print(f"TRAINING SAMPLE VERIFICATION: {signal_name}")
    print(f"{'='*60}")
    
    all_pass = True
    results = []
    
    for sid in samples:
        npz_path = base_dir / f"{sid}.npz"
        json_path = base_dir / f"{sid}.json"
        
        if not npz_path.exists() or not json_path.exists():
            print(f"\nWARNING: Sample {sid} missing files, skipping")
            continue
        
        # Load spectrogram
        with np.load(npz_path) as data:
            spec = data["spectrogram"]
        
        # Load boxes
        with open(json_path) as f:
            meta = json.load(f)
        
        boxes = meta.get("boxes", [])
        coord_format = meta.get("coordinate_format", "unknown")
        
        print(f"\n--- Sample: {sid} ---")
        print(f"  Spectrogram shape: {spec.shape}")
        print(f"  Spectrogram range: {spec.min()}-{spec.max()}")
        print(f"  Coordinate format: {coord_format}")
        print(f"  Number of boxes: {len(boxes)}")
        
        # Check each box
        sample_pass = True
        for i, box in enumerate(boxes):
            x_min = int(box['x_min'])
            y_min = int(box['y_min'])
            x_max = int(box['x_max'])
            y_max = int(box['y_max'])
            
            # Extract box region
            box_region = spec[y_min:y_max, x_min:x_max]
            
            if box_region.size == 0:
                print(f"  Box {i}: EMPTY REGION (coords out of bounds)")
                sample_pass = False
                continue
            
            mean_val = box_region.mean()
            max_val = box_region.max()
            min_val = box_region.min()
            
            # Determine if signal is present (bright region)
            # Signal regions should have mean > 150 typically
            is_signal = mean_val > 120
            status = "✅ PASS" if is_signal else "❌ FAIL"
            
            print(f"  Box {i}: ({x_min},{y_min})-({x_max},{y_max})")
            print(f"    Size: {x_max-x_min}x{y_max-y_min}")
            print(f"    Mean brightness: {mean_val:.1f}")
            print(f"    Range: {min_val}-{max_val}")
            print(f"    {status} (expect >120 for signal)")
            
            if not is_signal:
                sample_pass = False
        
        # Save visualization
        output_dir = Path("training_data/verification")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"verify_{signal_name}_{sid}.png"
        
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(spec, cmap='viridis', aspect='auto', origin='upper')
        ax.set_title(f'{signal_name} / {sid}\ncoord_format: {coord_format}')
        
        # Draw boxes
        for i, box in enumerate(boxes):
            x = box['x_min']
            y = box['y_min']
            w = box['x_max'] - box['x_min']
            h = box['y_max'] - box['y_min']
            
            # Color based on brightness
            box_region = spec[int(y):int(y+h), int(x):int(x+w)]
            mean_val = box_region.mean() if box_region.size > 0 else 0
            color = 'lime' if mean_val > 120 else 'red'
            
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=3, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x, y - 10, f'Box {i}: {mean_val:.0f}', 
                   color=color, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        plt.colorbar(ax.images[0], ax=ax, label='uint8 value')
        plt.savefig(str(output_path), dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_path}")
        
        if not sample_pass:
            all_pass = False
        
        results.append({
            'sample_id': sid,
            'passed': sample_pass,
            'boxes': len(boxes),
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    print(f"Samples verified: {total}")
    print(f"Passed: {passed}/{total}")
    
    if all_pass:
        print("\n✅ ALL SAMPLES PASS - Boxes land on signal regions!")
        print("   The coordinate fix is working correctly.")
    else:
        print("\n❌ SOME SAMPLES FAIL - Boxes may be misaligned!")
        print("   Check the verification images in training_data/verification/")
        print("\n   Possible causes:")
        print("   1. Old samples created before the coordinate fix")
        print("   2. Flutter not sending real-unit coordinates")
        print("   3. Python not receiving/parsing real units correctly")
        print("\n   To fix: Delete old samples and re-label in Flutter")
    
    return all_pass


def verify_all_signals(training_data_dir: str = "training_data/signals"):
    """Verify all signals in the training data directory."""
    base_dir = Path(training_data_dir)
    
    if not base_dir.exists():
        print(f"ERROR: Training data directory not found: {base_dir}")
        return
    
    signals = [d.name for d in base_dir.iterdir() if d.is_dir()]
    
    if not signals:
        print(f"No signals found in {base_dir}")
        return
    
    print(f"Found {len(signals)} signals: {signals}")
    
    all_pass = True
    for signal_name in signals:
        if not verify_sample(signal_name, training_data_dir=training_data_dir):
            all_pass = False
    
    return all_pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nRunning verification on all signals...")
        verify_all_signals()
    elif sys.argv[1] == "--all":
        verify_all_signals()
    else:
        signal_name = sys.argv[1]
        sample_id = sys.argv[2] if len(sys.argv) > 2 else None
        verify_sample(signal_name, sample_id)
