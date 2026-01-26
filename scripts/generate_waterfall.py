#!/usr/bin/env python3
"""
Generate waterfall image from IQ file using same method as freqhunter.
Outputs PNG that Flutter can display.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration - match freqhunter
FFT_SIZE = 4096
OVERLAP_RATIO = 0.5
WINDOW_FN = 'hanning'
NFFT = FFT_SIZE * 2  # Zero-padding for better resolution

# Input file
IQ_FILE = Path(__file__).parent / 'data' / '825MHz.sigmf-data'
OUTPUT_FILE = Path(__file__).parent / 'data' / 'waterfall.png'

def main():
    print(f"Loading {IQ_FILE}...")
    
    # Load IQ data (cf32_le = interleaved float32 I,Q)
    raw = np.memmap(IQ_FILE, dtype=np.complex64, mode='r')
    print(f"Loaded {len(raw)} samples ({len(raw) * 8 / 1e6:.1f} MB)")
    
    # Compute spectrogram
    hop_size = int(FFT_SIZE * (1 - OVERLAP_RATIO))
    num_frames = (len(raw) - FFT_SIZE) // hop_size
    print(f"Computing {num_frames} FFT frames...")
    
    # Window
    window = np.hanning(FFT_SIZE).astype(np.float32)
    
    # Compute STFT
    stft_list = []
    for i in range(min(num_frames, 2000)):  # Limit to 2000 rows for demo
        start = i * hop_size
        segment = raw[start:start + FFT_SIZE] * window
        
        # FFT with zero-padding
        fft_out = np.fft.fft(segment, n=NFFT)
        fft_out = np.fft.fftshift(fft_out)
        
        # Magnitude in dB (freqhunter method)
        mag = np.abs(fft_out)
        db = 20 * np.log10(mag + 1e-6)
        
        stft_list.append(db)
        
        if i % 500 == 0:
            print(f"  Frame {i}/{min(num_frames, 2000)}")
    
    spectrogram = np.vstack(stft_list)
    print(f"Spectrogram shape: {spectrogram.shape}")
    
    # Save as image using viridis colormap
    print(f"Saving to {OUTPUT_FILE}...")
    
    # Use percentile scaling like we want
    vmin = np.percentile(spectrogram, 2)
    vmax = np.percentile(spectrogram, 98)
    
    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    ax.imshow(spectrogram, aspect='auto', cmap='viridis', 
              vmin=vmin, vmax=vmax, origin='upper')
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    plt.savefig(OUTPUT_FILE, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    
    print(f"Done! Image saved to {OUTPUT_FILE}")
    print(f"Resolution: {spectrogram.shape[1]} x {spectrogram.shape[0]} pixels")

if __name__ == '__main__':
    main()
