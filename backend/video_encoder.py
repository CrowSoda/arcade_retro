"""
Video encoder abstraction for waterfall streaming.
- NVENC H.264 on Orin/RTX (production)
- JPEG fallback for development without FFmpeg
"""

import os
import io
import subprocess
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class BaseEncoder(ABC):
    """Abstract base class for video encoders."""
    
    @abstractmethod
    def encode(self, frame: np.ndarray) -> bytes:
        """Encode a single frame. Returns encoded bytes."""
        pass
    
    @abstractmethod
    def get_content_type(self) -> str:
        """Return MIME type for the encoded data."""
        pass
    
    @abstractmethod
    def close(self):
        """Clean up encoder resources."""
        pass


class NVENCEncoder(BaseEncoder):
    """
    Hardware H.264 encoder using NVENC via FFmpeg.
    Works on: RTX GPUs, Jetson Orin NX/AGX
    Does NOT work on: Jetson Orin Nano, systems without NVENC
    """
    
    def __init__(self, width: int, height: int, fps: int = 30, bitrate: int = 4_000_000):
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.process: Optional[subprocess.Popen] = None
        self._start_encoder()
    
    def _start_encoder(self):
        """Start FFmpeg encoder subprocess."""
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            # Input
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', 'pipe:0',
            # Encoder settings
            '-c:v', 'h264_nvenc',
            '-preset', 'p1',           # Lowest latency preset
            '-tune', 'll',             # Low latency tuning
            '-rc', 'cbr',              # Constant bitrate
            '-b:v', str(self.bitrate),
            '-maxrate', str(self.bitrate),
            '-bufsize', str(self.bitrate // 2),
            '-g', '30',                # Keyframe every 30 frames (1 sec)
            '-bf', '0',                # No B-frames
            '-zerolatency', '1',
            # Output
            '-f', 'h264',
            'pipe:1'
        ]
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        print(f"[NVENCEncoder] Started FFmpeg process, PID={self.process.pid}")
    
    def encode(self, frame: np.ndarray) -> bytes:
        """
        Encode frame to H.264.
        
        Args:
            frame: numpy array, shape (height, width, 3), dtype uint8, RGB format
            
        Returns:
            H.264 NAL units as bytes
        """
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Encoder process not running")
        
        # Ensure correct format
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Expected shape ({self.height}, {self.width}, 3), got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # Write frame to FFmpeg stdin
        self.process.stdin.write(frame.tobytes())
        self.process.stdin.flush()
        
        # Read available encoded data (non-blocking)
        # FFmpeg buffers internally, so we read what's available
        encoded = b''
        
        # Use select on Unix, different approach on Windows
        if os.name == 'nt':
            # Windows: use non-blocking read with timeout
            import msvcrt
            import ctypes
            from ctypes import wintypes
            
            # Try to read what's available
            try:
                # Simple approach: read1 if available
                chunk = self.process.stdout.read1(65536)
                if chunk:
                    encoded = chunk
            except Exception:
                pass
        else:
            # Unix: use select
            import select
            while select.select([self.process.stdout], [], [], 0)[0]:
                chunk = self.process.stdout.read(65536)
                if chunk:
                    encoded += chunk
                else:
                    break
        
        return encoded
    
    def get_content_type(self) -> str:
        return 'video/h264'
    
    def close(self):
        if self.process:
            try:
                self.process.stdin.close()
            except:
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                pass
            try:
                self.process.kill()
            except:
                pass
            self.process = None
        print("[NVENCEncoder] Closed", flush=True)


class LibX264Encoder(BaseEncoder):
    """
    Software H.264 encoder using libx264 via FFmpeg.
    Works everywhere FFmpeg is installed.
    """
    
    def __init__(self, width: int, height: int, fps: int = 30, bitrate: int = 4_000_000):
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.process: Optional[subprocess.Popen] = None
        self._start_encoder()
    
    def _start_encoder(self):
        """Start FFmpeg encoder subprocess."""
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            # Input
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', 'pipe:0',
            # Encoder settings - ultrafast for low latency
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-b:v', str(self.bitrate),
            '-g', '30',                # Keyframe every 30 frames (1 sec)
            '-bf', '0',                # No B-frames
            # Output
            '-f', 'h264',
            'pipe:1'
        ]
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        print(f"[LibX264Encoder] Started FFmpeg process, PID={self.process.pid}")
    
    def encode(self, frame: np.ndarray) -> bytes:
        """
        Encode frame to H.264.
        
        Args:
            frame: numpy array, shape (height, width, 3), dtype uint8, RGB format
            
        Returns:
            H.264 NAL units as bytes
        """
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Encoder process not running")
        
        # Ensure correct format
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Expected shape ({self.height}, {self.width}, 3), got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # Write frame to FFmpeg stdin
        self.process.stdin.write(frame.tobytes())
        self.process.stdin.flush()
        
        # Read available encoded data
        encoded = b''
        
        if os.name == 'nt':
            try:
                chunk = self.process.stdout.read1(65536)
                if chunk:
                    encoded = chunk
            except Exception:
                pass
        else:
            import select
            while select.select([self.process.stdout], [], [], 0)[0]:
                chunk = self.process.stdout.read(65536)
                if chunk:
                    encoded += chunk
                else:
                    break
        
        return encoded
    
    def get_content_type(self) -> str:
        return 'video/h264'
    
    def close(self):
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception as e:
                print(f"[LibX264Encoder] Error closing: {e}")
            print("[LibX264Encoder] Closed")


class JPEGEncoder(BaseEncoder):
    """
    JPEG fallback encoder for development/testing.
    Works everywhere, no FFmpeg required.
    Higher bandwidth than H.264 but simpler.
    """
    
    def __init__(self, width: int, height: int, quality: int = 80):
        self.width = width
        self.height = height
        self.quality = quality
        
        if not HAS_PIL:
            raise ImportError("PIL/Pillow is required for JPEGEncoder: pip install Pillow")
        
        print(f"[JPEGEncoder] Initialized, quality={quality}")
    
    def encode(self, frame: np.ndarray) -> bytes:
        """
        Encode frame to JPEG.
        
        Args:
            frame: numpy array, shape (height, width, 3), dtype uint8, RGB format
            
        Returns:
            JPEG bytes
        """
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        img = Image.fromarray(frame, mode='RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        return buffer.getvalue()
    
    def get_content_type(self) -> str:
        return 'image/jpeg'
    
    def close(self):
        pass


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available in PATH."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_nvenc_available() -> bool:
    """Check if FFmpeg has NVENC support."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return 'h264_nvenc' in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def create_encoder(
    width: int,
    height: int,
    fps: int = 30,
    prefer_nvenc: bool = True,
    fallback_jpeg: bool = True,
) -> BaseEncoder:
    """
    Factory function to create the best available encoder.
    
    Priority order:
    1. NVENC (if available and prefer_nvenc=True)
    2. libx264 (if FFmpeg available)
    3. JPEG (if fallback_jpeg=True)
    4. Raise error
    
    Args:
        width: Frame width
        height: Frame height
        fps: Target framerate
        prefer_nvenc: If True, try NVENC first
        fallback_jpeg: If True, fall back to JPEG if FFmpeg unavailable
        
    Returns:
        Encoder instance
    """
    # Check FFmpeg availability
    has_ffmpeg = check_ffmpeg_available()
    
    if has_ffmpeg:
        if prefer_nvenc and check_nvenc_available():
            print("[create_encoder] NVENC available, using H.264 hardware encoding")
            try:
                return NVENCEncoder(width, height, fps)
            except Exception as e:
                print(f"[create_encoder] NVENC failed: {e}, trying libx264")
        
        # Try libx264
        print("[create_encoder] Using libx264 software encoding")
        try:
            return LibX264Encoder(width, height, fps)
        except Exception as e:
            print(f"[create_encoder] libx264 failed: {e}")
    else:
        print("[create_encoder] FFmpeg not found")
    
    # Fall back to JPEG
    if fallback_jpeg:
        print("[create_encoder] Falling back to JPEG encoding")
        return JPEGEncoder(width, height)
    
    raise RuntimeError("No encoder available. Install FFmpeg or enable JPEG fallback.")


if __name__ == "__main__":
    # Test encoder creation
    print("Testing encoder creation...")
    
    try:
        encoder = create_encoder(2048, 1024, fps=30)
        print(f"Encoder type: {encoder.get_content_type()}")
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
        encoded = encoder.encode(test_frame)
        print(f"Encoded size: {len(encoded)} bytes")
        
        encoder.close()
        print("Test passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
