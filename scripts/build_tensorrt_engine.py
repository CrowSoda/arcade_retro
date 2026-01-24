#!/usr/bin/env python3
"""
TensorRT Engine Builder - Build .engine files ON THE TARGET SYSTEM.

CRITICAL: Never copy .engine files between machines!
Always run this script on the deployment target.

Usage:
    python build_tensorrt_engine.py --onnx model.onnx --output model.engine
    
    # With FP16 optimization
    python build_tensorrt_engine.py --onnx model.onnx --output model.engine --fp16
    
    # With INT8 calibration
    python build_tensorrt_engine.py --onnx model.onnx --output model.engine --int8 --calib-data calib/
"""

import argparse
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = False,
    int8: bool = False,
    calib_data_dir: str = None,
    max_batch_size: int = 1,
    workspace_mb: int = 4096,
    min_shape: tuple = (1, 3, 640, 640),
    opt_shape: tuple = (1, 3, 640, 640),
    max_shape: tuple = (1, 3, 640, 640),
) -> bool:
    """
    Build TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Path to input ONNX model
        engine_path: Path to output .engine file
        fp16: Enable FP16 precision
        int8: Enable INT8 precision (requires calibration data)
        calib_data_dir: Directory with calibration images (for INT8)
        max_batch_size: Maximum batch size
        workspace_mb: GPU memory workspace in MB
        min_shape: Minimum input shape (N, C, H, W)
        opt_shape: Optimal input shape
        max_shape: Maximum input shape
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.error("TensorRT not found. Please install TensorRT.")
        logger.error("This script must be run ON THE TARGET SYSTEM with TensorRT installed.")
        return False
    
    logger.info(f"TensorRT version: {trt.__version__}")
    logger.info(f"Input ONNX: {onnx_path}")
    logger.info(f"Output Engine: {engine_path}")
    logger.info(f"FP16: {fp16}, INT8: {int8}")
    
    # Verify ONNX exists
    if not os.path.exists(onnx_path):
        logger.error(f"ONNX file not found: {onnx_path}")
        return False
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Set workspace size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    
    # Parse ONNX
    logger.info("Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(error)}")
            return False
    
    # Get input name and set optimization profile
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    logger.info(f"Input tensor: {input_name}, shape: {input_tensor.shape}")
    
    # Dynamic shapes profile
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    # Precision settings
    if fp16 and builder.platform_has_fast_fp16:
        logger.info("Enabling FP16 precision")
        config.set_flag(trt.BuilderFlag.FP16)
    elif fp16:
        logger.warning("FP16 requested but not supported on this platform")
    
    if int8 and builder.platform_has_fast_int8:
        logger.info("Enabling INT8 precision")
        config.set_flag(trt.BuilderFlag.INT8)
        
        if calib_data_dir:
            # TODO: Implement INT8 calibrator
            logger.warning("INT8 calibration not yet implemented")
    elif int8:
        logger.warning("INT8 requested but not supported on this platform")
    
    # Build engine
    logger.info("Building TensorRT engine... (this may take several minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        logger.error("Failed to build engine")
        return False
    
    # Save engine
    logger.info(f"Saving engine to {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    # Verify
    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    logger.info(f"Engine built successfully! Size: {engine_size_mb:.2f} MB")
    
    return True


def verify_engine(engine_path: str) -> bool:
    """Verify an engine file can be loaded."""
    try:
        import tensorrt as trt
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            logger.error("Failed to deserialize engine")
            return False
        
        logger.info("Engine verification: OK")
        logger.info(f"  Bindings: {engine.num_bindings}")
        
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            is_input = engine.binding_is_input(i)
            logger.info(f"  [{i}] {name}: {shape} ({dtype}) {'INPUT' if is_input else 'OUTPUT'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Engine verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Build TensorRT engine from ONNX model',
        epilog='CRITICAL: Run this script ON THE TARGET SYSTEM. Never copy .engine files!',
    )
    
    parser.add_argument('--onnx', required=True, help='Input ONNX model path')
    parser.add_argument('--output', required=True, help='Output .engine file path')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    parser.add_argument('--int8', action='store_true', help='Enable INT8 precision')
    parser.add_argument('--calib-data', help='Calibration data directory (for INT8)')
    parser.add_argument('--workspace', type=int, default=4096, help='Workspace size in MB')
    parser.add_argument('--batch', type=int, default=1, help='Max batch size')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing engine')
    
    args = parser.parse_args()
    
    if args.verify_only:
        success = verify_engine(args.output)
    else:
        success = build_engine(
            onnx_path=args.onnx,
            engine_path=args.output,
            fp16=args.fp16,
            int8=args.int8,
            calib_data_dir=args.calib_data,
            workspace_mb=args.workspace,
            max_batch_size=args.batch,
        )
        
        if success:
            verify_engine(args.output)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
