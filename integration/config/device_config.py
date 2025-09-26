"""
Global device configuration for UniAD-ITRI integration.
Provides centralized device management with CLI support.
"""

import argparse
import torch
import logging
import time
import subprocess
import json
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Performance metrics collection and analysis."""
    
    def __init__(self):
        self.metrics = {
            'memory_timeline': [],
            'stage_timings': {},
            'frame_timings': [],
            'gpu_utilization': []
        }
        self.start_times = {}
        
    def start_timing(self, stage: str):
        """Start timing for a stage."""
        self.start_times[stage] = time.time()
        
    def end_timing(self, stage: str):
        """End timing for a stage and record duration."""
        if stage in self.start_times:
            duration = time.time() - self.start_times[stage]
            if stage not in self.metrics['stage_timings']:
                self.metrics['stage_timings'][stage] = []
            self.metrics['stage_timings'][stage].append(duration)
            del self.start_times[stage]
            return duration
        return None
        
    def record_memory_snapshot(self, stage: str = ""):
        """Record current memory usage."""
        if torch.cuda.is_available():
            snapshot = {
                'timestamp': time.time(),
                'stage': stage,
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'cached_gb': torch.cuda.memory_cached() / 1024**3
            }
            self.metrics['memory_timeline'].append(snapshot)
            
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            'total_frames': len(self.metrics['frame_timings']),
            'avg_frame_time': 0,
            'peak_memory_gb': 0,
            'stage_averages': {}
        }
        
        if self.metrics['frame_timings']:
            summary['avg_frame_time'] = sum(self.metrics['frame_timings']) / len(self.metrics['frame_timings'])
            summary['fps'] = 1.0 / summary['avg_frame_time'] if summary['avg_frame_time'] > 0 else 0
            
        if self.metrics['memory_timeline']:
            summary['peak_memory_gb'] = max(m['allocated_gb'] for m in self.metrics['memory_timeline'])
            
        for stage, timings in self.metrics['stage_timings'].items():
            if timings:
                summary['stage_averages'][stage] = {
                    'avg_time': sum(timings) / len(timings),
                    'min_time': min(timings),
                    'max_time': max(timings),
                    'count': len(timings)
                }
                
        return summary
        
    def save_metrics(self, output_path: str):
        """Save metrics to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'summary': self.get_summary()
            }, f, indent=2)
            
        logger.info(f"Performance metrics saved to {output_path}")


class DeviceConfig:
    """Centralized device configuration manager."""
    
    def __init__(self, mode: str = 'auto', enable_metrics: bool = False):
        """
        Initialize device configuration.
        
        Args:
            mode: Device mode - 'cpu', 'gpu', or 'auto'
            enable_metrics: Enable performance metrics collection
        """
        self.mode = mode
        self._config = self._setup_device_config()
        self.metrics = PerformanceMetrics() if enable_metrics else None
        
    def _setup_device_config(self) -> Dict[str, Any]:
        """Setup device configuration based on mode."""
        config = {
            'data_processing': 'cpu',      # Always CPU for file I/O
            'model_inference': 'cpu',      # Default to CPU for stability
            'tensor_operations': 'cpu',    # Default to CPU for safety
            'memory_monitoring': True,     # Always monitor memory usage
            'fallback_enabled': True,      # Auto fallback GPU→CPU on OOM
        }
        
        if self.mode == 'gpu':
            if torch.cuda.is_available():
                config.update({
                    'model_inference': 'cuda',
                    'tensor_operations': 'cuda',
                })
                logger.info(f"GPU mode enabled: {torch.cuda.get_device_name()}")
            else:
                logger.warning("GPU requested but CUDA not available, using CPU")
                
        elif self.mode == 'auto':
            if torch.cuda.is_available():
                # Conservative auto mode: GPU for inference only
                config.update({
                    'model_inference': 'cuda',
                    'tensor_operations': 'cpu',  # Keep tensors on CPU initially
                })
                logger.info(f"Auto mode: GPU inference + CPU tensor ops")
            else:
                logger.info("Auto mode: CPU only (CUDA not available)")
                
        elif self.mode == 'cpu':
            logger.info("CPU mode: All operations on CPU")
            
        else:
            raise ValueError(f"Invalid device mode: {mode}. Use 'cpu', 'gpu', or 'auto'")
            
        return config
        
    @property
    def data_device(self) -> str:
        """Device for data processing operations."""
        return self._config['data_processing']
        
    @property 
    def model_device(self) -> str:
        """Device for model inference operations."""
        return self._config['model_inference']
        
    @property
    def tensor_device(self) -> str:
        """Device for tensor operations."""
        return self._config['tensor_operations']
        
    @property
    def memory_monitoring(self) -> bool:
        """Whether to monitor memory usage."""
        return self._config['memory_monitoring']
        
    @property
    def fallback_enabled(self) -> bool:
        """Whether automatic GPU→CPU fallback is enabled."""
        return self._config['fallback_enabled']
        
    def log_memory_usage(self, stage: str = "", detailed: bool = False):
        """Log current GPU memory usage if monitoring enabled."""
        if not self.memory_monitoring:
            return
            
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            cached = torch.cuda.memory_cached() / 1024**3
            
            if detailed:
                logger.info(f"GPU Memory {stage}:")
                logger.info(f"  Allocated: {allocated:.3f}GB")
                logger.info(f"  Reserved:  {reserved:.3f}GB") 
                logger.info(f"  Cached:    {cached:.3f}GB")
                logger.info(f"  Summary:\n{torch.cuda.memory_summary()}")
            else:
                logger.info(f"GPU Memory {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
    def transfer_to_model_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transfer tensor to model device."""
        return tensor.to(self.model_device)
        
    def transfer_to_tensor_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transfer tensor to tensor operations device."""
        return tensor.to(self.tensor_device)
        
    def safe_cuda_operation(self, operation_fn, *args, **kwargs):
        """
        Execute operation with automatic CPU fallback on GPU OOM.
        
        Args:
            operation_fn: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Result of operation_fn, potentially with device fallback
        """
        if not self.fallback_enabled or self.model_device == 'cpu':
            return operation_fn(*args, **kwargs)
            
        try:
            return operation_fn(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"GPU OOM detected, falling back to CPU: {e}")
                # Move all tensor args to CPU
                cpu_args = []
                cpu_kwargs = {}
                
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        cpu_args.append(arg.cpu())
                    else:
                        cpu_args.append(arg)
                        
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor):
                        cpu_kwargs[key] = value.cpu()
                    else:
                        cpu_kwargs[key] = value
                        
                # Temporarily switch to CPU mode
                original_model_device = self._config['model_inference']
                self._config['model_inference'] = 'cpu'
                
                try:
                    result = operation_fn(*cpu_args, **cpu_kwargs)
                    return result
                finally:
                    # Restore original device config
                    self._config['model_inference'] = original_model_device
            else:
                raise


# Global instance
_global_device_config = None


def get_device_config() -> DeviceConfig:
    """Get global device configuration instance."""
    global _global_device_config
    if _global_device_config is None:
        _global_device_config = DeviceConfig()
    return _global_device_config


def set_device_config(mode: str):
    """Set global device configuration."""
    global _global_device_config
    _global_device_config = DeviceConfig(mode)
    logger.info(f"Global device config set to: {mode}")


def add_device_args(parser: argparse.ArgumentParser):
    """Add device configuration arguments to argument parser."""
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'gpu', 'auto'],
        default='auto',
        help='Device mode: cpu (all CPU), gpu (prefer GPU), auto (smart hybrid)'
    )
    parser.add_argument(
        '--memory-monitoring',
        action='store_true',
        default=True,
        help='Enable GPU memory monitoring'
    )
    parser.add_argument(
        '--no-fallback',
        action='store_true',
        help='Disable automatic GPU→CPU fallback on OOM'
    )


def configure_from_args(args: argparse.Namespace):
    """Configure device settings from command line arguments."""
    config = DeviceConfig(args.device)
    
    if hasattr(args, 'memory_monitoring'):
        config._config['memory_monitoring'] = args.memory_monitoring
        
    if hasattr(args, 'no_fallback'):
        config._config['fallback_enabled'] = not args.no_fallback
        
    set_device_config(config.mode)
    return config


# Convenience functions for common operations
def get_model_device() -> str:
    """Get device for model operations."""
    return get_device_config().model_device


def get_tensor_device() -> str:
    """Get device for tensor operations.""" 
    return get_device_config().tensor_device


def safe_to_device(tensor: torch.Tensor, target_device: str) -> torch.Tensor:
    """Safely move tensor to target device with error handling."""
    try:
        return tensor.to(target_device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"OOM moving to {target_device}, keeping on {tensor.device}")
            return tensor
        else:
            raise