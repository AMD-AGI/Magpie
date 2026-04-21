###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Image selector for benchmark mode.

Selects appropriate Docker images based on framework and GPU architecture.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import yaml

from ...utils.gpu import detect_gpu

logger = logging.getLogger(__name__)


class ImageSelector:
    """
    Selects Docker images based on framework and GPU architecture.
    
    Loads image mappings from benchmark_images.yaml and provides
    automatic selection based on detected GPU hardware.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the image selector.
        
        Args:
            config_path: Path to benchmark_images.yaml. If None, uses default
                        location relative to Magpie package.
        """
        if config_path is None:
            # Default: Magpie/benchmark_images.yaml
            config_path = str(Path(__file__).parent.parent.parent / "benchmark_images.yaml")
        
        self.config_path = config_path
        self._mapping: Dict[str, Dict[str, str]] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load image mapping configuration from YAML file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Image config not found: {self.config_path}, using empty mapping")
            self._mapping = {}
            return
        
        try:
            with open(self.config_path, 'r') as f:
                self._mapping = yaml.safe_load(f) or {}
            logger.info(f"Loaded image mapping from {self.config_path}")
            logger.debug(f"Available frameworks: {list(self._mapping.keys())}")
        except Exception as e:
            logger.error(f"Failed to load image config: {e}")
            self._mapping = {}
    
    def select_image(
        self,
        framework: str,
        gpu_arch: Optional[str] = None,
        override_image: Optional[str] = None,
    ) -> str:
        """
        Select appropriate Docker image for benchmark.
        
        Args:
            framework: Framework name ("vllm" or "sglang")
            gpu_arch: GPU architecture (e.g., "gfx942", "sm_90"). 
                     Auto-detected if not specified.
            override_image: If provided, use this image instead of auto-selection
        
        Returns:
            Docker image name
        
        Raises:
            ValueError: If no suitable image found
        """
        # Use override if provided
        if override_image:
            logger.info(f"Using override image: {override_image}")
            return override_image
        
        # Auto-detect GPU architecture if not provided
        if gpu_arch is None:
            vendor, detected_arch = detect_gpu()
            gpu_arch = detected_arch
            logger.info(f"Auto-detected GPU: {vendor.name}, arch: {gpu_arch}")
        
        # Normalize framework name
        framework = framework.lower()
        
        # Get framework mapping
        framework_mapping = self._mapping.get(framework, {})
        if not framework_mapping:
            raise ValueError(
                f"No image mapping found for framework '{framework}'. "
                f"Available frameworks: {list(self._mapping.keys())}. "
                f"Please add '{framework}' to {self.config_path}"
            )
        
        # Try to find image for the architecture
        if gpu_arch and gpu_arch in framework_mapping:
            image = framework_mapping[gpu_arch]
            logger.info(f"Selected image for {framework}/{gpu_arch}: {image}")
            return image
        
        # No matching architecture found - raise error
        available_archs = list(framework_mapping.keys())
        raise ValueError(
            f"No image found for GPU architecture '{gpu_arch}' with framework '{framework}'. "
            f"Available architectures: {available_archs}. "
            f"Please add '{gpu_arch}' to {self.config_path} or use --docker-image to override."
        )
    
    def get_runner_type(self, gpu_arch: Optional[str] = None) -> str:
        """
        Get InferenceX runner type based on GPU architecture.
        
        Args:
            gpu_arch: GPU architecture. Auto-detected if not specified.
        
        Returns:
            Runner type string (e.g., "mi300x", "h100")
        """
        if gpu_arch is None:
            _, gpu_arch = detect_gpu()
        
        # Map GPU architectures to runner types
        arch_to_runner = {
            # AMD GPUs
            "gfx942": "mi300x",   # MI300X
            "gfx950": "mi355x",   # MI355X
            "gfx1100": "mi325x",  # MI325X
            # NVIDIA GPUs
            "sm_80": "a100",      # A100
            "sm_90": "h100",      # H100
            "sm_90a": "h200",     # H200
            "sm_100": "b200",     # B200
        }
        
        runner = arch_to_runner.get(gpu_arch, None)
        if runner is None:
            logger.warning(f"No runner type found for {gpu_arch}")
            raise ValueError(f"No runner type found for {gpu_arch}")
        logger.debug(f"Runner type for {gpu_arch}: {runner}")
        return runner
    
    def list_available_images(self, framework: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """
        List all available images.
        
        Args:
            framework: Filter by framework name, or None for all
        
        Returns:
            Dictionary of framework -> arch -> image mappings
        """
        if framework:
            return {framework: self._mapping.get(framework.lower(), {})}
        return self._mapping.copy()
