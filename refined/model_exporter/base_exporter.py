"""Base model exporter class"""

import os
import sys
import torch
from typing import Dict, List, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod

# Add MMDetection3D to path
sys.path.append('/Users/kevinteng/src/kevinteng525/open-mmlab/mmdetection3d')

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.logging import print_log


class BaseModelExporter(ABC):
    """Base class for model exporters"""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda:0'
    ):
        """
        Initialize the base exporter

        Args:
            config_path: Path to the model config file
            checkpoint_path: Path to the model checkpoint (optional)
            device: Device to run the model on
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device

        self.cfg = None
        self.model = None
        self.pipeline = None
        self.input_analyzer = None
        self.wrapper_generator = None

    def setup(self):
        """Setup the exporter by loading config, model, and pipeline"""
        print_log("Setting up model exporter...", logger='current')

        # Load configuration
        self.load_config()

        # Build model
        self.build_model()

        # Load checkpoint if provided
        if self.checkpoint_path:
            self.load_checkpoint()

        # Setup data pipeline
        self.setup_pipeline()

        print_log("Setup completed!", logger='current')

    def load_config(self):
        """Load model configuration from file"""
        print_log(f"Loading config from {self.config_path}...", logger='current')

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.cfg = Config.fromfile(self.config_path)

        # Set work_dir if not in config
        if 'work_dir' not in self.cfg:
            self.cfg.work_dir = './work_dir'

        print_log("Config loaded successfully!", logger='current')

    @abstractmethod
    def build_model(self):
        """Build the model from config"""
        pass

    def load_checkpoint(self):
        """Load model weights from checkpoint"""
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            print_log("Checkpoint not provided or not found, skipping...", logger='current')
            return

        print_log(f"Loading checkpoint from {self.checkpoint_path}...", logger='current')

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load weights
        self.model.load_state_dict(state_dict, strict=False)

        print_log("Checkpoint loaded successfully!", logger='current')

    @abstractmethod
    def setup_pipeline(self):
        """Setup the data processing pipeline"""
        pass

    @abstractmethod
    def export(self, output_path: str, format: str = 'both'):
        """Export the model to specified format"""
        pass