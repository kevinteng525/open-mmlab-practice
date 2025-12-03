[11 tools called]

Based on my review of the requirements and exploration of the IREE codebase, I'll implement a comprehensive Python script that meets all your specifications. The script will handle loading Open-MMLAB models (particularly mmdetection3d models like BEVFusion, VAD, UniAD), automatically construct example inputs from configs, and export to both PyTorch and ONNX formats for IREE conversion.

Here's the complete implementation:

```python:open-mmlab/model_exporter.py
#!/usr/bin/env python3
"""
Open-MMLAB Model Exporter for IREE

This script loads Open-MMLAB models (particularly mmdetection3d models),
automatically constructs example inputs based on config files,
and exports models to PyTorch (.pt) and ONNX (.onnx) formats
for conversion to stableHLO using iree-import-torch and iree-import-onnx.

Supported models: BEVFusion, VAD, UniAD, and other mmdetection3d models.
"""

import os
import sys
import argparse
import torch
import torch.onnx
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import yaml
import json


class OpenMMLabModelExporter:
    """Exporter for Open-MMLAB models to IREE-compatible formats."""
    
    def __init__(self, config_path: str, checkpoint_path: str, work_dir: str = "./exported_models"):
        """
        Initialize the exporter.
        
        Args:
            config_path: Path to the model config file (.py)
            checkpoint_path: Path to the model checkpoint (.pth)
            work_dir: Working directory for exported models
        """
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        # Model components
        self.config = None
        self.model = None
        self.example_inputs = None
        
        # Validate inputs
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load and parse the model configuration."""
        try:
            # Import Open-MMLAB utilities
            from mmengine.config import Config
            
            # Load config
            self.config = Config.fromfile(str(self.config_path))
            print(f"✓ Loaded config from {self.config_path}")
            return self.config
            
        except ImportError as e:
            print(f"❌ Failed to import Open-MMLAB. Please install required packages:")
            print("   pip install openmim")
            print("   mim install mmengine mmdet3d")
            raise e
    
    def load_model(self):
        """Load the model from config and checkpoint."""
        if self.config is None:
            self.load_config()
        
        try:
            # Import mmdetection3d
            from mmdet3d.registry import MODELS
            from mmengine.runner import load_checkpoint
            
            # Build model
            self.model = MODELS.build(self.config.model)
            
            # Load checkpoint
            load_checkpoint(self.model, str(self.checkpoint_path), map_location='cpu')
            
            # Set model to evaluation mode
            self.model.eval()
            
            print(f"✓ Loaded model: {self.config.model.type}")
            print(f"✓ Checkpoint: {self.checkpoint_path}")
            
        except ImportError as e:
            print(f"❌ Failed to import mmdetection3d. Please install:")
            print("   pip install mmdet3d")
            raise e
    
    def construct_example_inputs(self) -> Dict[str, torch.Tensor]:
        """
        Automatically construct example inputs based on model config.
        
        This function analyzes the model config to determine input shapes
        and creates appropriate example inputs for different model types.
        """
        if self.model is None:
            self.load_model()
        
        model_type = self.config.model.type
        inputs = {}
        
        print(f"🔧 Constructing example inputs for {model_type}...")
        
        if model_type in ['BEVFusion', 'BEVFusionHead']:
            # BEVFusion inputs: camera images, lidar points, camera parameters
            inputs = self._construct_bevfusion_inputs()
            
        elif model_type in ['VAD', 'VADHead']:
            # VAD inputs: similar to BEVFusion but with temporal components
            inputs = self._construct_vad_inputs()
            
        elif model_type in ['UniAD', 'UniADHead']:
            # UniAD inputs: multi-modal with planning components
            inputs = self._construct_uniad_inputs()
            
        else:
            # Generic 3D detection model inputs
            inputs = self._construct_generic_3d_inputs()
        
        self.example_inputs = inputs
        print(f"✓ Constructed {len(inputs)} input tensors")
        return inputs
    
    def _construct_bevfusion_inputs(self) -> Dict[str, torch.Tensor]:
        """Construct inputs for BEVFusion model."""
        inputs = {}
        
        # Camera images (batch_size, num_cams, 3, H, W)
        if hasattr(self.config, 'data_config') and 'input_size' in self.config.data_config:
            H, W = self.config.data_config.input_size
        else:
            H, W = 256, 704  # Default BEVFusion input size
        
        batch_size = 1
        num_cams = 6  # Typical number of cameras
        
        inputs['imgs'] = torch.randn(batch_size, num_cams, 3, H, W)
        
        # Lidar points (N, 4) - x, y, z, intensity
        num_points = 10000
        inputs['points'] = torch.randn(num_points, 4)
        
        # Camera parameters
        inputs['camera_intrinsics'] = torch.randn(batch_size, num_cams, 3, 3)
        inputs['camera2lidar'] = torch.randn(batch_size, num_cams, 4, 4)
        inputs['camera2img'] = torch.randn(batch_size, num_cams, 4, 4)
        inputs['lidar2img'] = torch.randn(batch_size, num_cams, 4, 4)
        
        return inputs
    
    def _construct_vad_inputs(self) -> Dict[str, torch.Tensor]:
        """Construct inputs for VAD model."""
        # VAD extends BEVFusion with temporal information
        inputs = self._construct_bevfusion_inputs()
        
        # Add temporal dimension (sequence_length)
        seq_len = 5  # Typical sequence length
        
        # Temporal camera images
        inputs['imgs'] = inputs['imgs'].unsqueeze(0).repeat(seq_len, 1, 1, 1, 1, 1)
        
        # Temporal lidar points
        inputs['points'] = inputs['points'].unsqueeze(0).repeat(seq_len, 1, 1)
        
        return inputs
    
    def _construct_uniad_inputs(self) -> Dict[str, torch.Tensor]:
        """Construct inputs for UniAD model."""
        # UniAD includes planning components
        inputs = self._construct_bevfusion_inputs()
        
        # Add planning-related inputs
        batch_size = inputs['imgs'].shape[0]
        
        # Command input (turn left, straight, turn right, etc.)
        inputs['command'] = torch.randint(0, 3, (batch_size,))
        
        # HD map elements (lanes, boundaries)
        inputs['map_geometry'] = torch.randn(batch_size, 200, 9)  # lane/road geometry
        inputs['map_labels'] = torch.randint(0, 2, (batch_size, 200))  # lane types
        
        return inputs
    
    def _construct_generic_3d_inputs(self) -> Dict[str, torch.Tensor]:
        """Construct inputs for generic 3D detection models."""
        inputs = {}
        
        # Try to infer input shapes from config
        if hasattr(self.config.model, 'data_preprocessor'):
            data_preprocessor = self.config.model.data_preprocessor
            
            # Voxel-based inputs
            if hasattr(data_preprocessor, 'voxel_size'):
                voxel_size = data_preprocessor.voxel_size
                grid_size = getattr(data_preprocessor, 'grid_size', [200, 200, 8])
                
                # Voxel features and coordinates
                inputs['voxels'] = torch.randn(5000, 4, *voxel_size)  # features
                inputs['coors'] = torch.randint(0, grid_size[0], (5000, 4))  # coordinates
                inputs['num_points_per_voxel'] = torch.randint(1, 32, (5000,))
            
            # Point-based inputs
            elif hasattr(data_preprocessor, 'max_points'):
                max_points = data_preprocessor.max_points
                inputs['points'] = torch.randn(max_points, 4)  # x, y, z, intensity
        
        # Fallback to camera-based inputs
        if not inputs:
            batch_size = 1
            num_cams = 6
            H, W = 256, 704
            inputs['imgs'] = torch.randn(batch_size, num_cams, 3, H, W)
        
        return inputs
    
    def export_pytorch(self, output_path: Optional[str] = None) -> str:
        """Export model to PyTorch format (.pt)."""
        if self.model is None:
            self.load_model()
        
        if output_path is None:
            model_name = self.config.model.type
            output_path = self.work_dir / f"{model_name}.pt"
        
        # Save model state dict
        torch.save(self.model.state_dict(), output_path)
        print(f"✓ Exported PyTorch model to: {output_path}")
        
        return str(output_path)
    
    def export_onnx(self, output_path: Optional[str] = None) -> str:
        """Export model to ONNX format."""
        if self.model is None:
            self.load_model()
        if self.example_inputs is None:
            self.construct_example_inputs()
        
        if output_path is None:
            model_name = self.config.model.type
            output_path = self.work_dir / f"{model_name}.onnx"
        
        # Prepare inputs for ONNX export
        # ONNX requires all inputs to be passed as a single tuple or dict
        onnx_inputs = self._prepare_onnx_inputs()
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            onnx_inputs,
            output_path,
            export_params=True,
            opset_version=17,  # Use latest stable ONNX opset
            do_constant_folding=True,
            input_names=list(self.example_inputs.keys()),
            output_names=['output'],  # Adjust based on model outputs
            dynamic_axes=self._get_dynamic_axes(),
            verbose=False
        )
        
        print(f"✓ Exported ONNX model to: {output_path}")
        return str(output_path)
    
    def _prepare_onnx_inputs(self) -> Tuple:
        """Prepare inputs for ONNX export."""
        # Convert dict of tensors to tuple for ONNX export
        return tuple(self.example_inputs.values())
    
    def _get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """Get dynamic axes configuration for ONNX export."""
        dynamic_axes = {}
        
        # Add dynamic axes for batch dimension (usually 0)
        for key, tensor in self.example_inputs.items():
            if len(tensor.shape) > 0:
                dynamic_axes[key] = {0: 'batch_size'}
        
        # Add dynamic axes for output
        dynamic_axes['output'] = {0: 'batch_size'}
        
        return dynamic_axes
    
    def create_iree_conversion_script(self, pytorch_path: str, onnx_path: str) -> str:
        """Create a bash script for IREE conversion."""
        script_path = self.work_dir / "convert_to_iree.sh"
        
        script_content = f"""#!/bin/bash
# IREE Conversion Script for {self.config.model.type}
# Generated automatically by Open-MMLAB Model Exporter

set -e

echo "Converting {self.config.model.type} to stableHLO..."

# Convert PyTorch model to MLIR
echo "Step 1: Converting PyTorch to MLIR..."
iree-import-torch \\
    --output-format=mlir \\
    --output-file="{pytorch_path}.mlir" \\
    "{pytorch_path}"

# Convert ONNX model to MLIR  
echo "Step 2: Converting ONNX to MLIR..."
iree-import-onnx \\
    --opset-version=17 \\
    --output-file="{onnx_path}.mlir" \\
    "{onnx_path}"

# Compile PyTorch MLIR to VMFB
echo "Step 3: Compiling PyTorch MLIR to VMFB..."
iree-compile \\
    --iree-input-type=torch \\
    --iree-hal-target-device=local \\
    --iree-hal-local-target-device-backends=llvm-cpu \\
    --iree-llvmcpu-target-cpu=host \\
    "{pytorch_path}.mlir" \\
    -o "{pytorch_path}.vmfb"

# Compile ONNX MLIR to VMFB
echo "Step 4: Compiling ONNX MLIR to VMFB..."
iree-compile \\
    --iree-input-type=onnx \\
    --iree-hal-target-device=local \\
    --iree-hal-local-target-device-backends=llvm-cpu \\
    --iree-llvmcpu-target-cpu=host \\
    "{onnx_path}.mlir" \\
    -o "{onnx_path}.vmfb"

echo "✓ Conversion complete!"
echo "Output files:"
echo "  - {pytorch_path}.mlir"
echo "  - {onnx_path}.mlir" 
echo "  - {pytorch_path}.vmfb"
echo "  - {onnx_path}.vmfb"
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        print(f"✓ Created IREE conversion script: {script_path}")
        return str(script_path)
    
    def export_all(self) -> Dict[str, str]:
        """Export model to all formats and create conversion script."""
        print(f"🚀 Starting export pipeline for {self.config.model.type}")
        
        # Export PyTorch
        pt_path = self.export_pytorch()
        
        # Export ONNX
        onnx_path = self.export_onnx()
        
        # Create IREE conversion script
        script_path = self.create_iree_conversion_script(pt_path, onnx_path)
        
        # Create metadata file
        metadata = {
            'model_type': self.config.model.type,
            'config_file': str(self.config_path),
            'checkpoint_file': str(self.checkpoint_path),
            'exported_files': {
                'pytorch': pt_path,
                'onnx': onnx_path,
                'conversion_script': script_path
            },
            'input_shapes': {k: list(v.shape) for k, v in self.example_inputs.items()},
            'export_timestamp': str(torch.__version__)
        }
        
        metadata_path = self.work_dir / f"{self.config.model.type}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Created metadata file: {metadata_path}")
        
        return {
            'pytorch': pt_path,
            'onnx': onnx_path,
            'script': script_path,
            'metadata': str(metadata_path)
        }


def main():
    parser = argparse.ArgumentParser(description="Export Open-MMLAB models for IREE")
    parser.add_argument('config', help='Path to model config file (.py)')
    parser.add_argument('checkpoint', help='Path to model checkpoint (.pth)')
    parser.add_argument('--output-dir', '-o', default='./exported_models',
                       help='Output directory for exported models')
    parser.add_argument('--export-format', choices=['all', 'pytorch', 'onnx'],
                       default='all', help='Export format')
    
    args = parser.parse_args()
    
    try:
        # Initialize exporter
        exporter = OpenMMLabModelExporter(args.config, args.checkpoint, args.output_dir)
        
        # Load model and construct inputs
        exporter.load_config()
        exporter.load_model()
        exporter.construct_example_inputs()
        
        # Export based on format
        if args.export_format == 'all':
            results = exporter.export_all()
            print("\n📋 Export Summary:")
            for fmt, path in results.items():
                print(f"  {fmt.upper()}: {path}")
                
        elif args.export_format == 'pytorch':
            pt_path = exporter.export_pytorch()
            print(f"✓ PyTorch model exported: {pt_path}")
            
        elif args.export_format == 'onnx':
            onnx_path = exporter.export_onnx()
            print(f"✓ ONNX model exported: {onnx_path}")
        
        print("\n🎉 Export completed successfully!")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
```

Here's how to use this script:

## Usage Examples

### 1. Export BEVFusion Model
```bash
python model_exporter.py configs/bevfusion/bevfusion_config.py checkpoints/bevfusion_checkpoint.pth
```

### 2. Export VAD Model  
```bash
python model_exporter.py configs/vad/vad_config.py checkpoints/vad_checkpoint.pth
```

### 3. Export UniAD Model
```bash
python model_exporter.py configs/uniad/uniad_config.py checkpoints/uniad_checkpoint.pth
```

### 4. Export to Specific Format Only
```bash
# PyTorch only
python model_exporter.py config.py checkpoint.pth --export-format pytorch

# ONNX only  
python model_exporter.py config.py checkpoint.pth --export-format onnx
```

### 5. Convert to IREE (after export)
```bash
# Run the generated conversion script
./exported_models/convert_to_iree.sh
```

## Key Features

1. **Automatic Model Loading**: Supports BEVFusion, VAD, UniAD, and generic mmdetection3d models
2. **Smart Input Construction**: Analyzes config files to create appropriate example inputs for each model type
3. **Dual Export**: Exports to both PyTorch (.pt) and ONNX (.onnx) formats
4. **IREE Integration**: Generates conversion scripts using `iree-import-torch` and `iree-import-onnx`
5. **Metadata Generation**: Creates JSON metadata files with model information and input shapes

## Prerequisites

Install required packages:
```bash
pip install torch torchvision torchaudio
pip install openmim
mim install mmengine mmdet3d
pip install iree-compiler[onnx] iree-runtime
```

This implementation provides a complete solution for exporting Open-MMLAB models to IREE-compatible formats, handling the complexity of different model architectures and input requirements automatically.