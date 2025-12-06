"""MMDetection3D model exporter implementation"""

import os
import sys
import torch
import torch.onnx
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Add MMDetection3D to path
sys.path.append('/Users/kevinteng/src/kevinteng525/open-mmlab/mmdetection3d')

from mmengine.config import Config
from mmengine.logging import print_log
from mmdet3d.models import build_detector

from .base_exporter import BaseModelExporter
from .input_analyzer import InputAnalyzer
from .data_generator import DataGenerator
from .wrapper_generator import WrapperGenerator


class MMDet3DExporter(BaseModelExporter):
    """Exporter for MMDetection3D models"""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda:0'
    ):
        super().__init__(config_path, checkpoint_path, device)

        # Initialize components
        self.input_analyzer = None
        self.data_generator = None
        self.wrapper_generator = None
        self.analysis_result = None
        self.sample_inputs = None
        self.wrapper_model = None

    def build_model(self):
        """Build MMDetection3D model from config"""
        print_log("Building MMDetection3D model...", logger='current')

        # Build model from config
        self.model = build_detector(self.cfg.model)

        # Move to device
        self.model.to(self.device)

        # Set to evaluation mode
        self.model.eval()

        print_log("Model built successfully!", logger='current')

    def setup_pipeline(self):
        """Setup data processing pipeline and components"""
        print_log("Setting up pipeline and components...", logger='current')

        # Initialize input analyzer
        self.input_analyzer = InputAnalyzer()

        # Initialize data generator
        self.data_generator = DataGenerator(self.cfg, self.device)
        self.data_generator.setup()

        print_log("Pipeline setup completed!", logger='current')

    def prepare_for_export(self):
        """Prepare model for export by analyzing inputs and generating wrapper"""
        print_log("Preparing for export...", logger='current')

        # Step 1: Generate sample data
        self.sample_inputs = self.data_generator.get_sample_batch()
        print_log("Sample data generated", logger='current')

        # Step 2: Analyze input structure
        inputs_dict = self.sample_inputs['inputs']
        self.analysis_result = self.input_analyzer.analyze_inputs(inputs_dict)
        print_log(f"Input analysis completed. Found {self.analysis_result['total_tensors']} tensors", logger='current')

        # Step 3: Validate model with sample data
        self._validate_model_with_sample()

        # Step 4: Generate wrapper model
        self.wrapper_generator = WrapperGenerator(self.model, self.analysis_result)
        self.wrapper_model = self.wrapper_generator.generate_wrapper_model()

        # Step 5: Validate wrapper
        if self.wrapper_generator.validate_wrapper(self.wrapper_model):
            print_log("Wrapper model validation passed!", logger='current')
        else:
            print_log("Warning: Wrapper model validation failed!", logger='current')

        print_log("Export preparation completed!", logger='current')

    def _validate_model_with_sample(self):
        """Validate model with sample inputs"""
        try:
            print_log("Validating model with sample inputs...", logger='current')

            with torch.no_grad():
                # Try forward pass
                output = self.model(
                    self.sample_inputs['inputs'],
                    data_samples=self.sample_inputs['data_samples'],
                    mode='predict'
                )
                print_log("Model validation passed!", logger='current')

        except Exception as e:
            print_log(f"Model validation failed: {e}", logger='current')
            # Try with different mode
            try:
                with torch.no_grad():
                    output = self.model(
                        self.sample_inputs['inputs'],
                        mode='tensor'
                    )
                    print_log("Model validation with 'tensor' mode passed!", logger='current')
            except Exception as e2:
                print_log(f"Model validation with 'tensor' mode also failed: {e2}", logger='current')
                raise

    def export(self, output_dir: str, format: str = 'both'):
        """
        Export the model to specified format(s)

        Args:
            output_dir: Directory to save exported models
            format: Export format ('pt2', 'onnx', or 'both')
        """
        print_log(f"Starting export to {format} format(s)...", logger='current')

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare for export
        self.prepare_for_export()

        # Export based on format
        if format in ['pt2', 'both']:
            self._export_to_pt2(output_dir)

        if format in ['onnx', 'both']:
            self._export_to_onnx(output_dir)

        # Export metadata
        self._export_metadata(output_dir)

        print_log("Export completed!", logger='current')

    def _export_to_pt2(self, output_dir: Path):
        """Export model to PT2 format using torch.export"""
        print_log("Exporting to PT2 format...", logger='current')

        try:
            # Generate example inputs for export
            export_args = self.wrapper_generator.generate_export_args(self.device)

            # Export using torch.export
            exported_program = torch.export.export(
                self.wrapper_model,
                export_args
            )

            # Save exported program
            pt2_path = output_dir / "model.pt2"
            torch.export.save(exported_program, str(pt2_path))

            print_log(f"PT2 model saved to {pt2_path}", logger='current')

            # Try to reload and validate
            try:
                reloaded = torch.export.load(str(pt2_path))
                print_log("PT2 model reload validation passed!", logger='current')
            except Exception as e:
                print_log(f"PT2 model reload validation failed: {e}", logger='current')

        except Exception as e:
            print_log(f"PT2 export failed: {e}", logger='current')
            raise

    def _export_to_onnx(self, output_dir: Path):
        """Export model to ONNX format"""
        print_log("Exporting to ONNX format...", logger='current')

        try:
            # Generate example inputs for export
            export_args = self.wrapper_generator.generate_export_args(self.device)

            # Define dynamic axes
            dynamic_axes = self._get_dynamic_axes()

            # Create input names and output names
            input_names = [f"input_{i}" for i in range(len(export_args))]
            output_names = self._get_output_names()

            # Export to ONNX
            onnx_path = output_dir / "model.onnx"
            torch.onnx.export(
                self.wrapper_model,
                export_args,
                str(onnx_path),
                export_params=True,
                opset_version=17,  # Use recent opset version
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )

            print_log(f"ONNX model saved to {onnx_path}", logger='current')

            # Validate ONNX model
            self._validate_onnx_model(onnx_path, export_args, input_names, output_names)

        except Exception as e:
            print_log(f"ONNX export failed: {e}", logger='current')
            raise

    def _get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """Define dynamic axes for ONNX export"""
        dynamic_axes = {}

        # Define dynamic axes for inputs
        for i, tensor_info in enumerate(self.tensor_infos):
            input_name = f"input_{i}"
            dynamic_axes[input_name] = {}

            # Mark batch dimensions (usually 0) as dynamic
            shape = tensor_info.shape
            for dim, size in enumerate(shape):
                if size == 1 or dim == 0:  # Batch dimension
                    dynamic_axes[input_name][dim] = f"batch_{i}_{dim}"

        # Define dynamic axes for outputs (if known)
        # This is model-specific and would need to be adapted based on actual outputs
        dynamic_axes["output"] = {0: "batch_size"}

        return dynamic_axes

    def _get_output_names(self) -> List[str]:
        """Get output names for ONNX export"""
        # This would need to be adapted based on the actual model outputs
        # Common outputs for 3D detection models
        return [
            "pred_boxes_3d",
            "pred_scores_3d",
            "pred_labels_3d"
        ]

    @property
    def tensor_infos(self):
        """Get tensor infos from analysis result"""
        if self.analysis_result:
            return self.analysis_result['tensor_infos']
        return []

    def _validate_onnx_model(
        self,
        onnx_path: Path,
        export_args: Tuple[torch.Tensor, ...],
        input_names: List[str],
        output_names: List[str]
    ):
        """Validate exported ONNX model"""
        try:
            import onnx
            import onnxruntime as ort

            # Load ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)

            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(str(onnx_path))

            # Prepare inputs
            ort_inputs = {
                name: arg.cpu().numpy() for name, arg in zip(input_names, export_args)
            }

            # Run inference
            ort_outputs = ort_session.run(output_names, ort_inputs)

            # Compare with PyTorch output
            with torch.no_grad():
                torch_output = self.wrapper_model(*export_args)

            print_log("ONNX model validation passed!", logger='current')

        except ImportError:
            print_log("ONNX validation skipped: onnx or onnxruntime not installed", logger='current')
        except Exception as e:
            print_log(f"ONNX validation failed: {e}", logger='current')

    def _export_metadata(self, output_dir: Path):
        """Export metadata about the model and export process"""
        print_log("Exporting metadata...", logger='current')

        metadata = {
            'model_config': str(self.config_path),
            'checkpoint': self.checkpoint_path,
            'model_type': str(self.cfg.model.type) if hasattr(self.cfg, 'model') else 'Unknown',
            'export_info': {
                'total_inputs': self.analysis_result['total_tensors'] if self.analysis_result else 0,
                'input_structure': self.analysis_result['structure_tree'] if self.analysis_result else {},
                'tensor_details': [
                    {
                        'path': info.path,
                        'shape': list(info.shape),
                        'dtype': str(info.dtype),
                        'description': info.description
                    }
                    for info in self.tensor_infos
                ]
            },
            'notes': {
                'device': self.device,
                'wrapper_used': True,
                'supports_torch_export': True
            }
        }

        # Save metadata as JSON
        import json
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print_log(f"Metadata saved to {metadata_path}", logger='current')