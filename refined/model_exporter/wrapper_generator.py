"""Wrapper model generator for torch.export compatibility"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
from .input_analyzer import TensorInfo
from mmengine.logging import print_log


class WrapperGenerator:
    """Generate wrapper models that accept flattened tensor inputs"""

    def __init__(self, original_model: nn.Module, input_analysis: Dict[str, Any]):
        """
        Initialize wrapper generator

        Args:
            original_model: The original model to wrap
            input_analysis: Analysis results from InputAnalyzer
        """
        self.original_model = original_model
        self.input_analysis = input_analysis
        self.tensor_infos = input_analysis['tensor_infos']
        self.flatten_mapping = input_analysis['flatten_mapping']
        self.structure_tree = input_analysis['structure_tree']

    def generate_wrapper_model(self) -> nn.Module:
        """
        Generate a wrapper model that accepts flattened tensors

        Returns:
            Wrapped model compatible with torch.export
        """
        print_log("Generating wrapper model...", logger='current')

        # Create dynamic wrapper class
        wrapper_class = self._create_wrapper_class()

        # Create wrapper instance
        wrapper_model = wrapper_class(
            original_model=self.original_model,
            structure_tree=self.structure_tree,
            flatten_mapping=self.flatten_mapping
        )

        print_log("Wrapper model generated successfully!", logger='current')
        return wrapper_model

    def _create_wrapper_class(self) -> type:
        """Dynamically create a wrapper class based on input structure"""

        # Generate forward method arguments
        arg_names = []
        for idx in range(len(self.tensor_infos)):
            arg_names.append(f"tensor_{idx}")

        # Generate forward method code
        forward_code = self._generate_forward_method(arg_names)

        # Create the wrapper class
        class_dict = {
            '__init__': self._generate_init_method(),
            'forward': self._generate_forward_method_body(arg_names),
            '_reconstruct_inputs': self._generate_reconstruct_method(),
            'get_input_info': self._generate_get_info_method()
        }

        # Create the class dynamically
        wrapper_class = type('ModelWrapper', (nn.Module,), class_dict)
        wrapper_class._forward_source = forward_code  # Store for debugging

        return wrapper_class

    def _generate_init_method(self):
        """Generate __init__ method for wrapper class"""
        def __init__(self, original_model, structure_tree, flatten_mapping):
            super().__init__()
            self.original_model = original_model
            self.structure_tree = structure_tree
            self.flatten_mapping = flatten_mapping

        return __init__

    def _generate_forward_method_body(self, arg_names: List[str]):
        """Generate the forward method body"""
        def forward(self, *args):
            # Convert args to list for easier manipulation
            tensor_list = list(args)

            # Reconstruct the original input structure
            reconstructed_inputs = self._reconstruct_inputs(tensor_list)

            # Run the original model
            return self.original_model(reconstructed_inputs, mode='predict')

        return forward

    def _generate_forward_method(self, arg_names: List[str]) -> str:
        """Generate forward method source code (for documentation)"""
        args_str = ', '.join(arg_names)
        method_code = f"""
def forward(self, {args_str}):
    # Convert args to list for easier manipulation
    tensor_list = [{args_str}]

    # Reconstruct the original input structure
    reconstructed_inputs = self._reconstruct_inputs(tensor_list)

    # Run the original model
    return self.original_model(reconstructed_inputs, mode='predict')
        """
        return method_code.strip()

    def _generate_reconstruct_method(self):
        """Generate method to reconstruct original input structure"""
        def _reconstruct_inputs(self, tensor_list: List[torch.Tensor]) -> Dict[str, Any]:
            """Reconstruct original input structure from flattened tensors"""
            inputs = {}

            # Reconstruct based on flatten mapping
            for idx, path in self.flatten_mapping.items():
                if idx < len(tensor_list):
                    tensor = tensor_list[idx]
                    self._set_nested_value(inputs, path, tensor)

            return inputs

        return _reconstruct_inputs

    def _set_nested_value(self, obj: Dict[str, Any], path: str, value: Any):
        """Set nested value in dictionary based on path string"""
        keys = path.split('.')

        # Handle 'inputs.' prefix
        if keys[0] == 'inputs':
            keys = keys[1:]

        current = obj
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _generate_get_info_method(self):
        """Generate method to get input information"""
        def get_input_info(self) -> Dict[str, Any]:
            """Get information about the input structure"""
            return {
                'num_inputs': len(self.flatten_mapping),
                'flatten_mapping': self.flatten_mapping,
                'tensor_infos': self.tensor_infos
            }

        return get_input_info

    def generate_export_args(self, device: torch.device = None) -> Tuple[torch.ArgsType, ...]:
        """
        Generate example arguments for model export

        Args:
            device: Device to create tensors on

        Returns:
            Tuple of example tensors
        """
        print_log("Generating export arguments...", logger='current')

        if device is None:
            device = next(self.original_model.parameters()).device

        export_args = []

        for tensor_info in self.tensor_infos:
            # Create tensor with same shape and dtype
            tensor = torch.randn(tensor_info.shape, dtype=tensor_info.dtype, device=device)

            # Adjust range based on tensor type
            if 'coord' in tensor_info.path.lower() or 'point' in tensor_info.path.lower():
                # Coordinates need larger range
                tensor = tensor * 50.0
            elif 'voxel' in tensor_info.path.lower():
                # Voxel features might need different scaling
                tensor = tensor * 10.0
            elif 'num_points' in tensor_info.path.lower():
                # Number of points should be positive integers
                tensor = torch.randint(
                    low=1,
                    high=max(tensor_info.shape) + 1,
                    size=tensor_info.shape,
                    dtype=tensor_info.dtype,
                    device=device
                )

            export_args.append(tensor)

        print_log(f"Generated {len(export_args)} export arguments", logger='current')
        return tuple(export_args)

    def validate_wrapper(self, wrapper_model: nn.Module) -> bool:
        """
        Validate the wrapper model by comparing outputs

        Args:
            wrapper_model: The generated wrapper model

        Returns:
            True if validation passes, False otherwise
        """
        print_log("Validating wrapper model...", logger='current')

        try:
            # Generate test inputs
            test_args = self.generate_export_args()

            # Get outputs from original model (reconstructed)
            test_inputs = self._reconstruct_inputs_for_test(test_args)
            with torch.no_grad():
                original_output = self.original_model(test_inputs, mode='predict')

            # Get outputs from wrapper model
            with torch.no_grad():
                wrapper_output = wrapper_model(*test_args)

            # Compare outputs
            is_valid = self._compare_outputs(original_output, wrapper_output)

            if is_valid:
                print_log("Wrapper validation passed!", logger='current')
            else:
                print_log("Wrapper validation failed! Outputs differ.", logger='current')

            return is_valid

        except Exception as e:
            print_log(f"Wrapper validation error: {e}", logger='current')
            return False

    def _reconstruct_inputs_for_test(self, tensor_list: List[torch.Tensor]) -> Dict[str, Any]:
        """Reconstruct inputs for testing"""
        inputs = {}
        for idx, path in self.flatten_mapping.items():
            if idx < len(tensor_list):
                tensor = tensor_list[idx]
                self._set_nested_value(inputs, path, tensor)
        return inputs

    def _compare_outputs(self, original_output: Any, wrapper_output: Any, tolerance: float = 1e-5) -> bool:
        """Compare outputs from original and wrapper models"""
        try:
            if isinstance(original_output, dict):
                if not isinstance(wrapper_output, dict):
                    return False

                # Compare dictionary outputs
                for key in original_output:
                    if key not in wrapper_output:
                        return False

                    orig_val = original_output[key]
                    wrap_val = wrapper_output[key]

                    if isinstance(orig_val, torch.Tensor) and isinstance(wrap_val, torch.Tensor):
                        if not torch.allclose(orig_val, wrap_val, atol=tolerance):
                            print_log(f"Output difference in key '{key}'", logger='current')
                            return False
                    elif orig_val != wrap_val:
                        return False

                return True

            elif isinstance(original_output, torch.Tensor):
                if not isinstance(wrapper_output, torch.Tensor):
                    return False
                return torch.allclose(original_output, wrapper_output, atol=tolerance)

            else:
                return original_output == wrapper_output

        except Exception as e:
            print_log(f"Error comparing outputs: {e}", logger='current')
            return False