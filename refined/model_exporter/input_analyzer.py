"""Input data structure analyzer for MMDetection3D models"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Union, Optional
from dataclasses import dataclass
from mmengine.logging import print_log


@dataclass
class TensorInfo:
    """Information about a tensor in the input structure"""
    path: str  # e.g., 'inputs.voxels'
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    value_range: Optional[Tuple[float, float]] = None
    description: Optional[str] = None


class InputAnalyzer:
    """Analyze and extract tensor information from model inputs"""

    def __init__(self):
        self.tensor_infos: List[TensorInfo] = []
        self.structure_tree: Dict[str, Any] = {}
        self.flatten_mapping: Dict[int, str] = {}  # Maps flat index to path
        self.total_params: int = 0

    def analyze_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the input structure and extract all tensor information

        Args:
            inputs: Model inputs (typically a dict)

        Returns:
            Dictionary containing analysis results
        """
        print_log("Analyzing input structure...", logger='current')

        # Reset state
        self.tensor_infos = []
        self.structure_tree = {}
        self.flatten_mapping = {}
        self.total_params = 0

        # Recursively analyze the input structure
        self._analyze_recursive(inputs, path="inputs")

        # Create flatten mapping
        for idx, tensor_info in enumerate(self.tensor_infos):
            self.flatten_mapping[idx] = tensor_info.path

        result = {
            'tensor_infos': self.tensor_infos,
            'structure_tree': self.structure_tree,
            'flatten_mapping': self.flatten_mapping,
            'total_tensors': len(self.tensor_infos),
            'total_params': self.total_params,
            'input_summary': self._generate_summary()
        }

        print_log(f"Found {len(self.tensor_infos)} tensors in input structure", logger='current')
        return result

    def _analyze_recursive(self, obj: Any, path: str):
        """Recursively analyze object structure and extract tensor info"""

        if isinstance(obj, torch.Tensor):
            # Found a tensor
            tensor_info = self._extract_tensor_info(obj, path)
            self.tensor_infos.append(tensor_info)

            # Count parameters
            self.total_params += np.prod(tensor_info.shape)

            # Add to structure tree
            self._add_to_structure_tree(path, {
                'type': 'tensor',
                'shape': tensor_info.shape,
                'dtype': str(tensor_info.dtype),
                'description': tensor_info.description
            })

        elif isinstance(obj, dict):
            # Handle dictionary
            structure_node = {
                'type': 'dict',
                'keys': list(obj.keys()),
                'children': {}
            }

            for key, value in obj.items():
                child_path = f"{path}.{key}" if path != "" else key
                self._analyze_recursive(value, child_path)
                structure_node['children'][key] = child_path

            self._add_to_structure_tree(path, structure_node)

        elif isinstance(obj, list):
            # Handle list
            structure_node = {
                'type': 'list',
                'length': len(obj),
                'children': {}
            }

            for idx, item in enumerate(obj):
                child_path = f"{path}[{idx}]"
                self._analyze_recursive(item, child_path)
                structure_node['children'][f'[{idx}]'] = child_path

            self._add_to_structure_tree(path, structure_node)

        elif isinstance(obj, (int, float, str, bool)):
            # Handle primitive types
            self._add_to_structure_tree(path, {
                'type': type(obj).__name__,
                'value': obj
            })

        else:
            # Handle other types (e.g., custom objects)
            try:
                # Try to get dictionary representation
                if hasattr(obj, '__dict__'):
                    self._analyze_recursive(obj.__dict__, path)
                else:
                    self._add_to_structure_tree(path, {
                        'type': type(obj).__name__,
                        'description': str(obj)
                    })
            except:
                self._add_to_structure_tree(path, {
                    'type': type(obj).__name__,
                    'description': 'Unknown object'
                })

    def _extract_tensor_info(self, tensor: torch.Tensor, path: str) -> TensorInfo:
        """Extract information from a tensor"""

        # Determine value range if tensor has data
        value_range = None
        if tensor.numel() > 0:
            try:
                tensor_flat = tensor.flatten()
                if tensor_flat.numel() > 1000:
                    # Sample to avoid memory issues
                    indices = torch.randperm(tensor_flat.numel())[:1000]
                    tensor_sample = tensor_flat[indices]
                else:
                    tensor_sample = tensor_flat

                value_range = (float(tensor_sample.min()), float(tensor_sample.max()))
            except:
                pass

        # Generate description based on path
        description = self._generate_tensor_description(path)

        return TensorInfo(
            path=path,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device=tensor.device,
            value_range=value_range,
            description=description
        )

    def _generate_tensor_description(self, path: str) -> str:
        """Generate a description for the tensor based on its path"""
        path_lower = path.lower()

        if 'voxel' in path_lower:
            if 'feature' in path_lower:
                return "Voxel feature tensor"
            elif 'coord' in path_lower:
                return "Voxel coordinate tensor"
            elif 'num_points' in path_lower:
                return "Number of points per voxel"
            return "Voxel tensor"

        elif 'point' in path_lower:
            if 'coord' in path_lower:
                return "Point coordinates"
            elif 'feat' in path_lower or 'feature' in path_lower:
                return "Point features"
            return "Point cloud tensor"

        elif 'img' in path_lower:
            if 'fea' in path_lower:
                return "Image feature tensor"
            return "Image tensor"

        elif 'bbox' in path_lower:
            return "Bounding box tensor"

        elif 'label' in path_lower:
            return "Label tensor"

        elif 'gt_' in path_lower:
            return "Ground truth tensor"

        return "Input tensor"

    def _add_to_structure_tree(self, path: str, info: Dict[str, Any]):
        """Add information to the structure tree"""
        keys = path.split('.')
        current = self.structure_tree

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = info

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the input analysis"""

        # Categorize tensors by type
        categories = {
            'voxel': [],
            'point': [],
            'image': [],
            'bbox': [],
            'label': [],
            'other': []
        }

        for tensor_info in self.tensor_infos:
            path_lower = tensor_info.path.lower()

            if 'voxel' in path_lower:
                categories['voxel'].append(tensor_info)
            elif 'point' in path_lower:
                categories['point'].append(tensor_info)
            elif 'img' in path_lower:
                categories['image'].append(tensor_info)
            elif 'bbox' in path_lower:
                categories['bbox'].append(tensor_info)
            elif 'label' in path_lower:
                categories['label'].append(tensor_info)
            else:
                categories['other'].append(tensor_info)

        return {
            'categories': categories,
            'largest_tensor': max(self.tensor_infos, key=lambda x: np.prod(x.shape)) if self.tensor_infos else None,
            'device_distribution': self._get_device_distribution(),
            'dtype_distribution': self._get_dtype_distribution()
        }

    def _get_device_distribution(self) -> Dict[str, int]:
        """Get distribution of tensors across devices"""
        distribution = {}
        for tensor_info in self.tensor_infos:
            device_str = str(tensor_info.device)
            distribution[device_str] = distribution.get(device_str, 0) + 1
        return distribution

    def _get_dtype_distribution(self) -> Dict[str, int]:
        """Get distribution of tensor data types"""
        distribution = {}
        for tensor_info in self.tensor_infos:
            dtype_str = str(tensor_info.dtype)
            distribution[dtype_str] = distribution.get(dtype_str, 0) + 1
        return distribution

    def generate_random_inputs(self) -> List[torch.Tensor]:
        """
        Generate random tensors based on analyzed input structure

        Returns:
            List of random tensors in the order of flatten_mapping
        """
        print_log("Generating random input tensors...", logger='current')

        random_tensors = []

        for tensor_info in self.tensor_infos:
            random_tensor = self._generate_random_tensor(tensor_info)
            random_tensors.append(random_tensor)

        print_log(f"Generated {len(random_tensors)} random tensors", logger='current')
        return random_tensors

    def _generate_random_tensor(self, tensor_info: TensorInfo) -> torch.Tensor:
        """Generate a random tensor based on TensorInfo"""

        # Determine random range based on tensor type
        if tensor_info.value_range:
            min_val, max_val = tensor_info.value_range
            # Expand range slightly for safety
            range_expansion = (max_val - min_val) * 0.1
            min_val -= range_expansion
            max_val += range_expansion
        else:
            # Default ranges based on tensor type
            if 'coord' in tensor_info.path.lower() or 'point' in tensor_info.path.lower():
                min_val, max_val = -50.0, 50.0
            elif 'img' in tensor_info.path.lower():
                min_val, max_val = 0.0, 255.0
            else:
                min_val, max_val = 0.0, 1.0

        # Generate random tensor
        shape = tensor_info.shape

        # Special handling for specific tensor types
        if 'num_points' in tensor_info.path.lower():
            # Number of points per voxel should be integer
            random_tensor = torch.randint(
                low=1,
                high=max(shape) + 1,
                size=shape,
                dtype=torch.int32,
                device=self._get_device(tensor_info)
            )
        elif 'coord' in tensor_info.path.lower():
            # Coordinates typically need larger range
            random_tensor = torch.uniform(
                min_val,
                max_val,
                size=shape,
                dtype=tensor_info.dtype,
                device=self._get_device(tensor_info)
            )
        else:
            random_tensor = torch.empty(shape, dtype=tensor_info.dtype, device=self._get_device(tensor_info))
            random_tensor.uniform_(min_val, max_val)

        return random_tensor

    def _get_device(self, tensor_info: TensorInfo) -> torch.device:
        """Get device for tensor generation"""
        # For generation, use CPU first, then move to target device
        return torch.device('cpu')