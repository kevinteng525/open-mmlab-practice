"""Data generator for MMDetection3D models"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from mmengine.logging import print_log

# Add MMDetection3D to path
sys.path.append('/Users/kevinteng/src/kevinteng525/open-mmlab/mmdetection3d')

from mmengine.dataset import Compose
from mmdet3d.datasets import build_dataset


class DataGenerator:
    """Generate sample data for MMDetection3D models"""

    def __init__(self, cfg, device: str = 'cuda:0'):
        """
        Initialize data generator

        Args:
            cfg: Configuration object
            device: Device to run on
        """
        self.cfg = cfg
        self.device = device
        self.dataset = None
        self.test_pipeline = None

    def setup(self):
        """Setup dataset and data pipeline"""
        print_log("Setting up data generator...", logger='current')

        # Build dataset if possible
        self._setup_dataset()

        # Setup data pipeline
        self._setup_pipeline()

        print_log("Data generator setup completed!", logger='current')

    def _setup_dataset(self):
        """Setup dataset for getting real samples"""
        try:
            # Try to build dataset from config
            if 'test_dataloader' in self.cfg:
                dataset_cfg = self.cfg.test_dataloader.dataset
            elif 'val_dataloader' in self.cfg:
                dataset_cfg = self.cfg.val_dataloader.dataset
            elif 'train_dataloader' in self.cfg:
                dataset_cfg = self.cfg.train_dataloader.dataset
            else:
                print_log("No dataset config found, using synthetic data", logger='current')
                return

            # Override dataset settings for testing
            dataset_cfg.test_mode = True

            # Build dataset
            self.dataset = build_dataset(dataset_cfg)
            print_log(f"Dataset built successfully with {len(self.dataset)} samples", logger='current')

        except Exception as e:
            print_log(f"Failed to build dataset: {e}. Using synthetic data.", logger='current')
            self.dataset = None

    def _setup_pipeline(self):
        """Setup data processing pipeline"""
        try:
            # Get test pipeline from config
            if hasattr(self.cfg, 'test_pipeline'):
                self.test_pipeline = Compose(self.cfg.test_pipeline)
                print_log("Test pipeline loaded from config", logger='current')
            elif hasattr(self.cfg, 'pipeline'):
                # Use general pipeline as fallback
                pipeline_cfg = [p for p in self.cfg.pipeline if 'Load' not in str(p.get('type', ''))]
                self.test_pipeline = Compose(pipeline_cfg)
                print_log("General pipeline loaded from config", logger='current')
            else:
                print_log("No pipeline config found, will use synthetic data", logger='current')

        except Exception as e:
            print_log(f"Failed to setup pipeline: {e}", logger='current')
            self.test_pipeline = None

    def get_sample_batch(self) -> Dict[str, Any]:
        """
        Get a sample batch of data for testing

        Returns:
            Dictionary containing sample inputs
        """
        print_log("Generating sample data batch...", logger='current')

        if self.dataset is not None:
            # Try to get real sample from dataset
            try:
                return self._get_real_sample()
            except Exception as e:
                print_log(f"Failed to get real sample: {e}. Using synthetic data.", logger='current')

        # Fallback to synthetic data
        return self._generate_synthetic_sample()

    def _get_real_sample(self) -> Dict[str, Any]:
        """Get a real sample from the dataset"""
        # Get a sample from dataset
        sample_idx = 0
        raw_data = self.dataset[sample_idx]

        # Process through pipeline if available
        if self.test_pipeline is not None:
            processed_data = self.test_pipeline(raw_data)
        else:
            processed_data = raw_data

        # Add additional fields needed for model input
        inputs = processed_data.get('inputs', {})

        # Ensure inputs is a dictionary
        if isinstance(inputs, torch.Tensor):
            inputs = {'data': inputs}

        # Add metadata that might be needed
        metadata = self._generate_metadata()

        return {
            'inputs': inputs,
            'data_samples': processed_data.get('data_samples', None),
            'metadata': metadata
        }

    def _generate_synthetic_sample(self) -> Dict[str, Any]:
        """Generate synthetic sample data"""
        print_log("Generating synthetic data...", logger='current')

        # Get point cloud range from model config
        point_cloud_range = self._get_point_cloud_range()
        voxel_size = self._get_voxel_size()

        # Generate synthetic point cloud
        num_points = 10000
        points = self._generate_points(num_points, point_cloud_range)

        # Voxelize points if needed
        voxelize = self._check_if_needs_voxelization()

        if voxelize:
            voxel_dict = self._voxelize_points(points, voxel_size, point_cloud_range)
            inputs = voxel_dict
        else:
            inputs = {'points': points}

        # Add image data if it's a multi-modal model
        if self._is_multimodal():
            img = self._generate_image()
            inputs['img'] = img

        # Generate metadata
        metadata = self._generate_metadata()

        return {
            'inputs': inputs,
            'data_samples': None,
            'metadata': metadata
        }

    def _get_point_cloud_range(self) -> List[float]:
        """Get point cloud range from config"""
        # Try to get from different possible locations
        if hasattr(self.cfg, 'model'):
            model = self.cfg.model
            if hasattr(model, 'point_cloud_range'):
                return model.point_cloud_range
            if hasattr(model, 'data_preprocessor'):
                preprocessor = model.data_preprocessor
                if hasattr(preprocessor, 'point_cloud_range'):
                    return preprocessor.point_cloud_range

        # Default range for outdoor scenes
        return [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]

    def _get_voxel_size(self) -> List[float]:
        """Get voxel size from config"""
        if hasattr(self.cfg, 'model'):
            model = self.cfg.model
            if hasattr(model, 'voxel_layer'):
                return model.voxel_layer.voxel_size
            if hasattr(model, 'data_preprocessor'):
                preprocessor = model.data_preprocessor
                if hasattr(preprocessor, 'voxel_layer'):
                    return preprocessor.voxel_layer.voxel_size

        # Default voxel size
        return [0.25, 0.25, 8.0]

    def _check_if_needs_voxelization(self) -> bool:
        """Check if the model needs voxelized input"""
        if hasattr(self.cfg, 'model'):
            model_type = self.cfg.model.type
            # Most 3D detectors use voxelization
            voxelization_models = [
                'VoxelNet', 'SECOND', 'PointPillars', 'CenterPoint',
                'MVXFasterRCNN', 'MVXNet', 'PartA2', 'PVRCNN'
            ]
            return any(model in str(model_type) for model in voxelization_models)
        return True  # Assume voxelization is needed by default

    def _is_multimodal(self) -> bool:
        """Check if the model is multi-modal (uses images + LiDAR)"""
        if hasattr(self.cfg, 'model'):
            model_type = self.cfg.model.type
            multimodal_models = ['MVXFasterRCNN', 'MVXNet', 'BEVFusion', 'ImVoxelNet']
            return any(model in str(model_type) for model in multimodal_models)
        return False

    def _generate_points(self, num_points: int, point_cloud_range: List[float]) -> torch.Tensor:
        """Generate synthetic point cloud"""
        x_range = point_cloud_range[3] - point_cloud_range[0]
        y_range = point_cloud_range[4] - point_cloud_range[1]
        z_range = point_cloud_range[5] - point_cloud_range[2]

        # Generate random points
        points = torch.rand(num_points, 4)  # x, y, z, intensity
        points[:, 0] = points[:, 0] * x_range + point_cloud_range[0]
        points[:, 1] = points[:, 1] * y_range + point_cloud_range[1]
        points[:, 2] = points[:, 2] * z_range + point_cloud_range[2]
        points[:, 3] = points[:, 3] * 255  # intensity

        return points

    def _generate_image(self) -> torch.Tensor:
        """Generate synthetic image data"""
        # Typical image size for autonomous driving datasets
        img_shape = (3, 900, 1600)  # C, H, W
        img = torch.rand(img_shape) * 255
        return img.unsqueeze(0)  # Add batch dimension

    def _voxelize_points(
        self,
        points: torch.Tensor,
        voxel_size: List[float],
        point_cloud_range: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Voxelize point cloud"""
        # Simple voxelization implementation
        num_points_per_voxel = 64
        max_voxels = 20000

        # Convert points to voxel coordinates
        voxel_size = torch.tensor(voxel_size).view(1, 3)
        point_cloud_range = torch.tensor(point_cloud_range).view(1, 6)

        # Get voxel indices
        xyz = points[:, :3]
        voxel_indices = ((xyz - point_cloud_range[:, :3]) / voxel_size).long()

        # Create voxels (simplified implementation)
        # In practice, this would use the efficient voxelization from MMDetection3D
        coors = torch.cat([voxel_indices, torch.arange(len(voxel_indices)).unsqueeze(1)], dim=1)

        # For simplicity, create dummy voxel features
        voxels = torch.rand(max_voxels, num_points_per_voxel, points.shape[1])
        num_points = torch.randint(1, num_points_per_voxel, (max_voxels,))
        coors = torch.randint(0, 200, (max_voxels, 3))

        return {
            'voxels': voxels,
            'num_points': num_points,
            'coors': coors
        }

    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata for the sample"""
        return {
            'img_shape': [900, 1600, 3],
            'pad_shape': [900, 1600, 3],
            'scale_factor': np.array([1., 1., 1.]),
            'flip': False,
            'pcd_horizontal_flip': False,
            'pcd_vertical_flip': False,
            'box_mode_3d': 'LIDAR',
            'box_type_3d': 'LiDAR',
            'roi_shape': (200, 200),
            'roi_scale_factor': 1.0,
            'img_norm_cfg': {
                'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
                'std': np.array([58.395, 57.12, 57.375], dtype=np.float32),
                'to_rgb': True
            },
            'sample_idx': 0,
            'pts_filename': 'synthetic_points.bin',
            'timestamp': 0.0,
            'img_filename': 'synthetic_image.jpg',
            'lidar2img': torch.eye(4),
            'depth2img': torch.eye(4),
            'cam2img': torch.eye(3),
            'pcd_trans': torch.eye(4),
            'pcd_scale_factor': 1.0,
            'pcd_rotation': 0.0,
            'ltrb': False,
            'transformation_3d': {
                'sample_idx': 0,
                'pcd_scale_factor': 1.0,
                'pcd_rotation': 0.0,
                'pcd_trans': np.zeros(3)
            }
        }