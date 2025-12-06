#!/usr/bin/env python3
"""Test script for MMDetection3D model exporter"""

import os
import sys
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/Users/kevinteng/src/kevinteng525/open-mmlab/mmdetection3d')
sys.path.append('/Users/kevinteng/src/kevinteng525/open-mmlab/mmengine')

from model_exporter import MMDet3DExporter
from mmengine.logging import print_log


def test_pointpillars_export():
    """Test exporting PointPillars model"""
    print_log("Testing PointPillars export...", logger='current')

    try:
        # Use a simple PointPillars config
        config_path = "configs/_base_/models/pointpillars_hv_secfpn_kitti.py"
        output_dir = "./test_output/pointpillars"

        # Create exporter
        exporter = MMDet3DExporter(
            config_path=config_path,
            checkpoint_path=None,  # No checkpoint for testing
            device='cpu'  # Use CPU for testing
        )

        # Setup and export
        exporter.setup()
        exporter.export(output_dir=output_dir, format='both')

        print_log("PointPillars test passed!", logger='current')
        return True

    except Exception as e:
        print_log(f"PointPillars test failed: {e}", logger='current')
        return False


def test_centerpoint_export():
    """Test exporting CenterPoint model"""
    print_log("Testing CenterPoint export...", logger='current')

    try:
        # Use CenterPoint config
        config_path = "configs/_base_/models/centerpoint_pillar02_second_secfpn_nus.py"
        output_dir = "./test_output/centerpoint"

        # Create exporter
        exporter = MMDet3DExporter(
            config_path=config_path,
            checkpoint_path=None,
            device='cpu'
        )

        # Setup and export
        exporter.setup()
        exporter.export(output_dir=output_dir, format='both')

        print_log("CenterPoint test passed!", logger='current')
        return True

    except Exception as e:
        print_log(f"CenterPoint test failed: {e}", logger='current')
        return False


def test_input_analyzer():
    """Test input analyzer with sample data"""
    print_log("Testing input analyzer...", logger='current')

    try:
        from model_exporter.input_analyzer import InputAnalyzer

        # Create sample input data
        sample_inputs = {
            'voxels': torch.randn(100, 20, 5),
            'num_points': torch.randint(1, 20, (100,)),
            'coors': torch.randint(0, 100, (100, 3)),
            'metadata': {
                'shape': [900, 1600],
                'scale_factor': 1.0
            }
        }

        # Analyze inputs
        analyzer = InputAnalyzer()
        analysis = analyzer.analyze_inputs(sample_inputs)

        print_log(f"Found {analysis['total_tensors']} tensors", logger='current')
        print_log(f"Analysis summary: {analysis['input_summary']}", logger='current')

        # Test random input generation
        random_inputs = analyzer.generate_random_inputs()
        print_log(f"Generated {len(random_inputs)} random tensors", logger='current')

        print_log("Input analyzer test passed!", logger='current')
        return True

    except Exception as e:
        print_log(f"Input analyzer test failed: {e}", logger='current')
        return False


def test_wrapper_generator():
    """Test wrapper generator"""
    print_log("Testing wrapper generator...", logger='current')

    try:
        import torch.nn as nn
        from model_exporter.input_analyzer import InputAnalyzer
        from model_exporter.wrapper_generator import WrapperGenerator

        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3)

            def forward(self, inputs, mode='predict'):
                if mode == 'predict':
                    # Simple processing
                    if 'voxels' in inputs:
                        return {'pred': self.conv(inputs['voxels'].unsqueeze(0))}
                    else:
                        return {'pred': torch.randn(1, 64, 1, 1)}
                return inputs

        # Create sample input
        sample_inputs = {
            'voxels': torch.randn(100, 3, 32, 32),
            'metadata': {'info': 'test'}
        }

        # Analyze inputs
        analyzer = InputAnalyzer()
        analysis = analyzer.analyze_inputs(sample_inputs)

        # Generate wrapper
        model = SimpleModel()
        wrapper_generator = WrapperGenerator(model, analysis)
        wrapper_model = wrapper_generator.generate_wrapper_model()

        # Test wrapper
        test_args = wrapper_generator.generate_export_args()
        output = wrapper_model(*test_args)

        print_log(f"Wrapper output shape: {list(output.keys())}", logger='current')
        print_log("Wrapper generator test passed!", logger='current')
        return True

    except Exception as e:
        print_log(f"Wrapper generator test failed: {e}", logger='current')
        return False


def cleanup_test_outputs():
    """Clean up test output directories"""
    import shutil
    test_dirs = ['./test_output']
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print_log(f"Cleaned up {test_dir}", logger='current')


def main():
    """Run all tests"""
    print_log("=" * 60, logger='current')
    print_log("Running MMDetection3D Exporter Tests", logger='current')
    print_log("=" * 60, logger='current')

    # Clean up previous test outputs
    cleanup_test_outputs()

    # Run tests
    tests = [
        ("Input Analyzer", test_input_analyzer),
        ("Wrapper Generator", test_wrapper_generator),
        ("PointPillars Export", test_pointpillars_export),
        ("CenterPoint Export", test_centerpoint_export),
    ]

    results = []
    for test_name, test_func in tests:
        print_log(f"\nRunning {test_name} test...", logger='current')
        print_log("-" * 40, logger='current')
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_log(f"{test_name} test crashed: {e}", logger='current')
            results.append((test_name, False))

    # Print summary
    print_log("\n" + "=" * 60, logger='current')
    print_log("Test Summary", logger='current')
    print_log("=" * 60, logger='current')

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print_log(f"{test_name}: {status}", logger='current')
        if result:
            passed += 1

    print_log(f"\nResults: {passed}/{total} tests passed", logger='current')

    if passed == total:
        print_log("All tests passed! ✅", logger='current')
    else:
        print_log("Some tests failed. ❌", logger='current')


if __name__ == '__main__':
    import torch
    main()