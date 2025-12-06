#!/usr/bin/env python3
"""
MMDetection3D Model Exporter Tool

This tool exports MMDetection3D models to PT2 (torch.export) and ONNX formats.
It handles complex input structures by automatically generating wrapper models.

Usage:
    python ai/export_mmdet3d.py --config configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py --output-dir ./exported
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_exporter import MMDet3DExporter
from mmengine.logging import print_log


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Export MMDetection3D models to PT2/ONNX formats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--config',
        required=True,
        type=str,
        help='Path to model config file'
    )

    # Optional arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (optional)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./exported_models',
        help='Directory to save exported models'
    )

    parser.add_argument(
        '--format',
        choices=['pt2', 'onnx', 'both'],
        default='both',
        help='Export format(s)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to run the model on (cuda:0, cpu, etc.)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def validate_config_path(config_path: str) -> Path:
    """Validate and return config file path"""
    config_path = Path(config_path)

    # If relative path, try to find it in mmdetection3d directory
    if not config_path.is_absolute():
        mmdet3d_dir = Path(__file__).parent.parent / 'mmdetection3d'
        potential_paths = [
            mmdet3d_dir / config_path,
            mmdet3d_dir / 'configs' / config_path,
            Path(config_path)
        ]

        for path in potential_paths:
            if path.exists():
                config_path = path
                break
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return config_path


def validate_checkpoint_path(checkpoint_path: str) -> Path:
    """Validate and return checkpoint file path"""
    if checkpoint_path is None:
        return None

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    return checkpoint_path


def main():
    """Main export function"""
    # Parse arguments
    args = parse_args()

    # Setup logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    print_log("=" * 60, logger='current')
    print_log("MMDetection3D Model Exporter", logger='current')
    print_log("=" * 60, logger='current')

    try:
        # Validate paths
        config_path = validate_config_path(args.config)
        checkpoint_path = validate_checkpoint_path(args.checkpoint)

        print_log(f"Config file: {config_path}", logger='current')
        if checkpoint_path:
            print_log(f"Checkpoint file: {checkpoint_path}", logger='current')
        else:
            print_log("No checkpoint provided - exporting untrained model", logger='current')
        print_log(f"Output directory: {args.output_dir}", logger='current')
        print_log(f"Export format(s): {args.format}", logger='current')
        print_log(f"Device: {args.device}", logger='current')
        print_log("-" * 60, logger='current')

        # Create exporter
        exporter = MMDet3DExporter(
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
            device=args.device
        )

        # Setup exporter
        exporter.setup()

        # Export model
        exporter.export(
            output_dir=args.output_dir,
            format=args.format
        )

        print_log("-" * 60, logger='current')
        print_log("Export completed successfully!", logger='current')
        print_log(f"Check the output directory: {args.output_dir}", logger='current')

    except Exception as e:
        print_log(f"Export failed with error: {e}", logger='current')
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()