#!/usr/bin/env python3
"""
Simple MMDetection3D Model Exporter

简化版导出工具，核心功能：
1. 加载配置和模型（参考 data_loader.py）
2. 展平输入结构
3. 创建 Wrapper 模型
4. 导出为 PT2/ONNX
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import Runner, load_checkpoint
from mmengine.runner.utils import set_random_seed
from mmengine.logging import print_log

from mmengine.mmengine import init_default_scope


def extract_tensors(data, tensors=None, path=""):
    """递归提取所有张量"""
    if tensors is None:
        tensors = []

    if isinstance(data, torch.Tensor):
        # 跳过空张量
        if data.numel() == 0:
            return tensors
        tensors.append({
            'tensor': data,
            'path': path,
            'shape': data.shape,
            'dtype': data.dtype
        })
    elif isinstance(data, dict):
        for key, value in data.items():
            extract_tensors(value, tensors, f"{path}.{key}" if path else key)
    elif isinstance(data, (list, tuple)) and not isinstance(data, str):
        for i, item in enumerate(data):
            extract_tensors(item, tensors, f"{path}[{i}]" if path else f"[{i}]")

    return tensors


def reconstruct_inputs(tensor_list, tensor_info):
    """从张量列表重建输入结构"""
    inputs = {}

    for info in tensor_info:
        tensor = info['tensor']
        path = info['path'].split('.')

        current = inputs
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[path[-1]] = tensor

    return inputs


class WrappedModel(torch.nn.Module):
    """包装模型，接受展平的张量输入"""

    def __init__(self, model, tensor_info):
        super().__init__()
        self.model = model
        self.tensor_info = tensor_info

    def forward(self, *args):
        # 重建输入
        # 创建 tensor_info 的副本，并更新张量值
        updated_info = []
        for i, (info, tensor) in enumerate(zip(self.tensor_info, args)):
            new_info = info.copy()
            new_info['tensor'] = tensor
            updated_info.append(new_info)

        inputs = reconstruct_inputs(updated_info, updated_info)

        # 调用原始模型
        return self.model(**inputs, mode='tensor')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--checkpoint', help='checkpoint路径（可选）')
    parser.add_argument('--output', default='./exported', help='输出目录')
    parser.add_argument('--format', choices=['pt2', 'onnx', 'both'], default='both', help='导出格式')
    parser.add_argument('--device', default='cuda:0', help='设备')
    args = parser.parse_args()

    print("=" * 50)
    print("Simple MMDetection3D Exporter")
    print("=" * 50)

    # 1. 加载配置
    print("\n[1] 加载配置...")
    cfg = Config.fromfile(args.config)
    set_random_seed(0)

    # 2. 构建模型
    print("[2] 构建模型...")
    init_default_scope("mmdet3d")
    model = MODELS.build(cfg.model)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 加载 checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"  加载 checkpoint: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint, map_location="cpu")

    # 3. 获取输入数据
    print("[3] 准备输入数据...")
    try:
        # 尝试使用真实数据
        dataloader = Runner.build_dataloader(cfg.test_dataloader)
        data_batch = next(iter(dataloader))
        # 移动到设备
        def to_device(d):
            if isinstance(d, torch.Tensor):
                return d.to(device)
            elif isinstance(d, dict):
                return {k: to_device(v) for k, v in d.items()}
            elif isinstance(d, (list, tuple)):
                return [to_device(x) for x in d]
            return d
        data_batch = to_device(data_batch)
        print("  使用真实数据集")
    except Exception as e:
        print(f"  无法加载真实数据: {e}")
        print("  使用默认输入")
        # 创建默认输入结构
        data_batch = {
            'inputs': {
                'voxels': torch.randn(100, 20, 5).to(device),
                'num_points': torch.randint(1, 20, (100,)).to(device),
                'coors': torch.randint(0, 100, (100, 3)).to(device),
            }
        }

    # 4. 提取张量
    print("[4] 分析输入结构...")
    tensor_info = extract_tensors(data_batch)
    print(f"  发现 {len(tensor_info)} 个张量:")
    for i, info in enumerate(tensor_info):
        print(f"    [{i}] {info['path']}: {info['shape']} ({info['dtype']})")

    # 5. 验证原始模型
    print("[5] 验证原始模型...")
    try:
        with torch.no_grad():
            original_output = model(**data_batch, mode='tensor')
        print("  ✓ 原始模型运行成功")
    except Exception as e:
        print(f"  ✗ 原始模型运行失败: {e}")
        return

    # 6. 创建包装模型
    print("[6] 创建包装模型...")
    wrapped_model = WrappedModel(model, tensor_info).to(device)
    wrapped_model.eval()

    # 验证包装模型
    try:
        flat_inputs = [info['tensor'] for info in tensor_info]
        with torch.no_grad():
            wrapped_output = wrapped_model(*flat_inputs)
        print("  ✓ 包装模型运行成功")
    except Exception as e:
        print(f"  ✗ 包装模型运行失败: {e}")
        return

    # 7. 导出模型
    print("[7] 导出模型...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = {'pt2': False, 'onnx': False}

    # PT2 导出
    if args.format in ['pt2', 'both']:
        try:
            print("  导出 PT2...")
            exported = torch.export.export(wrapped_model, tuple(flat_inputs))
            pt2_path = output_dir / 'model.pt2'
            torch.export.save(exported, str(pt2_path))
            print(f"  ✓ PT2 导出成功: {pt2_path}")
            success['pt2'] = True
        except Exception as e:
            print(f"  ✗ PT2 导出失败: {e}")

    # ONNX 导出
    if args.format in ['onnx', 'both']:
        try:
            print("  导出 ONNX...")
            onnx_path = output_dir / 'model.onnx'
            input_names = [f'input_{i}' for i in range(len(flat_inputs))]
            output_names = ['output']

            torch.onnx.export(
                wrapped_model,
                tuple(flat_inputs),
                str(onnx_path),
                export_params=True,
                opset_version=17,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={name: {0: 'batch'} for name in input_names + output_names}
            )
            print(f"  ✓ ONNX 导出成功: {onnx_path}")
            success['onnx'] = True
        except Exception as e:
            print(f"  ✗ ONNX 导出失败: {e}")

    # 8. 保存元数据
    print("[8] 保存元数据...")
    metadata = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'model_type': cfg.model.type,
        'tensor_count': len(tensor_info),
        'tensors': [
            {
                'path': info['path'],
                'shape': list(info['shape']),
                'dtype': str(info['dtype'])
            }
            for info in tensor_info
        ],
        'export_success': success
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n导出完成！")
    print(f"  成功: PT2={'✓' if success['pt2'] else '✗'}, ONNX={'✓' if success['onnx'] else '✗'}")
    print(f"  输出目录: {output_dir}")


if __name__ == '__main__':
    main()