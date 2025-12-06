#!/usr/bin/env python3
"""
Improved MMDetection3D Model Exporter

基于已验证的 data_loader.py 实现，添加输入展平和 wrapper 功能
支持导出为 PT2 和 ONNX 格式
"""

import os
import sys
import argparse
import random
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import Runner, load_checkpoint
from mmengine.runner.utils import set_random_seed
from mmengine.logging import print_log

# 用于导出的额外导入
import torch.onnx
from typing import Dict, List, Any, Tuple, Optional

from mmengine import init_default_scope

import sys
print("当前Python路径:")
for p in sys.path:
    print(f"  {p}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Export MMDetection3D model')
    parser.add_argument('config', help='模型配置文件的路径')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='模型checkpoint路径（可选）')
    parser.add_argument(
        '--use-random-data',
        action='store_true',
        help='是否使用随机生成的数据代替真实数据集')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='推理和导出使用的设备')
    parser.add_argument(
        '--output-dir',
        default='./exported_models',
        help='导出模型保存目录')
    parser.add_argument(
        '--format',
        choices=['pt2', 'onnx', 'both'],
        default='both',
        help='导出格式')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='启用详细日志')
    args = parser.parse_args()
    return args


def to_device(data, device):
    """递归地将数据移动到指定设备"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, Mapping):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [to_device(item, device) for item in data]
    else:
        return data


def get_real_dataloader_inputs(cfg, device):
    """
    使用 Runner 的类方法来构建数据加载器
    """
    print("--- 1. 使用真实数据集加载输入 ---")
    print("正在使用 Runner 构建数据加载器...")

    dataloader = Runner.build_dataloader(cfg.test_dataloader)

    print("数据加载器构建成功，正在获取第一个 batch...")
    data_batch = next(iter(dataloader))

    data_batch = to_device(data_batch, device)

    print("成功加载一个 batch 的真实数据。")
    return data_batch


def create_random_like(data):
    """递归地创建与输入数据结构相同的随机数据"""
    if isinstance(data, torch.Tensor):
        if torch.is_floating_point(data):
            return torch.randn_like(data)
        else:
            return torch.randint_like(data, low=0, high=10)
    elif isinstance(data, Mapping):
        return {key: create_random_like(value) for key, value in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [create_random_like(item) for item in data]
    else:
        return data


def get_random_inputs(cfg, device):
    """生成与配置文件中定义的数据集具有相同结构和形状的随机输入"""
    print("--- 1. 使用随机数据生成输入 ---")
    print("首先，加载一个真实数据 batch 以获取其结构...")

    real_batch = get_real_dataloader_inputs(cfg, 'cpu')

    print("\n正在根据真实数据结构生成随机数据...")
    random_batch = create_random_like(real_batch)
    random_batch = to_device(random_batch, device)

    print("成功生成与真实数据结构相同的随机数据。")
    return random_batch


def print_data_structure(data, name="Data", indent=0):
    """辅助函数：递归打印数据结构"""
    prefix = '  ' * indent
    if isinstance(data, torch.Tensor):
        print(f"{prefix}- {name}: Tensor, shape={data.shape}, dtype={data.dtype}")
    elif isinstance(data, Mapping):
        print(f"{prefix}- {name}: Dict")
        for key, value in data.items():
            print_data_structure(value, f"'{key}'", indent + 1)
    elif isinstance(data, Sequence) and not isinstance(data, str) and len(data) > 0:
        print(f"{prefix}- {name}: List/Tuple (length: {len(data)})")
        print_data_structure(data[0], "item[0]", indent + 1)
    else:
        print(f"{prefix}- {name}: {type(data)}")


class InputFlattener:
    """输入展平器 - 将嵌套的输入结构展平为张量列表"""

    def __init__(self):
        self.tensor_info = []
        self.flatten_mapping = {}

    def analyze_and_flatten(self, data, path=""):
        """分析并展平输入数据"""
        self.tensor_info = []
        self.flatten_mapping = {}

        tensors = self._extract_tensors(data, path)

        # 创建反向映射
        for idx, info in enumerate(self.tensor_info):
            self.flatten_mapping[info['path']] = idx

        return tensors

    def _extract_tensors(self, data, path=""):
        """递归提取所有张量"""
        tensors = []

        if isinstance(data, torch.Tensor):
            # 找到张量
            info = {
                'path': path,
                'shape': data.shape,
                'dtype': data.dtype,
                'device': data.device
            }
            self.tensor_info.append(info)
            tensors.append(data)

        elif isinstance(data, Mapping):
            # 处理字典
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                tensors.extend(self._extract_tensors(value, new_path))

        elif isinstance(data, Sequence) and not isinstance(data, str):
            # 处理列表
            for idx, item in enumerate(data):
                new_path = f"{path}[{idx}]" if path else f"[{idx}]"
                tensors.extend(self._extract_tensors(item, new_path))

        return tensors

    def reconstruct_inputs(self, flat_tensors):
        """从展平的张量重建原始输入结构"""
        if not self.tensor_info:
            return {}

        # 创建重建后的输入字典
        inputs = {}

        for idx, tensor in enumerate(flat_tensors):
            if idx < len(self.tensor_info):
                info = self.tensor_info[idx]
                self._set_nested_value(inputs, info['path'], tensor)

        return inputs

    def _set_nested_value(self, obj, path, value):
        """在嵌套字典中设置值"""
        keys = path.split('.')

        current = obj
        for key in keys[:-1]:
            if '[' in key and key.endswith(']'):
                # 处理列表索引，如 'data[0]'
                base_key = key.split('[')[0]
                idx = int(key.split('[')[1].split(']')[0])
                if base_key not in current:
                    current[base_key] = []
                while len(current[base_key]) <= idx:
                    current[base_key] = {}
                current = current[base_key][idx]
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]

        final_key = keys[-1]
        if '[' in final_key and final_key.endswith(']'):
            # 处理最后一层的列表索引
            base_key = final_key.split('[')[0]
            idx = int(final_key.split('[')[1].split(']')[0])
            if base_key not in current:
                current[base_key] = []
            while len(current[base_key]) <= idx:
                current[base_key] = None
            current[base_key][idx] = value
        else:
            current[final_key] = value


class ModelWrapper(torch.nn.Module):
    """模型包装器 - 将原始模型包装为接受扁平张量输入的形式"""

    def __init__(self, original_model, flattener):
        super().__init__()
        self.original_model = original_model
        self.flattener = flattener
        self.num_inputs = len(flattener.tensor_info)

    def forward(self, *args):
        """前向传播 - 接受展平的张量，重建输入，调用原始模型"""
        # 将参数转换为列表（如果传入的是元组）
        flat_tensors = list(args)

        # 重建原始输入结构
        reconstructed_inputs = self.flattener.reconstruct_inputs(flat_tensors)

        # 调用原始模型
        return self.original_model(**reconstructed_inputs, mode='tensor')


def load_checkpoint(model, checkpoint_path):
    """加载checkpoint"""
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n正在加载 checkpoint: {checkpoint_path}")
        load_checkpoint(model, checkpoint_path, map_location="cpu")
    else:
        print("\n未提供 checkpoint 或文件不存在，使用未训练的模型。")


def export_to_pt2(model, inputs, output_path, verbose=False):
    """导出为 PT2 格式"""
    print(f"\n--- 导出 PT2 格式到 {output_path} ---")

    try:
        # 使用 torch.export 导出
        exported_program = torch.export.export(model, inputs)

        # 保存
        torch.export.save(exported_program, output_path)

        print(f"✅ PT2 导出成功！保存到: {output_path}")

        # 尝试重新加载验证
        if verbose:
            print("验证 PT2 重新加载...")
            reloaded = torch.export.load(output_path)
            print("✅ PT2 重新加载验证成功！")

        return True

    except Exception as e:
        print(f"❌ PT2 导出失败: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def export_to_onnx(model, inputs, output_path, verbose=False):
    """导出为 ONNX 格式"""
    print(f"\n--- 导出 ONNX 格式到 {output_path} ---")

    try:
        # 准备输出名称
        output_names = []
        if isinstance(model, ModelWrapper):
            # 对于包装模型，我们需要先运行一次来了解输出结构
            with torch.no_grad():
                sample_output = model(*inputs)
                if isinstance(sample_output, dict):
                    output_names = list(sample_output.keys())
                elif isinstance(sample_output, (list, tuple)):
                    output_names = [f"output_{i}" for i in range(len(sample_output))]
                else:
                    output_names = ["output"]
        else:
            output_names = ["output"]

        # 输入名称
        input_names = [f"input_{i}" for i in range(len(inputs))]

        # 动态轴配置
        dynamic_axes = {}
        for i, input_name in enumerate(input_names):
            dynamic_axes[input_name] = {0: 'batch_size'}
        for output_name in output_names:
            dynamic_axes[output_name] = {0: 'batch_size'}

        # 导出
        torch.onnx.export(
            model,
            inputs,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=verbose
        )

        print(f"✅ ONNX 导出成功！保存到: {output_path}")

        # 验证 ONNX 模型
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX 模型验证成功！")
        except ImportError:
            print("⚠️  未安装 onnx，跳过验证")
        except Exception as e:
            print(f"⚠️  ONNX 验证失败: {e}")

        return True

    except Exception as e:
        print(f"❌ ONNX 导出失败: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def save_metadata(output_dir, config_path, checkpoint_path, flattener, model_info):
    """保存导出元数据"""
    metadata = {
        'model_config': config_path,
        'checkpoint': checkpoint_path,
        'model_info': model_info,
        'tensor_info': flattener.tensor_info,
        'flatten_mapping': flattener.flatten_mapping,
        'num_inputs': len(flattener.tensor_info)
    }

    metadata_path = Path(output_dir) / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"元数据已保存到: {metadata_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("MMDetection3D 模型导出工具（改进版）")
    print("=" * 60)

    # 1. 加载配置文件
    print("\n正在加载配置文件...")
    cfg = Config.fromfile(args.config)
    cfg.work_dir = './work_dir'

    # 设置随机种子
    set_random_seed(0)

    # 2. 构建模型
    print("\n正在根据配置构建模型...")
    init_default_scope("mmdet3d")
    model = MODELS.build(cfg.model)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"模型构建成功，并已移动到 {device}，设置为 eval 模式。")

    # 加载 checkpoint（如果提供）
    load_checkpoint(model, args.checkpoint)

    # 3. 获取输入数据
    if args.use_random_data:
        inputs = get_random_inputs(cfg, device)
    else:
        inputs = get_real_dataloader_inputs(cfg, device)

    print("\n--- 输入数据结构预览 ---")
    print_data_structure(inputs)
    print("-" * 40)

    # 4. 验证原始模型前向传播
    print("\n--- 2. 验证原始模型前向传播 ---")
    try:
        with torch.no_grad():
            original_outputs = model(**inputs, mode='tensor')
        print("✅ 原始模型前向传播成功！")
        if args.verbose:
            print("\n--- 原始输出数据结构预览 ---")
            print_data_structure(original_outputs, name="Original Output")
            print("-" * 40)
    except Exception as e:
        print("❌ 原始模型前向传播失败！")
        print(f"错误信息: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return

    # 5. 展平输入
    print("\n--- 3. 展平输入数据 ---")
    flattener = InputFlattener()
    flat_inputs = flattener.analyze_and_flatten(inputs)

    print(f"成功将输入展平为 {len(flat_inputs)} 个张量")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: shape={info['shape']}, dtype={info['dtype']}")

    # 6. 创建并验证包装模型
    print("\n--- 4. 创建包装模型 ---")
    wrapper_model = ModelWrapper(model, flattener)
    wrapper_model = wrapper_model.to(device)
    wrapper_model.eval()

    # 验证包装模型
    print("验证包装模型前向传播...")
    try:
        with torch.no_grad():
            wrapper_outputs = wrapper_model(*flat_inputs)
        print("✅ 包装模型前向传播成功！")

        # 比较输出
        if isinstance(original_outputs, dict) and isinstance(wrapper_outputs, dict):
            print("输出结构对比:")
            print(f"  原始: {list(original_outputs.keys())}")
            print(f"  包装: {list(wrapper_outputs.keys())}")

    except Exception as e:
        print("❌ 包装模型前向传播失败！")
        print(f"错误信息: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return

    # 7. 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 8. 导出模型
    print("\n--- 5. 导出模型 ---")

    # 准备模型信息
    model_info = {
        'type': str(type(model).__name__),
        'config_type': cfg.model.type if hasattr(cfg.model, 'type') else 'Unknown',
        'device': str(device),
        'input_shapes': [info['shape'] for info in flattener.tensor_info],
        'input_dtypes': [str(info['dtype']) for info in flattener.tensor_info]
    }

    success_count = 0

    if args.format in ['pt2', 'both']:
        pt2_path = output_dir / 'model.pt2'
        if export_to_pt2(wrapper_model, tuple(flat_inputs), str(pt2_path), args.verbose):
            success_count += 1

    if args.format in ['onnx', 'both']:
        onnx_path = output_dir / 'model.onnx'
        if export_to_onnx(wrapper_model, tuple(flat_inputs), str(onnx_path), args.verbose):
            success_count += 1

    # 9. 保存元数据
    save_metadata(output_dir, args.config, args.checkpoint, flattener, model_info)

    # 10. 总结
    print("\n" + "=" * 60)
    print(f"导出完成！成功格式: {success_count}/{len(args.format.split(',')) if args.format != 'both' else 2}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()