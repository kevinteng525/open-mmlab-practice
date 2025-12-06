# MMDetection3D Model Exporter

这个工具可以将MMDetection3D模型导出为PT2（torch.export）和ONNX格式，特别针对IREE推理引擎优化。

## 功能特点

- 自动处理复杂的输入结构（如点云、体素、图像等多模态数据）
- 生成符合torch.export要求的Wrapper模型
- 支持所有MMDetection3D支持的模型架构
- 自动分析输入结构并生成对应的随机数据
- 支持模型验证和输出对比

## 支持的模型

- PointPillars
- CenterPoint
- SECOND
- VoxelNet
- PV-RCNN
- BEVFusion
- 其他所有MMDetection3D支持的模型

## 安装依赖

```bash
# 安装基础依赖
pip install torch torchvision
pip install mmengine mmcv mmdet3d

# 安装导出相关依赖（可选）
pip install onnx onnxruntime  # 用于ONNX导出和验证
```

## 使用方法

### 基本使用

```bash
# 导出PointPillars模型
python ai/export_mmdet3d.py \
    --config configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py \
    --output-dir ./exported_models/pointpillars

# 导出带checkpoint的模型
python refined/export_mmdet3d.py \
    --config configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py \
    --checkpoint path/to/checkpoint.pth \
    --output-dir ./exported_models/centerpoint

# 只导出ONNX格式
python refined/export_mmdet3d.py \
    --config configs/votenet/votenet_16x8_sunrgbd-3d-10class.py \
    --output-dir ./exported_models/votenet \
    --format onnx
```

### 参数说明

- `--config`: 模型配置文件路径（必需）
- `--checkpoint`: 模型checkpoint路径（可选）
- `--output-dir`: 导出模型保存目录（默认：./exported_models）
- `--format`: 导出格式，可选 'pt2', 'onnx', 'both'（默认：both）
- `--device`: 运行设备（默认：cuda:0）
- `--verbose`: 启用详细日志

## 导出流程

工具自动执行以下步骤：

1. **加载配置**：解析MMDetection3D配置文件
2. **构建模型**：根据配置构建模型
3. **加载checkpoint**：如果提供了checkpoint文件
4. **生成样本数据**：通过数据管道生成真实的输入样本
5. **分析输入结构**：递归分析输入中的所有张量和数据结构
6. **验证模型**：使用样本数据验证模型正常运行
7. **生成Wrapper**：创建接受扁平化张量输入的包装模型
8. **导出模型**：导出为PT2和/或ONNX格式
9. **验证导出**：验证导出模型的正确性

## 输出文件

导出成功后，输出目录将包含：

- `model.pt2`: PT2格式模型（如果选择了pt2）
- `model.onnx`: ONNX格式模型（如果选择了onnx）
- `metadata.json`: 包含模型信息和导出元数据

### metadata.json 示例

```json
{
  "model_config": "/path/to/config.py",
  "checkpoint": "/path/to/checkpoint.pth",
  "model_type": "PointPillars",
  "export_info": {
    "total_inputs": 3,
    "input_structure": {...},
    "tensor_details": [
      {
        "path": "inputs.voxels",
        "shape": [10000, 20, 5],
        "dtype": "torch.float32",
        "description": "Voxel feature tensor"
      }
    ]
  }
}
```

## 使用导出的模型

### PT2格式

```python
import torch

# 加载PT2模型
exported_program = torch.export.load("model.pt2")

# 运行推理（输入需要是扁平的张量列表）
# 具体的输入形状和类型需要参考metadata.json
results = exported_program(tensor_0, tensor_1, tensor_2, ...)
```

### ONNX格式

```python
import onnxruntime as ort

# 创建ONNX Runtime会话
session = ort.InferenceSession("model.onnx")

# 准备输入
inputs = {
    "input_0": numpy_array_0,
    "input_1": numpy_array_1,
    "input_2": numpy_array_2,
}

# 运行推理
outputs = session.run(None, inputs)
```

## 测试

运行测试脚本验证导出工具：

```bash
python refined/test_exporter.py
```

## 常见问题

### 1. 导出失败：模型包含不支持的操作

某些模型可能包含torch.export不支持的操作（如某些自定义CUDA操作）。解决方案：
- 检查模型是否使用了自定义的CUDA kernel
- 考虑用PyTorch原生操作替换自定义操作

### 2. 输入维度不匹配

导出时生成的随机输入可能与实际使用场景不符。解决方案：
- 使用真实数据样本进行导出验证
- 调整`metadata.json`中的输入尺寸信息

### 3. 内存不足

处理大型模型时可能出现内存不足。解决方案：
- 使用较小的batch size
- 在CPU上进行导出，然后在GPU上运行

## 技术细节

### Wrapper模型原理

MMDetection3D模型通常接受复杂的字典输入，例如：
```python
inputs = {
    'voxels': torch.Tensor(...),
    'num_points': torch.Tensor(...),
    'coors': torch.Tensor(...),
    'metadata': {...}
}
```

而torch.export需要扁平的张量输入。工具自动生成Wrapper模型：
```python
class WrapperModel(torch.nn.Module):
    def forward(self, tensor_0, tensor_1, tensor_2):
        # 重建原始输入结构
        reconstructed = {
            'voxels': tensor_0,
            'num_points': tensor_1,
            'coors': tensor_2
        }
        # 调用原始模型
        return self.original_model(reconstructed, mode='predict')
```

### 输入结构分析

工具递归分析输入数据，提取：
- 所有张量的形状和类型
- 张量在输入结构中的路径
- 张量的语义信息（基于路径推断）

## 贡献

欢迎提交Issue和Pull Request来改进这个工具！

## 许可证

遵循OpenMMLab的许可证条款。