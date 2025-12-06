# 改进的 MMDetection3D 模型导出工具

基于已验证的 `data_loader.py` 实现，添加输入展平和 wrapper 功能，支持导出为 PT2 和 ONNX 格式。

## 核心改进

1. **复用已验证的代码**：直接引用 `data_loader.py` 中的配置加载、模型构建、数据加载逻辑
2. **输入展平**：将复杂的嵌套输入结构（dict、list、tensor）展平为张量列表
3. **Wrapper 模型**：动态生成包装模型，将展平的张量重建为原始输入结构
4. **导出优化**：支持 PT2（torch.export）和 ONNX 格式导出

## 文件说明

- `improved_exporter.py` - 完整功能的导出工具
- `simple_exporter.py` - 简化版导出工具，核心功能
- `data_loader.py` - 已验证的模型加载和数据处理实现

## 快速开始

### 基本使用

```bash
# 导出 PointPillars 模型（使用真实数据）
python refined/simple_exporter.py \
    configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py \
    --output ./exported/pointpillars

# 导出 CenterPoint 模型（带 checkpoint）
python refined/simple_exporter.py \
    configs/centerpoint/centerpoint_02pillar_second_secfpn_nus.py \
    --checkpoint path/to/centerpoint.pth \
    --output ./exported/centerpoint \
    --format both

# 使用改进版导出工具（更多功能）
python refined/improved_exporter.py \
    configs/votenet/votenet_16x8_sunrgbd-3d-10class.py \
    --checkpoint path/to/votenet.pth \
    --output ./exported/votenet \
    --use-random-data \
    --verbose
```

## 主要功能对比

### data_loader.py（基础实现）
- ✅ 配置文件加载
- ✅ 模型构建和初始化
- ✅ 真实数据加载（通过 Runner）
- ✅ 随机数据生成
- ✅ 基本的模型前向传播

### simple_exporter.py（核心扩展）
- ✅ 所有 data_loader.py 的功能
- ✅ 输入结构分析（递归提取张量）
- ✅ 输入展平为张量列表
- ✅ Wrapper 模型动态生成
- ✅ PT2 格式导出
- ✅ ONNX 格式导出

### improved_exporter.py（完整功能）
- ✅ 所有 simple_exporter.py 的功能
- ✅ 详细的输入/输出结构分析
- ✅ Wrapper 模型验证和对比
- ✅ 更好的错误处理和日志
- ✅ 元数据保存
- ✅ 导出模型验证

## 核心实现原理

### 1. 输入展平过程

```python
# 原始输入（复杂结构）
inputs = {
    'inputs': {
        'voxels': torch.Tensor(100, 20, 5),
        'num_points': torch.Tensor(100),
        'coors': torch.Tensor(100, 3),
        'metadata': {...}
    },
    'data_samples': [...]
}

# 展平后（张量列表）
flat_inputs = [
    torch.Tensor(100, 20, 5),  # inputs.inputs.voxels
    torch.Tensor(100),         # inputs.inputs.num_points
    torch.Tensor(100, 3),      # inputs.inputs.coors
    # ... 其他张量
]
```

### 2. Wrapper 模型

```python
class WrappedModel(torch.nn.Module):
    def forward(self, tensor_0, tensor_1, tensor_2, ...):
        # 重建原始输入结构
        reconstructed = {
            'inputs': {
                'voxels': tensor_0,
                'num_points': tensor_1,
                'coors': tensor_2,
                'metadata': {...}
            }
        }
        # 调用原始模型
        return self.model(**reconstructed, mode='tensor')
```

## 输出文件

导出成功后，输出目录包含：

- `model.pt2` - PT2 格式模型（如果选择了 pt2）
- `model.onnx` - ONNX 格式模型（如果选择了 onnx）
- `metadata.json` - 导出元数据（输入结构、形状、类型等信息）

### metadata.json 示例

```json
{
  "config": "configs/pointpillars/hv_pointpillars_secfpn_kitti-3d-car.py",
  "checkpoint": null,
  "model_type": "VoxelNet",
  "tensor_count": 3,
  "tensors": [
    {
      "path": "inputs.inputs.voxels",
      "shape": [100, 20, 5],
      "dtype": "torch.float32"
    },
    {
      "path": "inputs.inputs.num_points",
      "shape": [100],
      "dtype": "torch.int64"
    },
    {
      "path": "inputs.inputs.coors",
      "shape": [100, 3],
      "dtype": "torch.int64"
    }
  ],
  "export_success": {
    "pt2": true,
    "onnx": true
  }
}
```

## 使用导出的模型

### PT2 格式

```python
import torch

# 加载导出的模型
exported_model = torch.export.load("model.pt2")

# 准备输入（根据 metadata.json 中的信息）
inputs = (
    torch.randn(100, 20, 5),    # tensor_0: voxels
    torch.randint(1, 20, (100,)), # tensor_1: num_points
    torch.randint(0, 100, (100, 3)) # tensor_2: coords
)

# 运行推理
outputs = exported_model(*inputs)
```

### ONNX 格式

```python
import onnxruntime as ort

# 创建推理会话
session = ort.InferenceSession("model.onnx")

# 准备输入
inputs = {
    "input_0": np.random.randn(1, 100, 20, 5).astype(np.float32),
    "input_1": np.random.randint(1, 20, (1, 100)).astype(np.int64),
    "input_2": np.random.randint(0, 100, (1, 100, 3)).astype(np.int64)
}

# 运行推理
outputs = session.run(None, inputs)
```

## 故障排除

### 1. 导出失败：torch.export 不支持

**问题**：模型包含 torch.export 不支持的操作
**解决方案**：
- 检查模型是否使用了自定义 CUDA 操作
- 尝试使用更简单的模型进行测试
- 查看错误信息，定位不支持的操作

### 2. 输入维度不匹配

**问题**：生成的输入与模型期望不符
**解决方案**：
- 使用真实数据集而不是随机数据
- 检查 metadata.json 中的输入信息
- 调整输入生成逻辑

### 3. 内存不足

**问题**：导出过程中内存溢出
**解决方案**：
- 使用 CPU 导出：`--device cpu`
- 减少批次大小
- 使用更小的模型进行测试

## 技术细节

### 与 MMDeploy 的对比

| 特性 | data_loader.py | 我们的实现 | MMDeploy |
|------|----------------|------------|----------|
| 配置加载 | ✅ | ✅ | ✅ |
| 模型构建 | ✅ | ✅ | ✅ |
| 输入展平 | ❌ | ✅ | ✅ |
| Wrapper 生成 | ❌ | ✅ | ✅ |
| PT2 导出 | ❌ | ✅ | ❌ |
| ONNX 导出 | ❌ | ✅ | ✅ |
| 多后端支持 | ❌ | ❌ | ✅ |
| 量化支持 | ❌ | ❌ | ✅ |

### 核心优势

1. **简单直接**：基于已验证的代码，转换路径短
2. **针对性强**：专门为 torch.export 和 IREE 优化
3. **易于理解**：代码结构清晰，便于维护
4. **快速验证**：适合快速实验和原型开发

## 未来改进

1. **支持更多模型**：测试更多 MMDetection3D 模型
2. **优化导出**：添加常量折叠、算子融合等优化
3. **精度验证**：添加导出前后的精度对比
4. **批量处理**：支持批量导出多个模型
5. **可视化工具**：输入结构可视化工具