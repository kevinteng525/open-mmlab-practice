from structures import Det3DDataSample背景： 
我需要测试我们自己研发的推理引擎，是基于IREE开发的，会将各种模型转成IR，再做一系列优化，最终跑在我们自己的国产GPU上。
因此我需要将各种模型转成pt2（通过torch.export），或者onnx（通过torch.onnx.export）

需求：
因为torch.export 不支持dict/list这种输入，所以不支持复杂结构体的输入，针对有些mmdet3d的网络，比如BEVFusion，它的输入是这样的：
```python
Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
            data_samples (list[:obj:`Det3DDataSample`],
                list[list[:obj:`Det3DDataSample`]], optional): The
                annotation data of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.
```
我们在torch.export前需要将model wrap一下，使得输入只有tensor，并且是展平的，比如
```python
class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y, z):
        self.model(generateInputs(x), generateDataSamples(y,z), mode='predict')
```
这样我们可以这样export：
```python
import torch

args = (torch.randn((1, 4, 10,10), dtype=torch.float16).cuda(),
        torch.tensor([128], dtype=torch.float32).cuda(),
        torch.randn((256, 768), dtype=torch.bfloat16).cuda())
graph = torch.export.export(WrapperModel(pipe.transformer).cuda(), args=args)
torch.export.save(graph, "xxx.pt2")
```

现在请根据你对这些open-mmlab各项目的理解，帮我实现一段这样的功能，可以基于project的config，实现自动导出成pt2或者onnx的功能，其中包括：
1. 加载配置文件
2. 加载模型
3. 加载checkpoint（可选）
4. 根据配置，通过pipeline自动生成真实输入
5. 根据真实输入，通过递归遍历，生成一份随机输入
6. 做一次forward，验证模型和输入没问题
7. 自动获取所有输入的tensor，及对应的shape和type
8. 展平并生成对应的wrapperModel
9. 导出成.pt2和.onnx

