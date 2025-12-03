You are a pyTorch expert.
I need a python file, which can:
1. load model from any Open-MMLAB models, such as those mmdetection3d models, e.g. BEVFusion, VAD, UniAD, etc.., 
2. auto construct the example inputs based on config, support all different models
3. export the entire model to .pt and .onnx, so that it can be converted by iree-import-torch and iree-import-onnx to stableHLO