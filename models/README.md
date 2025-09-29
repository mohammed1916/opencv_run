# DNN Models for Segmentation

This directory is for storing pre-trained DNN models for semantic segmentation.

### OpenCV Model Zoo - Manual Download

1. Visit [OpenCV Model Zoo](https://apple.github.io/ml-cvnets/en/general/README-model-zoo.html)
2. Navigate to semantic segmentation models
3. Download the ONNX files to this directory

### Own Models

Use tools like:

- `torch.onnx.export()` for PyTorch models
- `tf2onnx` for TensorFlow models
- OpenVINO Model Optimizer

## Usage

Once models are placed in this directory, the DNN segmentation program will automatically detect and use them.

- Program falls back to watershed segmentation for demonstration
