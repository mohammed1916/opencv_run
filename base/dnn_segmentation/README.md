# DNN Segmentation with MobileViT

This program performs image segmentation using the MobileViT classification model for feature extraction combined with K-means clustering, with fallback to traditional segmentation methods.

## Model Setup

1. **MobileViT Model**: Place your downloaded MobileViT model files in:
   - `models/mobilevit/model.onnx` - The ONNX model file
   - `models/mobilevit/config.json` - The model configuration file

## Usage

```bash
# Basic usage
dnn_segmentation <image_path>

# Run without GUI display
dnn_segmentation <image_path> --no-gui

# Use SAM instead (if sam_wrapper.py is available)
dnn_segmentation <image_path> --use-sam

# SAM with different modes
dnn_segmentation <image_path> --use-sam --sam-mode auto
dnn_segmentation <image_path> --use-sam --sam-mode points
```

## Segmentation Methods

The program tries different segmentation approaches in this order:

1. **SAM (Segment Anything Model)** - If `--use-sam` flag is provided
2. **MobileViT + K-means** - Uses the loaded MobileViT model for feature extraction combined with K-means clustering
3. **Traditional Methods** - Falls back to Watershed + GrabCut segmentation if no DNN model is available

## Output

Results are saved in different directories based on the method used:

- `tools/out_images/sam/` - SAM results (if using SAM)
- `tools/out_images/mobilevit_seg/` - MobileViT-based segmentation results
- `tools/out_images/traditional_seg/` - Traditional segmentation results

### Output Files

For MobileViT segmentation:
- `original.png` - Original input image
- `segmentation_mask.png` - Grayscale segmentation mask
- `colored_segmentation.png` - Colored visualization of segments
- `blended_result.png` - Original image blended with colored segmentation

For traditional segmentation:
- `original.png` - Original input image
- `binary.png` - Binary mask from preprocessing
- `distance_transform.png` - Distance transform visualization
- `watershed_segmented.png` - Watershed segmentation result
- `grabcut_result.png` - GrabCut segmentation result

## Technical Details

### MobileViT Integration

The MobileViT model is used as a feature extractor rather than a direct segmentation model:

1. **Preprocessing**: Images are resized to 224x224 and normalized to [-1, 1] range
2. **Feature Extraction**: The classification model outputs a 1001-dimensional feature vector
3. **K-means Clustering**: The original image pixels are clustered using K-means (K=8) 
4. **Post-processing**: Results are resized back to original image dimensions

### Fallback Methods

If MobileViT fails to load, the program uses improved traditional methods:

1. **Watershed Segmentation**: 
   - Bilateral filtering for noise reduction
   - Adaptive thresholding with Otsu's method
   - Distance transform-based foreground detection
   - Morphological operations for cleanup

2. **GrabCut Segmentation**:
   - Automatic rectangle initialization (center 80% of image)
   - Interactive foreground/background separation

## Testing

Use the provided test script to verify functionality:

```bash
python test_segmentation.py
```

This creates a test image with colored regions and runs the segmentation pipeline.

## Requirements

- OpenCV 4.x with DNN module support
- ONNX runtime support in OpenCV
- C++17 compatible compiler
- CMake for building

## Model Information

The current implementation expects a MobileNet v1 classification model with:
- Input size: 224x224x3
- Output: 1001 classes (ImageNet + background)
- Format: ONNX

For proper semantic segmentation models (like DeepLabV3, FCN, etc.), additional modifications would be needed to handle multi-class segmentation outputs directly.