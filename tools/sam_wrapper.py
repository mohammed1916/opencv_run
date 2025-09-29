#!/usr/bin/env python3
"""
SAM (Segment Anything Model) wrapper for C++ integration
"""
import sys
import os
import cv2
import numpy as np
import argparse
import json

sam_path = "C:/Users/abd/d/ai/segment-anything"
if os.path.exists(sam_path):
    sys.path.append(sam_path)

try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    import torch
except ImportError as e:
    print(f"Error importing SAM: {e}")
    print("Make sure segment-anything is properly installed and the path is correct")
    sys.exit(1)

def load_sam_model(model_path="models/sam_vit_h_4b8939.pth", use_cuda=True):
    """Load SAM model with CUDA support"""
    if not os.path.exists(model_path):
        alt_path = "C:/Users/abd/d/ai/segment-anything/sam_vit_h_4b8939.pth"
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"SAM model not found at {model_path}")
            return None, None
    
    print(f"Loading SAM model from: {model_path}")
    
    # Check CUDA availability
    if use_cuda and torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = "cpu"
        if use_cuda:
            print("CUDA not available, falling back to CPU")
        else:
            print("Using CPU")
    
    if "vit_h" in model_path:
        model_type = "vit_h"
    elif "vit_l" in model_path:
        model_type = "vit_l"
    elif "vit_b" in model_path:
        model_type = "vit_b"
    else:
        model_type = "vit_h"  # default
    
    try:
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        print(f"Successfully loaded SAM model ({model_type}) on {device.upper()}")
        return sam, predictor
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        return None, None

def segment_with_sam_auto(image_path, output_dir, use_cuda=True, max_size=1024):
    """Automatic segmentation using SAM with CUDA acceleration and memory management"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    original_image = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape
    print(f"Original image: {original_shape[1]}x{original_shape[0]} pixels")
    
    # Resize image if too large to avoid CUDA memory issues
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Resized image to: {new_w}x{new_h} pixels (scale: {scale:.3f}) to fit in GPU memory")
    else:
        scale = 1.0
        print(f"Using original size: {w}x{h} pixels")
    
    # Clear CUDA cache if using GPU
    if use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA memory before: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # Load SAM model with CUDA support
    sam, _ = load_sam_model(use_cuda=use_cuda)
    if sam is None:
        return False
    
    # Create mask generator with memory-efficient parameters
    mask_generator_params = {
        'model': sam,
        'points_per_side': 16,  # Reduced for memory efficiency
        'pred_iou_thresh': 0.86,
        'stability_score_thresh': 0.92,
        'crop_n_layers': 0,  # Disable cropping to save memory
        'crop_n_points_downscale_factor': 1,
        'min_mask_region_area': int(100 * (scale ** 2)),  # Scale min area
    }
    
    # Adjust parameters based on device and available memory
    if use_cuda and torch.cuda.is_available():
        # Check available GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {gpu_memory_gb:.1f} GB")
        
        if gpu_memory_gb >= 8:  # High memory GPU
            mask_generator_params.update({
                'points_per_side': 24,
                'pred_iou_thresh': 0.88,
            })
            print("Using high-memory GPU parameters")
        else:  # Low memory GPU
            mask_generator_params.update({
                'points_per_side': 12,
                'pred_iou_thresh': 0.84,
            })
            print("Using low-memory GPU parameters")
    else:
        # CPU parameters
        mask_generator_params.update({
            'points_per_side': 8,  # Very conservative for CPU
        })
        print("Using CPU parameters")
    
    mask_generator = SamAutomaticMaskGenerator(**mask_generator_params)
    
    print("Generating masks with SAM...")
    import time
    start_time = time.time()
    
    try:
        # Generate masks with error handling
        masks = mask_generator.generate(image_rgb)
        
        end_time = time.time()
        print(f"Generated {len(masks)} masks in {end_time - start_time:.2f} seconds")
        
        if use_cuda and torch.cuda.is_available():
            print(f"CUDA memory after: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA out of memory error: {e}")
        print("Trying with reduced parameters...")
        
        # Clear cache and try again with minimal parameters
        if use_cuda:
            torch.cuda.empty_cache()
        
        # Retry with very conservative parameters
        mask_generator_params.update({
            'points_per_side': 8,
            'crop_n_layers': 0,
            'min_mask_region_area': int(500 * (scale ** 2)),
        })
        
        mask_generator = SamAutomaticMaskGenerator(**mask_generator_params)
        
        try:
            masks = mask_generator.generate(image_rgb)
            end_time = time.time()
            print(f"Generated {len(masks)} masks in {end_time - start_time:.2f} seconds (reduced parameters)")
        except Exception as e2:
            print(f"Failed even with reduced parameters: {e2}")
            print("Falling back to CPU...")
            
            # Try CPU fallback
            sam, _ = load_sam_model(use_cuda=False)
            if sam is not None:
                mask_generator_params['model'] = sam
                mask_generator = SamAutomaticMaskGenerator(**mask_generator_params)
                masks = mask_generator.generate(image_rgb)
                end_time = time.time()
                print(f"Generated {len(masks)} masks in {end_time - start_time:.2f} seconds (CPU fallback)")
            else:
                return False
    except Exception as e:
        print(f"Unexpected error during mask generation: {e}")
        return False
    
    # Create visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original image (full resolution)
    cv2.imwrite(os.path.join(output_dir, "sam_original.png"), original_image)
    
    # If we resized, we need to scale masks back up
    if scale != 1.0:
        print(f"Scaling masks back to original resolution...")
        scaled_masks = []
        for mask_info in masks:
            mask = mask_info['segmentation']
            # Scale mask back to original size
            scaled_mask = cv2.resize(mask.astype(np.uint8), 
                                   (original_shape[1], original_shape[0]), 
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # Update mask info
            scaled_mask_info = mask_info.copy()
            scaled_mask_info['segmentation'] = scaled_mask
            scaled_mask_info['area'] = int(scaled_mask_info['area'] / (scale ** 2))
            scaled_masks.append(scaled_mask_info)
        
        masks = scaled_masks
        h, w = original_shape[:2]
        working_image = original_image
    else:
        h, w = image.shape[:2]
        working_image = image
    
    # Create combined mask
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    colored_masks = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Sort masks by area (largest first)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    print(f"Processing {len(masks)} masks...")
    for i, mask_info in enumerate(masks):
        mask = mask_info['segmentation']
        
        # Add to combined mask
        combined_mask[mask] = (i % 255) + 1
        
        # Color the mask with better color distribution
        np.random.seed(i)  # Consistent colors
        color = np.random.randint(0, 255, 3)
        colored_masks[mask] = color
    
    # Save masks
    cv2.imwrite(os.path.join(output_dir, "sam_combined_mask.png"), combined_mask)
    cv2.imwrite(os.path.join(output_dir, "sam_colored_masks.png"), colored_masks)
    
    # Create overlay
    overlay = cv2.addWeighted(working_image, 0.6, colored_masks, 0.4, 0)
    cv2.imwrite(os.path.join(output_dir, "sam_overlay.png"), overlay)
    
    # Save top masks separately with confidence scores
    top_masks = masks[:10]  # Top 10 masks
    for i, mask_info in enumerate(top_masks):
        mask = mask_info['segmentation']
        area = mask_info['area']
        stability_score = mask_info.get('stability_score', 0)
        
        masked_image = working_image.copy()
        masked_image[~mask] = 0
        
        filename = f"sam_mask_{i+1}_area{area}_score{stability_score:.3f}.png"
        cv2.imwrite(os.path.join(output_dir, filename), masked_image)
    
    # Create summary statistics
    areas = [m['area'] for m in masks]
    stability_scores = [m.get('stability_score', 0) for m in masks]
    
    print(f"SAM segmentation complete! Results saved to {output_dir}")
    print(f"Statistics:")
    print(f"  - Total masks: {len(masks)}")
    print(f"  - Average area: {np.mean(areas):.0f} pixels")
    print(f"  - Average stability: {np.mean(stability_scores):.3f}")
    print(f"  - Processing time: {end_time - start_time:.2f} seconds")
    
    return True

def segment_with_sam_points(image_path, output_dir, points=None, use_cuda=True):
    """Point-based segmentation using SAM with CUDA acceleration"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Load SAM model with CUDA support
    sam, predictor = load_sam_model(use_cuda=use_cuda)
    if sam is None or predictor is None:
        return False
    
    # Set image
    print("Setting image for SAM predictor...")
    import time
    start_time = time.time()
    predictor.set_image(image_rgb)
    set_image_time = time.time() - start_time
    print(f"Image processing took {set_image_time:.2f} seconds")
    
    # If no points provided, use strategic points
    if points is None:
        h, w = image.shape[:2]
        points = np.array([
            [w//2, h//2],      # center
            [w//4, h//4],      # top-left quadrant
            [3*w//4, h//4],    # top-right quadrant
            [w//4, 3*h//4],    # bottom-left quadrant
            [3*w//4, 3*h//4],  # bottom-right quadrant
            [w//2, h//4],      # top center
            [w//2, 3*h//4],    # bottom center
            [w//4, h//2],      # left center
            [3*w//4, h//2],    # right center
        ])
    
    labels = np.ones(len(points))  # All positive points
    
    print(f"Segmenting with {len(points)} points...")
    
    # Generate masks
    start_time = time.time()
    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )
    prediction_time = time.time() - start_time
    print(f"Prediction took {prediction_time:.2f} seconds")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "sam_original.png"), image)
    
    # Save each mask with detailed information
    best_mask_idx = np.argmax(scores)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        is_best = i == best_mask_idx
        suffix = "_BEST" if is_best else ""
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = image[mask]
        
        # Save mask
        filename = f"sam_point_mask_{i+1}_score{score:.3f}{suffix}.png"
        cv2.imwrite(os.path.join(output_dir, filename), colored_mask)
        
        # Create overlay with different colors for each mask
        overlay = image.copy()
        if i == 0:
            overlay_color = [0, 255, 0]  # Green
        elif i == 1:
            overlay_color = [255, 0, 0]  # Blue
        else:
            overlay_color = [0, 0, 255]  # Red
            
        overlay[mask] = cv2.addWeighted(overlay[mask], 0.5, 
                                       np.full_like(overlay[mask], overlay_color), 0.5, 0)
        
        overlay_filename = f"sam_point_overlay_{i+1}_score{score:.3f}{suffix}.png"
        cv2.imwrite(os.path.join(output_dir, overlay_filename), overlay)
        
        # Calculate mask statistics
        mask_area = np.sum(mask)
        mask_percentage = (mask_area / (mask.shape[0] * mask.shape[1])) * 100
        
        print(f"Mask {i+1}: confidence = {score:.3f}, area = {mask_area} pixels ({mask_percentage:.1f}%){' [BEST]' if is_best else ''}")
    
    # Draw points on original image with numbers
    image_with_points = image.copy()
    for idx, point in enumerate(points):
        cv2.circle(image_with_points, tuple(point.astype(int)), 8, (0, 0, 255), -1)
        cv2.putText(image_with_points, str(idx+1), 
                   (point[0].astype(int) + 12, point[1].astype(int) + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, "sam_input_points.png"), image_with_points)
    
    # Create combined result showing the best mask
    best_mask = masks[best_mask_idx]
    combined_result = image.copy()
    combined_result[best_mask] = cv2.addWeighted(combined_result[best_mask], 0.7,
                                               np.full_like(combined_result[best_mask], [0, 255, 0]), 0.3, 0)
    cv2.imwrite(os.path.join(output_dir, "sam_best_result.png"), combined_result)
    
    print(f"SAM point-based segmentation complete! Results saved to {output_dir}")
    print(f"Performance: Set image: {set_image_time:.2f}s, Prediction: {prediction_time:.2f}s")
    print(f"Best mask: #{best_mask_idx + 1} with confidence {scores[best_mask_idx]:.3f}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='SAM Segmentation Wrapper with CUDA Support')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output_dir', default='tools/out_images/sam', help='Output directory')
    parser.add_argument('--mode', choices=['auto', 'points'], default='auto', help='Segmentation mode')
    parser.add_argument('--points', help='JSON string of points [[x1,y1],[x2,y2],...]')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA and use CPU only')
    parser.add_argument('--device-info', action='store_true', help='Show CUDA device information and exit')
    parser.add_argument('--max-size', type=int, default=1024, help='Maximum image dimension to avoid memory issues (default: 1024)')
    
    args = parser.parse_args()
    
    # Show device information if requested
    if args.device_info:
        print("PyTorch and CUDA Information:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
        return 0
    
    # Determine whether to use CUDA
    use_cuda = not args.no_cuda
    
    # Parse points if provided
    points = None
    if args.points:
        try:
            points = np.array(json.loads(args.points))
        except:
            print("Invalid points format. Use: [[x1,y1],[x2,y2],...]")
            return 1
    
    # Run segmentation
    print(f"Starting SAM segmentation in {args.mode} mode...")
    print(f"CUDA enabled: {use_cuda}")
    
    if args.mode == 'auto':
        success = segment_with_sam_auto(args.image_path, args.output_dir, use_cuda, args.max_size)
    else:
        success = segment_with_sam_points(args.image_path, args.output_dir, points, use_cuda)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())