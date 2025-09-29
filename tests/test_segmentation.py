#!/usr/bin/env python3
"""
Simple test script to create a test image and run segmentation
"""

import cv2
import numpy as np
import os
import subprocess

def create_test_image():
    """Create a simple test image for segmentation testing"""
    # Create a 400x300 image with different colored regions
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add some colored regions
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(img, (200, 50), (300, 150), (0, 255, 0), -1) # Green rectangle
    cv2.rectangle(img, (125, 175), (275, 225), (0, 0, 255), -1) # Red rectangle
    
    # Add some circles
    cv2.circle(img, (100, 250), 30, (255, 255, 0), -1)  # Cyan circle
    cv2.circle(img, (300, 250), 30, (255, 0, 255), -1)  # Magenta circle
    
    # Add some noise to make it more realistic
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return img

def main():
    print("Creating test image for segmentation...")
    
    # Create test image
    test_img = create_test_image()
    
    # Save test image
    test_img_path = "test_image.jpg"
    cv2.imwrite(test_img_path, test_img)
    print(f"Test image saved as: {test_img_path}")
    
    # Run the segmentation executable
    exe_path = "build/Debug/dnn_segmentation.exe"
    if os.path.exists(exe_path):
        print(f"Running segmentation on test image...")
        try:
            # Test traditional segmentation (fallback when no model is found)
            result = subprocess.run([exe_path, test_img_path, "--no-gui"], 
                                  capture_output=True, text=True, timeout=60)
            print("STDOUT:")
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            print(f"Return code: {result.returncode}")
            
            if result.returncode == 0:
                print("✓ Segmentation completed successfully!")
                print("Check the following directories for results:")
                print("- tools/out_images/traditional_seg/")
                print("- tools/out_images/mobilevit_seg/ (if MobileViT model was loaded)")
            else:
                print("✗ Segmentation failed!")
        except subprocess.TimeoutExpired:
            print("✗ Segmentation timed out!")
        except Exception as e:
            print(f"✗ Error running segmentation: {e}")
    else:
        print(f"✗ Executable not found: {exe_path}")
        print("Make sure to build the project first.")

if __name__ == "__main__":
    main()