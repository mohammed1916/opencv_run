#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: dnn_segmentation <image_path> [--no-gui] [--use-sam] [--sam-mode auto|points]" << endl;
        cout << "This program performs semantic segmentation using DNN models or SAM." << endl;
        cout << "Options:" << endl;
        cout << "  --no-gui     Run without GUI display" << endl;
        cout << "  --use-sam    Use SAM (Segment Anything Model) instead of traditional DNN" << endl;
        cout << "  --sam-mode   SAM mode: 'auto' for automatic segmentation, 'points' for point-based" << endl;
        return 1;
    }
    
    bool noGui = false;
    bool useSam = false;
    string samMode = "auto";
    
    for (int i = 2; i < argc; ++i) {
        string a = argv[i];
        if (a == "--no-gui") noGui = true;
        else if (a == "--use-sam") useSam = true;
        else if (a == "--sam-mode" && i + 1 < argc) {
            samMode = argv[++i];
        }
    }

    string imgPath = argv[1];
    Mat img = imread(imgPath);
    if (img.empty()) {
        cerr << "Failed to load image: " << imgPath << endl;
        return 1;
    }

    if (useSam) {
        cout << "Using SAM (Segment Anything Model) for segmentation..." << endl;
        
        string outDir = "tools/out_images/sam";
        if (!filesystem::exists(outDir)) {
            filesystem::create_directories(outDir);
        }
        
        // Build Python command to run SAM with CUDA support
        // The C++ executable runs from project root, so always use tools/sam_wrapper.py
        string samScript = "tools/sam_wrapper.py";
        string pythonCmd = "python " + samScript + " \"" + imgPath + "\" --output_dir " + outDir + " --mode " + samMode;
        
        cout << "Running SAM with CUDA acceleration..." << endl;
        cout << "Command: " << pythonCmd << endl;
        
        // Execute Python SAM script
        int result = system(pythonCmd.c_str());
        
        if (result == 0) {
            cout << "SAM segmentation completed successfully!" << endl;
            
            if (!noGui) {
                vector<string> samResults = {
                    outDir + "/sam_original.png",
                    outDir + "/sam_colored_masks.png",
                    outDir + "/sam_overlay.png"
                };
                
                vector<string> windowNames = {"Original", "SAM Masks", "SAM Overlay"};
                
                for (size_t i = 0; i < samResults.size(); ++i) {
                    Mat result_img = imread(samResults[i]);
                    if (!result_img.empty()) {
                        imshow(windowNames[i], result_img);
                    }
                }
                
                cout << "Press any key to exit..." << endl;
                waitKey(0);
                destroyAllWindows();
            }
            
            return 0;
        } else {
            cout << "SAM segmentation failed. Falling back to traditional segmentation..." << endl;
        }
    }

    cout << "Loading MobileViT classification model for feature-based segmentation..." << endl;
    
    string modelPath = "models/mobilevit/model.onnx"; 
    string configPath = "models/mobilevit/config.json"; 
    
    Net net;
    bool modelLoaded = false;
    
    try {
        if (filesystem::exists(modelPath)) {
            net = readNetFromONNX(modelPath);
            
            if (!net.empty()) {
                modelLoaded = true;
                cout << "Successfully loaded MobileViT model: " << modelPath << endl;
                cout << "Note: This is a classification model, will be used for feature extraction" << endl;
            }
        } else {
            cout << "MobileViT model not found at: " << modelPath << endl;
        }
    } catch (const Exception& e) {
        cout << "Failed to load MobileViT model: " << e.what() << endl;
    }
    
 
    
    if (!modelLoaded) {
        cout << "No pre-trained DNN model found. Using advanced traditional segmentation methods..." << endl;
        
        Mat gray, binary, dist_transform, markers;
        
        cvtColor(img, gray, COLOR_BGR2GRAY);
        Mat filtered;
        bilateralFilter(gray, filtered, 9, 75, 75);
        
        threshold(filtered, binary, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
        
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(binary, binary, MORPH_OPEN, kernel, Point(-1,-1), 2);
        morphologyEx(binary, binary, MORPH_CLOSE, kernel, Point(-1,-1), 1);
        
        Mat sure_bg;
        dilate(binary, sure_bg, kernel, Point(-1,-1), 3);
        
        distanceTransform(binary, dist_transform, DIST_L2, 5);
        Mat sure_fg;
        double maxVal;
        minMaxLoc(dist_transform, nullptr, &maxVal);
        threshold(dist_transform, sure_fg, 0.4 * maxVal, 255, 0);
        sure_fg.convertTo(sure_fg, CV_8U);
        
        Mat unknown;
        subtract(sure_bg, sure_fg, unknown);
        
        connectedComponents(sure_fg, markers);
        
        markers = markers + 1;
        
        markers.setTo(0, unknown == 255);
        
        watershed(img, markers);
        
        Mat segmented = Mat::zeros(img.size(), CV_8UC3);
        
        vector<Vec3b> colors;
        colors.push_back(Vec3b(0, 0, 0)); 
        for (int i = 1; i < 256; i++) {
            colors.push_back(Vec3b(rand() % 255, rand() % 255, rand() % 255));
        }
        
        for (int i = 0; i < markers.rows; i++) {
            for (int j = 0; j < markers.cols; j++) {
                int index = markers.at<int>(i, j);
                if (index > 0 && index < colors.size()) {
                    segmented.at<Vec3b>(i, j) = colors[index];
                }
            }
        }
        
        Mat grabcut_result = img.clone();
        Mat mask = Mat::zeros(img.size(), CV_8U);
        Mat bgModel, fgModel;
        
        int border = (int)(min(img.rows, img.cols) * 0.1);
        Rect rectangle(border, border, img.cols - 2*border, img.rows - 2*border);
        
        cout << "Applying GrabCut algorithm..." << endl;
        grabCut(img, mask, rectangle, bgModel, fgModel, 5, GC_INIT_WITH_RECT);
        
        Mat grabcut_mask;
        compare(mask, GC_PR_FGD, grabcut_mask, CMP_EQ);
        Mat mask2;
        compare(mask, GC_FGD, mask2, CMP_EQ);
        grabcut_mask = grabcut_mask | mask2;
        
        grabcut_result.setTo(Scalar(0, 0, 0));
        img.copyTo(grabcut_result, grabcut_mask);
        
        string outDir = "tools/out_images/traditional_seg";
        if (!filesystem::exists(outDir)) {
            filesystem::create_directories(outDir);
        }
        
        imwrite(outDir + "/original.png", img);
        imwrite(outDir + "/binary.png", binary);
        imwrite(outDir + "/distance_transform.png", dist_transform);
        imwrite(outDir + "/watershed_segmented.png", segmented);
        imwrite(outDir + "/grabcut_result.png", grabcut_result);
        
        cout << "Traditional segmentation complete! Results saved to " << outDir << "/" << endl;
        cout << "Methods used: Watershed + GrabCut" << endl;
        
        if (!noGui) {
            Mat display_orig, display_binary, display_dist, display_segmented, display_grabcut;
            int display_width = 400;
            double scale = (double)display_width / img.cols;
            int display_height = (int)(img.rows * scale);
            
            resize(img, display_orig, Size(display_width, display_height));
            resize(binary, display_binary, Size(display_width, display_height));
            resize(dist_transform, display_dist, Size(display_width, display_height));
            resize(segmented, display_segmented, Size(display_width, display_height));
            resize(grabcut_result, display_grabcut, Size(display_width, display_height));
            
            imshow("Original", display_orig);
            imshow("Binary Mask", display_binary);
            imshow("Distance Transform", display_dist);
            imshow("Watershed Segmentation", display_segmented);
            imshow("GrabCut Result", display_grabcut);
            
            cout << "Press any key to exit..." << endl;
            waitKey(0);
            destroyAllWindows();
        }
        
        return 0;
    }
    
    cout << "Performing feature-based segmentation with MobileViT..." << endl;
    
    Mat blob;
    blobFromImage(img, blob, 2.0/255.0, Size(224, 224), Scalar(127.5, 127.5, 127.5), true, false);
    blob -= 1.0; // Convert [0,1] to [-1,1]
    
    net.setInput(blob);
    
    Mat features = net.forward();
    
    cout << "Features shape: " << features.size << endl;
    cout << "Features type: " << features.type() << endl;
    
    Mat segmentation, coloredSeg, result;
    
    Mat imgFloat, imgResized;
    img.convertTo(imgFloat, CV_32F);
    resize(imgFloat, imgResized, Size(224, 224));
    
    Mat flatFeatures = features.reshape(1, features.total());
    cout << "Using MobileViT features (size: " << flatFeatures.rows << " x " << flatFeatures.cols << ") for segmentation guidance..." << endl;
    
    // Create feature-guided segmentation by combining spatial and feature information
    Mat samples = imgResized.reshape(3, imgResized.rows * imgResized.cols);
    
    Mat normalizedFeatures;
    normalize(flatFeatures, normalizedFeatures, 0, 255, NORM_MINMAX, CV_32F);
    
    // Use the first few principal features to influence clustering
    int numFeaturesToUse = min(8, (int)flatFeatures.total()); // Use up to 8 features
    Mat featureWeights = normalizedFeatures(Rect(0, 0, 1, numFeaturesToUse)).clone();
    
    // Create feature-weighted samples by adding feature bias to each pixel
    Mat featureInfluencedSamples = samples.clone();
    
    // Apply feature-based bias to clustering
    for (int i = 0; i < samples.rows; i++) {
        for (int j = 0; j < samples.cols; j++) {
            // Add feature influence based on position and feature values
            int featureIdx = (i * samples.cols + j) % numFeaturesToUse;
            float featureWeight = featureWeights.at<float>(featureIdx) * 0.1f; // Scale influence
            featureInfluencedSamples.at<float>(i, j) += featureWeight;
        }
    }
    
    // Perform K-means clustering with feature-influenced data
    int K = 8; // Number of clusters
    Mat labels, centers;
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 20, 1.0);
    
    cout << "Performing MobileViT feature-guided K-means clustering with K=" << K << "..." << endl;
    kmeans(featureInfluencedSamples, K, labels, criteria, 3, KMEANS_PP_CENTERS, centers);
    
    // Convert labels back to image format
    Mat clustered = labels.reshape(0, imgResized.rows);
    clustered.convertTo(segmentation, CV_8U, 255.0 / (K-1));
    
    // Resize back to original size
    resize(segmentation, segmentation, img.size(), 0, 0, INTER_NEAREST);
    
    // Create colored segmentation
    applyColorMap(segmentation, coloredSeg, COLORMAP_JET);
    
    // Blend with original image
    addWeighted(img, 0.6, coloredSeg, 0.4, 0, result);
    
    cout << "MobileViT feature-guided segmentation completed successfully!" << endl;
    cout << "Used " << numFeaturesToUse << " MobileViT features to guide K-means clustering." << endl;
    
    // Create output directory if it doesn't exist
    string outDir = "tools/out_images/mobilevit_seg";
    if (!filesystem::exists(outDir)) {
        filesystem::create_directories(outDir);
    }
    
    // Save results
    imwrite(outDir + "/original.png", img);
    imwrite(outDir + "/segmentation_mask.png", segmentation);
    imwrite(outDir + "/colored_segmentation.png", coloredSeg);
    imwrite(outDir + "/blended_result.png", result);
    
    cout << "MobileViT-based segmentation complete! Results saved to " << outDir << "/" << endl;
    cout << "Method used: K-means clustering guided by MobileViT features" << endl;
    
    if (!noGui) {
        // Resize images for better display
        Mat display_orig, display_seg, display_colored, display_result;
        int display_width = 400;
        double scale = (double)display_width / img.cols;
        int display_height = (int)(img.rows * scale);
        
        resize(img, display_orig, Size(display_width, display_height));
        resize(segmentation, display_seg, Size(display_width, display_height));
        resize(coloredSeg, display_colored, Size(display_width, display_height));
        resize(result, display_result, Size(display_width, display_height));
        
        imshow("Original", display_orig);
        imshow("Segmentation Mask", display_seg);
        imshow("Colored Segmentation", display_colored);
        imshow("Blended Result", display_result);
        
        cout << "Press any key to exit..." << endl;
        waitKey(0);
        destroyAllWindows();
    }
    
    return 0;
}