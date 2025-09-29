#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: segmentation <image_path>" << endl;
        return 1;
    }
    bool noGui = false;
    for (int i = 2; i < argc; ++i) {
        string a = argv[i];
        if (a == "--no-gui") noGui = true;
    }

    string imgPath = argv[1];
    Mat img = imread(imgPath);
    if (img.empty()) {
        cerr << "Failed to load image: " << imgPath << endl;
        return 1;
    }
    

    // color-based segmentation in HSV + optional grabCut
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    // Ask user for lower/upper HSV ranges or use default for green-ish objects
    Scalar lower(35, 40, 40); // lower bound (H,S,V)
    Scalar upper(85, 255, 255); // upper bound

    Mat mask;
    inRange(hsv, lower, upper, mask);

    // refine with morphology
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    Mat res;
    img.copyTo(res, mask);

    // Create output directory if it doesn't exist
    string outDir = "tools/out_images";
    if (!filesystem::exists(outDir)) {
        filesystem::create_directories(outDir);
    }

    // Save segmentation results
    imwrite(outDir + "/segmentation_original.png", img);
    imwrite(outDir + "/segmentation_mask.png", mask);
    imwrite(outDir + "/segmentation_result.png", res);

    if (!noGui) {
        imshow("Original", img);
        imshow("Mask", mask);
        imshow("Segmented", res);
        cout << "Press 'g' to run GrabCut refinement, any other key to exit." << endl;
    }

    // Always run GrabCut for demonstration purposes
    Mat bgModel, fgModel;
    Mat grabMask;
    // convert mask to GC probable foreground/background
    grabMask.create(mask.size(), CV_8UC1);
    for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
            grabMask.at<uchar>(y,x) = (mask.at<uchar>(y,x) > 0) ? GC_PR_FGD : GC_PR_BGD;
        }
    }

    cout << "Running GrabCut refinement..." << endl;
    Rect rect(1,1, img.cols-2, img.rows-2);
    grabCut(img, grabMask, rect, bgModel, fgModel, 5, GC_INIT_WITH_MASK);

    Mat grabResult = (grabMask==GC_FGD) | (grabMask==GC_PR_FGD);
    Mat grabRes;
    img.copyTo(grabRes, grabResult);

    // Save GrabCut result
    imwrite(outDir + "/grabcut_result.png", grabRes);
    cout << "Segmentation complete! Results saved to " << outDir << "/" << endl;

    if (!noGui) {
        imshow("GrabCut Result", grabRes);
        waitKey(0);
    }

    return 0;
}
