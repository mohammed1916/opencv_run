#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    vector<string> candidates = {
        "dataset\\leftImg8bit_trainvaltest\\test\\berlin\\berlin_000000_000019_leftImg8bit.png",
        "..\\dataset\\leftImg8bit_trainvaltest\\test\\berlin\\berlin_000000_000019_leftImg8bit.png",
    };

    // CLI: if --no-gui is present, don't call imshow/waitKey; instead
    // write transformed images to the output directory `out_images`.
    bool noGui = false;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--no-gui") noGui = true;
    }

    Mat img;
    string usedPath;
    for (const auto &p : candidates) {
        img = imread(p);
        if (!img.empty()) {
            usedPath = p;
            break;
        }
    }

    if (img.empty()) {
        cout << "Image not found!\n";
        for (const auto &p : candidates) cout << "Tried: " << p << "\n";
        return -1;
    }

    cout << "Loaded image from: " << usedPath << "\n";

    // Display or save original image
    if (noGui) {
        std::filesystem::create_directories("out_images");
        imwrite("out_images/original.png", img);
    } else {
        imshow("Original Image", img);
    }

    // ----------------Transformations --------------------

    //Convert to Grayscale
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    if (noGui) imwrite("out_images/grayscale.png", gray);
    else imshow("Grayscale", gray);

    //Gaussian Blur
    Mat blurImg;
    GaussianBlur(img, blurImg, Size(7,7), 1.5);
    if (noGui) imwrite("out_images/gaussian_blur.png", blurImg);
    else imshow("Gaussian Blur", blurImg);

    //Edge Detection (Canny)
    Mat edges;
    Canny(gray, edges, 50, 150);
    if (noGui) imwrite("out_images/canny_edges.png", edges);
    else imshow("Canny Edges", edges);

    //Resize
    Mat resized;
    resize(img, resized, Size(img.cols/2, img.rows/2));
    if (noGui) imwrite("out_images/resized.png", resized);
    else imshow("Resized", resized);

    //Rotation
    Mat rotated;
    Point2f center(img.cols/2.0F, img.rows/2.0F);
    double angle = 45; 
    double scale = 1.0;
    Mat rotMat = getRotationMatrix2D(center, angle, scale);
    warpAffine(img, rotated, rotMat, img.size());
    if (noGui) imwrite("out_images/rotated.png", rotated);
    else imshow("Rotated 45 deg", rotated);

    //Perspective Transform
    Mat persp;
    Point2f srcPts[4] = { {0.0f, 0.0f}, {static_cast<float>(img.cols-1), 0.0f}, {static_cast<float>(img.cols-1), static_cast<float>(img.rows-1)}, {0.0f, static_cast<float>(img.rows-1)} };
    Point2f dstPts[4] = { {50.0f, 50.0f}, {static_cast<float>(img.cols-100), 30.0f}, {static_cast<float>(img.cols-50), static_cast<float>(img.rows-50)}, {30.0f, static_cast<float>(img.rows-30)} };
    Mat perspMat = getPerspectiveTransform(srcPts, dstPts);
    warpPerspective(img, persp, perspMat, img.size());
    if (noGui) imwrite("out_images/perspective.png", persp);
    else imshow("Perspective Transform", persp);

    // Color Space Conversion (HSV)
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    if (noGui) imwrite("out_images/hsv.png", hsv);
    else imshow("HSV Image", hsv);

    // Flip
    Mat flipped;
    flip(img, flipped, 1); // 1 = horizontal flip
    if (noGui) imwrite("out_images/flipped.png", flipped);
    else imshow("Flipped", flipped);

    if (!noGui) waitKey(0);
    else {
        // Print saved files so callers (like notebooks) can find them.
        cout << "Saved images to out_images/" << endl;
    }
    return 0;
}
