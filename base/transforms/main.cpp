#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    vector<string> candidates = {
        "dataset\\leftImg8bit_trainvaltest\\test\\berlin\\berlin_000000_000019_leftImg8bit.png",
    };

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

    // Display original image
    imshow("Original Image", img);

    // ----------------Transformations --------------------

    //Convert to Grayscale
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    imshow("Grayscale", gray);

    //Gaussian Blur
    Mat blurImg;
    GaussianBlur(img, blurImg, Size(7,7), 1.5);
    imshow("Gaussian Blur", blurImg);

    //Edge Detection (Canny)
    Mat edges;
    Canny(gray, edges, 50, 150);
    imshow("Canny Edges", edges);

    //Resize
    Mat resized;
    resize(img, resized, Size(img.cols/2, img.rows/2));
    imshow("Resized", resized);

    //Rotation
    Mat rotated;
    Point2f center(img.cols/2.0F, img.rows/2.0F);
    double angle = 45; // degrees
    double scale = 1.0;
    Mat rotMat = getRotationMatrix2D(center, angle, scale);
    warpAffine(img, rotated, rotMat, img.size());
    imshow("Rotated 45 deg", rotated);

    //Perspective Transform
    Mat persp;
    // Use explicit float literals to avoid narrowing conversion errors on MSVC when
    // initializing cv::Point2f from integer initializer lists.
    Point2f srcPts[4] = { {0.0f, 0.0f}, {static_cast<float>(img.cols-1), 0.0f}, {static_cast<float>(img.cols-1), static_cast<float>(img.rows-1)}, {0.0f, static_cast<float>(img.rows-1)} };
    Point2f dstPts[4] = { {50.0f, 50.0f}, {static_cast<float>(img.cols-100), 30.0f}, {static_cast<float>(img.cols-50), static_cast<float>(img.rows-50)}, {30.0f, static_cast<float>(img.rows-30)} };
    Mat perspMat = getPerspectiveTransform(srcPts, dstPts);
    warpPerspective(img, persp, perspMat, img.size());
    imshow("Perspective Transform", persp);

    // Color Space Conversion (HSV)
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    imshow("HSV Image", hsv);

    // Flip
    Mat flipped;
    flip(img, flipped, 1); // 1 = horizontal flip
    imshow("Flipped", flipped);

    waitKey(0);
    return 0;
}
