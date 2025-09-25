#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
#include <vector>

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

    imshow("Test Image", img);
    waitKey(0);
    return 0;
}