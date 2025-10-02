#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <memory>

namespace fs = std::filesystem;

class SessionManager {
public:
    SessionManager(const std::string& basePath) {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        char buf[32];
        strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));
        sessionPath = basePath + "/session_" + std::string(buf);
        fs::create_directories(sessionPath);
    }
    std::string getSessionPath() const { return sessionPath; }
private:
    std::string sessionPath;
};

class TrajectoryLogger {
public:
    TrajectoryLogger(const std::string& folder) : folderPath(folder) {}
    void log(int objectId, const cv::Point2f& translation, double timestamp) {
        std::ofstream ofs(folderPath + "/object_" + std::to_string(objectId) + ".csv", std::ios::app);
        ofs << timestamp << "," << translation.x << "," << translation.y << "\n";
    }
private:
    std::string folderPath;
};

class TrackerManager {
public:
    TrackerManager(const std::string& trackerType) : trackerType(trackerType), nextId(0) {}
    void update(const cv::Mat& frame, const std::vector<cv::Rect>& detections, double timestamp, TrajectoryLogger& logger) {
        std::vector<bool> matched(detections.size(), false);
        for (auto& obj : objects) obj.updated = false;
        for (size_t i = 0; i < objects.size(); ++i) {
            bool ok = objects[i].tracker->update(frame, objects[i].bbox);
            if (ok) {
                objects[i].updated = true;
                logger.log(objects[i].id, center(objects[i].bbox), timestamp);
            }
        }
        for (size_t i = 0; i < detections.size(); ++i) {
            int bestIdx = -1;
            double minDist = 1e9;
            for (size_t j = 0; j < objects.size(); ++j) {
                double d = cv::norm(center(detections[i]) - center(objects[j].bbox));
                if (d < minDist && d < 50.0 && !matched[i]) {
                    minDist = d;
                    bestIdx = j;
                }
            }
            if (bestIdx != -1 && !objects[bestIdx].updated) {
                objects[bestIdx].tracker = createTracker(trackerType);
                objects[bestIdx].tracker->init(frame, detections[i]);
                objects[bestIdx].bbox = detections[i];
                objects[bestIdx].updated = true;
                matched[i] = true;
                logger.log(objects[bestIdx].id, center(detections[i]), timestamp);
            }
        }
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!matched[i]) {
                TrackedObject obj;
                obj.id = nextId++;
                obj.bbox = detections[i];
                obj.tracker = createTracker(trackerType);
                obj.tracker->init(frame, detections[i]);
                obj.updated = true;
                objects.push_back(std::move(obj));
                logger.log(obj.id, center(detections[i]), timestamp);
            }
        }
        objects.erase(std::remove_if(objects.begin(), objects.end(),
            [](const TrackedObject& o) { return !o.updated; }), objects.end());
    }
private:
    struct TrackedObject {
        int id;
        cv::Rect2d bbox;
        std::shared_ptr<cv::Tracker> tracker;
        bool updated;
    };
    std::vector<TrackedObject> objects;
    int nextId;
    std::string trackerType;
    cv::Point2f center(const cv::Rect2d& r) { return cv::Point2f(r.x + r.width/2.0f, r.y + r.height/2.0f); }
    std::shared_ptr<cv::Tracker> createTracker(const std::string& type) {
        if (type == "KCF") return cv::TrackerKCF::create();
        if (type == "CSRT") return cv::TrackerCSRT::create();
        if (type == "MIL") return cv::TrackerMIL::create();
        if (type == "MOSSE") return cv::TrackerMOSSE::create();
        return cv::TrackerKCF::create();
    }
};

std::vector<cv::Rect> detectMovingObjects(const cv::Mat& frame, cv::Ptr<cv::BackgroundSubtractor> bgsub) {
    cv::Mat fgmask;
    bgsub->apply(frame, fgmask);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fgmask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> detections;
    for (const auto& c : contours) {
        if (cv::contourArea(c) > 500)
            detections.push_back(cv::boundingRect(c));
    }
    return detections;
}

int main() {
    SessionManager session("logs");
    TrajectoryLogger logger(session.getSessionPath());
    TrackerManager tracker("KCF");
    cv::VideoCapture cap(0);
    auto bgsub = cv::createBackgroundSubtractorMOG2();
    while (cap.isOpened()) {
        cv::Mat frame;
        if (!cap.read(frame)) break;
        auto now = std::chrono::system_clock::now();
        double timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();
        auto detections = detectMovingObjects(frame, bgsub);
        tracker.update(frame, detections, timestamp, logger);
        for (const auto& r : detections)
            cv::rectangle(frame, r, cv::Scalar(0,255,0), 2);
        cv::imshow("Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}
