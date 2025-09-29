cmake_minimum_required(VERSION 3.10)
project(OpenCVTest)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")

# set(OpenCV_DIR "C:/Users/abd/d/ai/build_opencv/build/install")
find_package(OpenCV REQUIRED)
message(STATUS "Using OpenCV: ${OpenCV_VERSION} at ${OpenCV_DIR}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# add_executable(img_load base/img_load/main.cpp)
# target_link_libraries(img_load ${OpenCV_LIBS})
# add_executable(transforms base/transforms/main.cpp)
# target_link_libraries(transforms ${OpenCV_LIBS})

add_executable(segmentation base/segmentation/main.cpp)
target_link_libraries(segmentation ${OpenCV_LIBS})