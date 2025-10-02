cmake_minimum_required(VERSION 3.10)
project(OpenCVTest)

set(CMAKE_CXX_STANDARD 17)

# set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")

set(OpenCV_DIR "C:/Users/abd/d/ai/opencv_build/install/x64/vc17/lib")
# find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio tracking)
find_package(OpenCV REQUIRED)
message(STATUS "Using OpenCV: ${OpenCV_VERSION} at ${OpenCV_DIR}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

add_executable(mtracking base/tracking/tutorial_multitracker.cpp)
# target_include_directories(mtracking PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mtracking PRIVATE ${OpenCV_LIBS})

# add_executable(img_load base/img_load/main.cpp)
# target_link_libraries(img_load ${OpenCV_LIBS})
# add_executable(transforms base/transforms/main.cpp)
# target_link_libraries(transforms ${OpenCV_LIBS})

# add_executable(segmentation base/segmentation/main.cpp)
# target_link_libraries(segmentation ${OpenCV_LIBS})

# add_executable(dnn_segmentation base/dnn_segmentation/main.cpp)
# target_link_libraries(dnn_segmentation ${OpenCV_LIBS})
# add_executable(tracking base/tracking/main.cpp)
# target_link_libraries(tracking ${OpenCV_LIBS})
