cmake_minimum_required(VERSION 3.5)
project( LandmarkDemo )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( kazemi-train kazemi-train.cpp)
add_executable( kazemi-demo kazemi-demo.cpp )
add_executable( kazemi-webcam-demo kazemi-webcam-demo.cpp )
target_link_libraries( kazemi-train ${OpenCV_LIBS} )
target_link_libraries( kazemi-demo ${OpenCV_LIBS} )
target_link_libraries( kazemi-webcam-demo ${OpenCV_LIBS} )
