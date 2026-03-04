cmake_minimum_required(VERSION 3.10)
project(yolo_ncnn_benchmark)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main ncnn ${OpenCV_LIBS})
