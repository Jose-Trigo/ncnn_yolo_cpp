cmake_minimum_required(VERSION 3.10)

project(nanodet_demo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# Path inside your Raspberry Pi sysroot
set(NCNN_DIR $ENV{HOME}/sysroot/pi/opt/ncnn)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${NCNN_DIR}/include
    ${NCNN_DIR}/include/ncnn
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(
    nanodet_demo
    main.cpp
    nanodet.cpp
)

target_link_libraries(
    nanodet_demo
    ${OpenCV_LIBS}
    ${NCNN_DIR}/lib/libncnn.so
    gomp
)