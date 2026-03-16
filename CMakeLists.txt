cmake_minimum_required(VERSION 3.10)
project(yolo_ncnn_benchmark)

# Optimization flags for Raspberry Pi 4 (Cortex-A72)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -ffast-math")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find NCNN (from your local build)
find_package(ncnn REQUIRED)

# Threads (recommended for NCNN)
find_package(Threads REQUIRED)

# Build executable
add_executable(main main.cpp)

# Link everything
target_link_libraries(main
    ncnn
    ${OpenCV_LIBS}
    Threads::Threads
)
