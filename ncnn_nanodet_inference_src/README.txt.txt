Fully working commands for cross compilation of ncnn and opencv with inference src code



cd ~/projects/ncnn_yolo_cpp/ncnn_nanodet_inference_src

mkdir build-arm
cd build-arm/

cmake .. -DCMAKE_TOOLCHAIN_FILE=$HOME/cross/arm64-rpi-toolchain.cmake -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)



after that go into :

cd ~/projects/ncnn_yolo_cpp/ncnn_nanodet_inference_src/build-arm/


copy the bin and other required files, like model and video, to the pi:

scp nanodet_demo ../nanodet.ncnn.param ../nanodet.ncnn.bin ../brt_presentation.mp4 efacec@192.168.1.75:~


ssh into pi:

ssh efacec@192.168.1.75

run it:

./nanodet_demo brt_presentation.mp4


go back to wsl2 and copy resulting output video file to wsl2:

scp efacec@192.168.1.75:~/output_with_detections.avi .





