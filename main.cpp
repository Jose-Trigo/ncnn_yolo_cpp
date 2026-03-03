#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

static const float CONF_THRESHOLD = 0.2f;
static const float NMS_THRESHOLD  = 0.2f;

struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

void nms(vector<Detection>& dets, float nms_thresh)
{
    sort(dets.begin(), dets.end(),
         [](const Detection& a, const Detection& b) { return a.score > b.score; });

    vector<Detection> result;
    vector<bool> removed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); i++) {
        if (removed[i]) continue;
        result.push_back(dets[i]);

        for (size_t j = i + 1; j < dets.size(); j++) {
            if (removed[j]) continue;

            float inter = (dets[i].box & dets[j].box).area();
            float uni   = dets[i].box.area() + dets[j].box.area() - inter;
            float iou   = inter / (uni + 1e-6f);

            if (iou > nms_thresh)
                removed[j] = true;
        }
    }

    dets = result;
}

vector<Detection> run_yolo(ncnn::Net& net, const cv::Mat& frame)
{
    int img_w = frame.cols;
    int img_h = frame.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        frame.data, ncnn::Mat::PIXEL_BGR,
        img_w, img_h,
        640, 640
    );

    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);

    ncnn::Mat out;
    ex.input("in0", in);
    ex.extract("out0", out);

    int num_boxes = out.w;

    const float* x_ptr     = out.row(0);
    const float* y_ptr     = out.row(1);
    const float* w_ptr     = out.row(2);
    const float* h_ptr     = out.row(3);
    const float* score_ptr = out.row(4);
    const float* cls_ptr   = out.row(5);

    vector<Detection> dets;
    dets.reserve(num_boxes);

    for (int i = 0; i < num_boxes; i++) {
        float score = score_ptr[i];
        if (score < CONF_THRESHOLD) continue;

        float x = x_ptr[i];
        float y = y_ptr[i];
        float w = w_ptr[i];
        float h = h_ptr[i];
        int class_id = (int)cls_ptr[i];

        float cx = x * img_w / 640.0f;
        float cy = y * img_h / 640.0f;
        float bw = w * img_w / 640.0f;
        float bh = h * img_h / 640.0f;

        int left = (int)(cx - bw / 2.0f);
        int top  = (int)(cy - bh / 2.0f);
        int rw   = (int)bw;
        int rh   = (int)bh;

        if (rw <= 0 || rh <= 0) continue;

        dets.push_back({cv::Rect(left, top, rw, rh), score, class_id});
    }

    nms(dets, NMS_THRESHOLD);
    return dets;
}

int main()
{
    string video_path = "data/brt_presentation.mp4";
    string model_param = "models/model.param";
    string model_bin   = "models/model.bin";

    cout << "YOLO NCNN Benchmark\n";
    cout << "Video: " << video_path << "\n";

    ncnn::Net net;
    net.opt.num_threads = 4;      // Raspberry Pi 4 = 4 cores
    net.opt.use_vulkan_compute = false; // Pi 4 has no GPU
    net.load_param(model_param.c_str());
    net.load_model(model_bin.c_str());

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cout << "Error: Could not open video.\n";
        return -1;
    }

    int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    cout << "Total frames: " << total_frames << "\n";

    vector<double> inference_times;
    inference_times.reserve(total_frames);

    cv::Mat frame;
    int frame_count = 0;

    while (true) {
        if (!cap.read(frame)) break;
        frame_count++;

        auto start = chrono::high_resolution_clock::now();
        auto dets = run_yolo(net, frame);
        auto end = chrono::high_resolution_clock::now();

        double ms = chrono::duration<double, milli>(end - start).count();
        inference_times.push_back(ms);

        if (frame_count % 100 == 0)
            cout << "Processed " << frame_count << " frames\n";
    }

    double avg = 0;
    for (double t : inference_times) avg += t;
    avg /= inference_times.size();

    cout << "\n=============================================\n";
    cout << "Frames processed: " << frame_count << "\n";
    cout << "Average inference time: " << avg << " ms\n";
    cout << "Theoretical FPS: " << 1000.0 / avg << "\n";
    cout << "=============================================\n";

    return 0;
}
