#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <ncnn/net.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

static const float CONF_THRESHOLD = 0.2f;
static const float NMS_THRESHOLD  = 0.45f;

struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

static void draw_detections(cv::Mat &frame, const vector<Detection> &dets)
{
    for (const auto &d : dets) {
        cv::rectangle(frame, d.box, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame,
                    cv::format("%.2f", d.score),
                    d.box.tl() + cv::Point(0, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);
    }
}

struct PreprocessState {
    ncnn::Mat input;
    float scale_x = 1.0f;
    float scale_y = 1.0f;
};

static PreprocessState preprocess(const cv::Mat &frame)
{
    PreprocessState st;

    const int img_w = frame.cols;
    const int img_h = frame.rows;
    const int target_w = 640;
    const int target_h = 640;

    st.scale_x = static_cast<float>(img_w) / target_w;
    st.scale_y = static_cast<float>(img_h) / target_h;

    st.input = ncnn::Mat::from_pixels_resize(
        frame.data, ncnn::Mat::PIXEL_BGR,
        img_w, img_h,
        target_w, target_h
    );

    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    st.input.substract_mean_normalize(mean_vals, norm_vals);

    return st;
}

static ncnn::Mat infer(ncnn::Net &net, const ncnn::Mat &input)
{
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.set_blob_allocator(ncnn::UnlockedPoolAllocator());

    ncnn::Mat out;
    ex.input("in0", input);
    ex.extract("out0", out);

    return out;
}

static vector<Detection> postprocess(
    const ncnn::Mat &out,
    const cv::Mat &frame,
    float scale_x,
    float scale_y,
    vector<cv::Rect> &boxes,
    vector<float> &scores,
    vector<int> &indices)
{
    boxes.clear();
    scores.clear();
    indices.clear();

    const int num_boxes = out.w;

    const float* x_ptr     = out.row(0);
    const float* y_ptr     = out.row(1);
    const float* w_ptr     = out.row(2);
    const float* h_ptr     = out.row(3);
    const float* score_ptr = out.row(4);
    const float* cls_ptr   = out.row(5);

    for (int i = 0; i < num_boxes; i++) {
        const float score = score_ptr[i];
        if (score < CONF_THRESHOLD) continue;

        const float x = x_ptr[i];
        const float y = y_ptr[i];
        const float w = w_ptr[i];
        const float h = h_ptr[i];

        const float cx = x * scale_x;
        const float cy = y * scale_y;
        const float bw = w * scale_x;
        const float bh = h * scale_y;

        const int left = std::max(0, std::min((int)(cx - bw * 0.5f), frame.cols - 1));
        const int top  = std::max(0, std::min((int)(cy - bh * 0.5f), frame.rows - 1));
        const int right = std::max(0, std::min((int)(cx + bw * 0.5f), frame.cols - 1));
        const int bottom = std::max(0, std::min((int)(cy + bh * 0.5f), frame.rows - 1));

        if (right <= left || bottom <= top) continue;

        boxes.emplace_back(left, top, right - left, bottom - top);
        scores.push_back(score);
    }

    vector<Detection> dets;
    if (!boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);

        dets.reserve(indices.size());
        for (int idx : indices) {
            int class_id = (int)cls_ptr[idx];
            dets.push_back({boxes[idx], scores[idx], class_id});
        }
    }

    return dets;
}

int main(int argc, char** argv)
{
    bool save_video = true;
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--no-video") {
            save_video = false;
        }
    }

    string video_path  = "data/brt_presentation.mp4";
    string model_param = "models/model.param";
    string model_bin   = "models/model.bin";
    string output_video = "output_ncnn.avi";

    cout << "YOLO NCNN Inference (timed stages)\n";
    cout << "Model: " << model_param << "\n";
    cout << "Video: " << video_path << "\n";
    cout << "Save video: " << (save_video ? "yes" : "no") << "\n";

    ncnn::Net net;
    net.opt.num_threads = 4;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_packing_layout = true;
    net.opt.use_winograd_convolution = true;

    net.load_param(model_param.c_str());
    net.load_model(model_bin.c_str());

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cout << "Error: Could not open video.\n";
        return -1;
    }

    int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter writer;
    if (save_video) {
        int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
        writer.open(output_video, fourcc, fps > 0 ? fps : 25.0, cv::Size(width, height));
        if (!writer.isOpened()) {
            cout << "H264 codec not available, falling back to MJPEG\n";
            fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            writer.open(output_video, fourcc, fps > 0 ? fps : 25.0, cv::Size(width, height));
            if (!writer.isOpened()) {
                cout << "Error: Could not open output file.\n";
                return -1;
            }
        }
    }

    vector<double> t_total, t_read, t_pre, t_infer, t_post, t_write;
    t_total.reserve(total_frames);
    t_read.reserve(total_frames);
    t_pre.reserve(total_frames);
    t_infer.reserve(total_frames);
    t_post.reserve(total_frames);
    t_write.reserve(total_frames);

    vector<cv::Rect> boxes;
    vector<float> scores;
    vector<int> indices;
    boxes.reserve(200);
    scores.reserve(200);
    indices.reserve(200);

    cv::Mat frame;
    int frame_count = 0;

    while (true) {
        auto t_frame_start = chrono::high_resolution_clock::now();

        auto t_read_start = chrono::high_resolution_clock::now();
        if (!cap.read(frame)) break;
        auto t_read_end = chrono::high_resolution_clock::now();

        frame_count++;

        auto t_pre_start = chrono::high_resolution_clock::now();
        PreprocessState pre = preprocess(frame);
        auto t_pre_end = chrono::high_resolution_clock::now();

        auto t_infer_start = chrono::high_resolution_clock::now();
        ncnn::Mat out = infer(net, pre.input);
        auto t_infer_end = chrono::high_resolution_clock::now();

        auto t_post_start = chrono::high_resolution_clock::now();
        auto dets = postprocess(out, frame, pre.scale_x, pre.scale_y, boxes, scores, indices);
        auto t_post_end = chrono::high_resolution_clock::now();

        draw_detections(frame, dets);

        auto t_write_start = chrono::high_resolution_clock::now();
        if (save_video) {
            writer.write(frame);
        }
        auto t_write_end = chrono::high_resolution_clock::now();

        auto t_frame_end = chrono::high_resolution_clock::now();

        t_total.push_back(chrono::duration<double>(t_frame_end - t_frame_start).count());
        t_read.push_back(chrono::duration<double>(t_read_end - t_read_start).count());
        t_pre.push_back(chrono::duration<double>(t_pre_end - t_pre_start).count());
        t_infer.push_back(chrono::duration<double>(t_infer_end - t_infer_start).count());
        t_post.push_back(chrono::duration<double>(t_post_end - t_post_start).count());
        t_write.push_back(chrono::duration<double>(t_write_end - t_write_start).count());

        if (frame_count % 50 == 0) {
            cout << "Processed " << frame_count << "/" << total_frames
                 << " - pre: " << t_pre.back() * 1000
                 << " ms, infer: " << t_infer.back() * 1000
                 << " ms, post: " << t_post.back() * 1000
                 << " ms, io: " << (t_read.back() + t_write.back()) * 1000
                 << " ms\n";
        }
    }

    cap.release();
    if (save_video) {
        writer.release();
    }

    if (!t_total.empty()) {
        auto avg = [](const vector<double> &v) {
            return accumulate(v.begin(), v.end(), 0.0) / v.size();
        };
        double avg_total = avg(t_total);
        double avg_read  = avg(t_read);
        double avg_pre   = avg(t_pre);
        double avg_infer = avg(t_infer);
        double avg_post  = avg(t_post);
        double avg_write = avg(t_write);

        cout << "\n=============================================\n";
        cout << "Frames processed: " << frame_count << "\n";
        cout << "Average total time: " << avg_total * 1000.0 << " ms\n";
        cout << "Average read time: " << avg_read * 1000.0 << " ms\n";
        cout << "Average preprocess time: " << avg_pre * 1000.0 << " ms\n";
        cout << "Average inference time: " << avg_infer * 1000.0 << " ms\n";
        cout << "Average postprocess time: " << avg_post * 1000.0 << " ms\n";
        cout << "Average write time: " << avg_write * 1000.0 << " ms\n";
        cout << "FPS (total): " << 1.0 / avg_total << "\n";
        cout << "FPS (inference only): " << 1.0 / avg_infer << "\n";
        cout << "Output: " << output_video << "\n";
        cout << "=============================================\n";
    }

    return 0;
}