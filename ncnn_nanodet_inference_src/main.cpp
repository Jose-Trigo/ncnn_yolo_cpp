// main.cpp
// Minimal launcher with draw_bboxes and video_demo included.
// Usage: ./nanodet_demo input_video_path
// Expects nanodet.ncnn.param and nanodet.ncnn.bin in the working directory.

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <ncnn/net.h>
#include "nanodet.h"

// If you want to set ncnn OpenMP threads manually, uncomment and set:
// #include <omp.h>
// ncnn::set_omp_num_threads(4);

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

// Keep the original resize_uniform to preserve letterbox behavior and effect_roi mapping
int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0f / h;
    float ratio_dst = dst_w * 1.0f / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0f / w) * h);
    }
    else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0f / h) * w);
    }
    else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) {
        int index_w = floor((dst_w - tmp_w) / 2.0f);
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h) {
        int index_h = floor((dst_h - tmp_h) / 2.0f);
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else {
        printf("error\n");
    }
    return 0;
}

// Full color palette preserved (80 colors) — used by draw_bboxes
const int color_list[80][3] =
{
    {216 , 82 , 24},
    {236 ,176 , 31},
    {125 , 46 ,141},
    {118 ,171 , 47},
    { 76 ,189 ,237},
    {238 , 19 , 46},
    { 76 , 76 , 76},
    {153 ,153 ,153},
    {255 ,  0 ,  0},
    {255 ,127 ,  0},
    {190 ,190 ,  0},
    {  0 ,255 ,  0},
    {  0 ,  0 ,255},
    {170 ,  0 ,255},
    { 84 , 84 ,  0},
    { 84 ,170 ,  0},
    { 84 ,255 ,  0},
    {170 , 84 ,  0},
    {170 ,170 ,  0},
    {170 ,255 ,  0},
    {255 , 84 ,  0},
    {255 ,170 ,  0},
    {255 ,255 ,  0},
    {  0 , 84 ,127},
    {  0 ,170 ,127},
    {  0 ,255 ,127},
    { 84 ,  0 ,127},
    { 84 , 84 ,127},
    { 84 ,170 ,127},
    { 84 ,255 ,127},
    {170 ,  0 ,127},
    {170 , 84 ,127},
    {170 ,170 ,127},
    {170 ,255 ,127},
    {255 ,  0 ,127},
    {255 , 84 ,127},
    {255 ,170 ,127},
    {255 ,255 ,127},
    {  0 , 84 ,255},
    {  0 ,170 ,255},
    {  0 ,255 ,255},
    { 84 ,  0 ,255},
    { 84 , 84 ,255},
    { 84 ,170 ,255},
    { 84 ,255 ,255},
    {170 ,  0 ,255},
    {170 , 84 ,255},
    {170 ,170 ,255},
    {170 ,255 ,255},
    {255 ,  0 ,255},
    {255 , 84 ,255},
    {255 ,170 ,255},
    { 42 ,  0 ,  0},
    { 84 ,  0 ,  0},
    {127 ,  0 ,  0},
    {170 ,  0 ,  0},
    {212 ,  0 ,  0},
    {255 ,  0 ,  0},
    {  0 , 42 ,  0},
    {  0 , 84 ,  0},
    {  0 ,127 ,  0},
    {  0 ,170 ,  0},
    {  0 ,212 ,  0},
    {  0 ,255 ,  0},
    {  0 ,  0 , 42},
    {  0 ,  0 , 84},
    {  0 ,  0 ,127},
    {  0 ,  0 ,170},
    {  0 ,  0 ,212},
    {  0 ,  0 ,255},
    {  0 ,  0 ,  0},
    { 36 , 36 , 36},
    { 72 , 72 , 72},
    {109 ,109 ,109},
    {145 ,145 ,145},
    {182 ,182 ,182},
    {218 ,218 ,218},
    {  0 ,113 ,188},
    { 80 ,182 ,188},
    {127 ,127 ,  0},
};

// draw_bboxes: uses effect_roi mapping from resize_uniform to map boxes back to original frame
void draw_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, object_rect effect_roi)
{
    static const char* class_names[] = { "sign" };

    // Draw directly on the input image (no imshow)
    cv::Mat image = bgr; // draw on original

    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(color_list[bbox.label][0], color_list[bbox.label][1], color_list[bbox.label][2]);
        cv::rectangle(image, cv::Rect(
            cv::Point((int)round((bbox.x1 - effect_roi.x) * width_ratio), (int)round((bbox.y1 - effect_roi.y) * height_ratio)),
            cv::Point((int)round((bbox.x2 - effect_roi.x) * width_ratio), (int)round((bbox.y2 - effect_roi.y) * height_ratio))
        ), color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)round((bbox.x1 - effect_roi.x) * width_ratio);
        int y = (int)round((bbox.y1 - effect_roi.y) * height_ratio) - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > image.cols) x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    }
    // intentionally do not call imshow here
}

// video_demo: reads frames, runs detector, draws boxes, writes output video
int video_demo(NanoDet& detector, const char* path)
{
    cv::Mat image;
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        fprintf(stderr, "Failed to open input video: %s\n", path);
        return -1;
    }

    int height = detector.input_size[0];
    int width = detector.input_size[1];

    // Get video properties
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0; // fallback
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Open output video writer (same codec as input if available)
    cv::VideoWriter writer("output_with_detections.avi", fourcc, fps, cv::Size(frame_width, frame_height));
    if (!writer.isOpened()) {
        // fallback to MJPG if input fourcc fails
        int fallback_fourcc = cv::VideoWriter::fourcc('M','J','P','G');
        writer.open("output_with_detections.avi", fallback_fourcc, fps, cv::Size(frame_width, frame_height));
        if (!writer.isOpened()) {
            fprintf(stderr, "Could not open the output video for write\n");
            return -1;
        }
    }

    while (true)
    {
        cap >> image;
        if (image.empty()) break;

        object_rect effect_roi;
        cv::Mat resized_img;
        resize_uniform(image, resized_img, cv::Size(width, height), effect_roi);

        auto results = detector.detect(resized_img, 0.2f, 0.45f);

        // Optional: print detections to console
        // printf("Detected %zu boxes\n", results.size());
        // for (const auto& box : results) {
        //     printf("label=%d score=%.2f box=[%.1f %.1f %.1f %.1f]\n",
        //         box.label, box.score, box.x1, box.y1, box.x2, box.y2);
        // }

        draw_bboxes(image, results, effect_roi); // Draw on the original frame

        writer.write(image); // Save frame to output video
    }

    writer.release();
    cap.release();
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: %s [input_video_path]\n", argv[0]);
        return -1;
    }

    // If you want to control number of CPU threads for ncnn, uncomment and set:
    // ncnn::set_omp_num_threads(4);

    NanoDet detector("./nanodet.ncnn.param", "./nanodet.ncnn.bin", true);

    const char* input_video = argv[1];
    int ret = video_demo(detector, input_video);
    return ret;
}
