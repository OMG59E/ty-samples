//
// Created  on 22-8-24.
//

#ifdef x86_64
#include <opencv2/opencv.hpp>
#endif

#include "models/yolov6_face.h"

#include "utils/device.h"
#include "utils/color.h"
#include "utils/image.h"
#include "bitmap_image.hpp"
#include "base_type.h"

int main(int argc, char** argv) {
    if (argc != 5) {
        DCL_APP_LOG(DCL_ERROR, "input param num(%d) must be == 5,\n"
                               "\t1 - sdk.config, 2 - input image path, 3 - model file path, 4 - result image path", argc);
        return -1;
    }

    const char *sdkCfg = argv[1];
    const char *imgPath = argv[2];
    const char *binFile = argv[3];
    const char *resFile = argv[4];

    bool enable_aipp = true;

    // sdk init
    dcl::deviceInit(sdkCfg);

    dcl::YoloV6Face model;
    std::vector<dcl::detection_t> detections;
    dcl::Mat vis, img;

    cv::Mat src = cv::imread(imgPath);
    if (src.empty()) {
        DCL_APP_LOG(DCL_ERROR, "Failed to read img, maybe filepath not exist -> %s", imgPath);
        goto exit;
    }
    vis.height = src.rows;
    vis.width = src.cols;
    vis.channels = src.channels();
    vis.original_height = src.rows;
    vis.original_width = src.cols;
    vis.data = src.data;
    vis.pixelFormat = DCL_PIXEL_FORMAT_BGR_888;
    vis.own = false;

    img.create(vis.h(), vis.w(), DCL_PIXEL_FORMAT_RGB_888_PLANAR);
    dcl::cvtColor(vis, img, IMAGE_COLOR_BGR888_TO_RGB888_PLANAR);

    // load model
    if (0 != model.load(binFile, enable_aipp)) {
        DCL_APP_LOG(DCL_ERROR, "Failed to load model");
        goto exit;
    }

    // inference
    if (0 != model.inference(img, detections)) {
        DCL_APP_LOG(DCL_ERROR, "Failed to inference");
        goto exit;
    }
    // unload
    if (0 != model.unload()) {
        DCL_APP_LOG(DCL_ERROR, "Failed to unload model");
        goto exit;
    }

    DCL_APP_LOG(DCL_INFO, "Found object num: %d", detections.size());

    // draw
    for (const auto& detection : detections) {
        dcl::rectangle(vis, dcl::Point(detection.box.x1, detection.box.y1),
                       dcl::Point(detection.box.x2, detection.box.y2), dcl::Color(0, 0, 255), 3);
        dcl::circle(vis, dcl::Point(detection.pts[0].x, detection.pts[0].y), 3, dcl::Color(0, 255, 0), -1);
        dcl::circle(vis, dcl::Point(detection.pts[1].x, detection.pts[1].y), 3, dcl::Color(0, 255, 0), -1);
        dcl::circle(vis, dcl::Point(detection.pts[2].x, detection.pts[2].y), 3, dcl::Color(0, 0, 255), -1);
        dcl::circle(vis, dcl::Point(detection.pts[3].x, detection.pts[3].y), 3, dcl::Color(255, 0, 0), -1);
        dcl::circle(vis, dcl::Point(detection.pts[4].x, detection.pts[4].y), 3, dcl::Color(255, 0, 0), -1);
    }

    cv::imwrite(resFile, src);

    exit:
    src.release();
    img.free();
    // sdk release
    dcl::deviceFinalize();
    return 0;
}
