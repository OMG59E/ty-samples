//
// Created on 23-2-21.
//

#include <opencv2/opencv.hpp>

#include "models/yolov5_seg.h"
#include "utils/device.h"
#include "utils/color.h"
#include "base_type.h"

const static std::vector<cv::Scalar> palette = {cv::Scalar(  0,   0, 255), cv::Scalar(  0, 255,   0), cv::Scalar(255,   0,   0),
                                                cv::Scalar(169, 169, 169), cv::Scalar(  0,   0, 139), cv::Scalar(  0,  69, 255),
                                                cv::Scalar( 30, 105, 210), cv::Scalar( 10, 215, 255), cv::Scalar(  0, 255, 255),
                                                cv::Scalar(  0, 128, 128), cv::Scalar(144, 238, 144), cv::Scalar(139, 139,   0),
                                                cv::Scalar(230, 216, 173), cv::Scalar(130,   0,  75), cv::Scalar(128,   0, 128),
                                                cv::Scalar(203, 192, 255), cv::Scalar(147,  20, 255), cv::Scalar(238, 130, 238)};



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

    dcl::YoloV5Seg model;
    std::vector<dcl::detection_t> detections;
    dcl::Mat tmp, img;

    cv::Mat mask;

    cv::Mat src = cv::imread(imgPath);
    if (src.empty()) {
        DCL_APP_LOG(DCL_ERROR, "Failed to read img, maybe filepath not exist -> %s", imgPath);
        goto exit;
    }
    tmp.height = src.rows;
    tmp.width = src.cols;
    tmp.channels = src.channels();
    tmp.original_height = src.rows;
    tmp.original_width = src.cols;
    tmp.data = src.data;
    tmp.pixelFormat = DCL_PIXEL_FORMAT_BGR_888;
    tmp.own = false;
    img.create(tmp.h(), tmp.w(), DCL_PIXEL_FORMAT_RGB_888_PLANAR);
    dcl::cvtColor(tmp, img, IMAGE_COLOR_BGR888_TO_RGB888_PLANAR);

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

    mask.create(src.rows, src.cols, CV_8UC1);
    for (const auto& detection : detections) {
        const cv::Scalar& color = palette[detection.cls % 18];
        cv::rectangle(src, cv::Point(detection.box.x1, detection.box.y1),
                       cv::Point(detection.box.x2, detection.box.y2), color, 2);
        memset(mask.data, 0, src.rows*src.cols);
        cv::fillPoly(mask, detection.contours, cv::Scalar(255));  // 填充mask
        // add
        for (int h=detection.box.y1; h<=detection.box.y2; ++h) {
            for (int w=detection.box.x1; w<=detection.box.x2; ++w) {
                if (mask.data[h * src.cols + w] == 255) {
                    src.data[h * src.cols * 3 + w * 3 + 0] = src.data[h * src.cols * 3 + w * 3 + 0] * 0.5f + color[0] * 0.5f;
                    src.data[h * src.cols * 3 + w * 3 + 1] = src.data[h * src.cols * 3 + w * 3 + 1] * 0.5f + color[1] * 0.5f;
                    src.data[h * src.cols * 3 + w * 3 + 2] = src.data[h * src.cols * 3 + w * 3 + 2] * 0.5f + color[2] * 0.5f;
                }
            }
        }
        cv::drawContours(src, detection.contours, -1, color, 2);
    }
    cv::imwrite(resFile, src);

exit:
    src.release();
    mask.release();
    img.free();

    // sdk release
    dcl::deviceFinalize();
    return 0;
}