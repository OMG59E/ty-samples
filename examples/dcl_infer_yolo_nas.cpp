//
// Created by intellif on 23-5-26.
//
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core/core.hpp>

#include "models/yolo_nas.h"
#include "utils/device.h"
#include "utils/utils.h"
#include "utils/image.h"
#include "bitmap_image.hpp"
#include "base_type.h"

int main(int argc, char** argv) {
    if (argc != 5) {
        printf("input param num(%d) must be == 5,\n"
               "\t1 - sdk.config, 2 - input image path, 3 - model file path, 4 - result image path\n");
        return -1;
    }

    const char *sdkCfg = argv[1];
    const char *imgPath = argv[2];
    const char *binFile = argv[3];
    const char *resFile = argv[4];

    // sdk init
    dcl::deviceInit(sdkCfg);

    dcl::YoloNas model;
    std::vector<dcl::detection_t> detections;
    dcl::Mat img;

    cv::Mat src = cv::imread(imgPath);
    if (src.empty()) {
        DCL_APP_LOG(DCL_ERROR, "Failed to read img, maybe filepath not exist -> %s", imgPath);
        goto exit;
    }

    img = cvMatToDclMat(src);

    // load model
    if (0 != model.load(binFile)) {
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

    for (const auto& detection : detections) {
        DCL_APP_LOG(DCL_INFO, "object: %f %d %d %d %d %d", detection.conf, detection.cls,
                    detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2);
        dcl::rectangle(img, dcl::Point(detection.box.x1, detection.box.y1),
                       dcl::Point(detection.box.x2, detection.box.y2), dcl::Color(0, 0, 255), 3);
    }

    cv::imwrite(resFile, src);

    exit:
    src.release();
    // sdk release
    dcl::deviceFinalize();
    return 0;
}