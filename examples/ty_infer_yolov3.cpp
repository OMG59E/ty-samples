//
// Created  on 22-9-16.
//

#include "models/yolov3.h"
#include "utils/device.h"
#include "utils/image.h"
#include "utils/utils.h"
#include "bitmap_image.hpp"
#include "base_type.h"

int main(int argc, char** argv) {
    if (argc != 5) {
        printf("input param num(%d) must be == 5,\n"
               "\t1 - sdk.config, 2 - input image path, 3 - model file path, 4 - result image path\n", argc);
        return -1;
    }

    const char *sdkCfg = argv[1];
    const char *imgPath = argv[2];
    const char *binFile = argv[3];
    const char *resFile = argv[4];
    // sdk init
    ty::deviceInit(sdkCfg);

    ty::YoloV3 model;
    std::vector<ty::detection_t> detections;
    ty::Mat img;

    cv::Mat src = cv::imread(imgPath);
    if (src.empty()) {
        DCL_APP_LOG(DCL_ERROR, "Failed to read img, maybe filepath not exist -> %s", imgPath);
        goto exit;
    }

    img.create(src.rows, src.cols, DCL_PIXEL_FORMAT_BGR_888_PACKED);
    memcpy(img.data, src.data, img.size());

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
        cv::rectangle(src, cv::Point(detection.box.x1, detection.box.y1),
                      cv::Point(detection.box.x2, detection.box.y2), cv::Scalar(0, 0, 255), 2);
    }

    cv::imwrite(resFile, src);

exit:
    src.release();
    img.free();
    // sdk release
    ty::deviceFinalize();
    return 0;
}