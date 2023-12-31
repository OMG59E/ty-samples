//
// Created on 23-2-20.
//
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "models/yolov7_pose.h"
#include "utils/device.h"
#include "utils/utils.h"
#include "utils/image.h"
#include "bitmap_image.hpp"
#include "base_type.h"

const static int skeleton[38] = {16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13,
                                 6, 7, 6, 8, 7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7};

const static std::vector<ty::Color> palette = {ty::Color(255, 128, 0), ty::Color(255, 153, 51), ty::Color(255, 178, 102),
                                               ty::Color(230, 230, 0), ty::Color(255, 153, 255), ty::Color(153, 204, 255),
                                               ty::Color(255, 102, 255), ty::Color(255, 51, 255), ty::Color(102, 178, 255),
                                               ty::Color(51, 153, 255), ty::Color(255, 153, 153), ty::Color(255, 102, 102),
                                               ty::Color(255, 51, 51), ty::Color(153, 255, 153), ty::Color(102, 255, 102),
                                               ty::Color(51, 255, 51), ty::Color(0, 255, 0), ty::Color(0, 0, 255),
                                               ty::Color(255, 0, 0), ty::Color(255, 255, 255)};

const static std::vector<int> pose_limb_color = {9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 1};
const static std::vector<int> pose_kpt_color = {16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9};


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

    ty::YoloV7Pose model;
    std::vector<ty::detection_t> detections;
    ty::Mat img;

    cv::Mat src = cv::imread(imgPath);
    ty::Mat vis = cvMatToDclMat(src);
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


    for (const auto &detection: detections) {
        ty::rectangle(vis, ty::Point(detection.box.x1, detection.box.y1),
                      ty::Point(detection.box.x2, detection.box.y2), ty::Color(0, 0, 255), 3);

        for (int k=0; k<19; ++k) {
            const ty::Point& p1 = detection.kpts[skeleton[k * 2 + 0] - 1];
            const ty::Point& p2 = detection.kpts[skeleton[k * 2 + 1] - 1];
            if (p1.score < 0.5f || p2.score < 0.5f)
                continue;
            const ty::Color& color = palette[pose_limb_color[k]];
            ty::lineBres(vis, p1, p2, color, 3);
        }

        for (int k=0; k<17; ++k) {
            const ty::Point& p = detection.kpts[k];
            if (p.score < 0.5f)
                continue;
            const ty::Color& color = palette[pose_kpt_color[k]];
            ty::circle(vis, p, 5, color, -1);
        }
    }

    cv::imwrite(resFile, src);

exit:
    src.release();
    img.free();
    // sdk release
    ty::deviceFinalize();
    return 0;
}