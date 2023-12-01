//
// Created  on 22-9-22.
//

#include "models/resnet.h"
#include "utils/device.h"
#include "utils/utils.h"
#include "bitmap_image.hpp"
#include "base_type.h"


int main(int argc, char** argv) {
    if (argc != 4) {
        printf("input param num(%d) must be == 4,\n"
               "\t1 - sdk.config, 2 - input image path, 3 - model file path\n", argc);
        return -1;
    }

    const char *sdkCfg = argv[1];
    const char *imgPath = argv[2];
    const char *binFile = argv[3];

    // sdk init
    ty::deviceInit(sdkCfg);

    ty::ResNet model;
    std::vector<ty::classification_t> classifications;
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
    if (0 != model.inference(img, classifications)) {
        DCL_APP_LOG(DCL_ERROR, "Failed to inference");
        goto exit;
    }
    // unload
    if (0 != model.unload()) {
        DCL_APP_LOG(DCL_ERROR, "Failed to unload model");
        goto exit;
    }

    for (const auto& classification : classifications) {
        DCL_APP_LOG(DCL_INFO, "cls: %d, name: %s, conf: %03f", classification.cls,
                    classification.name.c_str(), classification.conf);
    }

exit:
    src.release();
    img.free();
    // sdk release
    ty::deviceFinalize();
    return 0;
}
