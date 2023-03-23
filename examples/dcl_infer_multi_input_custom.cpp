//
// Created  on 22-9-21.
//

#ifdef x86_64
#include <opencv2/opencv.hpp>
#endif

#include "models/base/net_operator.h"

#include "utils/device.h"
#include "utils/color.h"
#include "utils/image.h"
#include "bitmap_image.hpp"
#include "base_type.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        DCL_APP_LOG(DCL_ERROR, "input param num(%d) must be == 4,\n"
                               "\t1 - sdk.config, 2 - input image path, 3 - model file path", argc);
        return -1;
    }

    const char *sdkCfg = argv[1];
    const char *imgPath = argv[2];
    const char *binFile = argv[3];

    bool enable_aipp = true;

    // sdk init
    dcl::deviceInit(sdkCfg);

    dcl::NetOperator net;
    std::vector<dcl::Tensor> vOutputTensors;
    dcl::Mat vis, img0, img1;

#ifdef x86_64
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
#else
    bitmap_image bmp(imgPath);
    vis.height = bmp.height();
    vis.width = bmp.width();
    vis.channels = bmp.bytes_per_pixel();
    vis.original_height = bmp.height();
    vis.original_width = bmp.width();
    vis.data = bmp.data();
    vis.pixelFormat = DCL_PIXEL_FORMAT_BGR_888;
    vis.own = false;
#endif
     img0.create(vis.h(), vis.w(), DCL_PIXEL_FORMAT_BGR_888_PLANAR);
     img1.create(vis.h(), vis.w(), DCL_PIXEL_FORMAT_RGB_888_PLANAR);
     dcl::cvtColor(vis, img0, IMAGE_COLOR_BGR888_TO_BGR888_PLANAR);
     dcl::cvtColor(vis, img1, IMAGE_COLOR_BGR888_TO_RGB888_PLANAR);

    // load model
    if (0 != net.load(binFile, enable_aipp)) {
        DCL_APP_LOG(DCL_ERROR, "Failed to load model");
        goto exit;
    }

    // inference
    if (0 != net.inference({img0, img1}, vOutputTensors)) {
        DCL_APP_LOG(DCL_ERROR, "Failed to inference");
        goto exit;
    }

    // unload
    if (0 != net.unload()) {
        DCL_APP_LOG(DCL_ERROR, "Failed to unload model");
        goto exit;
    }

    exit:
#ifdef x86_64
    src.release();
#endif
    img0.free();
    img1.free();

    // sdk release
    dcl::deviceFinalize();
    return 0;
}